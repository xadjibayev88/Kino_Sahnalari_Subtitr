"""
Kino lavhalaridagi yozuv (subtitr, matn) va watermark (logo) larni olib tashlash.

Ishlash prinsipi:
  1. Matn: har N-kadrda RapidOCR (PaddleOCR detektori, ONNX) bilan matn joylari topiladi.
     Subtitr bir necha kadr turadi, shuning uchun qo'shni namunalar maskasi birlashtiriladi.
  2. Watermark: videodan ~40 ta kadr olinib, qaysi piksellar deyarli har doim "chegara"
     (edge) bo'lib qolayotgani hisoblanadi. Sahna o'zgarsa ham o'zgarmaydigan narsa --
     bu ustiga qo'yilgan logo/watermark.
  3. Topilgan joylar OpenCV inpainting bilan atrofdagi piksellar asosida to'ldiriladi.
  4. Asl audio ffmpeg orqali qayta qo'shiladi.
"""

from __future__ import annotations

import json
import os
import logging
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[float], None]]

MODE_ALL = "all"          # matn + watermark
MODE_SUBS = "subs"        # faqat pastki qismdagi subtitr
MODE_TEXT = "text"        # kadrning istalgan joyidagi matn
MODE_WATERMARK = "wm"     # faqat watermark/logo

MODES = {
    MODE_ALL: "Hammasi (matn + watermark)",
    MODE_SUBS: "Faqat subtitr (pastki qism)",
    MODE_TEXT: "Barcha yozuvlar",
    MODE_WATERMARK: "Faqat watermark / logo",
}


@dataclass
class CleanOptions:
    mode: str = MODE_ALL
    det_every: int = 3            # har nechta kadrda matn qidirish
    det_max_side: int = 960       # detektor uchun kadrni kichraytirish
    subs_region: float = 0.35     # MODE_SUBS: pastdan qancha qism (0..1)
    mask_pad: int = 8             # maskani kengaytirish (px)
    inpaint_radius: int = 5
    wm_samples: int = 40
    wm_edge_ratio: float = 0.65   # piksel shuncha kadrda edge bo'lsa -> static
    wm_border: float = 0.30       # watermark faqat chekka qismlarda qidiriladi (0 = butun kadr)
    crf: int = 20
    # O'z belgingiz (masalan Instagram kanal nomi). Bo'sh bo'lsa qo'yilmaydi.
    brand_text: str = ""
    brand_opacity: float = 0.75
    brand_size: float = 0.045     # matn balandligi = kadr qisqa tomonining shu qismi


# --------------------------------------------------------------------------- #
# Matn detektori
# --------------------------------------------------------------------------- #

class TextDetector:
    _instance: "TextDetector | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self._run_lock = threading.Lock()

    @classmethod
    def get(cls) -> "TextDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = TextDetector()
            return cls._instance

    def detect(self, img: np.ndarray, max_side: int = 960) -> list[np.ndarray]:
        """Matn joylarini to'rtburchak poligonlar ro'yxati sifatida qaytaradi (asl o'lchamda)."""
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
        with self._run_lock:
            out = self._engine.text_detector(small)
        boxes = out[0] if isinstance(out, tuple) else out
        if boxes is None:
            return []
        return [np.asarray(b, dtype=np.float32) / scale for b in boxes]


def boxes_to_mask(boxes, shape, opts: CleanOptions) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    y_min = int(h * (1 - opts.subs_region)) if opts.mode == MODE_SUBS else 0
    for b in boxes:
        pts = b.astype(np.int32)
        if opts.mode == MODE_SUBS and pts[:, 1].mean() < y_min:
            continue
        cv2.fillPoly(mask, [pts], 255)
    if opts.mask_pad > 0 and mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * opts.mask_pad + 1,) * 2)
        mask = cv2.dilate(mask, k)
    return mask


# --------------------------------------------------------------------------- #
# Watermark detektori (statik elementlar)
# --------------------------------------------------------------------------- #

def detect_static_watermark(frames: list[np.ndarray], opts: CleanOptions) -> np.ndarray:
    h, w = frames[0].shape[:2]
    empty = np.zeros((h, w), np.uint8)
    if len(frames) < 8:
        return empty

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    # Sahna yetarlicha o'zgaradimi? Kamera qimirlamasa fon ham "statik" bo'lib qoladi.
    stack = np.stack(grays).astype(np.float32)
    motion = float(stack.std(axis=0).mean())
    if motion < 6.0:
        log.info("Video juda statik (std=%.1f), watermark aniqlash o'tkazib yuborildi", motion)
        return empty

    edge_count = np.zeros((h, w), np.float32)
    for g in grays:
        edge_count += cv2.Canny(g, 60, 160) > 0
    ratio = edge_count / len(grays)
    mask = (ratio >= opts.wm_edge_ratio).astype(np.uint8) * 255

    # Faqat chekka hududlarda (logo odatda burchakda bo'ladi)
    if opts.wm_border > 0:
        bx, by = int(w * opts.wm_border), int(h * opts.wm_border)
        mask[by:h - by, bx:w - bx] = 0

    # Edge piksellarini yaxlit obyektga aylantirish
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    # Juda kichik shovqin va juda katta (fon) bo'laklarni olib tashlash
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    frame_area = h * w
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if 0.0002 * frame_area <= area <= 0.06 * frame_area:
            # Logo ichidagi bo'shliqlar va zaif chiziqlar ham qoplanishi uchun qavariq qobiq
            pts = cv2.findNonZero((labels == i).astype(np.uint8))
            cv2.fillConvexPoly(out, cv2.convexHull(pts), 255)
    if out.any():
        out = cv2.dilate(out, np.ones((7, 7), np.uint8))
    return out


# --------------------------------------------------------------------------- #
# O'z belgingizni (brand) qo'yish
# --------------------------------------------------------------------------- #

_FONT_PATHS = [
    os.getenv("BRAND_FONT", ""),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _load_font(px: int):
    from PIL import ImageFont

    for path in _FONT_PATHS:
        if path and os.path.exists(path):
            return ImageFont.truetype(path, px)
    return ImageFont.load_default(size=px)


def pick_corner(mask: np.ndarray | None, default: str = "tr") -> str:
    """Olib tashlangan watermark qaysi burchakda bo'lganini aniqlaydi: tl/tr/bl/br."""
    if mask is None or not mask.any():
        return default
    h, w = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    return ("t" if cy < h / 2 else "b") + ("l" if cx < w / 2 else "r")


class Brand:
    """Matnni bir marta RGBA rasmga chizib, har kadrga tez alpha-blend qiladi."""

    def __init__(self, text: str, frame_w: int, frame_h: int, corner: str, opts: CleanOptions):
        from PIL import Image, ImageDraw

        px = max(14, int(min(frame_w, frame_h) * opts.brand_size))
        font = _load_font(px)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        l, t, r, b = tmp.textbbox((0, 0), text, font=font, stroke_width=max(1, px // 12))
        tw, th = r - l, b - t
        # Juda uzun bo'lsa, kadr kengligining 60% iga sig'diramiz
        if tw > frame_w * 0.6:
            px = max(12, int(px * frame_w * 0.6 / tw))
            font = _load_font(px)
            l, t, r, b = tmp.textbbox((0, 0), text, font=font, stroke_width=max(1, px // 12))
            tw, th = r - l, b - t
        pad = max(2, px // 6)
        img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((pad - l, pad - t), text, font=font, fill=(255, 255, 255, 255),
               stroke_width=max(1, px // 12), stroke_fill=(0, 0, 0, 255))
        rgba = np.asarray(img).astype(np.float32)
        self.rgb = rgba[..., 2::-1]                       # RGB -> BGR
        self.alpha = (rgba[..., 3:4] / 255.0) * opts.brand_opacity
        bh, bw = self.alpha.shape[:2]
        mx, my = int(frame_w * 0.03), int(frame_h * 0.03)
        self.x = mx if corner[1] == "l" else frame_w - bw - mx
        self.y = my if corner[0] == "t" else frame_h - bh - my
        self.x, self.y = max(0, self.x), max(0, self.y)
        self.bw, self.bh = min(bw, frame_w - self.x), min(bh, frame_h - self.y)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        y, x, h, w = self.y, self.x, self.bh, self.bw
        roi = frame[y:y + h, x:x + w].astype(np.float32)
        a = self.alpha[:h, :w]
        frame[y:y + h, x:x + w] = (roi * (1 - a) + self.rgb[:h, :w] * a).astype(np.uint8)
        return frame


# --------------------------------------------------------------------------- #
# Rasm
# --------------------------------------------------------------------------- #

def clean_image(src: str, dst: str, opts: CleanOptions) -> bool:
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Rasmni o'qib bo'lmadi")
    mask = np.zeros(img.shape[:2], np.uint8)
    if opts.mode != MODE_WATERMARK:
        mask |= boxes_to_mask(TextDetector.get().detect(img, opts.det_max_side), img.shape, opts)
    else:
        # Bitta rasmda statik tahlil iloji yo'q -- matnli logolarni OCR orqali chekkalardan izlaymiz
        o = CleanOptions(**{**opts.__dict__, "mode": MODE_TEXT})
        m = boxes_to_mask(TextDetector.get().detect(img, opts.det_max_side), img.shape, o)
        h, w = m.shape
        bx, by = int(w * opts.wm_border), int(h * opts.wm_border)
        m[by:h - by, bx:w - bx] = 0
        mask |= m
    found = bool(mask.any())
    if found:
        img = cv2.inpaint(img, mask, opts.inpaint_radius, cv2.INPAINT_TELEA)
    if opts.brand_text:
        h, w = img.shape[:2]
        # Rasmda watermark joyi: chekkadagi topilgan joylar (pastki subtitr hisobga olinmaydi)
        edge = mask.copy()
        edge[int(h * 0.7):, int(w * 0.2):int(w * 0.8)] = 0
        img = Brand(opts.brand_text, w, h, pick_corner(edge), opts).apply(img)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return found


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #

def _probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def _has_audio(info: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def _read_samples(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total > 0:
        for idx in np.linspace(0, max(total - 1, 0), num=min(n, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    cap.release()
    return frames


def clean_video(src: str, dst: str, opts: CleanOptions, progress: ProgressCb = None) -> bool:
    info = _probe(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError("Videoni ochib bo'lmadi")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    use_text = opts.mode != MODE_WATERMARK
    use_wm = opts.mode in (MODE_ALL, MODE_WATERMARK)

    # --- Watermark maskasi
    wm_mask = np.zeros((h, w), np.uint8)
    if use_wm:
        wm_mask = detect_static_watermark(_read_samples(src, opts.wm_samples), opts)
        log.info("Watermark maskasi: %d piksel", int((wm_mask > 0).sum()))

    # --- 1-o'tish: har N-kadrda matn joylarini aniqlash
    det = TextDetector.get() if use_text else None
    text_masks: dict[int, np.ndarray] = {}
    if use_text:
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % opts.det_every == 0:
                ok, frame = cap.retrieve()
                if ok:
                    m = boxes_to_mask(det.detect(frame, opts.det_max_side), frame.shape, opts)
                    if m.any():
                        text_masks[idx] = m
            idx += 1
            if progress and total and idx % 15 == 0:
                progress(0.5 * idx / total)
        total = total or idx
        cap.release()
        cap = cv2.VideoCapture(src)

    found = bool(wm_mask.any() or text_masks)
    brand = Brand(opts.brand_text, w, h, pick_corner(wm_mask), opts) if opts.brand_text else None

    # --- 2-o'tish: inpaint va ffmpeg ga yozish
    has_audio = _has_audio(info)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
        "-i", src,
        "-map", "0:v:0",
    ]
    if has_audio:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "160k"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(opts.crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", dst,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    step = opts.det_every
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = wm_mask.copy()
            if text_masks:
                prev_s = (idx // step) * step
                for s in (prev_s - step, prev_s, prev_s + step):
                    m = text_masks.get(s)
                    if m is not None:
                        mask |= m
            if mask.any():
                frame = cv2.inpaint(frame, mask, opts.inpaint_radius, cv2.INPAINT_TELEA)
            if brand is not None:
                frame = brand.apply(frame)
            proc.stdin.write(frame.tobytes())
            idx += 1
            if progress and total and idx % 15 == 0:
                base = 0.5 if use_text else 0.0
                progress(base + (1 - base) * idx / total)
    finally:
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg xatosi: {err[-500:]}")
    if progress:
        progress(1.0)
    return found
