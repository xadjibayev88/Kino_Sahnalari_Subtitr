"""Instagram havolasidan video/rasmni yuklab olish (yt-dlp orqali)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

URL_RE = r"https?://(?:www\.)?instagram\.com/(?:[\w.]+/)?(?:reel|reels|p|tv)/[\w-]+[^\s]*"

# Ixtiyoriy: Instagram cookies.txt fayli (Netscape format). Instagram serverdan
# login qilmasdan yuklashni ko'pincha bloklaydi -- shunda cookie kerak bo'ladi.
COOKIES_FILE = os.getenv("INSTAGRAM_COOKIES_FILE", "")
COOKIES_TEXT = os.getenv("INSTAGRAM_COOKIES", "")


class InstaError(Exception):
    """Foydalanuvchiga ko'rsatiladigan tushunarli xato."""


def _cookie_path(tmp: Path) -> str | None:
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        return COOKIES_FILE
    if COOKIES_TEXT.strip():
        p = tmp / "cookies.txt"
        p.write_text(COOKIES_TEXT.replace("\\n", "\n"))
        return str(p)
    return None


def download(url: str, tmp: Path, max_duration: int, max_mb: int) -> tuple[Path, str]:
    """Havoladagi birinchi videoni (yoki rasmni) yuklab oladi. (path, 'video'|'image') qaytaradi."""
    import yt_dlp

    opts = {
        "outtmpl": str(tmp / "insta.%(ext)s"),
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "playlist_items": "1",
        "quiet": True,
        "no_warnings": True,
        "max_filesize": max_mb * 1024 * 1024,
        "socket_timeout": 30,
    }
    cookies = _cookie_path(tmp)
    if cookies:
        opts["cookiefile"] = cookies

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist" and info.get("entries"):
                info = next(e for e in info["entries"] if e)
            dur = info.get("duration") or 0
            if dur and dur > max_duration:
                raise InstaError(f"Video juda uzun ({int(dur)} s). Maksimum: {max_duration} soniya.")
            ydl.process_info(info)
    except InstaError:
        raise
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        log.warning("Instagram yuklashda xato: %s", e)
        if "login" in msg or "rate" in msg or "cookies" in msg or "401" in msg or "403" in msg:
            raise InstaError(
                "Instagram bu videoni berishni rad etdi (login talab qilmoqda yoki server bloklangan). "
                "Videoni o'zingiz yuklab, botga fayl qilib yuboring."
            ) from e
        if "private" in msg:
            raise InstaError("Bu yopiq (private) akkaunt — yuklab bo'lmaydi.") from e
        if "no video" in msg:
            raise InstaError("Bu postda video topilmadi.") from e
        raise InstaError("Instagramdan yuklab bo'lmadi. Havolani tekshirib, qayta urinib ko'ring.") from e

    files = sorted(p for p in tmp.glob("insta.*") if p.suffix not in (".part", ".ytdl"))
    if not files:
        raise InstaError("Instagramdan fayl yuklanmadi.")
    f = files[0]
    kind = "image" if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".heic") else "video"
    return f, kind
