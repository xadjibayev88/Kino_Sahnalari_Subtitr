"""
Telegram bot: kino lavhasidan yozuv va watermark larni olib tashlaydi.

Ishga tushirish:
    export BOT_TOKEN=123456:ABC...
    python bot.py
"""

from __future__ import annotations

import asyncio
import re
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Awaitable, Callable

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

import cleaner
import insta
from cleaner import MODES, CleanOptions

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bot")

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
# Lokal Bot API server (katta fayllar uchun, 2 GB gacha). Bo'sh bo'lsa -- api.telegram.org
LOCAL_API_URL = os.getenv("TELEGRAM_API_URL", "").rstrip("/")
MAX_JOBS = int(os.getenv("MAX_JOBS", "1"))
# Tozalangan videoga qo'yiladigan o'z belgingiz. Bo'sh qilsangiz qo'yilmaydi.
BRAND_TEXT = os.getenv("BRAND_TEXT", "kino_sahnalari_uzsubtitr").strip()
MAX_DURATION = int(os.getenv("MAX_DURATION", "180"))  # soniya
# Oddiy Bot API: yuklab olish 20 MB, yuborish 50 MB
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "2000" if LOCAL_API_URL else "20"))
# Havoladan yuklashda Telegram 20 MB cheklovi yo'q
MAX_DOWNLOAD_MB_LINK = int(os.getenv("MAX_DOWNLOAD_MB_LINK", "200"))

user_modes: dict[int, str] = {}
job_sem = asyncio.Semaphore(MAX_JOBS)
dp = Dispatcher()


def mode_keyboard(current: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=("✅ " if k == current else "") + v, callback_data=f"mode:{k}")]
        for k, v in MODES.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def get_mode(user_id: int) -> str:
    return user_modes.get(user_id, cleaner.MODE_ALL)


@dp.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    await msg.answer(
        "👋 Salom! Menga kino lavhasini (video yoki rasm) yoki <b>Instagram havolasini</b> yuboring — "
        "ustidagi <b>yozuv, subtitr va watermark</b>larni olib tashlab qaytaraman.\n\n"
        f"📏 Video: {MAX_DURATION} soniyagacha, {MAX_DOWNLOAD_MB} MB gacha.\n"
        "⚙️ Rejimni tanlash: /mode\n"
        "ℹ️ Yordam: /help",
        parse_mode="HTML",
    )


@dp.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(
        "<b>Qanday ishlaydi?</b>\n"
        "• Matn/subtitr — har bir kadrda sun'iy intellekt (OCR detektori) yozuvni topadi.\n"
        "• Watermark — sahna o'zgarsa ham joyida qoladigan logo topiladi "
        "(odatda burchaklarda).\n"
        "• Topilgan joylar atrofdagi piksellar bilan to'ldiriladi.\n\n"
        "<b>Rejimlar</b> (/mode):\n"
        + "\n".join(f"• {v}" for v in MODES.values())
        + "\n\n💡 Maslahat: videoni <i>fayl (document)</i> sifatida yuborsangiz sifati yo'qolmaydi.",
        parse_mode="HTML",
    )


@dp.message(Command("mode"))
async def cmd_mode(msg: Message) -> None:
    await msg.answer("Rejimni tanlang:", reply_markup=mode_keyboard(get_mode(msg.from_user.id)))


@dp.callback_query(F.data.startswith("mode:"))
async def on_mode(cb: CallbackQuery) -> None:
    mode = cb.data.split(":", 1)[1]
    if mode in MODES:
        user_modes[cb.from_user.id] = mode
        await cb.message.edit_reply_markup(reply_markup=mode_keyboard(mode))
        await cb.answer(f"Tanlandi: {MODES[mode]}")
    else:
        await cb.answer()


def _pick_media(msg: Message):
    """(file_id, file_size, kind, duration) qaytaradi. kind: 'video' | 'image'."""
    if msg.video:
        return msg.video.file_id, msg.video.file_size, "video", msg.video.duration
    if msg.animation:
        return msg.animation.file_id, msg.animation.file_size, "video", msg.animation.duration
    if msg.video_note:
        return msg.video_note.file_id, msg.video_note.file_size, "video", msg.video_note.duration
    if msg.photo:
        p = msg.photo[-1]
        return p.file_id, p.file_size, "image", 0
    if msg.document:
        mime = msg.document.mime_type or ""
        if mime.startswith("video/"):
            return msg.document.file_id, msg.document.file_size, "video", 0
        if mime.startswith("image/"):
            return msg.document.file_id, msg.document.file_size, "image", 0
    return None


async def _download(bot: Bot, file_id: str, dst: Path) -> None:
    f = await bot.get_file(file_id)
    # Lokal API server --local rejimida fayl yo'lini to'g'ridan-to'g'ri beradi
    if LOCAL_API_URL and f.file_path and os.path.isabs(f.file_path) and os.path.exists(f.file_path):
        shutil.copy(f.file_path, dst)
    else:
        await bot.download_file(f.file_path, destination=dst)


@dp.message(F.video | F.animation | F.video_note | F.photo | F.document)
async def on_media(msg: Message, bot: Bot) -> None:
    media = _pick_media(msg)
    if not media:
        await msg.answer("Iltimos, video yoki rasm yuboring.")
        return
    file_id, size, kind, duration = media

    if size and size > MAX_DOWNLOAD_MB * 1024 * 1024:
        await msg.answer(
            f"❌ Fayl juda katta ({size / 1024 / 1024:.1f} MB). Maksimum: {MAX_DOWNLOAD_MB} MB.\n"
            "Videoni qisqartirib yoki siqib qayta yuboring."
        )
        return
    if kind == "video" and duration and duration > MAX_DURATION:
        await msg.answer(f"❌ Video juda uzun ({duration} s). Maksimum: {MAX_DURATION} soniya.")
        return

    async def fetch(tmp: Path) -> tuple[Path, str]:
        src = tmp / ("input.mp4" if kind == "video" else "input.jpg")
        await _download(bot, file_id, src)
        return src, kind

    await _process(msg, fetch)


Fetcher = Callable[[Path], Awaitable[tuple[Path, str]]]


async def _process(msg: Message, fetch: Fetcher) -> None:
    """Faylni olish (fetch), tozalash, o'z belgisini qo'yish va qaytarib yuborish."""
    mode = get_mode(msg.from_user.id)
    opts = CleanOptions(mode=mode, brand_text=BRAND_TEXT)
    status = await msg.answer(
        f"⏳ Navbatda... (rejim: {MODES[mode]})" if job_sem.locked() else "⏳ Yuklab olinmoqda..."
    )

    async with job_sem:
        tmp = Path(tempfile.mkdtemp(prefix="kino_"))
        try:
            await _safe_edit(status, "⏳ Yuklab olinmoqda...")
            src, kind = await fetch(tmp)
            dst = tmp / ("output.mp4" if kind == "video" else "output.jpg")
            await status.edit_text("🧹 Tozalanmoqda... 0%")

            loop = asyncio.get_running_loop()
            last = {"t": 0.0, "p": -1}

            def progress(p: float) -> None:
                pct = int(p * 100)
                now = time.monotonic()
                if pct != last["p"] and now - last["t"] > 3:
                    last.update(t=now, p=pct)
                    asyncio.run_coroutine_threadsafe(
                        _safe_edit(status, f"🧹 Tozalanmoqda... {pct}%"), loop
                    )

            if kind == "video":
                found = await asyncio.to_thread(cleaner.clean_video, str(src), str(dst), opts, progress)
            else:
                found = await asyncio.to_thread(cleaner.clean_image, str(src), str(dst), opts)

            caption = "✅ Tayyor!" if found else "ℹ️ Yozuv yoki watermark topilmadi, faqat belgingiz qo'yildi."
            await _safe_edit(status, "📤 Yuborilmoqda...")
            out = FSInputFile(dst)
            if kind == "video":
                out_mb = dst.stat().st_size / 1024 / 1024
                if out_mb > 50 and not LOCAL_API_URL:
                    await status.edit_text(
                        f"❌ Natija {out_mb:.1f} MB — Telegram bot 50 MB dan katta fayl yubora olmaydi."
                    )
                    return
                await msg.answer_video(out, caption=caption, supports_streaming=True)
            else:
                await msg.answer_photo(out, caption=caption)
                await msg.answer_document(FSInputFile(dst, filename="tozalangan.jpg"))
            await status.delete()
        except insta.InstaError as e:
            await _safe_edit(status, f"❌ {e}")
        except Exception as e:  # noqa: BLE001
            log.exception("Ishlov berishda xato")
            await _safe_edit(status, f"❌ Xatolik yuz berdi: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@dp.message(F.text.regexp(insta.URL_RE))
async def on_instagram(msg: Message) -> None:
    url = re.search(insta.URL_RE, msg.text).group(0)

    async def fetch(tmp: Path) -> tuple[Path, str]:
        return await asyncio.to_thread(insta.download, url, tmp, MAX_DURATION, MAX_DOWNLOAD_MB_LINK)

    await _process(msg, fetch)


async def _safe_edit(message: Message, text: str) -> None:
    try:
        await message.edit_text(text)
    except Exception:  # noqa: BLE001
        pass


@dp.message()
async def fallback(msg: Message) -> None:
    await msg.answer("Menga video, rasm yoki Instagram havolasini yuboring 🎬  (/help)")


async def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("BOT_TOKEN muhit o'zgaruvchisi o'rnatilmagan (.env faylga yozing).")
    session = None
    if LOCAL_API_URL:
        session = AiohttpSession(api=TelegramAPIServer.from_base(LOCAL_API_URL, is_local=True))
    bot = Bot(BOT_TOKEN, session=session)
    # Modelni oldindan yuklab qo'yamiz, birinchi so'rov tez bo'lsin
    await asyncio.to_thread(cleaner.TextDetector.get)
    log.info("Bot ishga tushdi")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
