# 🎬 Kino Sahnalari Subtitr — Telegram bot

Kino lavhasini (video yoki rasm) yuborasiz — bot ustidagi **subtitr, yozuv va watermark (logo)** larni olib tashlab, tozalangan faylni qaytaradi.

## Qanday ishlaydi

| Bosqich | Usul |
|---|---|
| Matn / subtitr | Har 3-kadrda [RapidOCR](https://github.com/RapidAI/RapidOCR) (PaddleOCR detektori, ONNX) matn joylarini topadi |
| Watermark / logo | Videodan 40 ta kadr olinadi; sahna o'zgarsa ham joyida qoladigan chegaralar (odatda burchaklarda) — logo deb topiladi |
| To'ldirish | OpenCV inpainting (TELEA) — atrofdagi piksellar bilan |
| Audio | ffmpeg orqali asl ovoz saqlanadi |
| O'z belgingiz | Olib tashlangan watermark turgan burchakka `BRAND_TEXT` (standart: `kino_sahnalari_uzsubtitr`) yarim shaffof yozuv sifatida qo'yiladi. Watermark topilmasa — yuqori o'ng burchak. |

Model fayllari `rapidocr-onnxruntime` paketi ichida keladi, GPU kerak emas.

### Rejimlar (`/mode`)
- **Hammasi** — matn + watermark (standart)
- **Faqat subtitr** — kadrning pastki 35% qismidagi matn
- **Barcha yozuvlar** — kadrning istalgan joyidagi matn
- **Faqat watermark / logo**

## O'rnatish

1. [@BotFather](https://t.me/BotFather) da `/newbot` orqali bot yarating va tokenni oling.
2. `.env.example` faylidan nusxa olib `.env` qiling va `BOT_TOKEN` ni yozing.

### Docker bilan (tavsiya etiladi)
```bash
docker build -t kino-bot .
docker run -d --restart=always --env-file .env --name kino-bot kino-bot
```

### Qo'lda
`ffmpeg` o'rnatilgan bo'lishi kerak (`sudo apt install ffmpeg`).
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python bot.py
```

## Cheklovlar
- Oddiy Telegram Bot API: bot **20 MB** gacha faylni yuklab oladi va **50 MB** gacha yubora oladi.
  Kattaroq fayllar uchun [lokal Bot API server](https://github.com/tdlib/telegram-bot-api) ni `--local` rejimida ishga tushirib, `.env` ga `TELEGRAM_API_URL=http://localhost:8081` yozing (2 GB gacha).
- Tezlik: 720p, 5 soniyali video oddiy CPU da ~15–20 soniya. 1 daqiqali video — bir necha daqiqa.
- Inpainting katta, murakkab fon ustidagi yozuvlarda biroz "surtilgan" iz qoldirishi mumkin.
  Yuqori sifat kerak bo'lsa, `cv2.inpaint` o'rniga LaMa yoki STTN kabi neyron modellarga almashtirish mumkin (GPU talab qilinadi).
- Yarim shaffof, butun kadr bo'ylab yoyilgan watermarklar statik tahlilda topilmasligi mumkin.

## Fayllar
- `bot.py` — Telegram bot (aiogram 3)
- `cleaner.py` — tasvir/videoni tozalash logikasi

⚠️ Botdan faqat sizga tegishli yoki foydalanishga ruxsat berilgan kontent bilan foydalaning — boshqalarning watermarkini olib tashlab qayta tarqatish mualliflik huquqini buzishi mumkin.
