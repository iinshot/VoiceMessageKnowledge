import os
from dotenv import load_dotenv
import logging
import random
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler
)
from converter import convert_ogg_to_wav
from work_with_files import (
    read_questions,
    get_next_counter,
    format_question_for_filename
)

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)
MAIN_MENU, CHOOSE_DIFFICULTY, WAITING_VOICE = range(3)
DIFFICULTY_FILES = {
    "Лёгкий": "question_files/questions_easy.csv",
    "Средний": "question_files/questions_medium.csv",
    "Сложный": "question_files/questions_hard.csv",
}

async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        ("start", "Начать"),
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [["Выбрать уровень сложности", "Я всемогущий"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Привет!\n"
        "Я бот, который задаёт вопросы по BigData.\n\n"
        "Выберите действие:",
        reply_markup=reply_markup
    )
    return MAIN_MENU

async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text

    if text == "Выбрать уровень сложности":
        keyboard = [["Лёгкий", "Средний", "Сложный"]]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        await update.message.reply_text(
            "Выберите уровень сложности:",
            reply_markup=reply_markup
        )
        return CHOOSE_DIFFICULTY


    elif text == "Я всемогущий":
        context.user_data["difficulty"] = "всемогущий"
        all_questions = set()

        for path in DIFFICULTY_FILES.values():
            all_questions.update(read_questions(path))

        question = random.choice(list(all_questions))
        context.user_data["current_question"] = question
        await update.message.reply_text(
            f"Отлично! Готов к самым сложным вопросам.\n\n"
            f"Вопрос: {question}\n\n"
            "Отправь голосовое сообщение с ответом",
            reply_markup=ReplyKeyboardRemove()
        )
        return WAITING_VOICE

async def handle_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    difficulty = update.message.text
    context.user_data["difficulty"] = difficulty
    questions = read_questions(DIFFICULTY_FILES[difficulty])
    question = random.choice(list(questions))
    context.user_data["current_question"] = question

    await update.message.reply_text(
        f"Выбран уровень: {difficulty}\n\n"
        f"Вопрос: {question}\n\n"
        "Отправь голосовое сообщение с ответом",
        reply_markup=ReplyKeyboardRemove()
    )
    return WAITING_VOICE

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    logger.info(f"Голосовое сообщение от {user.first_name} (id={user.id})")

    await update.message.reply_text("Получил голосовое, конвертирую...")

    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)

    question = context.user_data.get("current_question", "unknown")
    counter = get_next_counter(question)
    question_part = format_question_for_filename(question)
    filename = f"voice_{counter}_{question_part}"

    ogg_path = os.path.join(AUDIO_DIR, f"{filename}.ogg")
    wav_path = os.path.join(AUDIO_DIR, f"{filename}.wav")

    await file.download_to_drive(ogg_path)
    logger.info(f"Скачан файл: {ogg_path}")

    success = convert_ogg_to_wav(ogg_path, wav_path)

    if success:
        await update.message.reply_text("Конвертация завершена!")
        with open(wav_path, "rb") as wav_file:
            await update.message.reply_audio(
                audio=wav_file,
                filename=f"{filename}.wav",
            )
        logger.info(f"WAV отправлен пользователю: {wav_path}")
    else:
        await update.message.reply_text("Ошибка при конвертации. Попробуй ещё раз.")

    for path in [ogg_path]:
        if os.path.exists(path):
            os.remove(path)

    return WAITING_VOICE

async def handle_unexpected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Отправь голосовое сообщение"
    )

def main() -> None:
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .post_init(post_init)
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_main_menu)
            ],
            CHOOSE_DIFFICULTY: [
                MessageHandler(filters.Regex("^(Лёгкий|Средний|Сложный)$"), handle_difficulty)
            ],
            WAITING_VOICE: [
                MessageHandler(filters.VOICE, handle_voice),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unexpected)
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    app.add_handler(conv_handler)

    logger.info("Бот запущен...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()