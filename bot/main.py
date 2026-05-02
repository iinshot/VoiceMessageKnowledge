import os
from dotenv import load_dotenv
import logging
import random
from datetime import datetime
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    ForceReply
)
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
from services_client import transcribe_wav, score_answer

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

AUDIO_DIR = os.getenv("STT_RECORD_DIR")
ANSWER_DIR = os.getenv("STT_OUTPUT_DIR")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(ANSWER_DIR, exist_ok=True)
MAIN_MENU, CHOOSE_DIFFICULTY, WAITING_VOICE, WAITING_CORRECTION = range(4)
DIFFICULTY_FILES = {
    "Лёгкий": "data/question_files/questions_easy.csv",
    "Средний": "data/question_files/questions_medium.csv",
    "Сложный": "data/question_files/questions_hard.csv",
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
        all_questions = {}
        for path in DIFFICULTY_FILES.values():
            all_questions.update(read_questions(path))
        question = random.choice(list(all_questions.keys()))
        context.user_data["current_question"] = question
        context.user_data["reference_answer"] = all_questions[question]
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
    question = random.choice(list(questions.keys()))
    context.user_data["current_question"] = question
    context.user_data["reference_answer"] = questions[question]

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

    await update.message.reply_text("Получил голосовое, конвертирую")

    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)

    question = context.user_data.get("current_question", "unknown")
    counter = get_next_counter(question)
    question_part = format_question_for_filename(question)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"voice_{counter}_{question_part}_{timestamp}"

    ogg_path = os.path.join(AUDIO_DIR, f"{filename}.ogg")
    wav_path = os.path.join(AUDIO_DIR, f"{filename}.wav")

    await file.download_to_drive(ogg_path)
    logger.info(f"Скачан файл: {ogg_path}")

    success = convert_ogg_to_wav(ogg_path, wav_path)

    if not success:
        await update.message.reply_text("Ошибка при конвертации. Попробуй ещё раз.")
        return WAITING_VOICE

    await update.message.reply_text("Конвертация завершена! Распознаю речь...")

    try:
        recognized_text = await transcribe_wav(wav_path)
    except Exception as e:
        logger.error(f"Ошибка STT: {e}")
        await update.message.reply_text("Ошибка распознавания речи. Попробуй ещё раз.")
        return WAITING_VOICE
    finally:
        for path in [ogg_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)

    txt_path = os.path.join(ANSWER_DIR, f"{filename}.txt")
    context.user_data["txt_path"] = txt_path
    context.user_data["recognized_text"] = recognized_text

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(recognized_text)

    keyboard = [["Корректно", "Исправить"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        f"Распознанный текст:\n\n{recognized_text}\n\nКорректно ли бот понял вас?",
        reply_markup=reply_markup
    )
    return WAITING_CORRECTION

async def handle_unexpected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Отправь голосовое сообщение"
    )

async def handle_correction_choice(update, context):
    text = update.message.text
    if text == "Корректно":
        recognized_text = context.user_data.get("recognized_text", "")
        return await _finalize_answer(update, context, recognized_text)
    elif text == "Исправить":
        recognized_text = context.user_data.get("recognized_text", "")
        await update.message.reply_text(
            "Скопируйте текст ниже, исправьте и отправьте исправленную версию:",
            reply_markup=ForceReply(selective=True)
        )
        await update.message.reply_text(recognized_text)
        return WAITING_CORRECTION

async def handle_corrected_text(update, context):
    corrected_text = update.message.text
    return await _finalize_answer(update, context, corrected_text)

async def _finalize_answer(update, context, final_text):
    txt_path = context.user_data.get("txt_path", "")
    question = context.user_data.get("current_question", "")
    reference = context.user_data.get("reference_answer", "")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    try:
        result = await score_answer(question, reference, final_text)
        grade = result["grade"]
        await update.message.reply_text(
            f"Ответ принят!\n\n"
            f"Оценка: {grade}/5\n"
        )
    except Exception as e:
        logger.error(f"Ошибка оценки: {e}")
        await update.message.reply_text("Ответ принят! (оценка недоступна)")

    keyboard = [["Выбрать уровень сложности", "Я всемогущий"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Выберите следующее действие:", reply_markup=reply_markup)
    return MAIN_MENU

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
            WAITING_CORRECTION: [
                MessageHandler(filters.Regex("^(Корректно|Исправить)$"), handle_correction_choice),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_corrected_text)
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    app.add_handler(conv_handler)

    logger.info("Бот запущен...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()