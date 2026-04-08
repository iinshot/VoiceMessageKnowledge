import logging
import subprocess
import os

logger = logging.getLogger(__name__)

def convert_ogg_to_wav(ogg_path: str, wav_path: str) -> bool:
    """
    Конвертирует .ogg файл в .wav с помощью ffmpeg.

    Args:
        ogg_path: путь к входному .ogg файлу
        wav_path: путь к выходному .wav файлу

    Returns:
        True если конвертация прошла успешно, False если ошибка
    """
    if not os.path.exists(ogg_path):
        logger.error(f"Файл не найден: {ogg_path}")
        return False

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", ogg_path,
                "-vn",
                "-ar", "16000",
                "-ac", "1",
                wav_path
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info(f"Конвертация успешна: {ogg_path} -> {wav_path}")
            return True
        else:
            logger.error(f"ffmpeg ошибка: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout — файл слишком большой или завис")
        return False

    except Exception as e:
        logger.error(f"Неожиданная ошибка при конвертации: {e}")
        return False