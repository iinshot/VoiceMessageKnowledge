import os
import httpx

STT_URL = os.getenv("STT_URL")
ASS_URL = os.getenv("ASS_URL")

TIMEOUT = 120.0

async def transcribe_wav(wav_path: str) -> str:
    """Отправляет .wav файл в STT сервис, возвращает распознанный текст."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        with open(wav_path, "rb") as f:
            response = await client.post(
                f"{STT_URL}/transcribe",
                files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            )
        if response.status_code != 200:
            raise Exception(f"STT error {response.status_code}: {response.text}")
        return response.json()["text"]

async def score_answer(question: str, reference: str, student: str) -> dict:
    """
    Отправляет вопрос + эталон + ответ студента в scorer, возвращает оценку.

    Возвращает словарь:
    {
        "S": 0.85, "C_raw": 0.72, "C": 0.80,
        "H": 0.95, "score": 0.84, "grade": 4
    }
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{ASS_URL}/score",
            json={
                "question": question,
                "reference": reference,
                "student": student,
            },
        )
        response.raise_for_status()
        return response.json()