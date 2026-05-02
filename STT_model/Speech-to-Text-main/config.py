from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

MODEL_DIR = Path(os.getenv("STT_MODEL_DIR"))
OUTPUT_DIR = Path(__file__).parent.parent.parent / Path(os.getenv("STT_OUTPUT_DIR"))
RECORD_DIR = Path(__file__).parent.parent.parent / Path(os.getenv("STT_RECORD_DIR"))
OUTPUT_FILE = OUTPUT_DIR / "transcript.txt"
SILERO_DIR = Path(os.getenv("SILERO_DIR"))
LM_PATH = Path(os.getenv("LM_PATH"))

MIN_SPEAKERS = 1
MAX_SPEAKERS = 1
SAMPLE_RATE = 16000

MONOLOGUE_STD_THRESHOLD = 0.135

USE_LM = False
LM_ALPHA = 0.5
LM_BETA = 1.5
LM_BEAM_WIDTH = 100
HOTWORD_WEIGHT = 10.0
DEFAULT_HOTWORDS = [
    "егэ", "огэ", "кубгу", "фктипм", "фкт", "макрос",
    "краснодар", "университет", "институт", "факультет",
]

USE_PUNCTUATION = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)