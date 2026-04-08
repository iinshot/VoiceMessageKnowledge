import csv
import threading
import json
import os

def read_questions(questions_file):
    with open(questions_file, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        unique_questions = set(row["Вопрос"] for row in reader)
    return unique_questions

counter_lock = threading.Lock()
COUNTER_FILE = "question_files/counter.json"

def get_next_counter(question):
    with counter_lock:
        if os.path.exists(COUNTER_FILE):
            with open(COUNTER_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                counters = json.loads(content) if content else {}
        else:
            counters = {}

        counters[question] = counters.get(question, 0) + 1
        count = counters[question]

        with open(COUNTER_FILE, "w", encoding="utf-8") as f:
            json.dump(counters, f, ensure_ascii=False)

        return count

def format_question_for_filename(question):
    invalid_chars = '/\\:*?"<>|'
    clean = question.strip()
    for ch in invalid_chars:
        clean = clean.replace(ch, "")
    clean = clean.replace(" ", "_")
    return clean[:50]