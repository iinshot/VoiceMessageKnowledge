import re
import json
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False
    print("pymorphy3 не установлен. Установите: pip install pymorphy3")
    print("  C будет считаться без лемматизации, H — по списку слов.")

DEFAULT_WEIGHTS = {"w1": 0.8, "w2": 0.1, "w3": 0.1}
GRADE_THRESHOLDS = [
    (0.875, 5),
    (0.625, 4),
    (0.375, 3),
    (0.0,   2),
]

 
 
DEFAULT_ALPHA_C = 0.5

 
H_ALPHA = 3.0    
H_BETA  = 2.0    

 
PARASITE_POS = {"PRCL", "INTJ"}   

 
FILLER_WORDS_FALLBACK = {
    "ну", "вот", "это", "типа", "как", "бы", "значит", "общем", "короче",
    "так", "сказать", "есть", "собственно", "кстати", "буквально", "вообще",
    "просто", "конечно", "наверное", "вроде", "прямо", "ладно", "ага", "угу","эээ"
}

 
STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом",
    "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо",
    "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж",
    "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем",
    "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее",
    "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при",
    "наконец", "два", "об", "другой", "хоть", "после", "над", "больше",
    "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много",
    "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
    "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им",
    "более", "всегда", "конечно", "всю", "между",
}


@lru_cache(maxsize=50000)
def lemmatize(word: str) -> str:
    """Приводит слово к начальной форме. Результат кешируется."""
    if not PYMORPHY_AVAILABLE:
        return word.lower()
    return _morph.parse(word)[0].normal_form


@lru_cache(maxsize=50000)
def get_pos(word: str) -> str:
    """Возвращает часть речи слова (POS-тег pymorphy3)."""
    if not PYMORPHY_AVAILABLE:
        return "UNKN"
    parsed = _morph.parse(word)[0]
    return parsed.tag.POS or "UNKN"


def tokenize_text(text: str) -> list:
    """Разбивает текст на токены, убирает пунктуацию."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if t]


 
 
 

def extract_keywords(text: str, min_len: int = 3) -> set:
    """
    Извлекает ключевые термины из текста.
    Улучшение v2: работает с леммами, а не поверхностными формами.
    "объёмом", "объёма", "объём" → все дают лемму "объём"
    """
    tokens = tokenize_text(text)
    keywords = set()
    for t in tokens:
        if len(t) < min_len:
            continue
        lemma = lemmatize(t)
        if lemma not in STOP_WORDS and len(lemma) >= min_len:
            keywords.add(lemma)
    return keywords


def compute_C_raw(reference: str, student: str, idf: dict) -> float:

    ref_kw = extract_keywords(reference)
    stu_kw = extract_keywords(student)

    if not ref_kw:
        return 1.0

    overlap = ref_kw & stu_kw

    num = sum(idf.get(t, 1.0) for t in overlap)
    den = sum(idf.get(t, 1.0) for t in ref_kw)

    return round(num / den, 4)


 
 
 

def compute_H(text: str, alpha: float = H_ALPHA, beta: float = H_BETA) -> float:
    """
    H = 1 - clip(паразиты/всего * alpha + повторы/всего * beta, 0, 1)

    v2 улучшения:
    - Паразиты определяются через POS-теги (PRCL=частицы, INTJ=междометия)
      а не захардкоженным списком слов
    - Повторы считаются по леммам: "данных данные" тоже считается повтором
    """
    if not text or not text.strip():
        return 0.0

    tokens = tokenize_text(text)
    if not tokens:
        return 0.0

    total = len(tokens)

    if PYMORPHY_AVAILABLE:
         
        parasite_count = sum(1 for t in tokens if get_pos(t) in PARASITE_POS)

         
        lemmas = [lemmatize(t) for t in tokens]
        repeat_count = sum(
            1 for i in range(1, len(lemmas)) if lemmas[i] == lemmas[i - 1]
        )
    else:
         
        parasite_count = sum(1 for t in tokens if t in FILLER_WORDS_FALLBACK)
        repeat_count = sum(
            1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1]
        )

    parasite_ratio = parasite_count / total
    repeat_ratio   = repeat_count / total

    penalty = parasite_ratio * alpha + repeat_ratio * beta
    return round(1.0 - min(penalty, 1.0), 4)


 
 
 

class SiameseRuBERT(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

    def mean_pool(self, token_embeddings, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        count  = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(
        self,
        ref_input_ids, ref_attention_mask, ref_token_type_ids,
        stu_input_ids, stu_attention_mask, stu_token_type_ids,
    ):
        ref_emb = self.encode(ref_input_ids, ref_attention_mask, ref_token_type_ids)
        stu_emb = self.encode(stu_input_ids, stu_attention_mask, stu_token_type_ids)
        cosine  = (ref_emb * stu_emb).sum(dim=-1)
        return (cosine + 1) / 2


 
 
 

class StudentAnswerScorer:
    """
    Загружает обученную Siamese RuBERT и считает S, C, H → итоговую оценку.

    Параметры:
        model_dir  : папка с best_model.pt, config.json и токенизатором
        weights    : {'w1': float, 'w2': float, 'w3': float}, сумма = 1
        max_length : длина токенизации (должна совпадать с обучением — 96)
        alpha_c    : коэффициент поправки C (0..1)
        device     : 'cuda' / 'cpu' / None (авто)
    """

    def __init__(
        self,
        model_dir:  str,
        weights:    Optional[dict] = None,
        max_length: int   = 128,
        alpha_c:    float = DEFAULT_ALPHA_C,
        device:     Optional[str] = None,
    ):
        self.weights    = weights or DEFAULT_WEIGHTS
        self.max_length = max_length
        self.alpha_c    = alpha_c
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        with open("../data/idf.json", encoding="utf-8") as f:
            self.idf = json.load(f)
        with open(f"{model_dir}/config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model = cfg.get("model_name", "DeepPavlov/rubert-base-cased")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model = SiameseRuBERT(base_model).to(self.device)
        self.model.load_state_dict(
            torch.load(f"{model_dir}/best_model.pt", map_location=self.device)
        )
        self.model.eval()

        morph_status = "pymorphy3 ✓" if PYMORPHY_AVAILABLE else "без морфологии ⚠"
        print(f"✓ Модель загружена из {model_dir}")
        print(f"  Устройство : {self.device}")
        print(f"  Морфология : {morph_status}")

    def _tokenize(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(1, self.max_length, dtype=torch.long)
            ).to(self.device),
        }

    def _compute_S(self, question: str, reference: str, student: str) -> float:
        ref_enc = self._tokenize(question + " [SEP] " + reference)
        stu_enc = self._tokenize(question + " [SEP] " + student)
        with torch.no_grad():
            s = self.model(
                ref_enc["input_ids"], ref_enc["attention_mask"], ref_enc["token_type_ids"],
                stu_enc["input_ids"], stu_enc["attention_mask"], stu_enc["token_type_ids"],
            )
        return round(float(s.item()), 4)

    def _score_to_grade(self, score: float) -> int:
        grade = 2
        for threshold, g in sorted(GRADE_THRESHOLDS):
            if score >= threshold:
                grade = g
        return grade

    def score(
        self,
        question:  str,
        reference: str,
        student:   str,
        verbose:   bool = False,
    ) -> dict:
        """
        Считает все компоненты и итоговую оценку.

        Возвращает:
        {
            'S':     float,   
            'C_raw': float,   
            'C':     float,   
            'H':     float,   
            'score': float,   
            'grade': int,     
        }
        """
        w1 = self.weights["w1"]
        w2 = self.weights["w2"]
        w3 = self.weights["w3"]

        S     = self._compute_S(question, reference, student)
        C_raw = compute_C_raw(reference, student, self.idf)
        C     = round(max(0.0, min(1.0, S + self.alpha_c * (C_raw - S))), 4)
        H     = compute_H(student)

        def safe_pow(base, exp):
            if exp == 0:
                return 1.0
            return max(float(base), 1e-9) ** exp

        score = round(safe_pow(S, w1) * safe_pow(C, w2) * safe_pow(H, w3), 4)
        grade = self._score_to_grade(score)

        result = {"S": S, "C_raw": C_raw, "C": C, "H": H, "score": score, "grade": grade}

        if verbose:
            print(f"\n{'─' * 52}")
            print(f"Вопрос : {question[:70]}{'...' if len(question) > 70 else ''}")
            print(f"{'─' * 52}")
            print(f"S      : {S:.4f}  ← косинус эмбеддингов (RuBERT)")
            print(f"C_raw  : {C_raw:.4f}  ← покрытие терминов по леммам")
            print(f"C(adj) : {C:.4f}  ← C скорректированное (alpha={self.alpha_c})")
            print(f"H      : {H:.4f}  ← связность речи (POS-теги)")
            print(f"{'─' * 52}")
            print(f"Score  : {S:.4f}^{w1} × {C:.4f}^{w2} × {H:.4f}^{w3} = {score:.4f}")
            print(f"Оценка : {grade}")
            print(f"{'─' * 52}\n")

        return result

    def score_batch(self, rows: list, verbose: bool = False) -> list:
        """
        Оценивает список ответов.
        rows — список словарей с ключами: 'question', 'reference', 'student'
        """
        return [
            self.score(
                question  = r["question"],
                reference = r["reference"],
                student   = r["student"],
                verbose   = verbose,
            )
            for r in rows
        ]

if __name__ == "__main__":
    scorer = StudentAnswerScorer(
        model_dir = "models/siamese_rubert_s_128",
        weights   = {"w1": 0.8, "w2": 0.1, "w3": 0.1},
    )

     
    scorer.score(
        question  = "Что такое большие данные?",
        reference = (
            "Большие данные — это наборы данных большого объёма и высокой "
            "скорости накопления, для обработки которых недостаточно "
            "традиционных методов. Характеризуются моделью 4V: "
            "Volume, Velocity, Variety, Veracity."
        ),
        student = (
            "Большие данные — это огромные объёмы информации, которые "
            "сложно обработать стандартными инструментами. Они описываются "
            "четырьмя характеристиками: объём, скорость, разнообразие и достоверность."
        ),
        verbose = True,
    )

     
    scorer.score(
        question  = "Что такое большие данные?",
        reference = (
            "Большие данные — это наборы данных большого объёма и высокой "
            "скорости накопления, для обработки которых недостаточно "
            "традиционных методов. Характеризуются моделью 4V: "
            "Volume, Velocity, Variety, Veracity."
        ),
        student = (
            "Ну это типа очень много данных, которые сложно обрабатывать. "
            "Там есть ещё какие-то характеристики характеристики, "
            "но я не помню помню как они называются."
        ),
        verbose = True,
    )