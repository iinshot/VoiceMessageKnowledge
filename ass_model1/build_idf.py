import math
import json
import pandas as pd
from collections import Counter

from scorer import extract_keywords


def build_idf(dataset_path):

    df = pd.read_csv(dataset_path)

    docs = []

     
    for text in df["Эталонный ответ преподавателя"]:
        docs.append(set(extract_keywords(str(text))))

    df_counter = Counter()

    for doc in docs:
        for term in doc:
            df_counter[term] += 1

    N = len(docs)

    idf = {
        term: math.log((N + 1) / (freq + 1)) + 1
        for term, freq in df_counter.items()
    }

    with open("../data/idf.json", "w", encoding="utf-8") as f:
        json.dump(idf, f, ensure_ascii=False, indent=2)

    print("✓ IDF построен")
    print("документов:", N)
    print("уникальных терминов:", len(idf))


if __name__ == "__main__":
    build_idf("../data/datasets/train.csv")