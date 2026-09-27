import pandas as pd
from pythainlp import word_tokenize
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/sample.csv")

def preprocess(t: str) -> str:
    tokens = word_tokenize(str(t), engine="newmm")
    return " ".join(tokens)

df["text_clean"] = df["text"].apply(preprocess)
df.to_csv("data/clean.csv", index=False, encoding="utf-8-sig")
print("✅ saved -> data/clean.csv (ข้อมูลหลังตัดคำ)")
