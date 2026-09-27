# src/collect/merge_datasets.py (ถ้าอยากแยกไฟล์รวม)
import pandas as pd
from pathlib import Path

files = [
    "data/custom_generated.csv",   # ที่เพิ่งสร้าง
    "data/custom.csv",             # (ถ้ามี)
    "data/sample.csv",             # (ถ้ามี)
]
dfs = [pd.read_csv(f) for f in files if Path(f).exists()]
df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
df.to_csv("data/train_all.csv", index=False, encoding="utf-8-sig")
print("✅ merged -> data/train_all.csv", len(df), "rows")
