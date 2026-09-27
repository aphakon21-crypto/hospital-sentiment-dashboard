import joblib
import os

MODEL_PATH = "models/baseline.joblib"
if not os.path.exists(MODEL_PATH):
    raise SystemExit("❌ ไม่พบโมเดล baseline.joblib — โปรดรัน src/train/train_baseline.py ก่อน")

_model = joblib.load(MODEL_PATH)

def predict_sentiment(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    return _model.predict([text])[0]
import joblib, os
from typing import Tuple
from pathlib import Path

# โหลดโมเดลโลคัล
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "baseline.joblib"
_model = joblib.load(MODEL_PATH)

# >>> เพิ่ม: เรียกใช้ ChatGPT เมื่อจำเป็น
try:
    from src.providers.openai_sentiment import classify_with_openai
except Exception:
    classify_with_openai = None

def predict_local_with_conf(text: str) -> Tuple[str, float]:
    """ทำนายด้วยโมเดลโลคัล + คืนความมั่นใจ (max proba)"""
    if not isinstance(text, str) or not text.strip():
        return "neu", 0.0
    proba = _model.predict_proba([text])[0]
    labels = _model.classes_
    idx = proba.argmax()
    return labels[idx], float(proba[idx])

def predict_smart(text: str, threshold: float = 0.65) -> str:
    """
    1) ให้โมเดลโลคัลทำนายก่อน
    2) ถ้า confidence ต่ำกว่า threshold และตั้งค่า OpenAI ไว้ → ขอความเห็นจาก ChatGPT
    """
    label, conf = predict_local_with_conf(text)
    if conf >= threshold or classify_with_openai is None:
        return label
    # fallback ไปถาม ChatGPT
    try:
        gpt = classify_with_openai(text)
        # คุณอาจกำหนดนโยบาย merge เช่น ถ้า GPT มั่นใจ >0.7 ให้เชื่อ GPT
        return gpt.label
    except Exception:
        return label  # ถ้าเรียก GPT ล้มเหลว ก็ใช้ผลโลคัล
