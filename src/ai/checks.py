import re
import math
from dataclasses import dataclass
from typing import Dict, Any, List

from langdetect import detect, LangDetectException
from better_profanity import profanity

# ----- OPTIONAL: ใช้เมื่ออยากมี toxicity (pytorch หนักพอสมควร)
try:
    from detoxify import Detoxify
    _detox = Detoxify('original')  # โหลดครั้งเดียว
except Exception:
    _detox = None

# ----- OPTIONAL: keyphrase
try:
    from keybert import KeyBERT
    _kb = KeyBERT(model='sentence-transformers/all-MiniLM-L6-v2')
except Exception:
    _kb = None

profanity.load_censor_words()  # โหลด default list

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\d{2,4}[-.\s]?){2,4}\d{2,4}")
URL_RE   = re.compile(r"https?://\S+|www\.\S+")

@dataclass
class AICheckConfig:
    # thresholds
    toxicity_threshold: float = 0.6
    profanity_as_toxic: bool = True
    max_keyphrases: int = 5
    use_toxicity: bool = False      # เปิดเมื่อคุณติดตั้ง detoxify แล้ว
    use_keybert: bool = True
    # กฎง่าย ๆ สำหรับ spam
    min_chars_for_spam: int = 6
    max_repeated_char: int = 6

def safe_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "und"

def has_pii(text: str) -> Dict[str, bool]:
    return {
        "email": bool(EMAIL_RE.search(text)),
        "phone": bool(PHONE_RE.search(text)),
    }

def is_spam_like(text: str, cfg: AICheckConfig) -> Dict[str, Any]:
    lower = text.lower()
    has_link = bool(URL_RE.search(text))
    repeated = re.search(r"(.)\1{" + str(cfg.max_repeated_char) + ",}", text) is not None
    too_short = len(text.strip()) < cfg.min_chars_for_spam
    looks_spam = has_link or repeated or too_short
    return {
        "has_link": has_link,
        "repeated": repeated,
        "too_short": too_short,
        "spam_flag": looks_spam
    }

def toxicity_score(text: str, cfg: AICheckConfig) -> Dict[str, Any]:
    if not cfg.use_toxicity or _detox is None:
        # fallback: ใช้ profanity เป็นสัญญาณหยาบ
        prof = profanity.contains_profanity(text)
        score = 0.7 if prof and cfg.profanity_as_toxic else 0.0
        return {"toxicity": score, "toxic_flag": score >= cfg.toxicity_threshold, "by": "profanity"}
    # ถ้าใช้ detoxify
    pred = _detox.predict(text)
    tox = float(pred.get("toxicity", 0.0))
    return {"toxicity": tox, "toxic_flag": tox >= cfg.toxicity_threshold, "by": "detoxify"}

def key_phrases(text: str, cfg: AICheckConfig) -> List[str]:
    if not cfg.use_keybert or _kb is None:
        return []
    try:
        kws = _kb.extract_keywords(text, top_n=cfg.max_keyphrases, stop_words='english')
        # kws = [(phrase, score), ...]
        return [k for k, _ in kws]
    except Exception:
        return []

def risk_level(row: Dict[str, Any]) -> str:
    """จัดระดับความเสี่ยงง่าย ๆ ตามเงื่อนไข"""
    if row.get("toxic_flag") or row.get("pii_email") or row.get("pii_phone"):
        return "high"
    if row.get("spam_flag"):
        return "medium"
    return "low"

def analyze_text(text: str, cfg: AICheckConfig) -> Dict[str, Any]:
    lang = safe_lang(text)
    pii = has_pii(text)
    spam = is_spam_like(text, cfg)
    tox  = toxicity_score(text, cfg)
    kps  = key_phrases(text, cfg)

    result = {
        "lang": lang,
        "pii_email": pii["email"],
        "pii_phone": pii["phone"],
        "has_link": spam["has_link"],
        "spam_flag": spam["spam_flag"],
        "toxicity": tox["toxicity"],
        "toxic_flag": tox["toxic_flag"],
        "toxicity_by": tox["by"],
        "keyphrases": ", ".join(kps) if kps else "",
    }
    result["risk"] = risk_level(result)
    return result
