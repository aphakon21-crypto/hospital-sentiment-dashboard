# src/collect/generate_synthetic_th.py
import os, random, csv
from pathlib import Path
import pandas as pd

# ----------------- ปรับจำนวนตัวอย่างต่อคลาสที่นี่ -----------------
N_POS = 400     # จำนวนบรรทัด positive
N_NEU = 400     # จำนวนบรรทัด neutral
N_NEG = 400     # จำนวนบรรทัด negative
RANDOM_SEED = 42
# --------------------------------------------------------------------

random.seed(RANDOM_SEED)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUT_FILE = DATA_DIR / "custom_generated.csv"

# คำ/วลีพื้นฐาน
POS_BASE = [
    "ดีมาก", "ยอดเยี่ยม", "ประทับใจ", "อร่อยมาก", "คุณภาพดี", "บริการดี", "คุ้มค่า",
    "ชอบมาก", "แนะนำเลย", "เกินคาด", "รวดเร็ว", "ใช้งานง่าย", "ตรงปก", "สะอาด", "น่าประทับใจ",
]
NEU_BASE = [
    "เฉยๆ", "ปานกลาง", "ใช้ได้", "โอเค", "ธรรมดา", "เป็นมาตรฐาน", "ตามราคา",
    "ยังไม่แน่ใจ", "ต้องลองอีกสักพัก", "ไม่มีความเห็น", "ก็พอใช้ได้", "เรื่อยๆ", "กลางๆ",
]
NEG_BASE = [
    "แย่มาก", "ห่วย", "ผิดหวัง", "ไม่ดี", "ช้ามาก", "ไม่คุ้ม", "บริการแย่", "คุณภาพแย่",
    "ไม่อร่อย", "งงมาก", "แพงเกินไป", "ไม่ตรงปก", "ใช้งานไม่ได้", "แย่ลง", "ไม่แนะนำ",
]

# ตัวขยาย/คำวิเศษณ์
BOOSTERS_POS = ["สุดๆ", "มากๆ", "ที่สุด", "โคตรดี", "ประทับใจมาก", "เกินคาดจริงๆ"]
BOOSTERS_NEG = ["สุดๆ", "มากๆ", "ที่สุด", "โคตรแย่", "แย่มากๆ", "ผิดหวังมาก"]
BOOSTERS_NEU = ["พอได้", "ก็โอเค", "กลางๆ", "ทั่วไป"]

# บริบทสินค้า/บริการ (จะสุ่มมาประกอบ)
CONTEXTS = [
    "อาหาร", "เครื่องดื่ม", "แพ็กของ", "ส่งของ", "พนักงาน", "แอป", "หน้าเว็บ",
    "บริการหลังการขาย", "บรรยากาศ", "รสชาติ", "ราคา", "ความเร็ว", "คุณภาพ",
]

# อีโมจิ
EMO_POS = ["😊", "😍", "👍", "✨", "🥰", "🤩", "💯"]
EMO_NEU = ["🙂", "😐"]
EMO_NEG = ["😡", "😞", "👎", "🤦", "😤", "💢"]

# รูปแบบประโยค (templates)
def templates_pos():
    base = random.choice(POS_BASE)
    ctx = random.choice(CONTEXTS)
    booster = random.choice(BOOSTERS_POS + [""])
    emo = random.choice(EMO_POS + [""])
    patt = random.choice([
        f"{ctx}{base}{(' ' + booster) if booster else ''}",
        f"{base} {ctx}{(' ' + booster) if booster else ''}",
        f"{base}{(' ' + booster) if booster else ''}",
        f"{ctx} {base} {emo}",
        f"{base} {emo}",
        f"คุ้มค่ามาก {ctx} {emo}",
        f"บริการดี {emo}",
        f"ถูกใจ {ctx} {emo}",
    ])
    return jitter(patt)

def templates_neu():
    base = random.choice(NEU_BASE)
    ctx = random.choice(CONTEXTS)
    booster = random.choice(BOOSTERS_NEU + [""])
    emo = random.choice(EMO_NEU + [""])
    patt = random.choice([
        f"{ctx}{base}{(' ' + booster) if booster else ''}",
        f"{base} {ctx}{(' ' + booster) if booster else ''}",
        f"{base}",
        f"{base} {emo}",
        f"{ctx} {base}",
    ])
    return jitter(patt)

def templates_neg():
    base = random.choice(NEG_BASE)
    ctx = random.choice(CONTEXTS)
    booster = random.choice(BOOSTERS_NEG + [""])
    emo = random.choice(EMO_NEG + [""])
    patt = random.choice([
        f"{ctx}{base}{(' ' + booster) if booster else ''}",
        f"{base} {ctx}{(' ' + booster) if booster else ''}",
        f"{base}{(' ' + booster) if booster else ''}",
        f"{base} {emo}",
        f"{ctx} {base} {emo}",
        f"ไม่โอเค {ctx} {emo}",
        f"ไม่คุ้ม {ctx} {emo}",
    ])
    return jitter(patt)

# ใส่ความ “สมจริง” เช่น การซ้ำตัวอักษร, !?!, เว้นวรรค, คำลงท้าย
ENDINGS_POS = ["ครับ", "ค่ะ", "เลย", "นะ", "", "จริงๆ"]
ENDINGS_NEU = ["ครับ", "ค่ะ", "", "มั้ง", "นะ"]
ENDINGS_NEG = ["ครับ", "ค่ะ", "", "มาก", "สุดๆ", "เลย"]

PUNCTS_POS = ["", "!", "!!", "!!!", "~"]
PUNCTS_NEU = ["", ".", ".."]
PUNCTS_NEG = ["!", "!!", "!!!", "?!", "…"]

def repeat_chars(txt: str) -> str:
    # สุ่มยืดสระ/พยัญชนะท้ายเล็กน้อย เช่น ดีมากกกก, แย่มากก
    if not txt or len(txt) < 2: return txt
    if random.random() < 0.25:
        idx = random.randrange(max(1, len(txt)-3), len(txt))
        ch = txt[idx]
        if ch not in " .!?" and len(txt) < 40:
            txt = txt[:idx] + ch * random.randint(2, 4) + txt[idx+1:]
    return txt

def jitter(sentence: str, sentiment: str = None) -> str:
    sentence = sentence.strip()
    sentence = repeat_chars(sentence)
    # เติมเครื่องหมาย + คำลงท้าย
    if sentiment == "pos":
        sentence += random.choice(PUNCTS_POS)
        tail = random.choice(ENDINGS_POS)
    elif sentiment == "neg":
        sentence += random.choice(PUNCTS_NEG)
        tail = random.choice(ENDINGS_NEG)
    else:
        sentence += random.choice(PUNCTS_NEU)
        tail = random.choice(ENDINGS_NEU)
    if tail:
        sentence += " " + tail
    return sentence

def gen_many(n: int, fn, label: str):
    out = []
    for _ in range(n):
        s = fn()
        s = jitter(s, label)
        out.append({"text": s, "label": label})
    return out

# สร้างข้อมูล
rows = []
rows += gen_many(N_POS, templates_pos, "pos")
rows += gen_many(N_NEU, templates_neu, "neu")
rows += gen_many(N_NEG, templates_neg, "neg")

# ลบซ้ำ + สุ่มลำดับ
df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# บันทึก
df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ generated -> {OUT_FILE}  ({len(df)} rows)")
