import pandas as pd
import os

os.makedirs("data", exist_ok=True)

samples = [
    # Positive
    {"text": "ร้านนี้บริการดีมาก ชอบมาก", "label": "pos"},
    {"text": "อาหารอร่อย คุณภาพดี", "label": "pos"},
    {"text": "พนักงานพูดจาดีมาก", "label": "pos"},
    {"text": "คุ้มค่ากับราคาที่จ่าย", "label": "pos"},
    {"text": "บริการรวดเร็วทันใจ", "label": "pos"},

    # Negative
    {"text": "รอนานเกินไป ไม่ประทับใจ", "label": "neg"},
    {"text": "ราคาสูงเกินไป ไม่โอเค", "label": "neg"},
    {"text": "พนักงานพูดจาไม่ดีเลย", "label": "neg"},
    {"text": "อาหารไม่สดเลย", "label": "neg"},
    {"text": "ผิดหวังกับบริการ", "label": "neg"},

    # Neutral
    {"text": "ราคาปานกลาง คุณภาพโอเค", "label": "neu"},
    {"text": "ร้านเปิดปกติทุกวัน", "label": "neu"},
    {"text": "อาหารรสชาติปกติ", "label": "neu"},
    {"text": "บรรยากาศธรรมดา ๆ", "label": "neu"},
    {"text": "พนักงานเฉย ๆ", "label": "neu"}
]

df = pd.DataFrame(samples)
df.to_csv("data/sample.csv", index=False, encoding="utf-8-sig")
print("✅ saved -> data/sample.csv (ข้อมูลตัวอย่างที่เพิ่มแล้ว)")
