import os
from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

Label = Literal["pos", "neu", "neg"]

@dataclass
class SentimentResult:
    label: Label
    confidence: float
    reason: str

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# เรากำหนดโครงสร้างเอาท์พุตแบบ JSON ชัดเจน (Structured Output)
# เพื่อให้โมเดลส่งกลับมาเป็น JSON ที่ parse ได้แน่นอน
SENTIMENT_SCHEMA = {
    "name": "sentiment_schema",
    "schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["pos", "neu", "neg"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"}
        },
        "required": ["label", "confidence", "reason"],
        "additionalProperties": False
    },
    "strict": True
}

def classify_with_openai(text: str) -> SentimentResult:
    if not text or not text.strip():
        return SentimentResult(label="neu", confidence=0.0, reason="empty input")

    # (ทางเลือกแนะนำ) ตรวจผ่าน Moderation API ก่อนส่งไปใช้งานต่อ
    # ถ้าพบว่ามีเนื้อหาไม่เหมาะสม คุณอาจปฏิเสธ/เบลอข้อความ
    try:
        mod = client.moderations.create(model="omni-moderation-latest", input=text)
        flagged = any(cat for cat, v in mod.results[0].categories.__dict__.items() if v)
        if flagged:
            return SentimentResult(label="neu", confidence=0.0,
                                   reason="content flagged by moderation")
    except Exception:
        # ถ้าตรวจไม่สำเร็จ เราจะไม่ล้ม แต่ไปต่อด้วยการจัดหมวด
        pass

    # ใช้ “Structured outputs” เพื่อให้โมเดลตอบเป็น JSON ตาม schema
    # หมายเหตุ: เอกสารอย่างเป็นทางการแนะนำการใช้ structured outputs/JSON mode
    # สำหรับงานที่ต้องการผลลัพธ์เป็น JSON ที่คาดเดาได้. 
    resp = client.responses.create(
        model="gpt-5-mini",  # เลือกโมเดลรุ่นเล็กรวดเร็ว; เปลี่ยนได้ตามบัญชีของคุณ
        input=[{
            "role": "user",
            "content": f"Classify the sentiment of the following Thai text into one of "
                       f"[pos, neu, neg]. Be strict and return valid JSON only.\n\n{text}"
        }],
        response_format={"type": "json_schema", "json_schema": SENTIMENT_SCHEMA}
    )
    # โครงสร้างผลลัพธ์จาก Responses API จะมี output_text / output[0]
    # แต่เมื่อใช้ json_schema จะได้ข้อความเป็น JSON string
    data = resp.output[0].content[0].text  # JSON string
    import json
    parsed = json.loads(data)
    return SentimentResult(
        label=parsed["label"],
        confidence=float(parsed["confidence"]),
        reason=parsed["reason"]
    )
