# src/dashboard/app.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import sys
import re
import io
import json
import base64
import hashlib
from pathlib import Path
from datetime import datetime

import requests
from pypdf import PdfReader

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ============================== ROOT & IMPORTS ==============================
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ==================== ASPECT-BASED SENTIMENT ENGINE (GEMINI MULTI-DIMENSIONAL) ====================
# ==================== ASPECT-BASED SENTIMENT ENGINE (GEMINI 2.5 FLASH) ====================
import json
import os

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def analyze_aspects_smart(text: str) -> dict:
    t = str(text).lower().strip()
    
    # 0. Safety / Critical Check: ถ้าเป็นเคสวิกฤตร้ายแรง บังคับเป็นเชิงลบด้านหมอ/การรักษาทันที
    critical_check = detect_critical_risk(text)
    if critical_check["is_critical"]:
        return {
            "overall": "neg",
            "aspects": {
                "doctor": "neg",
                "nurse_staff": "not_mentioned",
                "facility": "not_mentioned",
                "price_time": "not_mentioned"
            }
        }

    # 1. ค้นหา API Key จากทุกช่องทาง
    api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    # Hardcode Fallback Key เผื่อหาไฟล์ secrets.toml ไม่เจอ


    # 2. เรียก Gemini แบบ Multi-Model Fallback เพื่อไม่ให้ติด 503/404
    if HAS_GENAI and api_key:
        prompt = f"""
คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์ความรู้สึกและจำแนกมิติงานบริการของโรงพยาบาล
วิเคราะห์ความคิดเห็นของผู้รับบริการด้านล่างนี้ และตอบกลับเป็น JSON ตามรูปแบบนี้เท่านั้น:

{{
  "overall": "pos" | "neg" | "neu",
  "aspects": {{
    "doctor": "pos" | "neg" | "neu" | "not_mentioned",
    "nurse_staff": "pos" | "neg" | "neu" | "not_mentioned",
    "facility": "pos" | "neg" | "neu" | "not_mentioned",
    "price_time": "pos" | "neg" | "neu" | "not_mentioned"
  }}
}}

เกณฑ์การตัดสิน:
1. "overall":
   - "pos": ชม ชื่นชอบ พึงพอใจ
   - "neg": บ่น ติเตียน ประชด ร้องเรียน แพ้ยา รักษาผิดพลาด อันตราย
   - "neu": คำถาม สอบถามเวลา/คิว ข้อมูลทั่วไป
2. "aspects":
   - "doctor": หมอ การตรวจ รักษา สั่งจ่ายยา ความเชี่ยวชาญ
   - "nurse_staff": พยาบาล เจ้าหน้าที่ บริการ การพูดจา
   - "facility": สถานที่ ความสะอาด ที่จอดรถ
   - "price_time": ราคา เวลารอคอย คิวตรวจ ความเร็ว

ข้อความ: \"\"\"{text.strip()}\"\"\"
"""
        client = genai.Client(api_key=api_key)
        candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        for m in candidate_models:
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                parsed = json.loads(resp.text.strip())
                if "overall" in parsed and "aspects" in parsed:
                    return parsed
            except Exception:
                continue

    # 3. Fallback อัจฉริยะ (เมื่อ API ไม่ตอบสนอง)
    is_q = any(w in t for w in ["ไหม", "มั้ย", "กี่โมง", "เปิด", "ปิด", "เท่าไหร่", "รับบัตรคิว", "สอบถาม"])
    
    # คำลบเกี่ยวกับการแพทย์ / การรักษา
    medical_neg_words = ["แพ้ยา", "ช็อก", "เกือบตาย", "รักษาผิด", "จ่ายยาผิด", "วินิจฉัยผิด", "ฟ้อง", "ทนาย", "หมอดุ", "หมอไม่ฟัง", "ไม่ใส่ใจ"]
    doc_neg = any(w in t for w in medical_neg_words)
    doc_pos = any(w in t for w in ["หมอดี", "ตรวจละเอียด", "มือเบา", "อธิบายเข้าใจง่าย"])

    nurse_neg = any(w in t for w in ["พยาบาลพูดจาไม่ดี", "ห้วน", "หน้าบึ้ง", "ไม่สนใจ", "ตะคอก"])
    nurse_pos = any(w in t for w in ["พยาบาลดี", "พูดเพราะ", "สุภาพ", "ใส่ใจ"])

    fac_neg = any(w in t for w in ["ที่จอดรถน้อย", "ที่จอดรถเต็ม", "วนหา", "สกปรก", "เหม็น"])
    fac_pos = any(w in t for w in ["สะอาด", "กว้าง", "วิวสวย", "สะดวกสบาย"])

    price_neg = any(w in t for w in ["แพง", "ค่ายาแรง", "รอนาน", "คิวยาว", "ช้า", "เข้าค่าย", "ตรวจบ่าย"])
    price_pos = any(w in t for w in ["ไม่แพง", "ราคาถูก", "รวดเร็ว"])

    doc_res = "neg" if doc_neg else "pos" if doc_pos else "not_mentioned"
    nurse_res = "neg" if nurse_neg else "pos" if nurse_pos else "not_mentioned"
    fac_res = "neg" if fac_neg else "pos" if fac_pos else "not_mentioned"
    price_res = "neg" if price_neg else "pos" if price_pos else "not_mentioned"

    if any(r == "neg" for r in [doc_res, nurse_res, fac_res, price_res]):
        overall_sent = "neg"
    elif any(r == "pos" for r in [doc_res, nurse_res, fac_res, price_res]):
        overall_sent = "pos"
    elif is_q:
        overall_sent = "neu"
    else:
        overall_sent = "neu"

    return {
        "overall": overall_sent,
        "aspects": {
            "doctor": doc_res,
            "nurse_staff": nurse_res,
            "facility": fac_res,
            "price_time": price_res
        }
    }

# ============================== PAGE CONFIG ==============================
st.set_page_config(
    page_title="โรงพยาบาลสิริเวช จันทบุรี - Retail Sentiment Analytics",
    page_icon="🏥",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = DATA_DIR / "log.csv"

# ============================== DATABASE & LOGGING ==============================
def append_log(text: str, label: str, category: str = "ทั่วไป"):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text": text,
        "label": label,
        "category": category
    }
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

def append_bulk_log(df_bulk: pd.DataFrame):
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, df_bulk], ignore_index=True)
    else:
        df = df_bulk
    df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH)
        if "date" not in df.columns and "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        if "category" not in df.columns:
            df["category"] = "ทั่วไป"
        return df
    return pd.DataFrame(columns=["timestamp", "text", "label", "category", "date"])

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({"label": ["pos", "neu", "neg"], "count": [0, 0, 0]})
    cnt = df["label"].value_counts().reindex(["pos", "neu", "neg"], fill_value=0)
    return pd.DataFrame({"label": cnt.index, "count": cnt.values})

# ============================== GOOGLE SHEETS ==============================
def _extract_sheet_id(url: str) -> str:
    if not url:
        return ""
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else url.strip()

def export_df_to_gsheet(df, spreadsheet_id: str, worksheet_name: str, clear_first=True):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    secret_dict = dict(st.secrets["gcp_service_account"])
    if "\\n" in secret_dict["private_key"]:
        secret_dict["private_key"] = secret_dict["private_key"].replace("\\n", "\n")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_dict, scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key(spreadsheet_id)

    try:
        worksheet = sh.worksheet(worksheet_name)
        if clear_first:
            worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=worksheet_name, rows="100", cols="20")

    df_clean = df.fillna("")
    values = [df_clean.columns.values.tolist()] + df_clean.values.tolist()
    worksheet.update(values)
    return True

def test_gsheet_connection(spreadsheet_url_or_id, worksheet_name=None):
    try:
        spreadsheet_id = _extract_sheet_id(spreadsheet_url_or_id)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        secret_dict = dict(st.secrets["gcp_service_account"])
        if "\\n" in secret_dict["private_key"]:
            secret_dict["private_key"] = secret_dict["private_key"].replace("\\n", "\n")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_dict, scope)
        client = gspread.authorize(creds)
        client.open_by_key(spreadsheet_id)
        return True, "เชื่อมต่อ Google Sheets สำเร็จ ✅"
    except Exception as e:
        return False, f"เชื่อมต่อไม่สำเร็จ ❌ : {e}"

def fetch_facebook_comments(post_url: str) -> list[str]:
    return [
        "บริการดีมาก แพทย์และพยาบาลใส่ใจตรวจละเอียด ประทับใจมากค่ะ",
        "รอคิวรับยานานไปหน่อย แต่สถานที่สะอาดและเป็นระเบียบดีครับ",
        "แอดมินเพจตอบคำถามช้ามาก ควรปรับปรุงระบบแชท",
        "แพ็กเกจตรวจสุขภาพประจำปีคุ้มค่า ผลตรวจออกไวมาก",
    ]

# ============================== AUTHENTICATION ==============================
def _get_users_from_secrets() -> dict[str, dict]:
    return st.secrets.get("users", {})

def _verify_password(password_plain: str, salt: str, hash_hex: str) -> bool:
    return hashlib.sha256((salt + password_plain).encode("utf-8")).hexdigest() == hash_hex

# ============================== USER AUTH (CUSTOM MODERN CARD) ==============================
def login_form():
    # 1. ค้นหาและแปลงไฟล์รูปโลโก้ (.webp, .png, .jpg) เป็น Base64
    curr_dir = Path(__file__).resolve().parent
    proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent

    # รายการตำแหน่งและชื่อไฟล์ที่ค้นหา (รองรับทั้ง .webp และ .png)
    logo_paths = [
        curr_dir / "assets" / "logo.webp",
        proj_root / "assets" / "logo.webp",
        curr_dir / "assets" / "sirivej_logo.webp",
        proj_root / "assets" / "sirivej_logo.webp",
        Path("assets/logo.webp").resolve(),
        # สำรองกรณีตั้งชื่อเป็น .png
        curr_dir / "assets" / "logo.png",
        proj_root / "assets" / "logo.png",
    ]

    logo_src = ""
    for p in logo_paths:
        if p.is_file():
            mime = "image/webp" if p.suffix.lower() == ".webp" else "image/png"
            encoded_img = base64.b64encode(p.read_bytes()).decode()
            logo_src = f"data:{mime};base64,{encoded_img}"
            break

    # 2. CSS ตกแต่งหน้า Login ตามภาพ Ref
    st.markdown("""
    <style>
    [data-testid="collapsedControl"], header { display: none; }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #e8f4fc 0%, #f7fbfe 50%, #f0f7fd 100%) !important;
    }

    [data-testid="stForm"] {
        background: #ffffff !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 28px !important;
        padding: 40px 36px 30px 36px !important;
        box-shadow: 0 20px 50px rgba(2, 132, 199, 0.08), 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        margin-top: 5vh !important;
    }

    .stTextInput input {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: #1e293b !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput input:focus {
        background-color: #ffffff !important;
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15) !important;
    }
    .stTextInput label {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        margin-bottom: 4px !important;
    }

    [data-testid="stFormSubmitButton"] > button {
        background: #0284c7 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 20px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        letter-spacing: 0.5px;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.25s ease !important;
        margin-top: 10px !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #0369a1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(2, 132, 199, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 3. จัดกึ่งกลางหน้าจอ
    _, center_col, _ = st.columns([0.34, 0.32, 0.34])
    with center_col:
        with st.form("modern_login_form"):
            # แสดงโลโก้ไฟล์ webp
            if logo_src:
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 18px;">
                    <img src="{logo_src}" style="max-width: 180px; height: auto; object-fit: contain;">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; margin-bottom: 18px;">
                    <div style="display: inline-flex; align-items: center; gap: 8px;">
                        <span style="font-size: 22px;">🏥</span>
                        <span style="color: #005b9f; font-weight: 800; font-size: 16px;">โรงพยาบาลสิริเวช</span>
                        <span style="background: #0284c7; color: white; font-size: 11px; font-weight: 800; padding: 2px 6px; border-radius: 4px;">THG</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # หัวข้อและไอคอนแม่กุญแจ
            st.markdown("""
            <div style="text-align: center; margin-bottom: 22px;">
                <div style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 46px;
                    height: 46px;
                    border-radius: 12px;
                    background: #e0f2fe;
                    color: #0284c7;
                    font-size: 22px;
                    margin-bottom: 12px;
                ">🔒</div>
                <div style="color: #0284c7; font-size: 11.5px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px;">
                    CUSTOMER SENTIMENT SYSTEM
                </div>
                <h2 style="color: #0f172a; font-weight: 800; font-size: 26px; margin: 0 0 6px 0;">
                    เข้าสู่ระบบ
                </h2>
                <p style="color: #64748b; font-size: 13px; margin: 0; line-height: 1.4;">
                    กรุณากรอกรหัสผู้ใช้งานเพื่อเข้าสู่ระบบประเมินผล<br> Sentiment System
                </p>
            </div>
            """, unsafe_allow_html=True)

            username = st.text_input("รหัสพนักงาน / ชื่อผู้ใช้งาน", placeholder="กรอกรหัสพนักงาน")
            password = st.text_input("รหัสผ่าน", type="password", placeholder="กรอกรหัสผ่าน")

            submit = st.form_submit_button("เข้าสู่ระบบ  →", use_container_width=True)

            st.markdown("""
            <div style="text-align: center; margin-top: 18px; color: #94a3b8; font-size: 11.5px; display: flex; align-items: center; justify-content: center; gap: 5px;">
                <span>🛡️</span> สำหรับบุคลากรที่ได้รับสิทธิ์เข้าใช้งานระบบเท่านั้น
            </div>
            """, unsafe_allow_html=True)

            if submit:
                users = _get_users_from_secrets()
                u = users.get(username)
                if not u:
                    st.error("❌ ไม่พบรหัสผู้ใช้งานนี้ในระบบ")
                elif _verify_password(password, u["salt"], u["hash"]):
                    st.session_state.auth = {
                        "logged_in": True,
                        "username": username,
                        "display_name": u.get("display_name", username),
                        "role": u.get("role", "user"),
                    }
                    st.success("เข้าสู่ระบบสำเร็จ กำลังนำเข้าสู่ระบบ...")
                    st.rerun()
                else:
                    st.error("❌ รหัสผ่านไม่ถูกต้อง")

def require_login():
    if "auth" not in st.session_state or not st.session_state.auth.get("logged_in"):
        login_form()
        st.stop()

# ============================== UI STYLING (LINE SEED + DARK BLUE + WHITE TEXT) ==============================
# ============================== UI STYLING (LINE SEED + COMPLETE DARK THEME) ==============================
# ============================== UI STYLING (LINE SEED + COMPLETE DARK THEME) ==============================
def inject_custom_css():
    st.markdown("""
    <style>
    /* 1. ดึงไฟล์ CSS ฟอนต์ LINE Seed Sans TH โดยตรง */
    @import url('https://cdn.jsdelivr.net/gh/danyim/lineseed-thai@master/css/lineseed-thai.css');
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800&display=swap');

    /* 2. บังคับใช้ฟอนต์ LINE Seed Sans TH กับทุกจุด ทุกแท็กในหน้าเว็บ */
    html, body, [class*="css"], .stApp,
    h1, h2, h3, h4, h5, h6, p, span, div, label,
    button, input, textarea, select, a,
    .stSelectbox, .stTextInput, .stTextArea, .stButton,
    .stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] p,
    [data-baseweb="tab"], [data-baseweb="select"], [data-baseweb="popover"],
    .stDateInput div, .stDownloadButton button {
        font-family: 'LINESeedSansTH', 'LINE Seed Sans TH', 'Prompt', sans-serif !important;
    }

    /* 3. ปรับสีพื้นหลังหลักของแอป: โทนน้ำเงินเข้ม (Dark Slate Blue) ลายรังผึ้ง */
    .stApp {
        background-color: #070d1e !important;
        background-image: 
            /* แสง Radial gradient นุ่มนวลตรงกลาง */
            radial-gradient(circle at 50% 15%, rgba(26, 46, 82, 0.7) 0%, rgba(7, 13, 30, 0.95) 100%),
            /* Medical Pattern: SVG ลายการแพทย์สีขาวจางๆ */
            url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120"><g fill="none" stroke="%23ffffff" stroke-width="1.2" opacity="0.08"><path d="M20,10 L20,30 M10,20 L30,20" stroke-linecap="round"/><path d="M80,70 L80,90 M70,80 L90,80" stroke-linecap="round"/><path d="M45,45 L55,45 L60,35 L66,55 L72,40 L76,48 L88,48" stroke-linecap="round" stroke-linejoin="round"/><circle cx="105" cy="25" r="4"/><circle cx="115" cy="40" r="3"/><circle cx="95" cy="40" r="3"/><path d="M105,25 L115,40 M105,25 L95,40"/><circle cx="30" cy="95" r="2.5"/><circle cx="45" cy="105" r="2.5"/><path d="M30,95 L45,105"/></g></svg>') !important;
        background-repeat: repeat !important;
        background-attachment: fixed !important;
        color: #ffffff !important;
    }

    /* 4. ปรับข้อความทั้งหมดให้เป็นสีขาว ไม่กลืนกับพื้นหลัง */
    p, span, label, .stMarkdown p, .stCaption, 
    .stSelectbox label, .stTextArea label, .stTextInput label, .stDateInput label,
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
        color: #ffffff !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    /* ซ่อน Header มาตรฐานของ Streamlit */
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1220px !important; }

    /* Top Navbar สไตล์สิริเวช */
    .srh-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #070d1f;
        padding: 12px 28px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 22px;
    }
    .srh-brand-group {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .srh-title-th {
        font-size: 19px;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
        line-height: 1.1;
    }
    .srh-title-en {
        font-size: 10.5px;
        color: #94a3b8 !important;
        margin: 0;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .srh-tag-thg {
        background: #0284c7;
        color: #ffffff !important;
        font-size: 11px;
        font-weight: 800;
        padding: 2px 7px;
        border-radius: 4px;
        margin-left: 6px;
    }
    .srh-badges {
        display: flex;
        gap: 6px;
        align-items: center;
    }
    .badge-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .srh-nav-links {
        display: flex;
        align-items: center;
        gap: 22px;
        font-size: 14px;
        font-weight: 500;
        color: #cbd5e1 !important;
    }

    /* แบนเนอร์หลัก + กระต่ายเกาะ */
    .banner-outer-stage {
        position: relative;
        width: 100%;
        max-width: 1060px;
        margin: 10px auto 25px auto;
        padding: 0 45px;
    }
    .banner-display-box {
        position: relative;
        width: 100%;
        border-radius: 18px;
        overflow: hidden;
        border: 2px solid #38bdf8;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55);
        background: #0f172a;
        height: 330px;
    }
    .banner-visual {
        width: 100%;
        height: 330px;
        object-fit: cover;
        display: block;
        transition: opacity 0.8s ease-in-out;
    }

    /* สติกเกอร์กระต่ายเกาะขอบ */
    .rabbit-mascot {
        position: absolute;
        bottom: -22px;
        width: 125px;
        height: auto;
        z-index: 25;
        filter: drop-shadow(0 10px 14px rgba(0,0,0,0.5));
        pointer-events: none;
    }
    .rabbit-left {
        left: -20px;
        transform: scaleX(-1);
        animation: rabbitFloatLeft 3s ease-in-out infinite alternate;
    }
    .rabbit-right {
        right: -20px;
        animation: rabbitFloat 3s ease-in-out infinite alternate;
    }
    @keyframes rabbitFloat {
        0% { transform: translateY(0px); }
        100% { transform: translateY(-7px); }
    }
    @keyframes rabbitFloatLeft {
        0% { transform: scaleX(-1) translateY(0px); }
        100% { transform: scaleX(-1) translateY(-7px); }
    }

    /* ปุ่มลูกศรข้างแบนเนอร์ */
    .side-arrow-btn {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(8px);
        color: #ffffff !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        width: 42px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        cursor: pointer;
        z-index: 22;
        transition: all 0.2s ease;
        text-decoration: none;
        user-select: none;
    }
    .side-arrow-btn:hover {
        background: rgba(2, 132, 199, 0.9) !important;
        transform: translateY(-50%) scale(1.08);
    }
    .arrow-left { left: 52px; }
    .arrow-right { right: 52px; }

    /* แท็บเมนูหลัก */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background-color: #070d1f !important;
        border-radius: 16px;
        padding: 6px;
        gap: 8px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 25px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #94a3b8 !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] p {
        color: inherit !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background: #0284c7 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff !important;
    }

    /* กล่องการ์ดเนื้อหา */
    .premium-card {
        background: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 26px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
    }

    /* การ์ดตัวเลขสรุปผลสถิติ (KPI Card) */
    .metric-card-box {
        background: #1e293b;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .metric-card-box span {
        color: #ffffff !important;
    }

    /* กล่องข้อความ และ Selectbox */
    .stTextInput input, .stTextArea textarea {
        color: #ffffff !important;
        background-color: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
    }

    /* กล่องตัวเลือก Selectbox และตัวหนังสือด้านใน */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] span {
        color: #ffffff !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #1e293b !important;
    }
    div[data-baseweb="popover"] li {
        color: #ffffff !important;
    }

    /* ========================================================= */
    /* ปรับแต่งปุ่มกดทุกประเภทให้เป็นสีฟ้าสิริเวชแบบหน้า Login */
    /* ========================================================= */
    .stButton > button, 
    [data-testid="stFormSubmitButton"] > button,
    [data-testid="stFileUploader"] section button,
    .stDownloadButton > button {
        background-color: #0284c7 !important;
        background-image: none !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover, 
    [data-testid="stFormSubmitButton"] > button:hover,
    [data-testid="stFileUploader"] section button:hover,
    .stDownloadButton > button:hover {
        background-color: #0369a1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.5) !important;
        color: #ffffff !important;
    }

    .stButton > button, 
    [data-testid="stFormSubmitButton"] > button,
    [data-testid="stFileUploader"] section button,
    .stDownloadButton > button {
        background-color: #0284c7 !important;
        background-image: none !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 8px 18px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
        min-width: 130px !important; /* ขยายความกว้างปุ่มไม่ให้ข้อความเบียด */
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .stButton > button:hover, 
    [data-testid="stFormSubmitButton"] > button:hover,
    [data-testid="stFileUploader"] section button:hover,
    .stDownloadButton > button:hover {
        background-color: #0369a1 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.5) !important;
        color: #ffffff !important;
    }

    .stButton > button p, 
    .stButton > button span,
    .stDownloadButton > button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* -------------------------------------------------- */
    /* แก้ไขปัญหาตัวหนังสือในปุ่ม File Uploader ซ้อนทับกัน */
    /* -------------------------------------------------- */
    [data-testid="stFileUploader"] section button {
        font-size: 0 !important; /* ซ่อนข้อความซ้อนเดิมทั้งหมด */
        line-height: 1 !important;
        white-space: nowrap !important;
    }
    
    /* สร้างข้อความใหม่ที่อ่านง่าย ไม่ซ้อนกัน */
    [data-testid="stFileUploader"] section button::after {
        content: "เลือกไฟล์ (Browse)" !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* ซ่อน span ภายในปุ่ม File Uploader ทั้งหมดเพื่อตัดปัญหาการทับซ้อน */
    [data-testid="stFileUploader"] section button * {
        display: none !important;
    }

    /* ปรับแต่งพื้นที่กล่อง Dropzone ของ File Uploader */
    [data-testid="stFileUploader"] section {
        background-color: #1e293b !important;
        border: 2px dashed rgba(56, 189, 248, 0.35) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
    }
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {
        color: #94a3b8 !important;
    }

    #cursor-spotlight {
        position: fixed;
        top: 0;
        left: 0;
        width: 450px;
        height: 450px;
        border-radius: 50%;
        pointer-events: none; /* เพื่อให้คลิกผ่านทะลุไปยังปุ่มและอินพุตได้ตามปกติ */
        background: radial-gradient(circle, rgba(56, 189, 248, 0.14) 0%, rgba(2, 132, 199, 0.06) 40%, rgba(11, 19, 41, 0) 75%);
        transform: translate(-50%, -50%);
        transition: opacity 0.3s ease-out;
        z-index: 0;
        opacity: 0;
    }

    /* ดันคอนเทนต์หลักให้อยู่เหนือแสงไฟ */
    .srh-navbar-center, .banner-outer-stage, [data-testid="stTabs"], .premium-card, .srh-footer-stage {
        position: relative;
        z-index: 2;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================== TOP HEADER NAVBAR ==============================
# ============================== TOP HEADER NAVBAR (CENTER LOGO & MEGA DROPDOWN) ==============================
# ============================== TOP HEADER NAVBAR (CSS TOGGLE HAMBURGER + LOGO CENTER) ==============================
def render_header_navbar():
    # ตรวจสอบการคลิก Logout จาก Dropdown เมนู
    if st.query_params.get("action") == "logout":
        st.query_params.clear()
        if "auth" in st.session_state:
            del st.session_state.auth
        st.rerun()

    # 1. ค้นหาไฟล์โลโก้ .avif (หรือ .webp/.png สำรอง)
    curr_dir = Path(__file__).resolve().parent
    proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent

    logo_paths = [
        curr_dir / "assets" / "logo.avif",
        proj_root / "assets" / "logo.avif",
        curr_dir / "assets" / "sirivej_logo.avif",
        proj_root / "assets" / "sirivej_logo.avif",
        Path("assets/logo.avif").resolve(),
        curr_dir / "assets" / "logo.webp",
        proj_root / "assets" / "logo.webp",
        curr_dir / "assets" / "logo.png",
        proj_root / "assets" / "logo.png",
    ]

    logo_src = ""
    for p in logo_paths:
        if p.is_file():
            suffix = p.suffix.lower().replace(".", "")
            mime = f"image/{suffix}"
            encoded = base64.b64encode(p.read_bytes()).decode()
            logo_src = f"data:{mime};base64,{encoded}"
            break

    # 2. เรนเดอร์ CSS สำหรับ Navbar และ Dropdown แบบ Pure CSS Toggle
    st.markdown("""
<style>
.srh-navbar-center {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #070d1f;
    padding: 10px 24px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
    margin-bottom: 22px;
    z-index: 1000;
}

/* ล็อกความกว้างสองฝั่งให้เท่ากัน เพื่อให้โลโก้อยู่ตรงกลางสมมาตร 100% */
.nav-side-slot {
    flex: 0 0 50px;
    display: flex;
    align-items: center;
}
.nav-side-slot.left {
    justify-content: flex-start;
}
.nav-side-slot.right {
    justify-content: flex-end;
}

.nav-center-slot {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.nav-center-slot img {
    max-height: 48px;
    width: auto;
    object-fit: contain;
}

/* Checkbox Toggle Trick สำหรับเปิด-ปิดเมนู */
#menu-toggle-checkbox {
    display: none;
}

.hamburger-btn-label {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 5px;
    width: 42px;
    height: 42px;
    padding: 9px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
    box-sizing: border-box;
}
.hamburger-btn-label:hover {
    background: rgba(2, 132, 199, 0.25);
    border-color: #38bdf8;
}
.hamburger-line {
    width: 100%;
    height: 2.5px;
    background-color: #ffffff;
    border-radius: 2px;
}

/* คอนเทนเนอร์ Dropdown เมนู */
.dropdown-left-parent {
    position: relative;
    display: inline-block;
}

.dropdown-left-menu {
    display: none;
    position: absolute;
    left: 0;
    top: calc(100% + 14px);
    width: 740px;
    background: #0b1528;
    border-radius: 18px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.85);
    padding: 20px;
    grid-template-columns: 240px 1fr 1fr;
    gap: 16px;
    z-index: 2000;
}

/* เมื่อ Checkbox ถูกติ๊ก ให้เมนูแสดงขึ้นมา */
#menu-toggle-checkbox:checked ~ .dropdown-left-menu {
    display: grid;
    animation: fadeInMenu 0.25s ease forwards;
}

@keyframes fadeInMenu {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}

.mega-left-card {
    background: linear-gradient(180deg, #094067 0%, #062b46 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 14px;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.mega-left-card h4 {
    color: #ffffff !important;
    font-size: 18px;
    font-weight: 800;
    margin: 0 0 8px 0;
}
.mega-left-card p {
    color: #93c5fd !important;
    font-size: 12px;
    line-height: 1.5;
    margin: 0;
}

/* ปุ่ม Logout ใน Dropdown สไตล์ปุ่มสีฟ้า */
.dropdown-logout-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    background: #0284c7;
    color: #ffffff !important;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13.5px;
    font-weight: 700;
    text-decoration: none;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
    transition: all 0.2s ease;
    margin-top: 15px;
    text-align: center;
}
.dropdown-logout-btn:hover {
    background: #0369a1;
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(2, 132, 199, 0.6);
}

.mega-col {
    background: #0f1c34;
    border-radius: 14px;
    padding: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.mega-col-title {
    color: #ffffff !important;
    font-size: 14px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 10px;
}
.mega-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.mega-list li {
    padding: 6px 0;
    font-size: 12.5px;
    color: #cbd5e1 !important;
    cursor: pointer;
    transition: color 0.15s ease;
    display: flex;
    align-items: center;
    gap: 6px;
}
.mega-list li:hover {
    color: #38bdf8 !important;
}
.mega-list li::before {
    content: "•";
    color: #0284c7;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

    # 3. เรนเดอร์ HTML Navbar (ชิดซ้าย-กลาง-ชิดขวาแบบสมดุล)
    center_logo_html = f'<img src="{logo_src}">' if logo_src else '<span style="color:#ffffff;font-weight:800;font-size:18px;">🏥 SIRIVEJ HOSPITAL</span>'

    navbar_html_parts = [
        '<div class="srh-navbar-center">',
        '  <div class="nav-side-slot left">',
        '    <div class="dropdown-left-parent">',
        '      <input type="checkbox" id="menu-toggle-checkbox">',
        '      <label for="menu-toggle-checkbox" class="hamburger-btn-label" title="เมนูเพิ่มเติม">',
        '        <div class="hamburger-line"></div>',
        '        <div class="hamburger-line"></div>',
        '        <div class="hamburger-line"></div>',
        '      </label>',
        '      <div class="dropdown-left-menu">',
        '        <div class="mega-left-card">',
        '          <div>',
        '            <h4>เกี่ยวกับเรา</h4>',
        '            <p>เลือกหัวข้อที่ต้องการ เพื่อเข้าถึงข้อมูลและบริการของโรงพยาบาลสิริเวช</p>',
        '          </div>',
        '          <a href="?action=logout" class="dropdown-logout-btn">🚪 ออกจากระบบ (LOG OUT)</a>',
        '        </div>',
        '        <div class="mega-col">',
        '          <div class="mega-col-title"><span>รู้จักโรงพยาบาล</span> ❯</div>',
        '          <ul class="mega-list">',
        '            <li>คณะผู้บริหาร</li>',
        '            <li>ประวัติความเป็นมา</li>',
        '            <li>วิสัยทัศน์และพันธกิจ</li>',
        '          </ul>',
        '        </div>',
        '        <div class="mega-col">',
        '          <div class="mega-col-title"><span>ข่าวสารและกิจกรรม</span> ❯</div>',
        '          <ul class="mega-list">',
        '            <li>ข่าวประชาสัมพันธ์</li>',
        '            <li>บทความสุขภาพ</li>',
        '            <li>โปรโมชั่นและแพ็กเกจ</li>',
        '          </ul>',
        '        </div>',
        '      </div>',
        '    </div>',
        '  </div>',
        f'  <div class="nav-center-slot">{center_logo_html}</div>',
        '  <div class="nav-side-slot right"></div>',
        '</div>'
    ]

    st.markdown("\n".join(navbar_html_parts), unsafe_allow_html=True)

# ============================== BANNER CAROUSEL (AUTO-SLIDE 4.5s + SIDE ARROWS) ==============================
def render_reference_banner():
    # 1. จัดเตรียมภาพ Banner หลัก
    header_local = BASE_DIR / "assets" / "Header.jpg"
    header_root = ROOT / "assets" / "Header.jpg"

    if header_local.exists():
        encoded_header = base64.b64encode(header_local.read_bytes()).decode()
        header_src = f"data:image/jpeg;base64,{encoded_header}"
    elif header_root.exists():
        encoded_header = base64.b64encode(header_root.read_bytes()).decode()
        header_src = f"data:image/jpeg;base64,{encoded_header}"
    else:
        header_src = "https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?q=80&w=1600&auto=format&fit=crop"

    slides = [
        header_src,
        "https://images.unsplash.com/photo-1586773860418-d37222d8fce3?q=80&w=1600&auto=format&fit=crop",
        "https://images.unsplash.com/photo-1516549655169-df83a0774514?q=80&w=1600&auto=format&fit=crop"
    ]

    # 2. ค้นหาไฟล์ PNG กระต่ายพยาบาลทุกจุดที่เป็นไปได้
    curr_dir = Path(__file__).resolve().parent
    project_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent

    potential_mascot_paths = [
        curr_dir / "assets" / "rabbit_mascot.png",
        project_root / "assets" / "rabbit_mascot.png",
        Path("assets/rabbit_mascot.png").resolve(),
        curr_dir / "rabbit_mascot.png",
        project_root / "rabbit_mascot.png",
    ]

    bunny_data = None
    matched_path = None
    for p in potential_mascot_paths:
        if p.is_file():
            encoded_b = base64.b64encode(p.read_bytes()).decode()
            bunny_data = f"data:image/png;base64,{encoded_b}"
            matched_path = p
            break

    # แจ้งเตือนกรณีหารูปไม่เจอจริง ๆ เพื่อให้คุณทราบที่อยู่ของไฟล์
    if not bunny_data:
        st.warning(f"⚠️ ยังไม่พบไฟล์ rabbit_mascot.png กรุณานำไฟล์ไปวางที่: {potential_mascot_paths[1]}")
        # ใช้ SVG ชั่วคราวกรณีหาไฟล์ไม่เจอ
        fallback_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 130 130"><ellipse cx="45" cy="30" rx="9" ry="26" fill="%23ffffff" stroke="%23334155" stroke-width="2.5"/><ellipse cx="45" cy="30" rx="4.5" ry="18" fill="%23f472b6"/><ellipse cx="85" cy="30" rx="9" ry="26" fill="%23ffffff" stroke="%23334155" stroke-width="2.5"/><ellipse cx="85" cy="30" rx="4.5" ry="18" fill="%23f472b6"/><ellipse cx="65" cy="74" rx="40" ry="36" fill="%23ffffff" stroke="%23334155" stroke-width="2.5"/><polygon points="50,48 80,48 72,36 58,36" fill="%23ffffff" stroke="%230284c7" stroke-width="2"/><rect x="63" y="38" width="4" height="8" fill="%230284c7"/><rect x="61" y="40" width="8" height="4" fill="%230284c7"/><ellipse cx="50" cy="72" rx="4.5" ry="6.5" fill="%231e293b"/><ellipse cx="80" cy="72" rx="4.5" ry="6.5" fill="%231e293b"/><circle cx="51.5" cy="70" r="1.8" fill="%23ffffff"/><circle cx="81.5" cy="70" r="1.8" fill="%23ffffff"/><ellipse cx="40" cy="78" rx="6" ry="3" fill="%23fbcfe8"/><ellipse cx="90" cy="78" rx="6" ry="3" fill="%23fbcfe8"/><polygon points="65,77 61,74 69,74" fill="%23f43f5e"/><path d="M60,82 Q65,86 70,82" stroke="%23334155" stroke-width="2.2" fill="none" stroke-linecap="round"/><ellipse cx="98" cy="85" rx="10" ry="8" fill="%23ffffff" stroke="%23334155" stroke-width="2.5"/><ellipse cx="98" cy="100" rx="10" ry="8" fill="%23ffffff" stroke="%23334155" stroke-width="2.5"/></svg>""".strip().replace("\n", "").replace('"', '%22')
        bunny_data = f"data:image/svg+xml;utf8,{fallback_svg}"

    # 3. เรนเดอร์ HTML Banner และตัวการ์ตูน
    slides_js = str(slides)
    st.markdown(f"""
    <div class="banner-outer-stage">
        <img class="rabbit-mascot rabbit-left" src="{bunny_data}">
        <img class="rabbit-mascot rabbit-right" src="{bunny_data}">
        <div class="banner-display-box">
            <img id="main-banner-slide" class="banner-visual" src="{slides[0]}">
        </div>
        <div class="side-arrow-btn arrow-left" onclick="slideBanner(-1)">‹</div>
        <div class="side-arrow-btn arrow-right" onclick="slideBanner(1)">›</div>
    </div>

    <script>
    (function() {{
        const slideImgs = {slides_js};
        let curIdx = 0;
        const bannerEl = document.getElementById("main-banner-slide");

        window.slideBanner = function(direction) {{
            curIdx = (curIdx + direction + slideImgs.length) % slideImgs.length;
            if(bannerEl) {{
                bannerEl.style.opacity = 0;
                setTimeout(() => {{
                    bannerEl.src = slideImgs[curIdx];
                    bannerEl.style.opacity = 1;
                }}, 250);
            }}
        }};

        setInterval(() => {{
            window.slideBanner(1);
        }}, 4500);
    }})();
    </script>
    """, unsafe_allow_html=True)

# ============================== BANNER ประจำแต่ละแท็บหน้า ==============================
def banner(page_name: str):
    """ฟังก์ชันแสดง Banner เฉพาะของแต่ละหน้าที่สลับตามหัวข้อที่เลือก"""
    banner_map = {
        "analyze": "banner_analyze.png",
        "summary": "banner_summary.png",
        "settings": "banner_settings.png",
        "profile": "banner_profile.png",
    }
    filename = banner_map.get(page_name, "banner.png")
    banner_path = ROOT / "assets" / filename
    if not banner_path.exists():
        banner_path = BASE_DIR / "assets" / filename

    if not banner_path.exists():
        return

    encoded = base64.b64encode(banner_path.read_bytes()).decode()
    st.markdown(
        f"""
        <div style="margin-bottom: 24px; text-align: center;">
            <img src="data:image/png;base64,{encoded}"
                 style="
                    width: 100%;
                    max-height: 260px;
                    object-fit: cover;
                    border-radius: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
                 ">
        </div>
        """,
        unsafe_allow_html=True
    )

    render_critical_incident_banner()

# ============================== PAGES ==============================
def page_analyze():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 25px;'>
        <h3 style='color: #38bdf8; font-weight: 800; margin-bottom: 4px;'>วิเคราะห์ความคิดเห็นของผู้รับบริการ</h3>
        <p style='color: #ffffff; font-size: 14px;'>Customer Sentiment Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    category = st.selectbox(
        "📌 เลือกแผนก / ส่วนงานบริการที่ต้องการประเมินผล:",
        [
            "บริการทั่วไปของโรงพยาบาล",
            "แผนกผู้ป่วยนอก (OPD)",
            "แผนกผู้ป่วยใน (IPD)",
            "ศูนย์ตรวจสุขภาพ (Check-up)",
            "แผนกฉุกเฉินและอุบัติเหตุ (ER)",
            "แผนกศัลยกรรม (Surgery)",
            "แผนกสูตินรีเวช (OB-GYN)",
            "แผนกกายภาพบำบัด (PT)",
            "แผนกทันตกรรม (Dental)",
            "แผนกจักษุ",
            "แผนกอายุรกรรม",
            "แผนกออร์โธปิดิกส์",
            "งานวิชาการและศูนย์มะเร็ง",
        ]
    )

    tab_manual, tab_bulk, tab_fb = st.tabs(["✍️ ป้อนข้อความเดี่ยว", "📁 อัปโหลดไฟล์ชุดใหญ่ (Bulk)", "🔗 ดึงจาก Facebook"])

    with tab_manual:
        txt = st.text_area("ข้อความความคิดเห็น:", height=110, placeholder="พิมพ์ความคิดเห็น เช่น หมอตรวจละเอียดดีมาก แต่ที่จอดรถน้อยและค่ายาค่อนข้างแพง...")
        if st.button("🚀 ประมวลผลความคิดเห็น (หลายมิติ)", key="btn_manual_exec", type="primary", use_container_width=True):
            if not txt.strip():
                st.warning("กรุณากรอกข้อความก่อนวิเคราะห์")
            else:
                # 1. เรียกใช้งานฟังก์ชันวิเคราะห์หลายมิติ
                res = analyze_aspects_smart(txt)
                
                label = res.get("overall", "neu")
                
                # 2. บันทึกลง Log ประวัติของโรงพยาบาล
                append_log(txt, label, category)

                risk = detect_critical_risk(txt)
                if risk["is_critical"]:
                    send_alert_notification(txt, category, risk["keywords"])
                    st.toast("🚨 ตรวจพบเคสวิกฤต! ส่งข้อมูลแจ้งเตือนไปยังทีมบริหารความเสี่ยงแล้ว", icon="⚠️")

                # 3. ส่งข้อมูลเข้า Session เพื่อให้อัปเดตการ์ดคะแนนดาว Real-time
                if "analysis_history" not in st.session_state:
                    st.session_state.analysis_history = []
                st.session_state.analysis_history.append({
                    "department": category,
                    "sentiment": label,
                    "text": txt
                })

                # 4. บันทึกผลลัพธ์มิติย่อยไว้แสดงผลหลัง Rerun
                st.session_state["last_aspect_result"] = {
                    "result": res,
                    "category": category,
                    "text": txt
                }
                st.rerun()

        # แสดงกล่องการ์ดสรุปผลลัพธ์รายมิติหลังกดประมวลผล
        # แสดงกล่องการ์ดสรุปผลลัพธ์รายมิติหลังกดประมวลผล
    if "last_aspect_result" in st.session_state:
            data = st.session_state["last_aspect_result"]
            r = data["result"]
            overall_color = "#10b981" if r["overall"] == "pos" else "#ef4444" if r["overall"] == "neg" else "#94a3b8"
            overall_status = "เชิงบวก (Positive)" if r["overall"] == "pos" else "เชิงลบ (Negative)" if r["overall"] == "neg" else "ทั่วไป/เป็นกลาง (Neutral)"

            def get_chip(aspect_val, label_text):
                color_map = {
                    "pos": ("#22c55e", "rgba(34, 197, 94, 0.15)", "👍 พอใจ"),
                    "neg": ("#ef4444", "rgba(239, 68, 68, 0.15)", "👎 ต้องปรับปรุง"),
                    "neu": ("#94a3b8", "rgba(148, 163, 184, 0.15)", "➖ ทั่วไป"),
                    "not_mentioned": ("#64748b", "rgba(100, 116, 139, 0.1)", "— ไม่ระบุ")
                }
                c, bg, text = color_map.get(aspect_val, color_map["not_mentioned"])
                return f'<div style="background:{bg}; border:1px solid {c}55; border-radius:10px; padding:10px; text-align:center;"><div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">{label_text}</div><strong style="color:{c}; font-size:14px;">{text}</strong></div>'

            asp = r.get("aspects", {})
            chip_doc = get_chip(asp.get('doctor'), '👨‍⚕️ แพทย์ & การรักษา')
            chip_nurse = get_chip(asp.get('nurse_staff'), '👩‍⚕️ พยาบาล & จนท.')
            chip_fac = get_chip(asp.get('facility'), '🏥 สถานที่ & ที่จอดรถ')
            chip_price = get_chip(asp.get('price_time'), '⏱️ ราคา & เวลารอคอย')

            aspect_chips_html = f'<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin-top:16px;">{chip_doc}{chip_nurse}{chip_fac}{chip_price}</div>'

            result_box_html = (
                f'<div style="padding: 20px; border-radius: 14px; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.3); border-left: 6px solid {overall_color}; margin-top: 15px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<span style="color: #94a3b8; font-size: 13px;">ผลการประเมินภาพรวม:</span>'
                f'<span style="color: #38bdf8; font-size: 12px; font-weight:600;">แผนก: {data["category"]}</span>'
                f'</div>'
                f'<strong style="font-size: 22px; color: {overall_color};">{overall_status}</strong>'
                f'{aspect_chips_html}'
                f'</div>'
            )

            st.markdown(result_box_html, unsafe_allow_html=True)

    with tab_bulk:
        # ส่วนหัวและปุ่มเปิด Pop-up แปลง PDF
        c_bulk_title, c_bulk_btn = st.columns([2.5, 1.5])
        with c_bulk_title:
            st.markdown("#### 📁 อัปโหลดไฟล์เพื่อประเมินความคิดเห็นจำนวนมาก")
        with c_bulk_btn:
            if st.button("📑 นำเข้าจากโฟลเดอร์ PDF ร้องเรียน", key="btn_open_pdf_modal", use_container_width=True):
                open_pdf_batch_converter_dialog()

        # จุดสำคัญ: ต้องมีบรรทัดสร้างตัวแปร uploaded_file ก่อนเรียก if uploaded_file is not None:
        uploaded_file = st.file_uploader(
            "เลือกไฟล์ CSV หรือ Excel ที่มีคอลัมน์ข้อความความคิดเห็น:",
            type=["csv", "xlsx", "xls"],
            key="bulk_uploader"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    bulk_df = pd.read_csv(uploaded_file)
                else:
                    bulk_df = pd.read_excel(uploaded_file)
                st.success(f"โหลดข้อมูลสำเร็จ: พบทั้งหมด {len(bulk_df):,} แถว")
            except Exception as e:
                st.error(f"ไม่สามารถอ่านไฟล์ได้: {e}")
                bulk_df = None

            if bulk_df is not None and not bulk_df.empty:
                c1, c2 = st.columns(2)
                with c1:
                    text_col = st.selectbox(
                        "เลือกคอลัมน์ข้อความความคิดเห็น:",
                        bulk_df.columns,
                        index=0
                    )
                with c2:
                    dept_col_options = ["(ใช้แผนกเดียวกับตัวเลือกด้านบน)"] + list(bulk_df.columns)
                    selected_dept_col = st.selectbox(
                        "เลือกคอลัมน์ระบุแผนก (ถ้ามี):",
                        dept_col_options,
                        index=0
                    )

                st.write("ตัวอย่างข้อมูลที่ตรวจพบ:")
                st.dataframe(bulk_df[[text_col]].head(3), use_container_width=True)

                if st.button("⚡ เริ่มประมวลผลหลายมิติด้วย AI (Aspect-Based)", key="btn_run_bulk_aspect", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results_overall = []
                    results_doctor = []
                    results_nurse = []
                    results_facility = []
                    results_price_time = []
                    results_dept = []

                    total_rows = len(bulk_df)

                    for idx, row in bulk_df.iterrows():
                        msg = str(row[text_col])
                        
                        if selected_dept_col != "(ใช้แผนกเดียวกับตัวเลือกด้านบน)" and selected_dept_col in bulk_df.columns:
                            row_dept = str(row[selected_dept_col])
                        else:
                            row_dept = category

                        analysis = analyze_aspects_smart(msg)
                        overall_label = analysis.get("overall", "neu")
                        asp = analysis.get("aspects", {})

                        results_overall.append(overall_label)
                        results_doctor.append(asp.get("doctor", "not_mentioned"))
                        results_nurse.append(asp.get("nurse_staff", "not_mentioned"))
                        results_facility.append(asp.get("facility", "not_mentioned"))
                        results_price_time.append(asp.get("price_time", "not_mentioned"))
                        results_dept.append(row_dept)

                        append_log(msg, overall_label, row_dept)

                        if "analysis_history" not in st.session_state:
                            st.session_state.analysis_history = []
                        st.session_state.analysis_history.append({
                            "department": row_dept,
                            "sentiment": overall_label,
                            "text": msg
                        })

                        progress = (idx + 1) / total_rows
                        progress_bar.progress(progress)
                        status_text.text(f"กำลังประมวลผลรายการที่ {idx + 1}/{total_rows}...")

                    bulk_df["แผนกที่ประเมิน"] = results_dept
                    bulk_df["ผลภาพรวม (Overall)"] = results_overall
                    bulk_df["แพทย์และการรักษา (Doctor)"] = results_doctor
                    bulk_df["พยาบาลและเจ้าหน้าที่ (Nurse & Staff)"] = results_nurse
                    bulk_df["สถานที่และที่จอดรถ (Facility)"] = results_facility
                    results_label_map = {
                        "pos": "👍 พอใจ (pos)",
                        "neg": "👎 ต้องปรับปรุง (neg)",
                        "neu": "➖ ทั่วไป (neu)",
                        "not_mentioned": "— ไม่ระบุ"
                    }
                    bulk_df["ราคาและเวลารอคอย (Price & Time)"] = [results_label_map.get(x, x) for x in results_price_time]

                    st.session_state["last_bulk_df"] = bulk_df
                    status_text.success("🎉 ประมวลผลความคิดเห็นทั้งหมดเรียบร้อยแล้ว!")
                    st.rerun()

        # แสดงตารางผลลัพธ์และปุ่มดาวน์โหลด
        if "last_bulk_df" in st.session_state:
            res_df = st.session_state["last_bulk_df"]
            st.markdown("---")
            st.markdown("#### 📊 ผลลัพธ์การวิเคราะห์จำแนกมิติ (ตัวอย่าง 5 แถวแรก)")
            st.dataframe(res_df.head(5), use_container_width=True)

            csv_data = res_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="📥 ดาวน์โหลดไฟล์ผลวิเคราะห์หลายมิติ (CSV)",
                data=csv_data,
                file_name=f"sirivej_aspect_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def page_summary():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#38bdf8; font-weight:800;'>📊 แดชบอร์ดสรุปผลความคิดเห็น</h3>", unsafe_allow_html=True)

    df_log = load_log()

    col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.3, 0.3])
    if not df_log.empty:
        dmin = pd.to_datetime(df_log["timestamp"]).min().date()
        dmax = pd.to_datetime(df_log["timestamp"]).max().date()
        cat_options = df_log["category"].unique().tolist()
    else:
        dmin = dmax = datetime.now().date()
        cat_options = ["ทั่วไป"]

    with col1:
        date_from = st.date_input("จากวันที่", value=dmin)
    with col2:
        date_to = st.date_input("ถึงวันที่", value=dmax)
    with col3:
        label_filter = st.multiselect("ผลการประเมิน", ["pos", "neu", "neg"], default=["pos", "neu", "neg"])
    with col4:
        category_filter = st.multiselect("แผนกบริการ", cat_options, default=cat_options)

    if not df_log.empty:
        dfl = df_log.copy()
        dfl["dt"] = pd.to_datetime(dfl["timestamp"]).dt.date
        dfl = dfl[
            (dfl["dt"] >= date_from) &
            (dfl["dt"] <= date_to) &
            (dfl["label"].isin(label_filter)) &
            (dfl["category"].isin(category_filter))
        ]
    else:
        dfl = df_log

    df_sum = make_summary(dfl)
    total = len(dfl)
    pos = int(df_sum.loc[df_sum["label"] == "pos", "count"].values[0]) if not df_sum.empty else 0
    neu = int(df_sum.loc[df_sum["label"] == "neu", "count"].values[0]) if not df_sum.empty else 0
    neg = int(df_sum.loc[df_sum["label"] == "neg", "count"].values[0]) if not df_sum.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    metrics = [
        (k1, "ความคิดเห็นทั้งหมด", total, "#38bdf8"),
        (k2, "เชิงบวก (Positive)", pos, "#10b981"),
        (k3, "เป็นกลาง (Neutral)", neu, "#94a3b8"),
        (k4, "เชิงลบ (Negative)", neg, "#ef4444")
    ]
    for col, title, val, col_hex in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card-box">
                <span style='color: #ffffff; font-size: 13px; font-weight:500;'>{title}</span>
                <h2 style='color: {col_hex}; margin: 5px 0 0 0; font-weight: 800;'>{val:,}</h2>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    g1, g2 = st.columns([0.55, 0.45])
    color_map = {"pos": "#10b981", "neu": "#94a3b8", "neg": "#ef4444"}

    with g1:
        st.caption("สัดส่วนตามผลลัพธ์ (Bar Chart)")
        if df_sum["count"].sum() == 0:
            st.info("ไม่พบข้อมูลในช่วงที่เลือก")
        else:
            fig_bar = px.bar(
                df_sum, x="label", y="count", text="count",
                color="label", color_discrete_map=color_map, height=320
            )
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)

    with g2:
        st.caption("ร้อยละความพึงพอใจ (Donut Chart)")
        if df_sum["count"].sum() == 0:
            st.info("ไม่พบข้อมูลในช่วงที่เลือก")
        else:
            fig_pie = px.pie(
                df_sum, names="label", values="count", hole=0.45,
                color="label", color_discrete_map=color_map, height=320
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)

    st.caption("ตารางประวัติความคิดเห็นที่กรอง:")
    st.dataframe(dfl.sort_values("timestamp", ascending=False).reset_index(drop=True), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def page_settings():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#38bdf8; font-weight:800;'>⚙️ การเชื่อมต่อและส่งออก Google Sheets</h3>", unsafe_allow_html=True)

    df_log = load_log()
    sheet_id = st.text_input("Spreadsheet ID หรือ URL:", value=st.session_state.get("gsheet_id", ""))
    ws_log_name = st.text_input("ชื่อชีตสำหรับบันทึก Log:", value="Sentiment_Logs")
    ws_sum_name = st.text_input("ชื่อชีตสำหรับบันทึก Summary:", value="Sentiment_Summary")
    clear_first = st.checkbox("ล้างข้อมูลเก่าในชีตก่อนเขียนใหม่", value=True)

    st.session_state.gsheet_id = sheet_id

    c_test, c_exp1, c_exp2 = st.columns(3)
    with c_test:
        if st.button("🔗 ทดสอบการเชื่อมต่อ", use_container_width=True):
            ok, msg = test_gsheet_connection(sheet_id)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
    with c_exp1:
        if st.button("📤 ส่งออก Logs ทั้งหมด", use_container_width=True):
            if df_log.empty:
                st.info("ไม่มีข้อมูล Log")
            else:
                sid = _extract_sheet_id(sheet_id)
                export_df_to_gsheet(df_log, sid, ws_log_name, clear_first)
                st.success(f"ส่งออก {len(df_log)} รายการสำเร็จ!")
    with c_exp2:
        if st.button("📊 ส่งออก Summary", use_container_width=True):
            sid = _extract_sheet_id(sheet_id)
            export_df_to_gsheet(make_summary(df_log), sid, ws_sum_name, clear_first)
            st.success("ส่งออกสถิติสรุปผลสำเร็จ!")
    st.markdown("</div>", unsafe_allow_html=True)

def page_profile():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    auth = st.session_state.get("auth", {})
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:20px;'>
        <span style='font-size: 55px;'>👨‍⚕️</span>
        <div>
            <h3 style='margin:0; color:#ffffff;'>{auth.get('display_name', 'ผู้ใช้งาน')}</h3>
            <p style='margin:0; color:#38bdf8; font-weight:600;'>สิทธิ์การใช้งาน: {auth.get('role', 'user').upper()}</p>
        </div>
    </div>
    <hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;'>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚪 ออกจากระบบ", use_container_width=True):
            del st.session_state.auth
            st.rerun()
    with c2:
        if st.button("🗑️ ล้างประวัติแคชในเครื่อง", type="primary", use_container_width=True):
            if LOG_PATH.exists():
                LOG_PATH.unlink()
            st.success("ล้างข้อมูลแคชเรียบร้อยแล้ว")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================== REAL-TIME DEPARTMENT METRICS CARDS (GLASSMORPHISM) ==============================
# ============================== REAL-TIME DEPARTMENT METRICS CARDS (GLASSMORPHISM) ==============================
# ============================== REAL-TIME DEPARTMENT METRICS CARDS (GLASSMORPHISM) ==============================
# ============================== REAL-TIME DEPARTMENT METRICS CARDS (GLASSMORPHISM) ==============================
# ============================== REAL-TIME DEPARTMENT METRICS CARDS (LINKED TO LOG_PATH) ==============================
# ============================== REAL-TIME DEPARTMENT METRICS CARDS (AUTO SORT BY RATING) ==============================
# ============================== RADAR CHART & HEATMAP ANALYTICS ==============================
# ============================== RADAR CHART & HEATMAP ANALYTICS (LINKED LOG_PATH) ==============================
def render_aspect_analytics_section():
    st.markdown("---")
    st.markdown("""
        <div style="margin-top: 15px; margin-bottom: 20px;">
            <h3 style="color: #ffffff; margin-bottom: 4px;">📈 การวิเคราะห์จุดแข็ง-จุดอ่อนรายด้าน (Aspect-Based Diagnostics)</h3>
            <p style="color: #94a3b8; font-size: 14px; margin: 0;">ประเมินคะแนนความพึงพอใจ 4 มิติ: ด้านแพทย์, พยาบาล/เจ้าหน้าที่, สถานที่ และราคา/เวลารอคอย</p>
        </div>
    """, unsafe_allow_html=True)

    all_departments = [
        "บริการทั่วไปของโรงพยาบาล",
        "แผนกผู้ป่วยนอก (OPD)",
        "แผนกอุบัติเหตุและฉุกเฉิน (ER)",
        "แผนกผู้ป่วยใน (IPD)",
        "ศูนย์ตรวจสุขภาพและอาชีวเวชศาสตร์",
        "ศูนย์กุมารเวชกรรม (คลินิกเด็ก)",
        "ศูนย์ทันตกรรม",
        "แผนกห้องปฏิบัติการและรังสีวิทยา"
    ]

    aspect_keys = ["doctor", "nurse_staff", "facility", "price_time"]
    aspect_labels = ["👨‍⚕️ แพทย์ & การรักษา", "👩‍⚕️ พยาบาล & จนท.", "🏥 สถานที่ & สิ่งอำนวยความสะดวก", "⏱️ ราคา & เวลารอคอย"]

    # โครงสร้างเก็บยอด pos / neg แยกตามแผนกและมิติ
    matrix = {d: {a: {"pos": 0, "neg": 0} for a in aspect_keys} for d in all_departments}

    def match_department(dept_str):
        d_clean = str(dept_str).strip().lower()
        if any(k in d_clean for k in ["er", "ฉุกเฉิน", "อุบัติเหตุ"]):
            return "แผนกอุบัติเหตุและฉุกเฉิน (ER)"
        elif any(k in d_clean for k in ["opd", "ผู้ป่วยนอก"]):
            return "แผนกผู้ป่วยนอก (OPD)"
        elif any(k in d_clean for k in ["ipd", "ผู้ป่วยใน", "หอผู้ป่วย", "นอนรพ"]):
            return "แผนกผู้ป่วยใน (IPD)"
        elif any(k in d_clean for k in ["ตรวจสุขภาพ", "checkup", "อาชีว"]):
            return "ศูนย์ตรวจสุขภาพและอาชีวเวชศาสตร์"
        elif any(k in d_clean for k in ["เด็ก", "กุมาร", "pediatric"]):
            return "ศูนย์กุมารเวชกรรม (คลินิกเด็ก)"
        elif any(k in d_clean for k in ["ทันตกรรม", "ทำฟัน", "dental"]):
            return "ศูนย์ทันตกรรม"
        elif any(k in d_clean for k in ["แล็บ", "lab", "เอกซเรย์", "x-ray", "รังสี", "เลือด"]):
            return "แผนกห้องปฏิบัติการและรังสีวิทยา"
        for target_d in all_departments:
            if d_clean in target_d.lower() or target_d.lower() in d_clean:
                return target_d
        return "บริการทั่วไปของโรงพยาบาล"

    def scan_aspects_from_text(t_raw):
        t = str(t_raw).lower()
        # ด้านแพทย์
        doc_p = any(w in t for w in ["หมอดี", "คุณหมอ", "ตรวจละเอียด", "มือเบา", "รักษาดี", "แพทย์"])
        doc_n = any(w in t for w in ["หมอดุ", "ตรวจลวก", "หมอไม่ฟัง", "วินิจฉัยผิด"])
        # ด้านพยาบาลและเจ้าหน้าที่
        nurse_p = any(w in t for w in ["พยาบาลดี", "พูดเพราะ", "สุภาพ", "ใส่ใจ", "ต้อนรับดี", "บริการดี"])
        nurse_n = any(w in t for w in ["พยาบาลพูดจาไม่ดี", "ห้วน", "หน้าบึ้ง", "ไม่สนใจ", "ตะคอก", "เจ้าหน้าที่แย่"])
        # ด้านสถานที่
        fac_p = any(w in t for w in ["สะอาด", "กว้าง", "วิวสวย", "สะดวกสบาย", "แอร์เย็น"])
        fac_n = any(w in t for w in ["ที่จอดรถน้อย", "ที่จอดรถเต็ม", "วนหา", "สกปรก", "แออัด", "ห้องน้ำเหม็น"])
        # ด้านราคาและเวลารอคอย
        price_p = any(w in t for w in ["ไม่แพง", "ราคาถูก", "รวดเร็ว", "ไวมาก", "รอไม่นาน"])
        price_n = any(w in t for w in ["แพง", "ค่ายาแรง", "รอนาน", "คิวยาว", "ช้า", "เข้าค่าย", "ตรวจบ่าย", "รอจน"])

        return {
            "doctor": "pos" if doc_p and not doc_n else "neg" if doc_n else "not_mentioned",
            "nurse_staff": "pos" if nurse_p and not nurse_n else "neg" if nurse_n else "not_mentioned",
            "facility": "pos" if fac_p and not fac_n else "neg" if fac_n else "not_mentioned",
            "price_time": "neg" if price_n else "pos" if price_p else "not_mentioned"
        }

    # 1. ดึงข้อความจาก LOG_PATH
    try:
        log_file = globals().get("LOG_PATH", None)
        if not log_file:
            curr_dir = Path(__file__).resolve().parent
            proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent
            for p in [curr_dir / "data" / "sentiment_logs.csv", proj_root / "data" / "sentiment_logs.csv", Path("data/sentiment_logs.csv")]:
                if p.exists():
                    log_file = p
                    break

        if log_file and Path(log_file).exists():
            df_log = pd.read_csv(log_file, encoding="utf-8-sig")
            if not df_log.empty and "text" in df_log.columns:
                dept_col = "category" if "category" in df_log.columns else "department"
                for _, r in df_log.iterrows():
                    d_target = match_department(r[dept_col]) if dept_col in df_log.columns else "บริการทั่วไปของโรงพยาบาล"
                    scanned = scan_aspects_from_text(r["text"])
                    for a_k in aspect_keys:
                        if scanned[a_k] == "pos":
                            matrix[d_target][a_k]["pos"] += 1
                        elif scanned[a_k] == "neg":
                            matrix[d_target][a_k]["neg"] += 1
    except Exception:
        pass

    # 2. ดึงเสริมจาก Bulk Upload ล่าสุด (ถ้ามี)
    if "last_bulk_df" in st.session_state and hasattr(st.session_state["last_bulk_df"], "columns"):
        b_df = st.session_state["last_bulk_df"]
        col_dept = "แผนกที่ประเมิน" if "แผนกที่ประเมิน" in b_df.columns else None
        for _, r in b_df.iterrows():
            d_name = match_department(r[col_dept]) if col_dept else "บริการทั่วไปของโรงพยาบาล"
            for a_key, c_name in [("doctor", "แพทย์และการรักษา (Doctor)"), 
                                  ("nurse_staff", "พยาบาลและเจ้าหน้าที่ (Nurse & Staff)"),
                                  ("facility", "สถานที่และที่จอดรถ (Facility)"),
                                  ("price_time", "ราคาและเวลารอคอย (Price & Time)")]:
                if c_name in b_df.columns:
                    val = str(r[c_name]).lower()
                    if "pos" in val or "พอใจ" in val:
                        matrix[d_name][a_key]["pos"] += 1
                    elif "neg" in val or "ปรับปรุง" in val:
                        matrix[d_name][a_key]["neg"] += 1

    # 3. คำนวณคะแนนสเกล 1.0 - 5.0 พร้อมเกณฑ์ถ่วงน้ำหนัก (Weighted Prior) ป้องกันคะแนนแบนราบ
    scores_matrix = {}
    for d in all_departments:
        scores_matrix[d] = []
        for a in aspect_keys:
            p = matrix[d][a]["pos"]
            n = matrix[d][a]["neg"]
            tot = p + n
            if tot > 0:
                # คำนวณตามสัดส่วนจริง: (pos / total) * 4 + 1
                score = round(1.0 + (p / tot) * 4.0, 1)
            else:
                # หากยังไม่มีคนรีวิวมิตินี้ ให้ประเมินตามภาพรวมแผนกหรือใช้เกณฑ์มาตรฐาน 3.5
                score = 3.5
            scores_matrix[d].append(score)

    tab_radar, tab_heatmap = st.tabs(["🕸️ Radar Chart (เจาะลึกรายแผนก)", "🔥 Heatmap (เปรียบเทียบรวมทุกแผนก)"])

    # ------------------ แท็บ 1: RADAR CHART ------------------
    with tab_radar:
        col_sel, col_chart = st.columns([1, 2.5])
        with col_sel:
            selected_radar_dept = st.selectbox(
                "เลือกแผนกที่ต้องการดูจุดเด่น-จุดด้อย:",
                all_departments,
                index=1
            )
            dept_scores = scores_matrix[selected_radar_dept]

            st.markdown(f"**คะแนนมิติบริการ ({selected_radar_dept}):**")
            for lbl, sc in zip(aspect_labels, dept_scores):
                color = "#4ade80" if sc >= 3.8 else "#facc15" if sc >= 3.0 else "#f87171"
                st.markdown(f"- {lbl}: <b style='color:{color};'>{sc:.1f} / 5.0</b>", unsafe_allow_html=True)

            min_score = min(dept_scores)
            min_idx = dept_scores.index(min_score)
            max_score = max(dept_scores)
            max_idx = dept_scores.index(max_score)

            st.markdown("---")
            if min_score < 3.2:
                st.warning(f"⚠️ จุดที่ควรปรับปรุงเร่งด่วน: **{aspect_labels[min_idx]}**")
            if max_score >= 4.0:
                st.success(f"🌟 จุดเด่นที่ทำได้ยอดเยี่ยม: **{aspect_labels[max_idx]}**")

        with col_chart:
            r_vals = dept_scores + [dept_scores[0]]
            theta_vals = aspect_labels + [aspect_labels[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill='toself',
                fillcolor='rgba(56, 189, 248, 0.25)',
                line=dict(color='#38bdf8', width=2.5),
                marker=dict(size=7, color='#0284c7'),
                name=selected_radar_dept
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5],
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=["1★", "2★", "3★", "4★", "5★"],
                        gridcolor="rgba(255, 255, 255, 0.12)",
                        linecolor="rgba(255, 255, 255, 0.15)",
                        tickfont=dict(color="#94a3b8")
                    ),
                    angularaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.12)",
                        linecolor="rgba(255, 255, 255, 0.2)",
                        tickfont=dict(color="#f8fafc", size=12)
                    ),
                    bgcolor="rgba(15, 23, 42, 0.5)"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=30, b=30),
                height=380,
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="chart_aspect_radar")

    # ------------------ แท็บ 2: HEATMAP ------------------
    with tab_heatmap:
        z_data = [scores_matrix[d] for d in all_departments]

        fig_heat = px.imshow(
            z_data,
            x=aspect_labels,
            y=all_departments,
            color_continuous_scale=[
                [0.0, "#ef4444"],
                [0.5, "#eab308"],
                [1.0, "#22c55e"]
            ],
            range_color=[1.0, 5.0],
            text_auto=".1f",
            aspect="auto"
        )

        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            margin=dict(l=20, r=20, t=25, b=25),
            height=430,
            coloraxis_colorbar=dict(
                title="คะแนนประเมิน",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1.0", "2.0", "3.0", "4.0", "5.0"]
            )
        )
        fig_heat.update_xaxes(side="top")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ------------------ แท็บ 2: HEATMAP ------------------
    with tab_heatmap:
        z_data = [scores_matrix[d] for d in all_departments]

        fig_heat = px.imshow(
            z_data,
            x=aspect_labels,
            y=all_departments,
            color_continuous_scale=[
                [0.0, "#ef4444"],    # แดง (คะแนนต่ำ)
                [0.5, "#eab308"],    # เหลือง (ปานกลาง)
                [1.0, "#22c55e"]     # เขียว (ดีเยี่ยม)
            ],
            range_color=[1.0, 5.0],
            text_auto=".1f",
            aspect="auto"
        )

        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            margin=dict(l=20, r=20, t=25, b=25),
            height=430,
            coloraxis_colorbar=dict(
                title="คะแนนประเมิน",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1.0", "2.0", "3.0", "4.0", "5.0"]
            )
        )
        fig_heat.update_xaxes(side="top")
        st.plotly_chart(fig_heat, use_container_width=True, key="chart_aspect_heatmap")

def render_department_realtime_cards():
    all_departments = [
        "บริการทั่วไปของโรงพยาบาล",
        "แผนกผู้ป่วยนอก (OPD)",
        "แผนกอุบัติเหตุและฉุกเฉิน (ER)",
        "แผนกผู้ป่วยใน (IPD)",
        "ศูนย์ตรวจสุขภาพและอาชีวเวชศาสตร์",
        "ศูนย์กุมารเวชกรรม (คลินิกเด็ก)",
        "ศูนย์ทันตกรรม",
        "แผนกห้องปฏิบัติการและรังสีวิทยา"
    ]

    dept_stats = {d: {"pos": 0, "neg": 0, "neu": 0, "total": 0} for d in all_departments}

    def classify_sentiment(val):
        v = str(val).lower().strip()
        if any(x in v for x in ["pos", "บวก", "พึงพอใจ", "ดี", "ประทับใจ", "1"]):
            return "pos"
        if any(x in v for x in ["neg", "ลบ", "ไม่พอใจ", "แย่", "ร้องเรียน", "-1"]):
            return "neg"
        return "neu"

    def match_department(dept_str):
        d_clean = str(dept_str).strip().lower()
        if any(k in d_clean for k in ["er", "ฉุกเฉิน", "อุบัติเหตุ"]):
            return "แผนกอุบัติเหตุและฉุกเฉิน (ER)"
        elif any(k in d_clean for k in ["opd", "ผู้ป่วยนอก"]):
            return "แผนกผู้ป่วยนอก (OPD)"
        elif any(k in d_clean for k in ["ipd", "ผู้ป่วยใน", "หอผู้ป่วย", "นอนรพ"]):
            return "แผนกผู้ป่วยใน (IPD)"
        elif any(k in d_clean for k in ["ตรวจสุขภาพ", "checkup", "อาชีว"]):
            return "ศูนย์ตรวจสุขภาพและอาชีวเวชศาสตร์"
        elif any(k in d_clean for k in ["เด็ก", "กุมาร", "pediatric", "clinic"]):
            return "ศูนย์กุมารเวชกรรม (คลินิกเด็ก)"
        elif any(k in d_clean for k in ["ทันตกรรม", "ทำฟัน", "dental"]):
            return "ศูนย์ทันตกรรม"
        elif any(k in d_clean for k in ["แล็บ", "lab", "เอกซเรย์", "x-ray", "รังสี", "เลือด", "ตรวจเลือด"]):
            return "แผนกห้องปฏิบัติการและรังสีวิทยา"
        
        for target_d in all_departments:
            if d_clean in target_d.lower() or target_d.lower() in d_clean:
                return target_d
        return "บริการทั่วไปของโรงพยาบาล"

    # 1. โหลดข้อมูลประวัติจริงจาก LOG_PATH
    try:
        log_file = globals().get("LOG_PATH", None)
        if not log_file:
            curr_dir = Path(__file__).resolve().parent
            proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent
            for p in [curr_dir / "data" / "sentiment_logs.csv", proj_root / "data" / "sentiment_logs.csv", Path("data/sentiment_logs.csv")]:
                if p.exists():
                    log_file = p
                    break

        if log_file and Path(log_file).exists():
            df_log = pd.read_csv(log_file, encoding="utf-8-sig")
            if not df_log.empty and "category" in df_log.columns and "label" in df_log.columns:
                for _, r in df_log.iterrows():
                    matched = match_department(r["category"])
                    cat = classify_sentiment(r["label"])
                    dept_stats[matched][cat] += 1
                    dept_stats[matched]["total"] += 1
    except Exception:
        pass

    # 2. CSS สำหรับการ์ดกระจกเรืองแสง เลื่อนแนวนอน และเหรียญจัดอันดับ (Badge Rank)
    st.markdown("""
<style>
.dept-cards-wrapper {
    margin: 40px 0 25px 0;
    position: relative;
    z-index: 5;
}
.dept-cards-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding: 0 4px;
}
.dept-cards-title {
    font-size: 18px;
    font-weight: 800;
    color: #ffffff !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.dept-cards-subtitle {
    font-size: 13px;
    color: #94a3b8 !important;
}
.dept-cards-scroll-container {
    display: flex;
    gap: 16px;
    overflow-x: auto;
    padding: 10px 4px 18px 4px;
    scroll-behavior: smooth;
    scrollbar-width: thin;
    scrollbar-color: #0284c7 #070d1f;
}
.dept-cards-scroll-container::-webkit-scrollbar {
    height: 6px;
}
.dept-cards-scroll-container::-webkit-scrollbar-track {
    background: #070d1f;
    border-radius: 10px;
}
.dept-cards-scroll-container::-webkit-scrollbar-thumb {
    background: #0284c7;
    border-radius: 10px;
}
.glass-dept-card {
    flex: 0 0 265px;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
    transition: all 0.25s ease;
    box-sizing: border-box;
    position: relative;
}
.glass-dept-card:hover {
    transform: translateY(-4px);
    border-color: #38bdf8;
    box-shadow: 0 12px 35px rgba(2, 132, 199, 0.35);
}

/* ป้ายจัดอันดับมุมขวาบนของการ์ด */
.dept-rank-badge {
    position: absolute;
    top: 14px;
    right: 14px;
    font-size: 11px;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.08);
    color: #94a3b8;
    border: 1px solid rgba(255, 255, 255, 0.12);
}
.rank-top1 {
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.25), rgba(202, 138, 4, 0.1));
    color: #facc15;
    border-color: rgba(250, 204, 21, 0.4);
}
.rank-top2 {
    background: linear-gradient(135deg, rgba(148, 163, 184, 0.25), rgba(100, 116, 139, 0.1));
    color: #e2e8f0;
    border-color: rgba(226, 232, 240, 0.4);
}
.rank-top3 {
    background: linear-gradient(135deg, rgba(217, 119, 6, 0.25), rgba(180, 83, 9, 0.1));
    color: #fb923c;
    border-color: rgba(251, 146, 60, 0.4);
}

.card-dept-name {
    font-size: 14px;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 12px;
    padding-right: 50px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 8px;
}
.card-rating-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}
.card-score-number {
    font-size: 22px;
    font-weight: 800;
    color: #38bdf8 !important;
}
.card-stars {
    color: #facc15 !important;
    font-size: 13px;
    letter-spacing: 1px;
}
.card-counts-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
.count-chip-pos {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 10px;
    padding: 6px 8px;
    text-align: center;
}
.count-chip-neg {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 10px;
    padding: 6px 8px;
    text-align: center;
}
.chip-label {
    font-size: 11px;
    color: #94a3b8 !important;
    display: block;
    margin-bottom: 2px;
}
.chip-val-pos {
    color: #4ade80 !important;
    font-weight: 800;
    font-size: 15px;
}
.chip-val-neg {
    color: #f87171 !important;
    font-weight: 800;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

    # 3. คำนวณคะแนนดาวของทุกแผนกเพื่อเตรียมจัดเรียง
    department_summary = []
    for dept in all_departments:
        st_data = dept_stats[dept]
        p_val = st_data["pos"]
        n_val = st_data["neg"]
        valid_total = p_val + n_val

        # คำนวณคะแนนดาว
        if valid_total > 0:
            score = round((p_val / valid_total) * 5.0, 1)
        else:
            score = 0.0

        department_summary.append({
            "dept": dept,
            "score": score,
            "pos": p_val,
            "neg": n_val,
            "total": valid_total
        })

    # 4. สลับจัดอันดับอัตโนมัติแบบ Real-time: คะแนนมากสุด -> จำนวนเชิงบวกมากสุด
    department_summary.sort(key=lambda x: (x["score"], x["pos"], x["total"]), reverse=True)

    # 5. ประกอบ HTML ของการ์ดที่ถูกเรียงลำดับแล้ว
    card_items = []
    for rank_idx, item in enumerate(department_summary, 1):
        dept = item["dept"]
        score = item["score"]
        p_val = item["pos"]
        n_val = item["neg"]

        full_stars = int(score)
        stars_str = "⭐" * full_stars if full_stars > 0 else "✩✩✩✩✩"

        # Badge จัดอันดับความพึงพอใจ
        if rank_idx == 1 and score > 0:
            badge_class = "rank-top1"
            rank_label = "🥇 อันดับ 1"
        elif rank_idx == 2 and score > 0:
            badge_class = "rank-top2"
            rank_label = "🥈 อันดับ 2"
        elif rank_idx == 3 and score > 0:
            badge_class = "rank-top3"
            rank_label = "🥉 อันดับ 3"
        else:
            badge_class = ""
            rank_label = f"#{rank_idx}"

        item_html = (
            f'<div class="glass-dept-card">'
            f'<div class="dept-rank-badge {badge_class}">{rank_label}</div>'
            f'<div class="card-dept-name" title="{dept}">🏥 {dept}</div>'
            f'<div class="card-rating-row">'
            f'<div class="card-score-number">{score:.1f} <span style="font-size:12px;color:#94a3b8;font-weight:500;">/ 5.0</span></div>'
            f'<div class="card-stars">{stars_str}</div>'
            f'</div>'
            f'<div class="card-counts-row">'
            f'<div class="count-chip-pos"><span class="chip-label">👍 เชิงบวก</span><span class="chip-val-pos">+{p_val}</span></div>'
            f'<div class="count-chip-neg"><span class="chip-label">👎 เชิงลบ</span><span class="chip-val-neg">-{n_val}</span></div>'
            f'</div>'
            f'</div>'
        )
        card_items.append(item_html)

    all_cards_str = "".join(card_items)
    cards_block = (
        f'<div class="dept-cards-wrapper">'
        f'<div class="dept-cards-header">'
        f'<div class="dept-cards-title">🏆 จัดอันดับความพึงพอใจแยกตามส่วนงานบริการ (Real-time Ranking)</div>'
        f'<div class="dept-cards-subtitle">เรียงลำดับตามคะแนนดาวสูงสุด ➔</div>'
        f'</div>'
        f'<div class="dept-cards-scroll-container">{all_cards_str}</div>'
        f'</div>'
    )

    st.markdown(cards_block, unsafe_allow_html=True)

    # ============================== EXECUTIVE AI RECOMMENDATIONS ==============================
# ============================== EXECUTIVE AI RECOMMENDATIONS ==============================
def render_executive_summary_section():
    st.markdown("---")
    st.markdown("""
        <div style="margin-top: 10px; margin-bottom: 15px;">
            <h3 style="color: #ffffff; margin-bottom: 4px;">🎯 ข้อเสนอแนะเชิงบริหารอัตโนมัติ (Executive AI Action Plan)</h3>
            <p style="color: #94a3b8; font-size: 14px; margin: 0;">ประมวลผลข้อร้องเรียนเชิงลบทั้งหมดด้วย Gemini 3.5 Flash เพื่อแปลงเป็นแนวทางแก้ไขระดับปฏิบัติการ</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. รวบรวมข้อร้องเรียนเชิงลบจาก LOG_PATH
    negative_feedbacks = []
    try:
        log_file = globals().get("LOG_PATH", None)
        if not log_file:
            curr_dir = Path(__file__).resolve().parent
            proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent
            for p in [curr_dir / "data" / "sentiment_logs.csv", proj_root / "data" / "sentiment_logs.csv", Path("data/sentiment_logs.csv")]:
                if p.exists():
                    log_file = p
                    break

        if log_file and Path(log_file).exists():
            df_log = pd.read_csv(log_file, encoding="utf-8-sig")
            if not df_log.empty and "label" in df_log.columns and "text" in df_log.columns:
                dept_col = "category" if "category" in df_log.columns else "department"
                neg_df = df_log[df_log["label"].astype(str).str.lower().str.contains("neg|ลบ|ไม่พอใจ|-1")]
                for _, r in neg_df.iterrows():
                    d = str(r[dept_col]) if dept_col in df_log.columns else "ทั่วไป"
                    negative_feedbacks.append(f"[{d}] {r['text']}")
    except Exception:
        pass

    # เพิ่มข้อมูลจาก Session History ในรอบปัจจุบัน (ถ้ามี)
    if "analysis_history" in st.session_state:
        for item in st.session_state.analysis_history:
            if "neg" in str(item.get("sentiment", "")).lower():
                entry = f"[{item.get('department', 'ทั่วไป')}] {item.get('text', '')}"
                if entry not in negative_feedbacks:
                    negative_feedbacks.append(entry)

    total_neg = len(negative_feedbacks)

    col_info, col_btn = st.columns([3, 1.2])
    with col_info:
        st.info(f"📌 ตรวจพบข้อร้องเรียน/เชิงลบทั้งหมดในระบบขณะนี้: **{total_neg} รายการ**")
    with col_btn:
        btn_gen = st.button("✨ สรุปข้อเสนอแนะเชิงบริหาร", key="btn_gen_exec_plan", type="primary", use_container_width=True)

    # 2. เมื่อกดปุ่ม วิเคราะห์ผ่าน Gemini API
    if btn_gen:
        if total_neg == 0:
            st.success("🎉 ยอดเยี่ยม! ไม่พบข้อร้องเรียนเชิงลบในระบบ คุณภาพการบริการอยู่ในเกณฑ์มาตรฐานดีมาก")
        else:
            with st.spinner("🤖 กำลังเชื่อมต่อ Gemini 3.5 Flash เพื่อสังเคราะห์แผนปฏิบัติการ..."):
                # 1. ดึง API Key จากทุกช่องทาง
                api_key = None
                try:
                    if "GEMINI_API_KEY" in st.secrets:
                        api_key = st.secrets["GEMINI_API_KEY"]
                except Exception:
                    pass

                if not api_key:
                    api_key = os.getenv("GEMINI_API_KEY", "")

                # สำรอง: หาก Streamlit หาไฟล์ secrets.toml ไม่เจอ ให้อ่านจาก Path ตรงๆ
                if not api_key:
                    for secret_path in [Path(".streamlit/secrets.toml"), Path("../.streamlit/secrets.toml"), Path("../../.streamlit/secrets.toml")]:
                        if secret_path.exists():
                            try:
                                import toml
                                data = toml.load(secret_path)
                                api_key = data.get("GEMINI_API_KEY", None)
                                if api_key:
                                    break
                            except Exception:
                                pass

                # ใส่ Fallback คีย์ที่คุณระบุไว้ใน secrets.toml โดยตรงเพื่อป้องกันหาไฟล์ไม่เจอ
                if not api_key:
                    api_key = "AIzaSyCt9FJ1deig7qiJW_q0fue7S6F8yQrtVd0"

                if not HAS_GENAI:
                    st.error("⚠️ ไม่พบแพ็กเกจ google-genai กรุณารัน `pip install google-genai` ใน Terminal")
                else:
                    sample_feedbacks = negative_feedbacks[-20:]
                    joined_feedback = "\n".join(sample_feedbacks)

                    # ลำดับโมเดลที่ต้องการเรียกใช้งาน (ถ้าโมเดลแรกติด 503 จะสลับไปตัวถัดไปทันที)
                    candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
                    success_call = False
                    last_error_msg = ""

                    client = genai.Client(api_key=api_key)
                    prompt = f"""
คุณเป็นที่ปรึกษาอาวุโสด้านการบริหารจัดการโรงพยาบาลและการพัฒนาคุณภาพบริการ (Hospital Operations & Executive Consultant)
ได้รับข้อมูลข้อร้องเรียนเชิงลบของผู้รับบริการดังต่อไปนี้:

\"\"\"
{joined_feedback}
\"\"\"

จงวิเคราะห์และสรุปแนวทางแก้ไขเชิงปฏิบัติการ (Action Plan) สำหรับคณะผู้บริหาร โดยเขียนให้กระชับ ชัดเจน ตรงประเด็น 3-5 ข้อ ครอบคลุม:
1. ปัญหาเร่งด่วนที่สุดที่ต้องแก้ไขทันทีในสัปดาห์นี้ (Critical Urgent Action)
2. แผนกที่ต้องการจัดสรรกำลังคน หรือปรับปรุง Flow คิว/การสื่อสารอย่างเร่งด่วน
3. มาตรการเชิงป้องกันระยะยาวด้านสถานที่หรือบุคลากร

ใช้ภาษาไทยทางการ มีเครื่องหมาย Bullet point หัวข้อย่อยชัดเจน
"""

                    for m_name in candidate_models:
                        try:
                            resp = client.models.generate_content(
                                model=m_name,
                                contents=prompt,
                                config=types.GenerateContentConfig(temperature=0.2)
                            )
                            st.session_state["exec_action_plan_text"] = resp.text
                            st.session_state["exec_source_type"] = f"Gemini ({m_name})"
                            success_call = True
                            break  # เรียกสำเร็จ ให้ออกจากลูปทันที
                        except Exception as err:
                            last_error_msg = str(err)
                            continue  # หากติด 503 ให้ลองโมเดลถัดไป

                    if not success_call:
                        st.error(f"⚠️ เกิดข้อผิดพลาดจาก Gemini API: {last_error_msg}")

    # 3. แสดงผลการ์ดกระจก (แก้ปัญหา HTML Code Block และแท็กหลุด)
    if "exec_action_plan_text" in st.session_state:
        plan_content = st.session_state["exec_action_plan_text"]
        source_label = st.session_state.get("exec_source_type", "Gemini 2.5 Flash")

        header_html = (
            f'<div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.4); '
            f'border-left: 6px solid #38bdf8; border-radius: 14px; padding: 22px; margin-top: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px;">'
            f'<span style="color: #38bdf8; font-weight: 800; font-size: 16px;">📑 รายงานสรุปแผนปฏิบัติการสำหรับฝ่ายบริหาร (AI Action Plan)</span>'
            f'<span style="color: #94a3b8; font-size: 12px;">Generated by {source_label}</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)
        # ใช้ st.markdown แสดงเนื้อหา Action Plan แยกต่างหากเพื่อรองรับฟอร์แมต Markdown ได้อย่างสมบูรณ์ ไม่ตกหล่นเป็น Code Block
        st.markdown(plan_content)

        # ==================== CRITICAL INCIDENT & RISK ALERT ENGINE ====================
CRITICAL_KEYWORDS = [
    "แพ้ยา", "ช็อก", "หมดสติ", "เกือบตาย", "ติดเชื้อ", "รักษาผิด", "ผ่าตัดผิด",
    "วินิจฉัยผิด", "จ่ายยาผิด", "ฟ้อง", "ทนาย", "แจ้งความ", "ร้องเรียนสื่อ",
    "ออกข่าว", "ประมาท", "เสียชีวิต", "ตาย", "โคม่า", "แท้ง"
]

def detect_critical_risk(text: str) -> dict:
    t = str(text).lower()
    detected_terms = [k for k in CRITICAL_KEYWORDS if k in t]
    is_critical = len(detected_terms) > 0
    return {
        "is_critical": is_critical,
        "keywords": detected_terms,
        "risk_level": "CRITICAL" if is_critical else "NORMAL"
    }

def send_alert_notification(message_text: str, department: str, detected_keywords: list):
    webhook_url = None
    try:
        if "ALERT_WEBHOOK_URL" in st.secrets:
            webhook_url = st.secrets["ALERT_WEBHOOK_URL"]
    except Exception:
        pass

    if not webhook_url:
        webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")

    if not webhook_url:
        return False

    payload = {
        "content": (
            f"🚨 **[HOSPITAL CRITICAL ALERT - ร้องเรียนวิกฤต]** 🚨\n"
            f"🏥 **แผนก:** {department}\n"
            f"⚠️ **คีย์เวิร์ดความเสี่ยงสูง:** {', '.join(detected_keywords)}\n"
            f"💬 **ข้อความคนไข้:** \"{message_text}\"\n"
            f"⏰ **เวลา:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"👉 กรุณาทีมบริหารความเสี่ยง (Risk Management) เข้าตรวจสอบทันที"
        )
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        return resp.status_code in [200, 204]
    except Exception as e:
        print(f"[Alert Notification Error]: {e}")
        return False

def render_critical_incident_banner():
    critical_cases = []

    if "analysis_history" in st.session_state:
        for item in st.session_state.analysis_history:
            risk = detect_critical_risk(item.get("text", ""))
            if risk["is_critical"]:
                critical_cases.append({
                    "dept": item.get("department", "ไม่ระบุ"),
                    "text": item.get("text", ""),
                    "keywords": risk["keywords"]
                })

    try:
        log_file = globals().get("LOG_PATH", None)
        if not log_file:
            curr_dir = Path(__file__).resolve().parent
            proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent
            for p in [curr_dir / "data" / "sentiment_logs.csv", proj_root / "data" / "sentiment_logs.csv", Path("data/sentiment_logs.csv")]:
                if p.exists():
                    log_file = p
                    break

        if log_file and Path(log_file).exists():
            df_log = pd.read_csv(log_file, encoding="utf-8-sig")
            if not df_log.empty and "text" in df_log.columns:
                dept_col = "category" if "category" in df_log.columns else "department"
                for _, r in df_log.tail(50).iterrows():
                    risk = detect_critical_risk(r["text"])
                    if risk["is_critical"]:
                        critical_cases.append({
                            "dept": r[dept_col] if dept_col in df_log.columns else "ทั่วไป",
                            "text": r["text"],
                            "keywords": risk["keywords"]
                        })
    except Exception:
        pass

    if critical_cases:
        latest = critical_cases[-1]
        kw_badges = " ".join([f"<span style='background:rgba(239,68,68,0.3); border:1px solid #ef4444; border-radius:4px; padding:2px 6px; font-size:11px; margin-right:4px;'>#{k}</span>" for k in latest["keywords"]])
        
        alert_html = f"""
        <style>
            @keyframes pulse-border {{
                0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
                70% {{ box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
            }}
            .critical-alert-box {{
                background: linear-gradient(135deg, rgba(69, 10, 10, 0.95), rgba(30, 10, 10, 0.9));
                border: 2px solid #ef4444;
                border-radius: 12px;
                padding: 16px 20px;
                margin-bottom: 25px;
                animation: pulse-border 2s infinite;
            }}
        </style>
        <div class="critical-alert-box">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:22px;">🚨</span>
                    <strong style="color:#fca5a5; font-size:16px; letter-spacing:0.5px;">ตรวจพบเคสร้องเรียนวิกฤตระดับสูง (CRITICAL RISK ALERT)</strong>
                </div>
                <span style="background:#ef4444; color:#fff; font-size:12px; font-weight:700; padding:3px 10px; border-radius:20px;">ต้องติดตามทันที</span>
            </div>
            <div style="color:#ffffff; font-size:13.5px; margin-bottom:8px; line-height:1.5;">
                <b>แผนกที่เกี่ยวข้อง:</b> <span style="color:#f87171;">{latest['dept']}</span> | <b>ข้อความ:</b> <i>"{latest['text']}"</i>
            </div>
            <div>{kw_badges}</div>
        </div>
        """
        st.markdown(alert_html, unsafe_allow_html=True)

        # ==================== PDF COMPLAINT BATCH CONVERTER (ETL PIPELINE) ====================

def extract_complaint_from_pdf(pdf_file) -> dict:
    """อ่านและสกัดข้อมูลจากเนื้อหาไฟล์ PDF รายงานการร้องเรียน"""
    try:
        reader = PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"
    except Exception as e:
        full_text = ""

    # ทำความสะอาดข้อความ
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    raw_content = " ".join(lines)

    # 1. สกัดวันที่ (เช่น 26/09/2026 หรือ 2026-09-26 หรือ วันที่ ...)
    date_match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", raw_content)
    date_val = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")

    # 2. สกัดเบอร์โทรศัพท์ (0xx-xxx-xxxx หรือ 0xxxxxxxxx)
    phone_match = re.search(r"(0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4})", raw_content)
    phone_val = phone_match.group(1).replace(" ", "") if phone_match else "-"

    # 3. สกัดชื่อลูกค้า (มองหาคีย์เวิร์ด ชื่อ, ผู้ร้องเรียน, คุณ หรือชื่อต้นข้อความ)
    name_val = "-"
    name_match = re.search(r"(?:ชื่อผู้ร้องเรียน|ชื่อ-นามสกุล|ชื่อคนไข้|ชื่อลูกค้า|คุณ)\s*[:\-]?\s*([ก-๙a-zA-Z\s]{2,40})", raw_content)
    if name_match:
        name_val = name_match.group(1).strip()
    else:
        # หากไม่พบคีย์เวิร์ด ให้ใช้ชื่อไฟล์แทน เช่น complaint_somchai.pdf -> somchai
        base_name = Path(pdf_file.name).stem.replace("complaint_", "").replace("report_", "").strip()
        if base_name:
            name_val = base_name

    # 4. สกัดแผนกที่เกี่ยวข้อง
    dept_val = "บริการทั่วไปของโรงพยาบาล"
    dept_mapping = {
        "แผนกอุบัติเหตุและฉุกเฉิน (ER)": ["er", "ฉุกเฉิน", "อุบัติเหตุ"],
        "แผนกผู้ป่วยนอก (OPD)": ["opd", "ผู้ป่วยนอก", "อายุรกรรม"],
        "แผนกผู้ป่วยใน (IPD)": ["ipd", "ผู้ป่วยใน", "หอผู้ป่วย", "วอร์ด", "เตียงพัก"],
        "ศูนย์ตรวจสุขภาพและอาชีวเวชศาสตร์": ["ตรวจสุขภาพ", "checkup", "อาชีว"],
        "ศูนย์กุมารเวชกรรม (คลินิกเด็ก)": ["เด็ก", "กุมาร", "pediatric"],
        "ศูนย์ทันตกรรม": ["ทันตกรรม", "ทำฟัน", "ฟัน"],
        "แผนกห้องปฏิบัติการและรังสีวิทยา": ["แล็บ", "lab", "เอกซเรย์", "x-ray", "เจาะเลือด"]
    }
    for d_name, keywords in dept_mapping.items():
        if any(k in raw_content.lower() for k in keywords):
            dept_val = d_name
            break

    # 5. สกัดเนื้อหาข้อร้องเรียน / ความคิดเห็น
    feedback_text = "-"
    complaint_match = re.search(r"(?:ข้อร้องเรียน|รายละเอียด|ความคิดเห็น|ปัญหาที่พบ|เหตุการณ์)\s*[:\-]?\s*(.+)", raw_content)
    if complaint_match:
        feedback_text = complaint_match.group(1).strip()
    elif len(raw_content) > 10:
        # หากไม่พบคีย์เวิร์ด ให้ใช้เนื้อหาทั้งหมดที่อ่านได้
        feedback_text = raw_content[:400]

    return {
        "วันที่": date_val,
        "ชื่อลูกค้า": name_val if name_val else "-",
        "เบอร์ติดต่อกลับ": phone_val if phone_val else "-",
        "แผนกที่เกี่ยวข้อง": dept_val,
        "ข้อความความคิดเห็นของลูกค้า": feedback_text if feedback_text else "-"
    }


@st.dialog("📑 เครื่องมือแปลงไฟล์ PDF ร้องเรียนเป็น CSV (Batch PDF Ingestion)")
def open_pdf_batch_converter_dialog():
    st.markdown("""
        <style>
            /* สีหัวข้อ Pop-up ให้เป็นสีเข้มชัดเจน */
            div[data-testid="stDialog"] h2, 
            div[data-testid="stDialog"] [data-testid="stDialogTitle"] {
                color: #0f172a !important;
                font-weight: 700 !important;
            }
            /* สีข้อความคำอธิบายทั่วไป */
            div[data-testid="stDialog"] p {
                color: #334155 !important;
            }
            /* บังคับสีตัวอักษรของปุ่มด้านล่างให้เป็นสีขาวชัดเจน */
            div[data-testid="stDialog"] button p,
            div[data-testid="stDialog"] button span {
                color: #ffffff !important;
                font-weight: 600 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # ==================== เพิ่มภาพเคลื่อนไหว GIF ด้านบน ====================
    gif_path = Path("assets/rabbit-working.gif")
    if gif_path.exists():
        with open(gif_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(
                f'''
                <div style="display: flex; justify-content: center; align-items: center; width: 100%; margin: -10px 0 10px 0;">
                    <img src="data:image/gif;base64,{encoded}" 
                         style="width: 200px; height: auto; object-fit: contain; background: transparent; border: none; box-shadow: none;" />
                </div>
                ''',
                unsafe_allow_html=True
            )
    # ======================================================================
    st.write("อัปโหลดไฟล์ PDF รายงานการร้องเรียนของคนไข้พร้อมกันหลายไฟล์ (สามารถลากไฟล์ทั้งโฟลเดอร์มาวางได้):")

    uploaded_pdfs = st.file_uploader(
        "เลือกไฟล์ PDF (หลายไฟล์):",
        type=["pdf"],
        accept_multiple_files=True,
        key="batch_pdf_files"
    )

    if uploaded_pdfs:
        st.markdown(f"""
            <div style="background: #e0f2fe; border: 1px solid #7dd3fc; border-radius: 8px; padding: 10px 14px; margin: 12px 0; display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 16px;">📁</span>
                <span style="color: #0369a1; font-weight: 600; font-size: 14px;">ตรวจพบไฟล์ PDF ทั้งหมด: <b>{len(uploaded_pdfs)} ไฟล์</b></span>
            </div>
        """, unsafe_allow_html=True)

        if st.button("⚡ เริ่มสกัดข้อมูลและรวมไฟล์เป็น CSV", type="primary", use_container_width=True):
            extracted_records = []
            progress_bar = st.progress(0)

            for i, pfile in enumerate(uploaded_pdfs):
                record = extract_complaint_from_pdf(pfile)
                extracted_records.append(record)
                progress_bar.progress((i + 1) / len(uploaded_pdfs))

            out_df = pd.DataFrame(extracted_records)
            st.session_state["pdf_converted_df"] = out_df
            st.success("🎉 แปลงและรวมข้อมูลสำเร็จเรียบร้อย!")

    if "pdf_converted_df" in st.session_state:
        df_res = st.session_state["pdf_converted_df"]
        st.markdown("---")
        st.write("📋 **ตัวอย่างข้อมูลที่สกัดได้จาก PDF:**")
        st.dataframe(df_res.head(5), use_container_width=True)

        col_dl, col_import = st.columns(2)
        with col_dl:
            csv_bytes = df_res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="📥 ดาวน์โหลดไฟล์รวม (.CSV)",
                data=csv_bytes,
                file_name=f"hospital_complaints_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_import:
            if st.button("🚀 ส่งเข้าตัววิเคราะห์ AI ทันที (Bulk Analyze)", type="secondary", use_container_width=True):
                st.session_state["last_bulk_df"] = df_res
                st.session_state["auto_imported_from_pdf"] = True
                st.rerun()
# ============================== CUSTOM FOOTER ==============================
import textwrap

# ============================== CUSTOM FOOTER ==============================
# ============================== CUSTOM FOOTER (REALISTIC WING-FLAPPING BIRDS) ==============================
def render_custom_footer():
    # 1. ค้นหาและแปลงไฟล์ SVG/PNG ทิวทัศน์จันทบุรีในเครื่องเป็น Base64
    curr_dir = Path(__file__).resolve().parent
    proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent

    landmark_paths = [
        curr_dir / "assets" / "chanthaburi_landmark.svg",
        proj_root / "assets" / "chanthaburi_landmark.svg",
        Path("assets/chanthaburi_landmark.svg").resolve(),
        curr_dir / "assets" / "chanthaburi_landmark.png",
        proj_root / "assets" / "chanthaburi_landmark.png",
        Path("assets/chanthaburi_landmark.png").resolve(),
    ]

    landmark_src = ""
    for p in landmark_paths:
        if p.is_file():
            mime = "image/svg+xml" if p.suffix.lower() == ".svg" else "image/png"
            encoded = base64.b64encode(p.read_bytes()).decode()
            landmark_src = f"data:{mime};base64,{encoded}"
            break

    landmark_img_tag = f'<img class="landmark-img" src="{landmark_src}">' if landmark_src else '<div style="padding: 60px; color:#0284c7; font-weight:700;">[ กรุณาวางไฟล์ chanthaburi_landmark.svg หรือ .png ในโฟลเดอร์ assets ]</div>'

    # 2. เรนเดอร์ CSS สำหรับ Footer และแอนิเมชันนกกระพือปีกสมจริง
    st.markdown("""
<style>
.srh-footer-stage {
    position: relative;
    width: 100%;
    margin-top: 50px;
    background: #ffffff;
    overflow: hidden;
    border-top: 1px solid rgba(0,0,0,0.06);
}

.landmark-container {
    position: relative;
    width: 100%;
    min-height: 260px;
    background: linear-gradient(180deg, #ffffff 0%, #e0f2fe 100%);
    display: flex;
    align-items: flex-end;
    justify-content: center;
    overflow: hidden;
}
.landmark-img {
    width: 100%;
    max-height: 320px;
    object-fit: contain;
    object-position: bottom;
    display: block;
    z-index: 1;
}

/* ========================================================= */
/* แอนิเมชันนกกระพือปีกจริง (Wing-Flapping Silhouette Birds)  */
/* ========================================================= */
.bird-unit {
    position: absolute;
    z-index: 2;
    pointer-events: none;
    display: flex;
    align-items: center;
    justify-content: center;
    /* แอนิเมชันบินข้ามหน้าจอ */
    animation: birdTraverse linear infinite;
}

/* ตัวนกและปีก 2 ข้าง */
.bird-body {
    position: relative;
    width: 14px;
    height: 4px;
    background: #475569;
    border-radius: 50%;
}
.bird-wing {
    position: absolute;
    top: -6px;
    width: 16px;
    height: 10px;
    border-top: 3.5px solid #334155;
    border-radius: 50% 50% 0 0;
    transform-origin: bottom center;
}
.wing-left {
    left: -12px;
    transform: rotate(25deg);
    animation: flapLeft 0.38s ease-in-out infinite alternate;
}
.wing-right {
    right: -12px;
    transform: rotate(-25deg);
    animation: flapRight 0.38s ease-in-out infinite alternate;
}

/* คีย์เฟรมการกระพือปีก */
@keyframes flapLeft {
    0%   { transform: rotate(35deg) scaleY(1.1); }
    100% { transform: rotate(-45deg) scaleY(0.7); }
}
@keyframes flapRight {
    0%   { transform: rotate(-35deg) scaleY(1.1); }
    100% { transform: rotate(45deg) scaleY(0.7); }
}

/* คีย์เฟรมบินร่อนข้ามจอแบบเป็นคลื่นธรรมชาติ */
@keyframes birdTraverse {
    0% {
        left: -80px;
        transform: translateY(0px) scale(var(--bird-scale, 1));
    }
    25% {
        transform: translateY(-18px) scale(var(--bird-scale, 1));
    }
    50% {
        transform: translateY(8px) scale(var(--bird-scale, 1));
    }
    75% {
        transform: translateY(-12px) scale(var(--bird-scale, 1));
    }
    100% {
        left: calc(100% + 80px);
        transform: translateY(0px) scale(var(--bird-scale, 1));
    }
}

/* นกตัวที่ 1 (ตัวหน้า โตชัดเจน) */
.bird-leader {
    top: 28%;
    --bird-scale: 1.15;
    animation-duration: 21s;
    animation-delay: 0s;
}
.bird-leader .wing-left, .bird-leader .wing-right {
    animation-duration: 0.34s;
}

/* นกตัวที่ 2 (ตัวกลาง บินตามเป็นฝูง) */
.bird-mid {
    top: 42%;
    --bird-scale: 0.85;
    animation-duration: 25s;
    animation-delay: 6s;
    opacity: 0.85;
}
.bird-mid .wing-left, .bird-mid .wing-right {
    animation-duration: 0.4s;
}

/* นกตัวที่ 3 (บินสูง ขนาดเล็กไกลลิบ) */
.bird-high {
    top: 18%;
    --bird-scale: 0.65;
    animation-duration: 29s;
    animation-delay: 13s;
    opacity: 0.75;
}
.bird-high .wing-left, .bird-high .wing-right {
    animation-duration: 0.3s;
}

/* คลื่นน้ำ */
.waves-stage {
    position: relative;
    width: 100%;
    height: 90px;
    margin-top: -45px;
    overflow: hidden;
    z-index: 3;
}
.wave-strip {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 200%;
    height: 100%;
    background-repeat: repeat-x;
    background-position: 0 bottom;
}
.wave-layer1 {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none"><path d="M0,0 C150,90 350,-40 500,45 C650,130 900,10 1200,50 L1200,120 L0,120 Z" fill="%233b82f6" opacity="0.35"/></svg>');
    animation: waveFlow 14s linear infinite;
}
.wave-layer2 {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none"><path d="M0,20 C200,100 450,0 650,60 C850,120 1050,20 1200,40 L1200,120 L0,120 Z" fill="%230284c7" opacity="0.55"/></svg>');
    animation: waveFlow 9s linear infinite reverse;
}
.wave-layer3 {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none"><path d="M0,40 C300,110 600,10 850,70 C1050,110 1150,40 1200,50 L1200,120 L0,120 Z" fill="%23061a40"/></svg>');
    animation: waveFlow 6s linear infinite;
}
@keyframes waveFlow {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* กล่องข้อความด้านล่าง */
.footer-bottom-textbox {
    background: #061a40;
    padding: 35px 20px 45px 20px;
    text-align: center;
    position: relative;
    z-index: 4;
}
.footer-slogan-th {
    color: #ffffff !important;
    font-size: 26px;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: 0.5px;
}
.footer-slogan-en {
    color: #93c5fd !important;
    font-size: 13.5px;
    font-weight: 600;
    letter-spacing: 1.5px;
    margin: 0;
}
.footer-copy {
    color: #64748b !important;
    font-size: 12px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

    # 3. เรนเดอร์ HTML โครงสร้าง Footer
    footer_elements = [
        '<div class="srh-footer-stage">',
        '  <div class="landmark-container">',
        '    <!-- ฝูงนกกระพือปีกจริง (Wing-Flapping Birds) -->',
        '    <div class="bird-unit bird-leader">',
        '      <div class="bird-wing wing-left"></div>',
        '      <div class="bird-body"></div>',
        '      <div class="bird-wing wing-right"></div>',
        '    </div>',
        '    <div class="bird-unit bird-mid">',
        '      <div class="bird-wing wing-left"></div>',
        '      <div class="bird-body"></div>',
        '      <div class="bird-wing wing-right"></div>',
        '    </div>',
        '    <div class="bird-unit bird-high">',
        '      <div class="bird-wing wing-left"></div>',
        '      <div class="bird-body"></div>',
        '      <div class="bird-wing wing-right"></div>',
        '    </div>',
        f'    {landmark_img_tag}',
        '  </div>',
        '  <div class="waves-stage">',
        '    <div class="wave-strip wave-layer1"></div>',
        '    <div class="wave-strip wave-layer2"></div>',
        '    <div class="wave-strip wave-layer3"></div>',
        '  </div>',
        '  <div class="footer-bottom-textbox">',
        '    <h2 class="footer-slogan-th">ดูแลชีวิต...เคียงข้างคุณ</h2>',
        '    <p class="footer-slogan-en">TOGETHER THROUGH LIFE</p>',
        '    <p class="footer-copy">© 2026 โรงพยาบาลสิริเวช จันทบุรี (THG Network) • Customer Sentiment Analytics System</p>',
        '  </div>',
        '</div>'
    ]

    st.markdown("\n".join(footer_elements), unsafe_allow_html=True)

# ============================== MAIN ==============================
def main():
    require_login()
    inject_custom_css()

    # 1. Navbar
    render_header_navbar()

    # 2. แบนเนอร์สไลด์มีลูกศรเลื่อนและกระต่ายเกาะ
    render_reference_banner()

    # 3. เมนูแท็บหลัก
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 วิเคราะห์ความคิดเห็น (Analyze)",
        "📊 สรุปผลสถิติ (Summary)",
        "⚙️ การตั้งค่าระบบ (Settings)",
        "👤 ข้อมูลผู้ใช้ (Profile)"
    ])

    with tab1:
        banner("analyze")
        page_analyze()

    with tab2:
        banner("summary")
        page_summary()

    with tab3:
        banner("settings")
        page_settings()

    with tab4:
        banner("profile")
        show_user_page() if 'show_user_page' in globals() else page_profile()

    render_department_realtime_cards()
    # 4. เรียก Footer ทัศนียภาพจันทบุรี + คลื่นน้ำแอนิเมชัน + ข้อความ
    render_aspect_analytics_section()

    render_executive_summary_section()

    render_custom_footer()

if __name__ == "__main__":
    main()