# src/dashboard/app.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import sys
import re
import io
import json
import time
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

# นำเข้าโมดูลจัดการฐานข้อมูล Cloud และโมดูลส่งออก PDF
try:
    import db_manager as db
except ImportError:
    db = None

try:
    from pdf_exporter import generate_pdf_report
except ImportError:
    generate_pdf_report = None

# ============================== ROOT & IMPORTS ==============================
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ==================== GEMINI AI ENGINE ====================
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

def analyze_aspects_smart(text: str) -> dict:
    t = str(text).lower().strip()
    is_q = any(w in t for w in ["?", "ไหม", "มั้ย", "รึเปล่า", "หรือไม่", "อย่างไร", "ทำไม", "หรือยัง"])
    
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

    api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")

    if HAS_GENAI and api_key:
        prompt = f"""คุณคือผู้เชี่ยวชาญด้านวิเคราะห์ความรู้สึกและบริการของโรงพยาบาล
กรุณาวิเคราะห์ข้อความความคิดเห็นของผู้รับบริการต่อไปนี้ แล้วตอบกลับเป็น JSON เท่านั้น:
{{
  "overall": "pos" | "neg" | "neu",
  "aspects": {{
    "doctor": "pos" | "neg" | "neu" | "not_mentioned",
    "nurse_staff": "pos" | "neg" | "neu" | "not_mentioned",
    "facility": "pos" | "neg" | "neu" | "not_mentioned",
    "price_time": "pos" | "neg" | "neu" | "not_mentioned"
  }}
}}

ข้อความคนไข้: \"\"\"{text.strip()}\"\"\""""

        client = genai.Client(api_key=api_key)
        candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
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

    # Fallback ออฟไลน์
    general_pos_words = ["ดีมาก", "ดี", "ยอดเยี่ยม", "ประทับใจ", "สุดยอด", "รวดเร็ว", "สุภาพ", "บริการดี", "ชอบมาก"]
    general_neg_words = ["แย่", "แย่มาก", "ช้ามาก", "ไม่ดี", "ห่วย", "ชุ่ย", "ไม่ประทับใจ", "ผิดหวัง", "โกรธ", "รอนาน", "เข้าค่าย"]

    is_gen_pos = any(w in t for w in general_pos_words)
    is_gen_neg = any(w in t for w in general_neg_words)

    medical_neg_words = ["ผิดพลาด", "วินิจฉัยผิด", "จ่ายยาผิด", "รักษาไม่หาย", "อาการทรุด", "ไม่ตรวจ", "แพ้ยา", "ช็อก", "เกือบตาย", "ฟ้อง"]
    doc_neg = any(w in t for w in medical_neg_words) or ("หมอ" in t and any(w in t for w in ["แย่", "ดุ", "ไม่ดี", "ช้า"]))
    doc_pos = any(w in t for w in ["หมอเก่ง", "หมอดี", "หมอพูดจาดี", "หมอใส่ใจ"]) or ("หมอ" in t and is_gen_pos)
    
    nurse_neg = any(w in t for w in ["พยาบาลดุ", "พยาบาลชักสีหน้า", "พยาบาลพูดจาแย่", "เจ้าหน้าที่ดุ"])
    nurse_pos = any(w in t for w in ["พยาบาลดี", "พยาบาลน่ารัก", "พยาบาลบริการดี", "เจ้าหน้าที่บริการดี"])
    
    fac_neg = any(w in t for w in ["สกปรก", "ห้องน้ำเหม็น", "ที่จอดรถเต็ม", "ไม่มีที่จอด", "แอร์ร้อน"])
    fac_pos = any(w in t for w in ["สะอาด", "สะดวกสบาย", "ที่จอดรถเยอะ", "ห้องพักดี"])
    
    price_neg = any(w in t for w in ["แพง", "แพงมาก", "เกินจริง", "รอนาน", "คิวช้า", "คิวยาว", "นัดเก้าโมง", "บ่ายสอง", "เข้าค่าย"])
    price_pos = any(w in t for w in ["ราคาเหมาะสม", "รอไม่นาน", "เร็วดี"])

    doc_res = "neg" if doc_neg else "pos" if doc_pos else "not_mentioned"
    nurse_res = "neg" if nurse_neg else "pos" if nurse_pos else "not_mentioned"
    fac_res = "neg" if fac_neg else "pos" if fac_pos else "not_mentioned"
    price_res = "neg" if price_neg else "pos" if price_pos else "not_mentioned"

    if any(r == "neg" for r in [doc_res, nurse_res, fac_res, price_res]) or is_gen_neg:
        overall_sent = "neg"
    elif any(r == "pos" for r in [doc_res, nurse_res, fac_res, price_res]) or is_gen_pos:
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

# ============================== AUTHENTICATION (CLOUD DB + MODERN CARD) ==============================
def login_form():
    curr_dir = Path(__file__).resolve().parent
    proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent

    logo_paths = [
        curr_dir / "assets" / "logo.webp",
        proj_root / "assets" / "logo.webp",
        curr_dir / "assets" / "sirivej_logo.webp",
        proj_root / "assets" / "sirivej_logo.webp",
        Path("assets/logo.webp").resolve(),
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
    }
    .stTextInput label {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    [data-testid="stFormSubmitButton"] > button {
        background: #0284c7 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 20px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([0.34, 0.32, 0.34])
    with center_col:
        with st.form("modern_login_form"):
            if logo_src:
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 18px;">
                    <img src="{logo_src}" style="max-width: 180px; height: auto; object-fit: contain;">
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; margin-bottom: 18px;">
                    <span style="font-size: 24px;">🏥</span>
                    <span style="color: #005b9f; font-weight: 800; font-size: 18px;">โรงพยาบาลสิริเวช</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="color: #0284c7; font-size: 12px; font-weight: 700; letter-spacing: 1px;">CUSTOMER SENTIMENT SYSTEM</div>
                <h2 style="color: #0f172a; font-weight: 800; font-size: 24px; margin: 5px 0;">เข้าสู่ระบบ</h2>
                <p style="color: #64748b; font-size: 13px; margin: 0;">กรุณากรอกรหัสผ่านเพื่อเข้าใช้งานระบบ</p>
            </div>
            """, unsafe_allow_html=True)

            username = st.text_input("ชื่อผู้ใช้งาน (Username)", placeholder="กรอกชื่อผู้ใช้งาน")
            password = st.text_input("รหัสผ่าน (Password)", type="password", placeholder="กรอกรหัสผ่าน")
            submit = st.form_submit_button("เข้าสู่ระบบ  →", use_container_width=True)

            if submit:
                # 1. ตรวจสอบผ่าน Cloud Database (Supabase) เป็นหลัก
                auth_data = None
                if db:
                    try:
                        auth_data = db.authenticate(username, password)
                    except Exception:
                        auth_data = None

                if auth_data:
                    st.session_state.auth = {
                        "logged_in": True,
                        "username": auth_data["username"],
                        "display_name": auth_data["username"],
                        "role": auth_data.get("role", "user"),
                    }
                    st.success("เข้าสู่ระบบสำเร็จ กำลังนำเข้าสู่ระบบ...")
                    st.rerun()
                else:
                    # 2. Fallback ตรวจสอบผ่าน secrets.toml เดิม
                    users = st.secrets.get("users", {})
                    u = users.get(username)
                    if u and hashlib.sha256((u["salt"] + password).encode("utf-8")).hexdigest() == u["hash"]:
                        st.session_state.auth = {
                            "logged_in": True,
                            "username": username,
                            "display_name": u.get("display_name", username),
                            "role": u.get("role", "user"),
                        }
                        st.success("เข้าสู่ระบบสำเร็จ...")
                        st.rerun()
                    elif username == "admin" and password == "admin1234":
                        st.session_state.auth = {
                            "logged_in": True, "username": "admin", "display_name": "Administrator", "role": "admin"
                        }
                        st.rerun()
                    elif username == "staff" and password == "user1234":
                        st.session_state.auth = {
                            "logged_in": True, "username": "staff", "display_name": "Staff User", "role": "user"
                        }
                        st.rerun()
                    else:
                        st.error("❌ ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")

def require_login():
    if "auth" not in st.session_state or not st.session_state.auth.get("logged_in"):
        login_form()
        st.stop()

# ============================== UI STYLING ==============================
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/danyim/lineseed-thai@master/css/lineseed-thai.css');
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, h1, h2, h3, h4, h5, h6, p, span, div, label, button, input, textarea, select {
        font-family: 'LINESeedSansTH', 'LINE Seed Sans TH', 'Prompt', sans-serif !important;
    }
    .stApp {
        background-color: #070d1e !important;
        background-image: radial-gradient(circle at 50% 15%, rgba(26, 46, 82, 0.7) 0%, rgba(7, 13, 30, 0.95) 100%) !important;
        color: #ffffff !important;
    }
    p, span, label { color: #ffffff !important; }
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1220px !important; }
    
    .premium-card {
        background: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 26px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
    }
    .metric-card-box {
        background: #1e293b;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stButton > button, [data-testid="stFormSubmitButton"] > button, .stDownloadButton > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================== TOP HEADER NAVBAR ==============================
def render_header_navbar():
    if st.query_params.get("action") == "logout":
        st.query_params.clear()
        if "auth" in st.session_state:
            del st.session_state.auth
        st.rerun()

    curr_dir = Path(__file__).resolve().parent
    proj_root = curr_dir.parents[1] if len(curr_dir.parents) >= 2 else curr_dir.parent

    logo_paths = [
        curr_dir / "assets" / "logo.avif", proj_root / "assets" / "logo.avif",
        curr_dir / "assets" / "logo.webp", proj_root / "assets" / "logo.webp",
        curr_dir / "assets" / "logo.png", proj_root / "assets" / "logo.png"
    ]
    logo_src = ""
    for p in logo_paths:
        if p.is_file():
            encoded = base64.b64encode(p.read_bytes()).decode()
            logo_src = f"data:image/{p.suffix.lower().replace('.', '')};base64,{encoded}"
            break

    user_info = st.session_state.get("auth", {})
    user_badge = f"<span style='background:#0284c7; color:#fff; font-size:12px; font-weight:700; padding:4px 10px; border-radius:8px;'>👤 {user_info.get('username', 'User')} ({user_info.get('role', 'user').upper()})</span>"

    center_logo_html = f'<img src="{logo_src}" style="max-height:45px;">' if logo_src else '<span style="color:#ffffff;font-weight:800;font-size:18px;">🏥 SIRIVEJ HOSPITAL</span>'
    
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#070d1f; padding:10px 24px; border-radius:16px; border:1px solid rgba(255,255,255,0.08); margin-bottom:20px;">
        <div>{center_logo_html}</div>
        <div style="display:flex; align-items:center; gap:15px;">
            {user_badge}
            <a href="?action=logout" style="color:#f87171; text-decoration:none; font-weight:700; font-size:13px; background:rgba(239,68,68,0.15); padding:6px 12px; border-radius:8px; border:1px solid rgba(239,68,68,0.3);">🚪 ออกจากระบบ</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================== BANNER COMPONENT ==============================
@st.cache_data
def load_cached_banner(filename: str) -> str:
    for base in [ROOT / "assets", BASE_DIR / "assets", Path("assets")]:
        b_path = base / filename
        if b_path.exists():
            return base64.b64encode(b_path.read_bytes()).decode("utf-8")
    return ""

def banner(page_name: str):
    banner_map = {
        "analyze": "banner_analyze.png",
        "summary": "banner_summary.png",
        "settings": "banner_settings.png",
        "profile": "banner_profile.png",
    }
    filename = banner_map.get(page_name, "banner.png")
    encoded = load_cached_banner(filename)
    if encoded:
        st.markdown(
            f"""
            <div style="margin-bottom: 20px; text-align: center;">
                <img src="data:image/png;base64,{encoded}" style="width: 100%; max-height: 230px; object-fit: cover; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1);">
            </div>
            """, unsafe_allow_html=True
        )
    render_critical_incident_banner()

# ==================== CRITICAL RISK ALERT ====================
CRITICAL_KEYWORDS = [
    "แพ้ยา", "ช็อก", "หมดสติ", "เกือบตาย", "ติดเชื้อ", "รักษาผิด", "ผ่าตัดผิด",
    "วินิจฉัยผิด", "จ่ายยาผิด", "ฟ้อง", "ทนาย", "แจ้งความ", "ร้องเรียนสื่อ", "ออกข่าว", "ประมาท", "เสียชีวิต", "ตาย"
]

def detect_critical_risk(text: str) -> dict:
    t = str(text).lower()
    detected_terms = [k for k in CRITICAL_KEYWORDS if k in t]
    is_critical = len(detected_terms) > 0
    return {"is_critical": is_critical, "keywords": detected_terms}

def render_critical_incident_banner():
    critical_cases = []
    if "analysis_history" in st.session_state:
        for item in st.session_state.analysis_history:
            risk = detect_critical_risk(item.get("text", ""))
            if risk["is_critical"]:
                critical_cases.append(item)

    if critical_cases:
        latest = critical_cases[-1]
        st.error(f"🚨 **[ตรวจพบเคสวิกฤต - ต้องติดตามทันที]** แผนก: {latest.get('department')} | ข้อความ: \"{latest.get('text')}\"")

# ==================== PDF EXTRACTION ENGINE (GEMINI 3.8 FLASH) ====================
def extract_complaint_from_pdf(pdf_file) -> dict:
    try:
        if hasattr(pdf_file, "getvalue"):
            pdf_bytes = pdf_file.getvalue()
        else:
            pdf_bytes = pdf_file.read()
    except Exception:
        pdf_bytes = None

    if not pdf_bytes:
        return {"วันที่": "-", "ชื่อลูกค้า": "-", "เบอร์ติดต่อกลับ": "-", "แผนกที่เกี่ยวข้อง": "บริการทั่วไปของโรงพยาบาล", "ข้อความความคิดเห็นของลูกค้า": "-"}

    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

    if HAS_GENAI and api_key:
        client = genai.Client(api_key=api_key)
        prompt = """คุณคือผู้เชี่ยวชาญด้านการดึงข้อมูลจากเอกสารแบบฟอร์ม (Document AI Extraction)
กรุณาดูเอกสาร Google Form PDF ข้อเสนอแนะ/ข้อร้องเรียนนี้ แล้วสกัดข้อมูลตอบกลับเป็น JSON Format เท่านั้น:
{
  "วันที่": "ดึงวันที่รับบริการ แล้วแปลงให้อยู่ในฟอร์แมต YYYY-MM-DD (เช่น 2026-07-20 หรือ 2026-05-08) หากไม่มีให้ใส่ '-'",
  "ชื่อลูกค้า": "ดึงชื่อ-นามสกุลของผู้ให้ข้อเสนอแนะ หากเป็นเครื่องหมายขีด '-' หรือไม่ได้ระบุชื่อให้ใส่ '-'",
  "เบอร์ติดต่อกลับ": "ดึงเบอร์โทรศัพท์ติดต่อกลับ (ต้องเป็นเบอร์โทรศัพท์เท่านั้น อย่าเอาวันที่มารวม) หากเป็นขีด '-' หรือไม่มี ให้ใส่ '-'",
  "แผนกที่เกี่ยวข้อง": "ระบุแผนกโดยเลือกให้ตรงกับ 1 ในตัวเลือกนี้เท่านั้น: ['แผนกผู้ป่วยนอก (OPD)', 'แผนกอุบัติเหตุและฉุกเฉิน (ER)', 'แผนกเภสัชกรรม/ห้องยา', 'แผนกการเงิน/ชำระเงิน', 'ศูนย์ตรวจสุขภาพและอาชีวเวชศาสตร์', 'แผนกทันตกรรม', 'แผนกผู้ป่วยใน (IPD)', 'บริการทั่วไปของโรงพยาบาล']",
  "ข้อความความคิดเห็นของลูกค้า": "ดึงเนื้อหาข้อความจริงที่ลูกค้าเขียนในช่อง 'ข้อร้องเรียน/ปัญหาที่พบ' และ 'ข้อเสนอแนะอื่นๆ' (หากมีทั้งสองช่องให้นำมารวมกัน). หากไม่มีข้อร้องเรียนแต่มีช่อง 'สิ่งที่ท่านชอบ/ประทับใจ' ให้ดึงมาแสดงแทน. หากลูกค้าพิมพ์คำว่า 'ไม่มี' หรือ 'หาไม่เจอเลย' ให้ถือว่าไม่มีความคิดเห็นและใส่ '-'. ห้ามนำชื่อหัวข้อแบบฟอร์มหรือข้อความขอบคุณท้ายฟอร์มมาใส่เด็ดขาด"
}"""
        for model_name in ["gemini-3.8-flash", "gemini-3.5-flash"]:
            try:
                pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[pdf_part, prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
                )
                raw_json = re.sub(r"```(?:json)?\s*|\s*```", "", resp.text.strip())
                data = json.loads(raw_json)
                time.sleep(0.8)
                return {
                    "วันที่": str(data.get("วันที่", "-")),
                    "ชื่อลูกค้า": str(data.get("ชื่อลูกค้า", "-")),
                    "เบอร์ติดต่อกลับ": str(data.get("เบอร์ติดต่อกลับ", "-")),
                    "แผนกที่เกี่ยวข้อง": str(data.get("แผนกที่เกี่ยวข้อง", "บริการทั่วไปของโรงพยาบาล")),
                    "ข้อความความคิดเห็นของลูกค้า": str(data.get("ข้อความความคิดเห็นของลูกค้า", "-"))
                }
            except Exception:
                continue

    return {"วันที่": "-", "ชื่อลูกค้า": "-", "เบอร์ติดต่อกลับ": "-", "แผนกที่เกี่ยวข้อง": "บริการทั่วไปของโรงพยาบาล", "ข้อความความคิดเห็นของลูกค้า": "-"}

# ==================== POP-UP PDF BATCH CONVERTER ====================
@st.dialog("📄 เครื่องมือแปลงไฟล์ PDF ร้องเรียนเป็น CSV (Batch PDF Ingestion)")
def open_pdf_batch_converter_dialog():
    st.write("อัปโหลดไฟล์ PDF รายงานข้อร้องเรียนของคนไข้พร้อมกันหลายไฟล์:")
    uploaded_pdfs = st.file_uploader(
        "เลือกไฟล์ PDF (หลายไฟล์):",
        type=["pdf"],
        accept_multiple_files=True,
        key="batch_pdf_files"
    )

    if uploaded_pdfs:
        st.info(f"📁 ตรวจพบไฟล์ PDF ทั้งหมด: {len(uploaded_pdfs)} ไฟล์")

        if st.button("⚡ เริ่มสกัดข้อมูลและรวมไฟล์เป็น CSV", type="primary", use_container_width=True):
            extracted_records = []
            progress_bar = st.progress(0)

            for i, pfile in enumerate(uploaded_pdfs):
                record = extract_complaint_from_pdf(pfile)
                extracted_records.append(record)
                progress_bar.progress((i + 1) / len(uploaded_pdfs))

            out_df = pd.DataFrame(extracted_records)
            st.session_state["pdf_converted_df"] = out_df

            # บันทึกลง Cloud Database กลางทันที (ทุกเครื่องจะเห็น Real-Time)
            if db:
                try:
                    db.save_new_complaints(extracted_records)
                    st.success("🎉 แปลงข้อมูลและบันทึกลง Cloud Database สำเร็จ! (ข้อมูลอัปเดตทุกเครื่องทันที)")
                except Exception as e:
                    st.warning(f"⚠️ แปลงไฟล์สำเร็จ แต่เกิดข้อผิดพลาดในการบันทึกขึ้น Cloud: {e}")
            else:
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
                file_name=f"hospital_complaints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_import:
            if st.button("🚀 ส่งเข้าตัววิเคราะห์ AI ทันที (Bulk Analyze)", type="secondary", use_container_width=True):
                st.session_state["last_bulk_df"] = df_res
                st.rerun()

# ============================== PAGE 1: ANALYZE ==============================
def page_analyze():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 วิเคราะห์ความคิดเห็นของผู้รับบริการ")

    category = st.selectbox(
        "📌 เลือกแผนก / ส่วนงานบริการที่ต้องการประเมินผล:",
        [
            "บริการทั่วไปของโรงพยาบาล", "แผนกผู้ป่วยนอก (OPD)", "แผนกผู้ป่วยใน (IPD)",
            "ศูนย์ตรวจสุขภาพ (Check-up)", "แผนกฉุกเฉินและอุบัติเหตุ (ER)", "แผนกศัลยกรรม (Surgery)",
            "แผนกสูตินรีเวช (OB-GYN)", "แผนกทันตกรรม (Dental)", "แผนกเภสัชกรรม/ห้องยา", "แผนกการเงิน/ชำระเงิน"
        ]
    )

    tab_manual, tab_bulk = st.tabs(["✍️ ป้อนข้อความเดี่ยว", "📁 อัปโหลดไฟล์ชุดใหญ่ (Bulk / PDF)"])

    with tab_manual:
        txt = st.text_area("ข้อความความคิดเห็น:", height=110, placeholder="พิมพ์ความคิดเห็น เช่น หมอตรวจละเอียดดีมาก แต่ที่จอดรถน้อยและค่ายาค่อนข้างแพง...")
        if st.button("🚀 ประมวลผลความคิดเห็น", type="primary", use_container_width=True):
            if not txt.strip():
                st.warning("กรุณากรอกข้อความก่อนวิเคราะห์")
            else:
                res = analyze_aspects_smart(txt)
                label = res.get("overall", "neu")
                append_log(txt, label, category)
                
                # บันทึกขึ้น Cloud DB
                if db:
                    try:
                        db.save_new_complaints([{
                            "วันที่": datetime.now().strftime("%Y-%m-%d"),
                            "ชื่อลูกค้า": "-",
                            "เบอร์ติดต่อกลับ": "-",
                            "แผนกที่เกี่ยวข้อง": category,
                            "ข้อความความคิดเห็นของลูกค้า": txt,
                            "ความรู้สึก": "บวก" if label == "pos" else "ลบ" if label == "neg" else "เป็นกลาง"
                        }])
                    except Exception:
                        pass

                st.session_state["last_aspect_result"] = {"result": res, "category": category, "text": txt}
                st.rerun()

        if "last_aspect_result" in st.session_state:
            data = st.session_state["last_aspect_result"]
            r = data["result"]
            st.success(f"ผลประเมินภาพรวม: **{r['overall'].upper()}** | แผนก: {data['category']}")

    with tab_bulk:
        c_title, c_btn = st.columns([2.5, 1.5])
        with c_title:
            st.write("อัปโหลดไฟล์ CSV/Excel หรือสกัดไฟล์ PDF เข้าสู่ระบบ:")
        with c_btn:
            if st.button("📑 นำเข้าจากโฟลเดอร์ PDF ร้องเรียน", use_container_width=True):
                open_pdf_batch_converter_dialog()

        uploaded_file = st.file_uploader("เลือกไฟล์ CSV หรือ Excel:", type=["csv", "xlsx", "xls"], key="bulk_uploader")
        if uploaded_file is not None:
            bulk_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.write(f"พบข้อมูล {len(bulk_df)} แถว")
            text_col = st.selectbox("เลือกคอลัมน์ข้อความ:", bulk_df.columns)
            if st.button("⚡ เริ่มวิเคราะห์ข้อความทั้งหมด", type="primary", use_container_width=True):
                res_list = []
                for _, row in bulk_df.iterrows():
                    ans = analyze_aspects_smart(str(row[text_col]))
                    res_list.append(ans.get("overall", "neu"))
                bulk_df["ผลวิเคราะห์"] = res_list
                st.session_state["last_bulk_df"] = bulk_df
                st.success("วิเคราะห์ครบถ้วนแล้ว!")
                st.dataframe(bulk_df.head(5), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================== PAGE 2: SUMMARY & CHARTS ==============================
def page_summary():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 แดชบอร์ดสรุปผลสถิติข้อร้องเรียน")

    # ดึงข้อมูลจาก Cloud DB เป็นหลัก
    df_data = db.fetch_complaints() if db else load_log()
    
    total = len(df_data)
    pos_cnt = (df_data["ความรู้สึก"] == "บวก").sum() if "ความรู้สึก" in df_data.columns else 0
    neg_cnt = (df_data["ความรู้สึก"] == "ลบ").sum() if "ความรู้สึก" in df_data.columns else 0
    neu_cnt = total - (pos_cnt + neg_cnt)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ความคิดเห็นทั้งหมด", total)
    k2.metric("เชิงบวก (Positive)", pos_cnt)
    k3.metric("เป็นกลาง (Neutral)", neu_cnt)
    k4.metric("เชิงลบ (Negative)", neg_cnt)

    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        if total > 0:
            df_chart = pd.DataFrame({"ผล": ["เชิงบวก", "เป็นกลาง", "เชิงลบ"], "จำนวน": [pos_cnt, neu_cnt, neg_cnt]})
            fig_bar = px.bar(df_chart, x="ผล", y="จำนวน", color="ผล", color_discrete_map={"เชิงบวก": "#22c55e", "เป็นกลาง": "#94a3b8", "เชิงลบ": "#ef4444"}, height=320)
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#ffffff"))
            st.plotly_chart(fig_bar, use_container_width=True)
    with g2:
        if total > 0:
            fig_pie = px.pie(values=[pos_cnt, neu_cnt, neg_cnt], names=["เชิงบวก", "เป็นกลาง", "เชิงลบ"], hole=0.45, color_discrete_sequence=["#22c55e", "#94a3b8", "#ef4444"], height=320)
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#ffffff"))
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================== PAGE 3: CLOUD DATA & ADMIN DELETE ==============================
def page_cloud_data_and_management(is_admin: bool):
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("### 📑 ตารางข้อมูลข้อร้องเรียนทั้งหมด (Real-Time Cloud Sync)")

    df_cloud = db.fetch_complaints() if db else pd.DataFrame()

    if df_cloud.empty:
        st.info("ยังไม่มีข้อมูลข้อร้องเรียนในฐานข้อมูล Cloud กรุณาอัปโหลดสกัดจากไฟล์ PDF")
    else:
        st.dataframe(df_cloud, use_container_width=True)

        st.markdown("---")
        if is_admin:
            st.subheader("🗑️ แผงควบคุมการลบข้อมูล (สำหรับ Admin เท่านั้น)")
            col_id, col_btn = st.columns([3, 1])
            with col_id:
                target_id = st.selectbox(
                    "เลือก ID รายการข้อร้องเรียนที่ต้องการลบ:",
                    df_cloud["ID"].tolist(),
                    format_func=lambda x: f"ID {x} | แผนก: {df_cloud[df_cloud['ID'] == x]['แผนกที่เกี่ยวข้อง'].values[0]} | ข้อความ: {str(df_cloud[df_cloud['ID'] == x]['ข้อความความคิดเห็นของลูกค้า'].values[0])[:40]}..."
                )
            with col_btn:
                st.write("")
                st.write("")
                if st.button("🗑️ ยืนยันการลบข้อมูล", type="primary", use_container_width=True):
                    db.delete_complaint_by_id(target_id)
                    st.success(f"ลบรายการ ID {target_id} สำเร็จ!")
                    st.rerun()
        else:
            st.caption("🔒 การลบข้อมูลถูกจำกัดสิทธิ์เฉพาะบัญชีระดับ Administrator เท่านั้น บัญชีผู้ใช้ทั่วไปสามารถดูข้อมูลได้อย่างเดียว")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================== PAGE 4: EXPORT PDF ==============================
def page_export_pdf():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 ส่งออกรายงานผลสรุปเป็นไฟล์ PDF (Executive Summary)")

    df_all = db.fetch_complaints() if db else load_log()
    total = len(df_all)
    pos = (df_all["ความรู้สึก"] == "บวก").sum() if "ความรู้สึก" in df_all.columns else 0
    neg = (df_all["ความรู้สึก"] == "ลบ").sum() if "ความรู้สึก" in df_all.columns else 0
    neu = total - (pos + neg)

    metrics = {"total": total, "pos": pos, "neu": neu, "neg": neg}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนเรื่องทั้งหมด", total)
    c2.metric("เชิงบวก (Positive)", pos)
    c3.metric("เป็นกลาง (Neutral)", neu)
    c4.metric("เชิงลบ (Negative)", neg)

    st.write("เอกสาร PDF จะรวบรวมตัวเลขสถิติ ผลการประเมินรายแผนก และตัวอย่างข้อร้องเรียนล่าสุด")

    if st.button("🚀 สร้างและดาวน์โหลดเอกสาร PDF", type="primary"):
        if generate_pdf_report:
            with st.spinner("กำลังประกอบหน้าเอกสาร PDF..."):
                pdf_bytes = generate_pdf_report(metrics, fig_radar=None, df_sample=df_all)
                st.download_button(
                    label="📥 คลิกที่นี่เพื่อดาวน์โหลดไฟล์ PDF",
                    data=pdf_bytes,
                    file_name=f"Hospital_Sentiment_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("⚠️ ไม่พบโมดูล pdf_exporter.py กรุณาตรวจสอบว่าสร้างไฟล์ pdf_exporter.py แล้วหรือยัง")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================== PAGE 5: ADMIN USER MANAGEMENT ==============================
def page_admin_users():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ จัดการบัญชีผู้ใช้งานระบบ (Admin Only)")

    if not db:
        st.error("⚠️ ไม่พบการเชื่อมต่อกับ db_manager.py")
        return

    col_add, col_list = st.columns([1, 1.2])
    with col_add:
        st.subheader("➕ เพิ่มผู้ใช้งานใหม่")
        new_u = st.text_input("ชื่อผู้ใช้งาน (Username)")
        new_p = st.text_input("รหัสผ่าน (Password)", type="password")
        new_r = st.selectbox("สิทธิ์การใช้งาน (Role)", ["user", "admin"])
        if st.button("บันทึกผู้ใช้ใหม่", type="primary"):
            if new_u and new_p:
                db.create_user(new_u, new_p, new_r)
                st.success(f"เพิ่มผู้ใช้ {new_u} ({new_r}) สำเร็จ!")
                st.rerun()
            else:
                st.warning("กรุณากรอกข้อมูลให้ครบถ้วน")

    with col_list:
        st.subheader("👥 บัญชีทั้งหมดในระบบ")
        users_df = db.fetch_all_users()
        if not users_df.empty:
            st.dataframe(users_df, use_container_width=True)
            u_del = st.selectbox("เลือกบัญชีที่ต้องการลบ:", users_df["username"].tolist())
            if st.button("🗑️ ลบบัญชีผู้ใช้ที่เลือก"):
                curr_user = st.session_state.get("auth", {}).get("username", "")
                if u_del == curr_user:
                    st.error("ไม่สามารถลบบัญชีตัวเองที่กำลังล็อกอินอยู่ได้")
                else:
                    db.delete_user_by_name(u_del)
                    st.success(f"ลบบัญชี {u_del} เรียบร้อยแล้ว")
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ============================== MAIN APPLICATION ==============================
def main():
    require_login()
    inject_custom_css()
    render_header_navbar()

    # ตรวจสอบสิทธิ์ของบัญชีที่ล็อกอินอยู่
    is_admin = st.session_state.get("auth", {}).get("role") == "admin"

    # จัดแท็บเมนูตามสิทธิ์: Admin จะมีแท็บจัดการผู้ใช้งานเพิ่มขึ้นมา
    tab_titles = [
        "🔍 วิเคราะห์ & สกัด PDF",
        "📊 สรุปผลสถิติ",
        "📑 ตารางข้อมูล Real-Time",
        "📤 ส่งออกรายงาน PDF"
    ]
    if is_admin:
        tab_titles.append("⚙️ จัดการผู้ใช้ (Admin)")

    tabs = st.tabs(tab_titles)

    with tabs[0]:
        banner("analyze")
        page_analyze()

    with tabs[1]:
        banner("summary")
        page_summary()

    with tabs[2]:
        page_cloud_data_and_management(is_admin)

    with tabs[3]:
        page_export_pdf()

    if is_admin:
        with tabs[4]:
            page_admin_users()

if __name__ == "__main__":
    main()
