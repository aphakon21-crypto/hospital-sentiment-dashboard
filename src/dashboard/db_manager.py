import streamlit as st
from supabase import create_client, Client
import pandas as pd

# เชื่อมต่อฐานข้อมูล Supabase
@st.cache_resource
def get_db_client() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# --- จัดการข้อร้องเรียน (Complaints) ---
def fetch_complaints() -> pd.DataFrame:
    """ดึงข้อมูลล่าสุดจาก Cloud เสมอ ทำให้ทุกเครื่องเห็นตรงกันแบบ Real-time"""
    supabase = get_db_client()
    res = supabase.table("complaints").select("*").order("id", desc=True).execute()
    if res.data:
        df = pd.DataFrame(res.data)
        # เปลี่ยนชื่อคอลัมน์ให้ตรงกับที่ Dashboard ใช้แสดงผล
        rename_map = {
            "id": "ID",
            "date": "วันที่",
            "patient_name": "ชื่อลูกค้า",
            "phone": "เบอร์ติดต่อกลับ",
            "department": "แผนกที่เกี่ยวข้อง",
            "feedback": "ข้อความความคิดเห็นของลูกค้า",
            "sentiment": "ความรู้สึก"
        }
        return df.rename(columns=rename_map)
    return pd.DataFrame(columns=["ID", "วันที่", "ชื่อลูกค้า", "เบอร์ติดต่อกลับ", "แผนกที่เกี่ยวข้อง", "ข้อความความคิดเห็นของลูกค้า", "ความรู้สึก"])

def save_new_complaints(records: list[dict]):
    """บันทึกข้อมูลที่สกัดจาก PDF ขึ้น Cloud"""
    supabase = get_db_client()
    db_records = []
    for r in records:
        db_records.append({
            "date": r.get("วันที่", "-"),
            "patient_name": r.get("ชื่อลูกค้า", "-"),
            "phone": r.get("เบอร์ติดต่อกลับ", "-"),
            "department": r.get("แผนกที่เกี่ยวข้อง", "บริการทั่วไปของโรงพยาบาล"),
            "feedback": r.get("ข้อความความคิดเห็นของลูกค้า", "-"),
            "sentiment": r.get("ความรู้สึก", "รอดำเนินการ")
        })
    if db_records:
        supabase.table("complaints").insert(db_records).execute()

def delete_complaint_by_id(record_id: int):
    """ลบข้อร้องเรียน (สิทธิ์ Admin เท่านั้น)"""
    supabase = get_db_client()
    supabase.table("complaints").delete().eq("id", record_id).execute()

# --- จัดการผู้ใช้งาน (Users) ---
def authenticate(username, password):
    supabase = get_db_client()
    res = supabase.table("app_users").select("*").eq("username", username).eq("password_hash", password).execute()
    if res.data:
        return res.data[0]
    return None

def fetch_all_users() -> pd.DataFrame:
    supabase = get_db_client()
    res = supabase.table("app_users").select("username, role, created_at").execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

def create_user(username, password, role="user"):
    supabase = get_db_client()
    supabase.table("app_users").insert({"username": username, "password_hash": password, "role": role}).execute()

def delete_user_by_name(username):
    supabase = get_db_client()
    supabase.table("app_users").delete().eq("username", username).execute()
