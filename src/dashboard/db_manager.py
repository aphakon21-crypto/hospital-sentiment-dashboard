# src/dashboard/db_manager.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from supabase import create_client, Client

# --- 1. ฟังก์ชันเชื่อมต่อ Supabase Client ---
@st.cache_resource
def get_db_client() -> Client:
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        raise ValueError("ไม่พบคีย์ SUPABASE_URL หรือ SUPABASE_KEY ใน st.secrets")
    return create_client(url, key)

# --- 2. ฟังก์ชันยืนยันตัวตนผู้ใช้งาน ---
def authenticate(username, password):
    """ตรวจสอบชื่อผู้ใช้และรหัสผ่าน พร้อมดักจับข้อผิดพลาดการเชื่อมต่อ"""
    try:
        supabase = get_db_client()
        res = (
            supabase.table("app_users")
            .select("*")
            .eq("username", username)
            .eq("password_hash", password)
            .execute()
        )
        if res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception as e:
        st.sidebar.error(f"⚠️ เกิดข้อผิดพลาดจากฐานข้อมูล: {e}")
        return None

# --- 3. ฟังก์ชันจัดการข้อมูลข้อร้องเรียน (Complaints) ---
def fetch_complaints() -> pd.DataFrame:
    """ดึงข้อมูลล่าสุดจาก Cloud เสมอ ทำให้ทุกเครื่องเห็นตรงกันแบบ Real-time"""
    try:
        supabase = get_db_client()
        res = supabase.table("complaints").select("*").order("id", desc=True).execute()
        if res.data:
            df = pd.DataFrame(res.data)
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
    except Exception as e:
        st.error(f"⚠️ ดึงข้อมูลจาก Cloud ไม่สำเร็จ: {e}")
    
    return pd.DataFrame(columns=["ID", "วันที่", "ชื่อลูกค้า", "เบอร์ติดต่อกลับ", "แผนกที่เกี่ยวข้อง", "ข้อความความคิดเห็นของลูกค้า", "ความรู้สึก"])

def save_new_complaints(records: list[dict]):
    """บันทึกข้อมูลที่สกัดจาก PDF ขึ้น Cloud"""
    try:
        supabase = get_db_client()
        db_records = []
        for r in records:
            db_records.append({
                "date": str(r.get("วันที่", "-")),
                "patient_name": str(r.get("ชื่อลูกค้า", "-")),
                "phone": str(r.get("เบอร์ติดต่อกลับ", "-")),
                "department": str(r.get("แผนกที่เกี่ยวข้อง", "บริการทั่วไปของโรงพยาบาล")),
                "feedback": str(r.get("ข้อความความคิดเห็นของลูกค้า", "-")),
                "sentiment": str(r.get("ความรู้สึก", "รอดำเนินการ"))
            })
        if db_records:
            supabase.table("complaints").insert(db_records).execute()
    except Exception as e:
        raise RuntimeError(f"ไม่สามารถบันทึกข้อมูลขึ้น Supabase ได้: {e}")

def delete_complaint_by_id(record_id: int):
    """ลบข้อร้องเรียน (สิทธิ์ Admin เท่านั้น)"""
    try:
        supabase = get_db_client()
        supabase.table("complaints").delete().eq("id", record_id).execute()
    except Exception as e:
        st.error(f"⚠️ ลบข้อมูลไม่สำเร็จ: {e}")

# --- 4. ฟังก์ชันจัดการบัญชีผู้ใช้งาน (Users) ---
def fetch_all_users() -> pd.DataFrame:
    """ดึงรายชื่อผู้ใช้ทั้งหมด"""
    try:
        supabase = get_db_client()
        res = supabase.table("app_users").select("username, role, created_at").execute()
        return pd.DataFrame(res.data) if res.data else pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ ดึงรายชื่อผู้ใช้ไม่สำเร็จ: {e}")
        return pd.DataFrame()

def create_user(username, password, role="user"):
    """สร้างผู้ใช้งานใหม่"""
    try:
        supabase = get_db_client()
        supabase.table("app_users").insert({
            "username": username,
            "password_hash": password,
            "role": role
        }).execute()
    except Exception as e:
        st.error(f"⚠️ เพิ่มผู้ใช้ไม่สำเร็จ: {e}")

def delete_user_by_name(username):
    """ลบผู้ใช้งานตามชื่อ"""
    try:
        supabase = get_db_client()
        supabase.table("app_users").delete().eq("username", username).execute()
    except Exception as e:
        st.error(f"⚠️ ลบผู้ใช้ไม่สำเร็จ: {e}")
