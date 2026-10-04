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
