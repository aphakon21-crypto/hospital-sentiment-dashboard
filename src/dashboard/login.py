import streamlit as st
import time

def show_login():
    # 1. CSS สำหรับจัดระเบียบหน้า Login โดยเฉพาะ
    st.markdown("""
    <style>
    /* ซ่อนแถบเมนูด้านซ้ายและแถบด้านบนสุดตอนที่ยังไม่ Login */
    [data-testid="collapsedControl"] { display: none; }
    header[data-testid="stHeader"] { display: none; }
    
    /* ดันเนื้อหาลงมาให้อยู่กึ่งกลางหน้าจอมากขึ้น */
    .block-container {
        padding-top: 10vh !important;
        max-width: 800px;
    }
    
    /* แต่งปุ่ม Submit ใน Form ให้เป็นปุ่มหลัก (Primary) */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #0284c7, #005b9f) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 2. ใช้ Columns เพื่อบีบเนื้อหาให้อยู่ตรงกลาง
    _, center_col, _ = st.columns([0.2, 0.6, 0.2])
    
    with center_col:
        # ส่วนหัวของหน้า Login (โลโก้ และ ชื่อระบบ)
        st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <div style='font-size: 55px; margin-bottom: 10px;'>🏥</div>
            <h1 style='color: #005b9f; font-family: "Prompt", sans-serif; font-size: 32px; font-weight: 700; margin-bottom: 5px;'>
                Sirivej Analytics
            </h1>
            <p style='color: #64748b; font-size: 15px;'>ระบบวิเคราะห์ความคิดเห็นและข้อมูลลูกค้า</p>
        </div>
        """, unsafe_allow_html=True)

        # 3. สร้างกล่องฟอร์มสำหรับกรอกข้อมูล
        with st.form("login_form", clear_on_submit=False):
            st.markdown("<h4 style='font-family: \"Prompt\", sans-serif; font-size: 18px; margin-bottom: 5px; color: #334155;'>เข้าสู่ระบบ</h4>", unsafe_allow_html=True)
            
            username = st.text_input("ชื่อผู้ใช้งาน (Username)", placeholder="กรอกชื่อผู้ใช้งานของคุณ")
            password = st.text_input("รหัสผ่าน (Password)", type="password", placeholder="••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ปุ่มเข้าสู่ระบบ
            submitted = st.form_submit_button("เข้าสู่ระบบ (Login)", use_container_width=True)

            if submitted:
                # =======================================================
                # 📌 กำหนดเงื่อนไขการตรวจสอบ Username / Password ของคุณที่นี่
                # =======================================================
                if username == "admin" and password == "1234":  # เปลี่ยนเป็นรหัสผ่านจริงของคุณ
                    st.success("✅ เข้าสู่ระบบสำเร็จ! กำลังพาท่านเข้าสู่หน้าหลัก...")
                    st.session_state["logged_in"] = True
                    time.sleep(1) # หน่วงเวลาให้ผู้ใช้เห็นข้อความ Success ก่อนเปลี่ยนหน้า
                    st.rerun()
                elif username == "" or password == "":
                    st.warning("⚠️ กรุณากรอกข้อมูลให้ครบถ้วน")
                else:
                    st.error("❌ ชื่อผู้ใช้งานหรือรหัสผ่านไม่ถูกต้อง กรุณาลองใหม่อีกครั้ง")
        
        # ส่วนท้าย (Footer)
        st.markdown("""
        <div style='text-align: center; margin-top: 25px;'>
            <p style='color: #94a3b8; font-size: 12px;'>© 2026 Customer Sentiment Analysis System</p>
        </div>
        """, unsafe_allow_html=True)