import streamlit as st
import tempfile
import numpy as np
import os
# Import your custom modules
from processor import extract_face_sequence
from model_logic import deepfake_detector

# --- تنظیمات صفحه ---
st.set_page_config(page_title="Deepfake Detection Lab", layout="wide")

# --- بخش دیزاین اختصاصی (Glassmorphism UI) ---
st.markdown("""
    <style>
    /* تغییر فونت و رنگ پس‌زمینه کل سایت */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    
    /* استایل شیشه‌ای برای کادرها */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* استایل دکمه آنالیز */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #00dbde, #fc00ff);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(252, 0, 255, 0.4);
    }

    /* استایل تیترها */
    h1, h3 {
        color: #00dbde !important;
        text-shadow: 0px 0px 10px rgba(0, 219, 222, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI Header ---
st.title("✨ AI Deepfake Detection Lab")
st.markdown(f"**Researcher:** `Esmatullah Ekhlas` | **Course:** `Görüntü İşleme`")
st.divider()

# --- ساخت ستون‌ها ---
col1, col2 = st.columns([1.3, 0.7], gap="large")

# --- ستون ۱: آپلود ویدیو ---
with col1:
    st.markdown("### 📤 Video Source")
    uploaded_file = st.file_uploader("فایل ویدیویی را اینجا رها کنید", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        # نمایش ویدیو با گوشه‌های گرد (توسط CSS بالا)
        st.video(video_path)
    else:
        # نمایش یک تصویر پیش‌فرض یا باکس خالی زیبا
        st.info("در انتظار بارگذاری ویدیو برای شروع پردازش تصویر...")

# --- ستون ۲: آنالیز و نتایج ---
with col2:
    st.markdown("### 🧠 Intelligent Processing")
    
    if uploaded_file:
        if st.button('🚀 Run Deepfake Analysis'):
            with st.status("در حال آنالیز لایه‌های تصویر...", expanded=True) as status:
                st.write("🔍 شناسایی چهره (Face Extraction)...")
                faces = extract_face_sequence(video_path)
                
                if len(faces) == 10:
                    st.write("⚡ استخراج ویژگی‌ها با CNN...")
                    st.write("🤖 طبقه‌بندی با Transformer...")
                    
                    input_data = np.expand_dims(faces, axis=0)
                    prediction = deepfake_detector.predict(input_data)
                    confidence = float(prediction[0][0])
                    
                    status.update(label="آنالیز با موفقیت تکمیل شد!", state="complete", expanded=False)

                    st.markdown("---")
                    st.subheader("Final Verdict")
                    
                    if confidence > 0.5:
                        st.success(f"✅ REAL VIDEO")
                        st.metric(label="درصد اصالت (Authenticity)", value=f"{confidence*100:.1f}%")
                    else:
                        st.error(f"🚨 FAKE / DEEPFAKE")
                        st.metric(label="احتمال دستکاری (Manipulation)", value=f"{(1-confidence)*100:.1f}%")
                    
                    st.balloons()
                else:
                    st.warning(f"تعداد {len(faces)} چهره یافت شد. برای آنالیز دقیق به ۱۰ فریم نیاز است.")
    else:
        st.write("لطفاً ابتدا ویدیو را آپلود کنید تا سیستم پردازش تصویر فعال شود.")
        # نمایش وضعیت سیستم به صورت گرافیکی
        st.info("System Status: IDLE")