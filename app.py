import streamlit as st
import tempfile
import numpy as np
import os
# Import your custom modules
from processor import extract_face_sequence
from model_logic import deepfake_detector

st.set_page_config(page_title="Deepfake Detection Lab", layout="wide")

# --- UI Header ---
st.title("✨ AI Deepfake Detection Lab")
st.write("Researcher: Esmatullah Ekhlas")

# --- CRITICAL: DEFINE COLUMNS FIRST ---
col1, col2 = st.columns([1.3, 0.7], gap="large")

# --- COLUMN 1: UPLOAD ---
with col1:
    st.markdown("### 📤 Video Source")
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        st.video(video_path)

# --- COLUMN 2: ANALYSIS ---
with col2:
    st.markdown("### 🧠 Intelligent Processing")
    if uploaded_file:
        if st.button('Run Deepfake Analysis'):
            with st.status("Analyzing...", expanded=True):
                st.write("🔍 Extracting face sequence...")
                faces = extract_face_sequence(video_path)
                
                if len(faces) == 10:
                    st.write("⚡ Running CNN-Transformer Inference...")
                    input_data = np.expand_dims(faces, axis=0)
                    prediction = deepfake_detector.predict(input_data)
                    confidence = float(prediction[0][0])
                    
                    st.subheader("Final Verdict")
                    if confidence > 0.5:
                        st.success(f"REAL VIDEO")
                        st.metric(label="Authenticity Score", value=f"{confidence*100:.1f}%")
                    else:
                        st.error(f"FAKE / DEEPFAKE")
                        st.metric(label="Manipulation Probability", value=f"{(1-confidence)*100:.1f}%")
                    st.balloons()
                else:
                    st.warning(f"Detected only {len(faces)} faces. Need 10 for full analysis.")
    else:
        st.info("Waiting for upload...")