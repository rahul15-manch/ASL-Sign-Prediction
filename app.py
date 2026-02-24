import sys
import os

# Ensure src/ is on the path for preprocessing utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import cv2
from tensorflow.keras.models import load_model
from gtts import gTTS

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ASL Sign Predictor",
    page_icon="🖐️",
    layout="wide",
)

# ─────────────────────────────────────────
# Load model & class mapping (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model("models/sign_model.h5")
    with open("models/class_indices.json", "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_resources()

# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────
def preprocess_image(image: Image.Image, target_size=(64, 64)) -> np.ndarray:
    """Convert PIL image → normalised numpy array ready for inference."""
    frame = np.array(image.convert("RGB"))
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)


def predict(image: Image.Image):
    """Return (label, confidence, top3_list) for a PIL image."""
    inp = preprocess_image(image)
    probs = model.predict(inp, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(idx_to_class.get(i, "?"), float(probs[i])) for i in top3_idx]
    label, confidence = top3[0]
    return label, confidence, top3


def text_to_speech_audio(text: str) -> bytes:
    """Generate gTTS audio bytes for a given text string."""
    tts = gTTS(text=text, lang="en")
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def confidence_bar(label: str, confidence: float, primary: bool = False):
    """Render a styled confidence bar."""
    color = "#4CAF50" if primary else "#90CAF9"
    bar_html = f"""
    <div style="margin:4px 0">
      <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
        <span><b>{label}</b></span>
        <span>{confidence*100:.1f}%</span>
      </div>
      <div style="background:#e0e0e0; border-radius:6px; height:10px;">
        <div style="width:{confidence*100:.1f}%; background:{color};
                    border-radius:6px; height:10px;"></div>
      </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def show_prediction_card(image: Image.Image, label: str, confidence: float, top3, tts_enabled: bool):
    """Display image + prediction result in a two-column card."""
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.image(image, use_column_width=True)

    with col_res:
        st.markdown(f"## 🔤 `{label}`")
        st.markdown(f"**Confidence:** {confidence * 100:.1f}%")

        st.markdown("#### Top 3 Predictions")
        for i, (lbl, prob) in enumerate(top3):
            confidence_bar(lbl, prob, primary=(i == 0))

        if tts_enabled:
            st.markdown("#### 🔊 Speech Output")
            audio_bytes = text_to_speech_audio(label)
            st.audio(audio_bytes, format="audio/mp3")

    st.divider()


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.radio(
        "Input Mode",
        ("📷 Webcam", "🖼️ Single Image", "🗂️ Multiple Images"),
    )
    tts_enabled = st.toggle("🔊 Text-to-Speech", value=False)
    st.divider()

    st.markdown("### 🧠 Model Info")
    st.markdown(
        """
        - **Architecture**: Custom CNN
        - **Input**: 64×64 RGB
        - **Classes**: 29 (A–Z + space, delete, nothing)
        - **Val Accuracy**: ~79%
        - **Test Accuracy**: ~93%
        - **Regularization**: BatchNorm + Dropout
        """
    )

# ─────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────
st.title("🖐️ ASL Sign Language Predictor")
st.markdown(
    "Upload a hand sign image or take a photo to get an instant ASL letter prediction."
)

st.divider()

# ────── Webcam ──────
if mode == "📷 Webcam":
    st.subheader("📷 Capture via Webcam")
    uploaded_file = st.camera_input("Take a picture")
    if uploaded_file:
        image = Image.open(uploaded_file)
        label, confidence, top3 = predict(image)
        show_prediction_card(image, label, confidence, top3, tts_enabled)

# ────── Single Image ──────
elif mode == "🖼️ Single Image":
    st.subheader("🖼️ Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        label, confidence, top3 = predict(image)
        show_prediction_card(image, label, confidence, top3, tts_enabled)

# ────── Multiple Images ──────
else:
    st.subheader("🗂️ Upload Multiple Images")
    uploaded_files = st.file_uploader(
        "Choose images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"#### Image {i + 1} — `{uploaded_file.name}`")
            image = Image.open(uploaded_file)
            label, confidence, top3 = predict(image)
            show_prediction_card(image, label, confidence, top3, tts_enabled)
