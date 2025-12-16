import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# PAGE CONFIG
st.set_page_config(
    page_title="Human vs Non-Human Detector",
    page_icon="👤",
    layout="wide"
)

# SIDEBAR THEME SELECTION
st.sidebar.header("Appearance")
theme = st.sidebar.radio("Choose theme:", ["Light ☀️", "Dark 🌙"], index=0)

# THEME COLORS
if theme == "Light ☀️":
    primary_bg = "#eef1f6"
    card_bg = "#ffffff"
    text_color = "#1c2333"
    title_color = "#1c2333"
    button_bg1 = "#4A73FF"
    button_bg2 = "#3358E0"
    preview_bg = "#f9fafc"
    progress_color = "#4A73FF"
else:
    primary_bg = "#0f1116"
    card_bg = "#1a1c23"
    text_color = "#e5e7eb"
    title_color = "#ffffff"
    button_bg1 = "#6366f1"
    button_bg2 = "#4f46e5"
    preview_bg = "#111317"
    progress_color = "#6366f1"

# CSS STYLING
st.markdown(f"""
<style>

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {primary_bg};
}}

/* FIX — force title color change in dark mode */
h1.main-title, .main-title, .main-title h1 {{
    color: {title_color} !important;
    text-align: center !important;
    padding-top: 20px;
}}

.card {{
    background: {card_bg};
    padding: 25px 30px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    margin-bottom: 25px;
    color: {text_color};
}}

.stButton>button {{
    width: 100%;
    background: linear-gradient(135deg, {button_bg1}, {button_bg2});
    color: white;
    border-radius: 10px;
    padding: 10px;
    border: none;
    font-weight: 600;
    transition: 0.2s ease-in-out;
}}

.stButton>button:hover {{
    transform: scale(1.03);
    background: linear-gradient(135deg, {button_bg2}, {button_bg1});
}}

.preview-box {{
    border: 2px dashed #4d5561;
    border-radius: 16px;
    padding: 12px;
    background: {preview_bg};
}}

h2, h3, label, p, span, .stMetric {{
    color: {text_color} !important;
}}

/* Transparent sidebar */
[data-testid="stSidebar"], .stSidebar {{
    background: rgba(0,0,0,0) !important;
    box-shadow: none !important;
}}

.stProgress > div > div > div > div {{
    background-color: {progress_color};
}}

/* Section divider fix */
.section-divider {{
    margin: 20px 0;
    border-top: 1px solid #4d5561;
}}

</style>
""", unsafe_allow_html=True)

# PAGE TITLE
st.markdown("<h1 class='main-title'>Human vs Non-Human Detector</h1>", unsafe_allow_html=True)
st.write("This AI checks if your photo contains a human. Friendly, fast, and surprisingly honest.")
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# MODEL SELECTION SIDEBAR
st.sidebar.header("Model Selection")

model_choice = st.sidebar.radio(
    "Choose which AI model to use:",
    ["MobileNetV2", "YoloV8"],
    index=0
)

MODEL_PATHS = {
    "MobileNetV2": "human_classifier_model.keras",
    "YoloV8": "human_classifier_yolov8.keras" #FOR THE CHAR
}

@st.cache_resource
def load_selected_model(model_path):
    return tf.keras.models.load_model(model_path)

with st.spinner(f"Loading {model_choice}..."):
    try:
        model = load_selected_model(MODEL_PATHS[model_choice])
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# FIXED CONFIDENCE THRESHOLD (since slider removed)
confidence_threshold = 0.70

# PREDICTION FUNCTION
def import_and_predict(image_data, model):
    image_data = image_data.convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)

    img = img / 255.0
    img_reshape = np.expand_dims(img, axis=0)

    prediction = model.predict(img_reshape)
    return prediction

# MAIN UI
st.markdown("<div class='card'>", unsafe_allow_html=True)
file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

if file is None:
    st.info("Upload a photo to begin.")
else:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Uploaded Image")

        image = Image.open(file)

        st.markdown("<div class='preview-box'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("AI Analysis")

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):

                predictions = import_and_predict(image, model)
                probability = predictions[0][0]

                is_human_prob = 1 - probability

                if is_human_prob > confidence_threshold:
                    st.success("Result: HUMAN 👤")
                    st.metric("Confidence", f"{is_human_prob:.2%}")
                    st.progress(int(is_human_prob * 100))
                else:
                    st.error("Result: NON-HUMAN 🚫")
                    st.metric("Confidence", f"{(1 - is_human_prob):.2%}")
                    st.progress(int((1 - is_human_prob) * 100))

        st.markdown("</div>", unsafe_allow_html=True)
