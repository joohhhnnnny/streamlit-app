import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from ultralytics import YOLO
import numpy as np
import cv2

# PAGE CONFIG
st.set_page_config(page_title="Human vs Non-Human Detector", page_icon="👤", layout="wide")

# THEME (same as before)
st.sidebar.header("Appearance")
theme = st.sidebar.radio("Choose theme:", ["Light", "Dark"], index=0)
if theme == "Light":
    primary_bg = "#eef1f6"; card_bg = "#ffffff"; text_color = "#1c2333"
    button_bg1 = "#4A73FF"; button_bg2 = "#3358E0"; preview_bg = "#f9fafc"; progress_color = "#4A73FF"
else:
    primary_bg = "#0f1116"; card_bg = "#1a1c23"; text_color = "#e5e7eb"
    button_bg1 = "#6366f1"; button_bg2 = "#4f46e5"; preview_bg = "#111317"; progress_color = "#6366f1"

# CSS (same as before)
st.markdown(f"<style>body{{background-color:{primary_bg};color:{text_color};}}</style>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>Human Face Classifier and Detector</h1>", unsafe_allow_html=True)

# MODEL SELECTION
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio("Choose AI model:", ["MobileNetV2", "YoloV8"], index=0)

MODEL_PATHS = {
    "MobileNetV2": "./models/mobilenetv2.h5",
    "YoloV8": "./models/best-yolov8s-v2.pt"
}

# LOAD MODELS
@st.cache_resource
def load_mobilenet_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

with st.spinner(f"Loading {model_choice}..."):
    try:
        if model_choice == "MobileNetV2":
            model = load_mobilenet_model(MODEL_PATHS[model_choice])
        else:
            model = load_yolo_model(MODEL_PATHS[model_choice])
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

confidence_threshold = 0.70

# MobileNetV2 prediction
def predict_mobilenet(image_data, model):
    image_data = image_data.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction[0][0]

# YOLOv8 prediction (returns image with bounding boxes)
def detect_yolo_frame(frame, model):
    results = model(frame)
    annotated_frame = results[0].plot()  # adds bounding boxes directly
    return annotated_frame

# MAIN UI
if model_choice == "MobileNetV2":
    file = st.file_uploader("Upload a photo", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)
        display_width = 400

        # Center the image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", width=display_width)

        if st.button("Analyze Image"):
            probability = predict_mobilenet(image, model)
            human_prob = 1 - probability
            non_human_prob = probability

            # Display both probabilities
            st.metric("Human Probability", f"{human_prob:.2%}")
            st.metric("Non-Human Probability", f"{non_human_prob:.2%}")

            # Display overall result
            if human_prob > non_human_prob:
                st.success("Result: HUMAN")
            else:
                st.error("Result: NON-HUMAN")

            # Optional progress bar (show human probability)
            st.progress(int(human_prob * 100))

else:  # YOLOv8 live webcam

    frame_width = 800  # desired display width

    # Create columns outside the loop
    col1, col2, col3 = st.columns([1, 2, 1])
    # Create the placeholder inside the middle column
    stframe = col2.empty()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect humans and draw bounding boxes
        annotated_frame = detect_yolo_frame(frame, model)

        # Update the image in the centered placeholder
        stframe.image(annotated_frame, channels="RGB", width=frame_width)

        # Stop if user closes the app or presses 'q' (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
