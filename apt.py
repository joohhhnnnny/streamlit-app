import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Human vs Non-Human Detector",
    page_icon="👤",
    layout="wide"
)

# =============================
# TITLE
# =============================
st.markdown("<h1 class='main-title'>Human vs Non-Human Detector</h1>", unsafe_allow_html=True)
st.write("Upload an image and let AI decide if it contains a human face or not.")
st.markdown("---")

# =============================
# MODEL SELECTION
# =============================
st.sidebar.header("Model Selection")

model_choice = st.sidebar.radio(
    "Choose AI model:",
    ["MobileNetV2", "YOLOv8"],
    index=0
)

MODEL_PATHS = {
    "YOLOv8": "models/best-yolov8s-v2.pt"
}

# =============================
# MODEL LOADERS
# =============================
@st.cache_resource
def load_mobilenet():
    """Load MobileNetV2 directly from Keras applications with ImageNet weights."""
    return MobileNetV2(weights='imagenet', include_top=True)

@st.cache_resource
def load_yolo(path):
    return YOLO(path)

with st.spinner(f"Loading {model_choice}..."):
    try:
        if model_choice == "MobileNetV2":
            model = load_mobilenet()
        else:
            model = load_yolo(MODEL_PATHS["YOLOv8"])
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

# =============================
# PREDICTION FUNCTIONS
# =============================
CONFIDENCE_THRESHOLD = 0.70

def mobilenet_predict(image, model):
    image = image.convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = model.predict(img)
    prob = preds[0].max()  # highest class probability
    return prob  # probability of top predicted class

def yolo_detect_and_draw(image, model):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model(img)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return img, 0

    h_img, w_img = img.shape[:2]
    scale = max(w_img, h_img) / 1000
    box_thickness = max(2, int(scale * 3))
    text_thickness = max(2, int(scale * 2))
    font_scale = max(0.6, scale * 0.7)

    person_count = 0
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == 0:  # class 0 = person
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"Human {conf:.2%}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

            # Text size
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)

            # Decide label position
            if y1 - h - 10 > 0:
                label_y1 = y1 - h - 10
                label_y2 = y1
                text_y = y1 - 5
            else:
                label_y1 = y1 + 5
                label_y2 = y1 + h + 15
                text_y = y1 + h + 10

            # Draw label background
            cv2.rectangle(
                img,
                (x1, label_y1),
                (x1 + w + 6, label_y2),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                img,
                label,
                (x1 + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                text_thickness,
                cv2.LINE_AA
            )

    return img, person_count

# =============================
# UI
# =============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if file:
    image = Image.open(file)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Uploaded Image")
        st.markdown("<div class='preview-box'>", unsafe_allow_html=True)
        st.image(image, width=600)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("AI Analysis")

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                if model_choice == "MobileNetV2":
                    prob = mobilenet_predict(image, model)
                    if prob >= CONFIDENCE_THRESHOLD:
                        st.success("Result: HUMAN 👤")
                        st.metric("Confidence", f"{prob:.2%}")
                        st.progress(int(prob * 100))
                    else:
                        st.error("Result: NON-HUMAN 🚫")
                        st.metric("Confidence", f"{(1 - prob):.2%}")
                        st.progress(int((1 - prob) * 100))
                else:
                    annotated_img, person_count = yolo_detect_and_draw(image, model)
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                    st.subheader("Detection Result")
                    if person_count > 0:
                        st.success(f"Detected {person_count} human(s)")
                    else:
                        st.warning("No humans detected")

                    st.image(annotated_img, width=600)

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Upload an image to begin.")
