"""
Image Processing App - Home
Run with: streamlit run app.py
"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import time
import adapter
import pathlib

from auth import require_login, logout_button
from db import add_run
from fruit_ripeness_object_detection.detection import detect_fruit_ripeness, draw_detections

# ---------------------------------------------------------
# BACKEND: fruit detection + ripeness classification
# ---------------------------------------------------------

@st.cache_data(show_spinner="Preprocessing and segmenting...")
def run_stages_1_2(file_bytes: bytes, suffix: str):
    return adapter.preprocess_and_segment(file_bytes, suffix)

def process_image(image: Image.Image, confidence_threshold: float):
    """
    Runs the real YOLO fruit ripeness/defect detection model.

    Returns:
        (annotated_image, detections) where annotated_image is a PIL image
        with bounding boxes drawn, and detections is the raw list of
        detected fruits from detect_fruit_ripeness().
    """
    # PIL gives RGB; OpenCV/YOLO expects BGR
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    detections = detect_fruit_ripeness(cv_image, confidence_threshold=confidence_threshold)
    annotated_cv = draw_detections(cv_image, detections)

    # Convert back to RGB for display in Streamlit
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))

    return annotated_image, detections


def get_metrics(detections: list) -> dict:
    """
    Summarizes detection output. No ground-truth accuracy/precision/recall
    is available here (that would require labeled test data) -- this
    reports what the model actually gives us: counts and confidence.
    """
    if not detections:
        return {"count": 0, "avg_confidence": 0.0}

    confidences = [d["confidence"] for d in detections]
    return {
        "count": len(detections),
        "avg_confidence": sum(confidences) / len(confidences),
    }


# ---------------------------------------------------------
# AUTH GATE
# ---------------------------------------------------------

st.set_page_config(page_title="Image Processing Tool", layout="wide")

require_login()

st.sidebar.write(f"Logged in as **{st.session_state.username}**")
logout_button()

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.title("Image Processing Tool")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    original = Image.open(uploaded).convert("RGB")

    st.sidebar.header("Controls")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.40, 0.05)
    use_preprocessing = st.sidebar.checkbox("Use preprocessed image for detection", value=True)

    suffix = pathlib.Path(uploaded.name).suffix or ".jpg"
    stages = run_stages_1_2(uploaded.getvalue(), suffix)

    if stages["is_blurry"]:
        st.warning(f"Image may be too blurry (score: {stages['blur_score']:.1f})")

    detect_input = (
        Image.fromarray(adapter.to_rgb(stages["working_image"]))
        if use_preprocessing else original
    )

    start = time.time()
    annotated, detections = process_image(detect_input, confidence_threshold)
    elapsed_ms = (time.time() - start) * 1000

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption="Detected", use_container_width=True)

    st.divider()
    st.subheader("Detections")

    metrics = get_metrics(detections)

    m1, m2, m3 = st.columns(3)
    m1.metric("Fruits detected", metrics["count"])
    m2.metric("Avg. confidence", f"{metrics['avg_confidence']*100:.1f}%")
    m3.metric("Time", f"{elapsed_ms:.0f} ms")

    if detections:
        st.dataframe(
            pd.DataFrame(detections)[["fruit_type", "ripeness", "confidence"]],
            use_container_width=True,
        )

    with st.expander("Pipeline stages"):
        s1, s2, s3, s4 = st.columns(4)
        s1.image(adapter.to_rgb(stages["working_image"]), caption="Preprocessed")
        s2.image(stages["gray_mask"], caption=f"Otsu gray ({stages['gray_threshold']:.0f})")
        s3.image(stages["saturation_mask"], caption=f"Otsu saturation ({stages['saturation_threshold']:.0f})")
        s4.image(stages["fruit_mask"], caption="Refined mask")
        st.caption(f"Projected fruit area: {stages['fruit_area_pixels']:,} px²")

    if st.button("Save this run to history"):
        add_run(metrics["count"], metrics["avg_confidence"], None, elapsed_ms)
        st.success("Run saved. Check the Dashboard page for history.")

    st.caption("See the **Dashboard** page in the sidebar for accuracy history across saved runs.")

else:
    st.info("Upload an image to get started.")