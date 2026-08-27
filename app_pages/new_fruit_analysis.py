"""
New Fruit Analysis - Upload an image and run fruit detection + ripeness classification.
"""

import pathlib
import time

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import cv2

from db import add_run
from fruit_ripeness_object_detection.detection import detect_fruit_ripeness, draw_detections
import adapter

# ---------------------------------------------------------
# BACKEND: fruit detection + ripeness classification
# ---------------------------------------------------------

def process_image(image: Image.Image, confidence_threshold: float):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    detections = detect_fruit_ripeness(cv_image, confidence_threshold=confidence_threshold)
    annotated_cv = draw_detections(cv_image, detections)
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))
    return annotated_image, detections


def get_metrics(detections: list) -> dict:
    if not detections:
        return {"count": 0, "avg_confidence": 0.0}
    confidences = [d["confidence"] for d in detections]
    return {
        "count": len(detections),
        "avg_confidence": sum(confidences) / len(confidences),
    }


@st.cache_data(show_spinner="Preprocessing and segmenting...")
def run_stages_1_2(file_bytes: bytes, suffix: str):
    return adapter.preprocess_and_segment(file_bytes, suffix)


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.title("New Fruit Analysis")

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
        add_run(st.session_state.user_email, metrics["count"], metrics["avg_confidence"], elapsed_ms)
        st.success("Run saved. Check the Dashboard or Analysis History page for it.")

    st.caption("See **Analysis History** in the sidebar for past runs.")

else:
    st.info("Upload an image to get started.")