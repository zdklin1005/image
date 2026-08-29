"""
New Fruit Analysis - Upload an image and run the full detection +
ripeness + blemish pipeline.
"""

import pathlib
import time

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import cv2

from db import add_run, add_feedback
import pipeline
import adapter

# ---------------------------------------------------------
# BACKEND
# ---------------------------------------------------------

def process_image(
    image: Image.Image,
    conf_a: float,
    conf_c: float,
    conf_d: float,
    iou_threshold: float,
    pixels_per_cm: float,
):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results, annotated_cv = pipeline.analyze_image(
        cv_image,
        confidence_threshold_a=conf_a,
        confidence_threshold_c=conf_c,
        confidence_threshold_d=conf_d,
        iou_threshold=iou_threshold,
        pixels_per_cm=pixels_per_cm,
    )
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))
    return annotated_image, results


def get_metrics(results: list) -> dict:
    if not results:
        return {"count": 0, "avg_confidence": 0.0, "avg_blemish_pct": 0.0}

    det_conf = [r["detection_confidence"] for r in results]
    blemish_vals = [
        r["blemish_percentage"] for r in results if r["blemish_percentage"] is not None
    ]
    return {
        "count": len(results),
        "avg_confidence": sum(det_conf) / len(det_conf),
        "avg_blemish_pct": (sum(blemish_vals) / len(blemish_vals)) if blemish_vals else 0.0,
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
    confidence_threshold = st.sidebar.slider(
        "Detection confidence threshold", 0.0, 1.0, 0.40, 0.05,
        help="Applied to Models A and C. Model D uses a slightly lower threshold by default.",
    )
    use_preprocessing = st.sidebar.checkbox("Use preprocessed image for detection", value=True)

    with st.sidebar.expander("Advanced"):
        conf_d = st.slider("Model D confidence threshold", 0.0, 1.0, 0.30, 0.05)
        iou_threshold = st.slider(
            "Fusion IoU threshold", 0.0, 1.0, 0.30, 0.05,
            help="How much bounding-box overlap is required to treat detections from different models as the same physical fruit.",
        )
        pixels_per_cm = st.number_input(
            "Size calibration (pixels per cm)", min_value=1.0, value=20.0, step=1.0,
            help="Fixed scale used to convert fruit size from pixels to cm². Adjust if your camera setup/distance changes.",
        )

    suffix = pathlib.Path(uploaded.name).suffix or ".jpg"
    stages = run_stages_1_2(uploaded.getvalue(), suffix)

    if stages["is_blurry"]:
        st.warning(f"Image may be too blurry (score: {stages['blur_score']:.1f})")

    detect_input = (
        Image.fromarray(adapter.to_rgb(stages["working_image"]))
        if use_preprocessing else original
    )

    start = time.time()
    annotated, results = process_image(
        detect_input, confidence_threshold, confidence_threshold, conf_d, iou_threshold, pixels_per_cm
    )
    elapsed_ms = (time.time() - start) * 1000

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption="Detected", use_container_width=True)

    st.divider()
    st.subheader("Detections")

    metrics = get_metrics(results)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Fruits detected", metrics["count"])
    m2.metric("Avg. confidence", f"{metrics['avg_confidence']*100:.1f}%")
    m3.metric("Avg. blemish", f"{metrics['avg_blemish_pct']:.1f}%")
    m4.metric("Blur score", f"{stages['blur_score']:.1f}", help="Lower means blurrier. A warning is shown above if this is too low.")
    m5.metric("Time", f"{elapsed_ms:.0f} ms")

    if results:
        table = pd.DataFrame(results)[
            ["fruit_type", "ripeness", "ripeness_confidence", "detection_confidence", "blemish_percentage", "fruit_area_cm2", "agreement"]
        ]
        st.dataframe(table, use_container_width=True)

        with st.expander("Per-fruit detail"):
            for i, r in enumerate(results, start=1):
                st.markdown(f"**Fruit {i}: {r['fruit_type']} ({r['ripeness']})**")

                d1, d2, d3 = st.columns(3)
                d1.metric("Ripeness confidence", f"{r['ripeness_confidence']*100:.0f}%")
                if r["blemish_percentage"] is not None:
                    severity = (
                        "Low" if r["blemish_percentage"] < 5
                        else "Moderate" if r["blemish_percentage"] < 15
                        else "High"
                    )
                    d2.metric("Blemish", f"{r['blemish_percentage']:.1f}%", severity)
                else:
                    d2.metric("Blemish", "N/A")
                d3.metric("Size", f"{r['fruit_area_cm2']:.1f} cm²" if r["fruit_area_cm2"] is not None else "N/A")

                c1, c2 = st.columns(2)
                c1.image(cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB), caption="Crop")
                if r["blemish_overlay"] is not None:
                    c2.image(
                        cv2.cvtColor(r["blemish_overlay"], cv2.COLOR_BGR2RGB),
                        caption="Blemish overlay",
                    )
                else:
                    c2.caption("Blemish detection unavailable for this crop.")
                st.caption(f"Detected by: {r['agreement']}")
                st.divider()

    with st.expander("Pipeline stages (whole image)"):
        s1, s2, s3, s4 = st.columns(4)
        s1.image(adapter.to_rgb(stages["working_image"]), caption="Preprocessed")
        s2.image(stages["gray_mask"], caption=f"Otsu gray ({stages['gray_threshold']:.0f})")
        s3.image(stages["saturation_mask"], caption=f"Otsu saturation ({stages['saturation_threshold']:.0f})")
        s4.image(stages["fruit_mask"], caption="Refined mask")
        st.caption(f"Projected fruit area: {stages['fruit_area_pixels']:,} px²")

    st.divider()
    st.subheader("How did the model do on this run?")

    with st.form("feedback_form"):
        rating_label = st.radio(
            "Rating",
            ["👍 Good", "👎 Bad"],
            horizontal=True,
            label_visibility="collapsed",
        )
        comment = st.text_area(
            "Comments (optional)",
            placeholder="e.g. missed a fruit in the corner, ripeness looked wrong, blemish % seemed too high...",
        )
        save_submitted = st.form_submit_button("Save this run to history")

    if save_submitted:
        rating = "good" if rating_label == "👍 Good" else "bad"
        run_id = add_run(st.session_state.user_email, metrics["count"], metrics["avg_confidence"], elapsed_ms)
        add_feedback(run_id, st.session_state.user_email, rating, comment)
        st.success("Run and feedback saved. Check Analysis History or Dashboard for it.")

    st.caption("See **Analysis History** in the sidebar for past runs.")

else:
    st.info("Upload an image to get started.")