"""
New Fruit Analysis - Upload an image and run fruit detection + ripeness classification.
"""

import io
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

CONFIDENCE_THRESHOLD = 0.40
LOW_CONFIDENCE_WARNING = 0.50


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


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


st.title("New Fruit Analysis")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    original = Image.open(uploaded).convert("RGB")

    suffix = pathlib.Path(uploaded.name).suffix or ".jpg"
    stages = run_stages_1_2(uploaded.getvalue(), suffix)

    if stages["is_blurry"]:
        st.warning(f"Image may be too blurry (score: {stages['blur_score']:.1f}). Results may be unreliable.")

    detect_input = Image.fromarray(adapter.to_rgb(stages["working_image"]))

    start = time.time()
    annotated, detections = process_image(detect_input, CONFIDENCE_THRESHOLD)
    elapsed_ms = (time.time() - start) * 1000
    metrics = get_metrics(detections)

    if not detections:
        st.warning("No fruit detected. Try a clearer image or check the fruit is one of the supported types.")

    # -------------------------------------------------------------
    # Summary cards
    # -------------------------------------------------------------
    st.divider()
    st.subheader("Analysis Results")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Fruits detected", metrics["count"])
    s2.metric("Avg. confidence", f"{metrics['avg_confidence']*100:.1f}%")
    s3.metric("Processing time", f"{elapsed_ms:.0f} ms")
    s4.metric("Image quality", "Blurry" if stages["is_blurry"] else "OK")
    s5.metric("Calibration", "Off")

    # -------------------------------------------------------------
    # Annotated image
    # -------------------------------------------------------------
    st.image(annotated, caption="Detected fruits", use_container_width=True)

    # -------------------------------------------------------------
    # Per-fruit table (with real per-fruit area from bbox crop)
    # -------------------------------------------------------------
    if detections:
        rows = []
        fruit_details = []
        for i, d in enumerate(detections, start=1):
            detail = adapter.compute_fruit_detail(stages["working_image"], d["bounding_box"], d["fruit_type"])
            fruit_details.append(detail)
            rows.append({
                "Fruit ID": f"Fruit {i}",
                "Fruit type": d["fruit_type"],
                "Ripeness": d["ripeness"],
                "Confidence": f"{d['confidence']*100:.1f}%",
                "Bounding box": str(d["bounding_box"]),
                "Area (px²)": f"{detail['area_pixels']:,}" if detail["area_pixels"] else "N/A",
                "Projected area (cm²)": "N/A (calibration off)",
                "Blemish area (px²)": f"{detail['blemish_area_pixels']:,}" if detail["blemish_area_pixels"] is not None else "N/A",
                "Blemish %": f"{detail['blemish_percentage']:.2f}%" if detail["blemish_percentage"] is not None else "N/A",
                "Mask status": detail["mask_status"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        low_conf = [d for d in detections if d["confidence"] < LOW_CONFIDENCE_WARNING]
        if low_conf:
            st.info(f"{len(low_conf)} detection(s) below {LOW_CONFIDENCE_WARNING*100:.0f}% confidence — manual review recommended.")

        failed_masks = [r for r in rows if r["Mask status"] not in ("Valid",)]
        if failed_masks:
            st.warning(f"{len(failed_masks)} detection(s) had segmentation mask issues — area/blemish figures may be unavailable.")
    # -------------------------------------------------------------
    # Processing-stage tabs
    # -------------------------------------------------------------
    st.divider()
    st.subheader("Processing Stages")
    tab_names = ["Original", "Preprocessed", "Detection", "Grayscale Otsu", "Saturation Otsu", "Refined Mask", "Blemish Result"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.image(original, use_container_width=True)
    with tabs[1]:
        st.image(adapter.to_rgb(stages["working_image"]), use_container_width=True)
    with tabs[2]:
        st.image(annotated, use_container_width=True)
    with tabs[3]:
        st.image(stages["gray_mask"], caption=f"Threshold: {stages['gray_threshold']:.0f}", use_container_width=True)
    with tabs[4]:
        st.image(stages["saturation_mask"], caption=f"Threshold: {stages['saturation_threshold']:.0f}", use_container_width=True)
    with tabs[5]:
        st.image(stages["fruit_mask"], use_container_width=True)
    with tabs[6]:
        if detections and fruit_details:
            overlay_cols = st.columns(min(len(fruit_details), 4))
            for i, (d, detail) in enumerate(zip(detections, fruit_details)):
                col = overlay_cols[i % len(overlay_cols)]
                with col:
                    if detail["blemish_overlay"] is not None:
                        st.image(adapter.to_rgb(detail["blemish_overlay"]), caption=f"Fruit {i+1}: {detail['blemish_percentage']:.1f}% blemish", use_container_width=True)
                    else:
                        st.caption(f"Fruit {i+1}: blemish detection unavailable")
        else:
            st.info("No detections to show blemish results for.")

    st.caption("Calibration stage will appear here once safely enabled.")
    
    # -------------------------------------------------------------
    # Result actions
    # -------------------------------------------------------------
    st.divider()
    st.subheader("Actions")

    a1, a2, a3 = st.columns(3)

    with a1:
        if st.button("Save this analysis", use_container_width=True):
            detections_summary = [
                {"fruit_type": d["fruit_type"], "ripeness": d["ripeness"], "confidence": d["confidence"]}
                for d in detections
            ]
            add_run(
                created_by=st.session_state.user_email,
                fruit_count=metrics["count"],
                avg_confidence=metrics["avg_confidence"],
                processing_time_ms=elapsed_ms,
                is_blurry=stages["is_blurry"],
                calibrated=False,
                detections=detections_summary,
            )
            st.success("Saved. Check Dashboard or Analysis History.")

    with a2:
        st.download_button(
            "Download annotated image",
            data=pil_to_png_bytes(annotated),
            file_name="annotated_result.png",
            mime="image/png",
            use_container_width=True,
        )

    with a3:
        if detections:
            csv_data = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="detection_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Download CSV", disabled=True, use_container_width=True)

else:
    st.info("Upload an image to get started.")