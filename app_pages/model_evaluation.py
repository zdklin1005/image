"""
Model Evaluation - admin-only.

Two independent sections:
1. User feedback (thumbs up/down + comments) submitted after each
   analysis -- always available, no labeled data required.
2. Objective 2: Segmentation IoU -- runs the team's
   evaluate_segmentation.py against labeled ground-truth masks, if
   available. Triggered on demand since it re-runs segmentation on
   every test image and can take a while.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

from db import get_feedback_history

st.title("Model Evaluation")

# ===========================================================
# SECTION 1: User feedback
# ===========================================================

st.subheader("User feedback")
st.caption("Based on user-submitted feedback after each analysis run.")

feedback = get_feedback_history()

if feedback.empty:
    st.info("No feedback submitted yet. It'll show up here once users start rating their analyses.")
else:
    feedback["timestamp"] = pd.to_datetime(feedback["timestamp"])

    total = len(feedback)
    good_count = (feedback["rating"] == "good").sum()
    bad_count = (feedback["rating"] == "bad").sum()
    good_pct = (good_count / total * 100) if total else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total feedback", total)
    m2.metric("Rated good", f"{good_count} ({good_pct:.0f}%)")
    m3.metric("Rated bad", f"{bad_count} ({100 - good_pct:.0f}%)")

    st.progress(good_pct / 100, text=f"{good_pct:.0f}% positive")

    daily = (
        feedback
        .assign(date=feedback["timestamp"].dt.date, is_good=(feedback["rating"] == "good").astype(int))
        .groupby("date")
        .agg(total=("is_good", "count"), good=("is_good", "sum"))
    )
    daily["good_pct"] = daily["good"] / daily["total"] * 100

    if len(daily) >= 2:
        st.line_chart(daily["good_pct"])
        st.caption("% rated good, per day")

    complaints = feedback[(feedback["rating"] == "bad") & (feedback["comment"].str.strip() != "")]
    if not complaints.empty:
        with st.expander(f"{len(complaints)} written complaint(s)"):
            st.dataframe(
                complaints[["timestamp", "created_by", "comment", "fruit_count", "avg_confidence"]],
                use_container_width=True,
                hide_index=True,
            )

# ===========================================================
# SECTION 2: Objective 2 - Segmentation IoU
# ===========================================================

st.divider()
st.subheader("Objective 2: Segmentation Accuracy (IoU)")
st.caption(
    "Compares the segmentation module's predicted fruit mask against "
    "hand-labeled ground-truth masks. Target: mean IoU ≥ 0.70 across "
    "at least 30 images."
)

EVAL_DIR = Path(__file__).resolve().parents[1] / "calibration_segmentation" / "segmentation_evaluation"

if st.button("Run Segmentation Evaluation"):
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))

    try:
        import evaluate_segmentation as seg_eval
    except Exception as e:
        st.error(f"Could not load evaluate_segmentation.py: {e}")
        st.session_state["seg_eval_results"] = None
        st.stop()

    try:
        image_paths = seg_eval.find_test_images()
    except FileNotFoundError as e:
        st.error(str(e))
        st.session_state["seg_eval_results"] = None
        st.stop()

    if not image_paths:
        st.warning("No test images found in segmentation_evaluation/test_images.")
        st.session_state["seg_eval_results"] = None
    else:
        results = []
        failures = []
        progress = st.progress(0, text="Running segmentation evaluation...")

        for i, path in enumerate(image_paths, start=1):
            try:
                results.append(seg_eval.evaluate_image(path))
            except Exception as e:
                failures.append({"image": path.name, "error": str(e)})
            progress.progress(i / len(image_paths), text=f"Evaluated {i}/{len(image_paths)}")

        progress.empty()

        if results:
            seg_eval.save_results_csv(results)

        st.session_state["seg_eval_results"] = results
        st.session_state["seg_eval_failures"] = failures

results = st.session_state.get("seg_eval_results")
failures = st.session_state.get("seg_eval_failures")

if results is not None:
    if failures:
        with st.expander(f"⚠️ {len(failures)} image(s) failed to evaluate"):
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
            st.caption(
                "The most common cause is a missing ground-truth mask for that "
                "image -- each test image needs a matching file (same name, "
                ".png) in the ground_truth folder."
            )

    if not results:
        st.warning("No images were successfully evaluated.")
    else:
        df = pd.DataFrame(results)
        mean_iou = df["iou"].mean()
        target_met = len(results) >= 30 and mean_iou >= 0.70

        c1, c2, c3 = st.columns(3)
        c1.metric("Images evaluated", len(results))
        c2.metric("Mean IoU", f"{mean_iou:.3f}")
        c3.metric("Objective 2 target", "Achieved" if target_met else "Not yet")

        if not target_met:
            reasons = []
            if len(results) < 30:
                reasons.append(f"needs at least 30 evaluated images (currently {len(results)})")
            if mean_iou < 0.70:
                reasons.append("mean IoU must reach 0.70")
            st.caption("Target not met: " + "; ".join(reasons))

        st.dataframe(
            df[["image", "iou", "predicted_pixels", "ground_truth_pixels"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Worst-performing samples")
        st.caption("Green = correctly segmented, red = over-segmented, blue = missed fruit area.")

        worst = df.nsmallest(min(3, len(df)), "iou")
        overlay_dir = EVAL_DIR / "overlays"
        cols = st.columns(len(worst))
        for col, (_, row) in zip(cols, worst.iterrows()):
            overlay_path = overlay_dir / f"{Path(row['image']).stem}_overlay.png"
            if overlay_path.exists():
                col.image(str(overlay_path), caption=f"{row['image']} (IoU {row['iou']:.2f})")
            else:
                col.caption(f"{row['image']}: overlay image not found")