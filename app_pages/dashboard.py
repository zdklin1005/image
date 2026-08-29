"""
Dashboard - business landing page after login.
"""

import json
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from db import get_run_history

st.title("Dashboard")

role = st.session_state.get("role", "client")

if role == "admin":
    history = get_run_history()
    st.caption("Showing analyses from all users.")
else:
    history = get_run_history(created_by=st.session_state.user_email)
    st.caption("Showing your analyses.")

if history.empty:
    st.info("No analyses saved yet — head to **New Fruit Analysis** in the navbar to run your first one.")
else:
    history = history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Total analyses performed", len(history))
    m2.metric("Total fruits detected", int(history["fruit_count"].sum()))
    m3.metric("Average model confidence", f"{history['avg_confidence'].mean() * 100:.1f}%")

    m4, m5, m6 = st.columns(3)
    m4.metric("Average processing time", f"{history['processing_time_ms'].mean():.0f} ms")
    m5.metric("Blurry / rejected images", int(history["is_blurry"].sum()))
    m6.metric("Calibrated analyses", int(history["calibrated"].sum()))

    st.divider()
    st.subheader("Detection Distribution")

    all_detections = []
    for raw in history["detections_json"].dropna():
        try:
            all_detections.extend(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            pass

    if all_detections:
        det_df = pd.DataFrame(all_detections)
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            det_df["fruit_type"].value_counts().plot(kind="bar", ax=ax1, color="#4CAF50")
            ax1.set_ylabel("Count")
            ax1.set_xlabel("")
            ax1.tick_params(axis="x", rotation=45)
            fig1.tight_layout()
            st.pyplot(fig1)
            st.caption("Fruit-type distribution")

        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            det_df["ripeness"].value_counts().plot(kind="bar", ax=ax2, color="#FF9800")
            ax2.set_ylabel("Count")
            ax2.set_xlabel("")
            ax2.tick_params(axis="x", rotation=45)
            fig2.tight_layout()
            st.pyplot(fig2)
            st.caption("Ripeness distribution")
    else:
        st.info("No fruit detections recorded yet — older saved runs predate this tracking.")
        
    st.divider()
    st.subheader("Recent analyses")

    cols = ["id", "timestamp", "fruit_count", "avg_confidence", "processing_time_ms", "is_blurry", "calibrated"]
    if role == "admin":
        cols.insert(1, "created_by")
    st.dataframe(history[cols].head(10), use_container_width=True, hide_index=True)