"""
Dashboard - business landing page after login.
"""

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
    st.info("No analyses saved yet — head to **New Fruit Analysis** in the sidebar to run your first one.")
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Total analyses", len(history))
    m2.metric("Avg. fruits per analysis", f"{history['fruit_count'].mean():.1f}")
    m3.metric("Avg. confidence", f"{history['avg_confidence'].mean()*100:.1f}%")

    st.divider()
    st.subheader("Trends")

    history = history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(history.set_index("timestamp")["fruit_count"])
        st.caption("Fruits detected per run")
    with c2:
        st.line_chart(history.set_index("timestamp")["avg_confidence"])
        st.caption("Average confidence per run")

    st.divider()
    st.subheader("Recent analyses")
    cols = ["id", "timestamp", "fruit_count", "avg_confidence", "processing_time_ms"]
    if role == "admin":
        cols.insert(1, "created_by")
    st.dataframe(history[cols].head(10), use_container_width=True, hide_index=True)