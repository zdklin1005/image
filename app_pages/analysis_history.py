"""
Analysis History - past runs. Clients see their own; admins see everyone's.
"""

import streamlit as st
from db import get_run_history

st.title("Analysis History")

role = st.session_state.get("role", "client")

if role == "admin":
    history = get_run_history()
    st.caption("Showing analyses from all users.")
else:
    history = get_run_history(created_by=st.session_state.user_email)
    st.caption("Showing your analyses.")

if history.empty:
    st.info("No analyses yet.")
else:
    cols = ["id", "timestamp", "fruit_count", "avg_confidence", "processing_time_ms"]
    if role == "admin":
        cols.insert(1, "created_by")
    st.dataframe(history[cols], use_container_width=True, hide_index=True)