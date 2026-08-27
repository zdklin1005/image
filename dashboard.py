"""
Dashboard - Accuracy and performance history across saved runs.
"""

import streamlit as st
import matplotlib.pyplot as plt

from auth import require_login, logout_button
from db import get_run_history

st.set_page_config(page_title="Dashboard", layout="wide")

require_login()

st.sidebar.write(f"Logged in as **{st.session_state.username}**")
logout_button()

st.title("Dashboard")
st.caption("Accuracy and performance history across all saved runs")

history = get_run_history()

if history.empty:
    st.info("No runs saved yet — process an image on the Home page and click **Save this run to history**.")
else:
    st.subheader("Accuracy over time")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history["id"], history["accuracy"], marker="o")
    ax.set_xlabel("Run")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Precision & recall over time")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(history["id"], history["precision_score"], marker="o", label="Precision")
    ax2.plot(history["id"], history["recall"], marker="o", label="Recall")
    ax2.set_xlabel("Run")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    st.subheader("All runs")
    st.dataframe(history, use_container_width=True)