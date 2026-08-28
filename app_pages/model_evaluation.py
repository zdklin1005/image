"""
Model Evaluation - admin-only. Shows how the model is doing based on
real user feedback (thumbs up/down + comments) submitted after each
analysis, since no labeled ground-truth test set is available.
"""

import streamlit as st
import pandas as pd

from db import get_feedback_history

st.title("Model Evaluation")
st.caption("Based on user-submitted feedback after each analysis run — not a labeled test set.")

feedback = get_feedback_history()

if feedback.empty:
    st.info("No feedback submitted yet. It'll show up here once users start rating their analyses.")
    st.stop()

feedback["timestamp"] = pd.to_datetime(feedback["timestamp"])

# ---------------------------------------------------------
# Overall satisfaction
# ---------------------------------------------------------

total = len(feedback)
good_count = (feedback["rating"] == "good").sum()
bad_count = (feedback["rating"] == "bad").sum()
good_pct = (good_count / total * 100) if total else 0.0

st.subheader("Overall satisfaction")

m1, m2, m3 = st.columns(3)
m1.metric("Total feedback", total)
m2.metric("Rated good", f"{good_count} ({good_pct:.0f}%)")
m3.metric("Rated bad", f"{bad_count} ({100 - good_pct:.0f}%)")

st.progress(good_pct / 100, text=f"{good_pct:.0f}% positive")

# ---------------------------------------------------------
# Trend over time
# ---------------------------------------------------------

st.divider()
st.subheader("Satisfaction over time")

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
else:
    st.caption("Not enough days of data yet for a trend line.")

# ---------------------------------------------------------
# Complaints
# ---------------------------------------------------------

st.divider()
st.subheader("Complaints")

complaints = feedback[(feedback["rating"] == "bad") & (feedback["comment"].str.strip() != "")]

if complaints.empty:
    st.info("No written complaints on bad-rated runs yet.")
else:
    st.caption(f"{len(complaints)} bad-rated run(s) with comments")
    st.dataframe(
        complaints[["timestamp", "created_by", "comment", "fruit_count", "avg_confidence"]],
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------
# All feedback
# ---------------------------------------------------------

st.divider()
with st.expander("All feedback"):
    st.dataframe(
        feedback[["timestamp", "created_by", "rating", "comment", "fruit_count", "avg_confidence"]],
        use_container_width=True,
        hide_index=True,
    )