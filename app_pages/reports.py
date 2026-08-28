"""
Reports - admin-only. Filter analysis history and export as CSV or PDF.
"""

import datetime

import streamlit as st
import pandas as pd
from fpdf import FPDF

from db import get_run_history

st.title("Reports")

history = get_run_history()

if history.empty:
    st.info("No analyses recorded yet.")
    st.stop()

history["timestamp"] = pd.to_datetime(history["timestamp"])

# ---------------------------------------------------------
# Filters
# ---------------------------------------------------------

st.subheader("Filter")

users = sorted(history["created_by"].dropna().unique().tolist())
selected_user = st.selectbox("User", ["All users"] + users)

date_min = history["timestamp"].min().date()
date_max = history["timestamp"].max().date()
date_range = st.date_input(
    "Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)

filtered = history.copy()
if selected_user != "All users":
    filtered = filtered[filtered["created_by"] == selected_user]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["timestamp"].dt.date >= start_date)
        & (filtered["timestamp"].dt.date <= end_date)
    ]

# ---------------------------------------------------------
# Summary + export
# ---------------------------------------------------------

st.divider()
st.subheader("Summary")

if filtered.empty:
    st.warning("No analyses match this filter.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Total analyses", len(filtered))
m2.metric("Avg. fruits per analysis", f"{filtered['fruit_count'].mean():.1f}")
m3.metric("Avg. confidence", f"{filtered['avg_confidence'].mean()*100:.1f}%")

st.dataframe(filtered, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Export")

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv_bytes,
    file_name=f"fruit_analysis_report_{datetime.date.today()}.csv",
    mime="text/csv",
)


def build_pdf(df: pd.DataFrame) -> bytes:
    date_label = (
        f"{date_range[0]} to {date_range[1]}"
        if isinstance(date_range, tuple) and len(date_range) == 2
        else "N/A"
    )

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Fruit Quality Assessment - Analysis Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.date.today()}", ln=True)
    pdf.cell(0, 8, f"User filter: {selected_user}", ln=True)
    pdf.cell(0, 8, f"Date range: {date_label}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Total analyses: {len(df)}", ln=True)
    pdf.cell(0, 7, f"Avg. fruits per analysis: {df['fruit_count'].mean():.1f}", ln=True)
    pdf.cell(0, 7, f"Avg. confidence: {df['avg_confidence'].mean()*100:.1f}%", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recent analyses (up to 50 rows)", ln=True)

    col_widths = [10, 35, 45, 25, 30, 30]
    headers = ["ID", "Timestamp", "User", "Fruits", "Confidence", "Time (ms)"]

    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, h, border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for _, row in df.head(50).iterrows():
        pdf.cell(col_widths[0], 7, str(row["id"]), border=1)
        pdf.cell(col_widths[1], 7, str(row["timestamp"])[:16], border=1)
        pdf.cell(col_widths[2], 7, str(row.get("created_by", ""))[:22], border=1)
        pdf.cell(col_widths[3], 7, str(row["fruit_count"]), border=1)
        pdf.cell(col_widths[4], 7, f"{row['avg_confidence']*100:.1f}%", border=1)
        pdf.cell(col_widths[5], 7, f"{row['processing_time_ms']:.0f}", border=1)
        pdf.ln()

    return bytes(pdf.output())


pdf_bytes = build_pdf(filtered)
st.download_button(
    "Download PDF summary",
    data=pdf_bytes,
    file_name=f"fruit_analysis_report_{datetime.date.today()}.pdf",
    mime="application/pdf",
)
