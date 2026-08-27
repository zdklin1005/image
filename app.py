"""
Fruit Quality Assessment System - Router
Run with: streamlit run app.py
"""

import streamlit as st

from firebase_auth import require_login, logout_button
from db import init_db, get_role

st.set_page_config(page_title="Fruit Quality Assessment System", layout="wide")

init_db()
require_login()

role = get_role(st.session_state.user_email)
st.session_state.role = role

shared_pages = [
    st.Page("app_pages/dashboard.py", title="Dashboard", icon="📊"),
    st.Page("app_pages/new_fruit_analysis.py", title="New Fruit Analysis", icon="🍎"),
    st.Page("app_pages/analysis_history.py", title="Analysis History", icon="🕘"),
    st.Page("app_pages/settings.py", title="Settings", icon="⚙️"),
    st.Page("app_pages/help_about.py", title="Help / About", icon="❓"),
]

admin_only_pages = [
    st.Page("app_pages/model_evaluation.py", title="Model Evaluation", icon="🧪"),
    st.Page("app_pages/reports.py", title="Reports", icon="📄"),
]

pages = shared_pages + admin_only_pages if role == "admin" else shared_pages

nav = st.navigation(pages, position="top")

st.sidebar.write(f"Logged in as **{st.session_state.user_email}** ({role})")
logout_button()

nav.run()