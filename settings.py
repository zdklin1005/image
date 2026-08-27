"""
Settings - Change your login password.
"""

import streamlit as st

from auth import require_login, logout_button
from db import update_password, verify_user

st.set_page_config(page_title="Settings", layout="wide")

require_login()

st.sidebar.write(f"Logged in as **{st.session_state.username}**")
logout_button()

st.title("Settings")

st.subheader("Change password")
with st.form("change_password_form"):
    current = st.text_input("Current password", type="password")
    new = st.text_input("New password", type="password")
    confirm = st.text_input("Confirm new password", type="password")
    submitted = st.form_submit_button("Update password")

if submitted:
    if not verify_user(st.session_state.username, current):
        st.error("Current password is incorrect.")
    elif new != confirm:
        st.error("New passwords don't match.")
    elif len(new) < 6:
        st.error("New password should be at least 6 characters.")
    else:
        update_password(st.session_state.username, new)
        st.success("Password updated.")