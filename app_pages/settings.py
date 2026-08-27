"""
Settings - Change your Firebase account password.
"""

import streamlit as st

from firebase_auth import change_password

st.title("Settings")

st.subheader("Change password")
with st.form("change_password_form"):
    current = st.text_input("Current password", type="password")
    new = st.text_input("New password", type="password")
    confirm = st.text_input("Confirm new password", type="password")
    submitted = st.form_submit_button("Update password")

if submitted:
    if new != confirm:
        st.error("New passwords don't match.")
    else:
        success, message = change_password(current, new)
        if success:
            st.success(message)
        else:
            st.error(message)