"""
auth.py - Simple single-user login gate for the app.
Call require_login() at the top of every page.
"""

import streamlit as st
from db import verify_user, init_db


def require_login():
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if st.session_state.logged_in:
        return

    st.title("Login")
    st.caption("Default login: admin / changeme — change this from Settings after logging in.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()  # prevents the rest of the page from rendering until logged in


def logout_button():
    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()