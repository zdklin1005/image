"""
firebase_auth.py - Firebase Authentication for the app.

Two Firebase pieces are used for different jobs:

- Web API Key + REST API (accounts:signInWithPassword): the Admin SDK
  cannot verify a plaintext password -- only Firebase's REST endpoint
  can. This is how the actual login check happens.

- Service account + Admin SDK: used for privileged actions that don't
  involve checking a password, like changing a password once the user
  is already authenticated.

Credentials are read from Streamlit's secrets file (.streamlit/secrets.toml),
which must never be committed to git. See secrets.toml.example for the
required format.
"""

import requests
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth as firebase_admin_auth

FIREBASE_WEB_API_KEY = st.secrets["firebase_web_api_key"]

# Initialize the Admin SDK once per app process
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase_service_account"]))
    firebase_admin.initialize_app(cred)


def _sign_in_with_email_password(email: str, password: str):
    """
    Calls Firebase's REST API to verify an email/password pair.
    Returns the response dict (containing idToken, localId/uid, email)
    on success, or None on failure.
    """
    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:"
        f"signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    )
    payload = {"email": email, "password": password, "returnSecureToken": True}

    try:
        response = requests.post(url, json=payload, timeout=10)
    except requests.RequestException:
        return None

    if response.status_code == 200:
        return response.json()
    return None


def require_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.user_uid = None

    if st.session_state.logged_in:
        return

    st.title("Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        result = _sign_in_with_email_password(email, password)
        if result is not None:
            st.session_state.logged_in = True
            st.session_state.user_email = result["email"]
            st.session_state.user_uid = result["localId"]
            st.rerun()
        else:
            st.error("Invalid email or password")

    st.stop()  # prevents the rest of the page from rendering until logged in


def logout_button():
    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.session_state.user_uid = None
        st.rerun()


def change_password(current_password: str, new_password: str) -> tuple[bool, str]:
    """
    Verifies the current password (by re-attempting sign-in), then
    updates to the new password via the Admin SDK.

    Returns (success, message).
    """
    email = st.session_state.get("user_email")
    uid = st.session_state.get("user_uid")

    if not email or not uid:
        return False, "You must be logged in to change your password."

    # Re-verify the current password before allowing a change
    if _sign_in_with_email_password(email, current_password) is None:
        return False, "Current password is incorrect."

    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    try:
        firebase_admin_auth.update_user(uid, password=new_password)
    except Exception as e:
        return False, f"Failed to update password: {e}"

    return True, "Password updated."