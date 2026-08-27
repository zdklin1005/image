"""
firebase_auth.py - Firebase Authentication (identity only).

Roles/authorization are NOT handled here anymore -- they live in
db.py's user_roles table, which updates instantly without needing a
re-login. This module only answers "who is this person" via Firebase:

- Web API Key + REST API (accounts:signInWithPassword): the Admin SDK
  cannot verify a plaintext password -- only Firebase's REST endpoint
  can. This is how the actual login check happens.

- Service account + Admin SDK: used for privileged actions that don't
  involve checking a password -- changing a password once logged in,
  and creating/listing/deleting accounts from the Manage Users page.

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
    """
    Shows a login form and halts the script (st.stop()) until the user
    is authenticated. Does NOT render any sidebar/navigation -- call
    this before building your page list so nothing is visible pre-login.
    """
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

    st.stop()  # prevents anything else (including nav) from rendering pre-login


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

    if _sign_in_with_email_password(email, current_password) is None:
        return False, "Current password is incorrect."

    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    try:
        firebase_admin_auth.update_user(uid, password=new_password)
    except Exception as e:
        return False, f"Failed to update password: {e}"

    return True, "Password updated."


# ---------------------------------------------------------
# Admin-only account management (used by app_pages/manage_users.py)
# Identity only -- roles are read/written via db.get_role / db.set_role
# ---------------------------------------------------------

def list_firebase_users() -> list[dict]:
    """Returns [{uid, email}] for every Firebase account."""
    return [
        {"uid": u.uid, "email": u.email}
        for u in firebase_admin_auth.list_users().iterate_all()
    ]


def create_user(email: str, password: str) -> tuple[bool, str]:
    try:
        firebase_admin_auth.create_user(email=email, password=password)
    except Exception as e:
        return False, f"Failed to create user: {e}"
    return True, f"Created account for {email}."


def delete_user_account(uid: str) -> tuple[bool, str]:
    try:
        firebase_admin_auth.delete_user(uid)
    except Exception as e:
        return False, f"Failed to delete user: {e}"
    return True, "User deleted."