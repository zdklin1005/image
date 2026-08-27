"""
Manage Users - admin-only page. Create/delete Firebase accounts and
assign roles (roles are stored locally in db.py, not in Firebase).
"""

import streamlit as st

from firebase_auth import list_firebase_users, create_user, delete_user_account
from db import get_role, set_role

st.title("Manage Users")

st.subheader("All users")

users = list_firebase_users()
for u in users:
    u["role"] = get_role(u["email"])

if not users:
    st.info("No users found.")
else:
    for u in users:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        col1.write(u["email"])
        col2.write(f"Role: **{u['role']}**")

        new_role = col3.selectbox(
            "Change role",
            ["client", "admin"],
            index=["client", "admin"].index(u["role"]),
            key=f"role_{u['uid']}",
            label_visibility="collapsed",
        )
        if new_role != u["role"]:
            if col3.button("Save", key=f"save_{u['uid']}"):
                set_role(u["email"], new_role)
                st.success(f"Updated {u['email']} to {new_role}.")
                st.rerun()

        if u["email"] != st.session_state.user_email:
            if col4.button("Delete", key=f"delete_{u['uid']}"):
                success, message = delete_user_account(u["uid"])
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            col4.caption("(you)")

st.divider()
st.subheader("Create a new user")

with st.form("create_user_form"):
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")
    new_user_role = st.selectbox("Role", ["client", "admin"])
    submitted = st.form_submit_button("Create user")

if submitted:
    if not new_email or not new_password:
        st.error("Email and password are required.")
    elif len(new_password) < 6:
        st.error("Password must be at least 6 characters.")
    else:
        success, message = create_user(new_email, new_password)
        if success:
            set_role(new_email, new_user_role)
            st.success(f"{message} Role set to {new_user_role}.")
            st.rerun()
        else:
            st.error(message)
