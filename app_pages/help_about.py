"""
Help / About - static reference page.
"""

import streamlit as st

st.title("Help / About")

tab_about, tab_howto, tab_roles, tab_faq, tab_contact = st.tabs(
    ["About", "How to Use", "Roles & Permissions", "FAQ", "Contact"]
)

with tab_about:
    st.subheader("About this system")
    st.markdown(
        """
        The **Fruit Quality Assessment System** analyzes photos of fruit to:

        - **Detect** fruits in an image and identify their type
        - **Classify ripeness** (Overripe / Ripe / Rotten / Unripe)
        - **Detect blemishes and defects** on the fruit's surface

        Detection results come from an ensemble of multiple models working
        together and cross-checking each other, rather than a single model,
        to make results more reliable.

        *Built by: [TEAM NAME HERE]*
        *Version: [VERSION NUMBER HERE]*
        """
    )

with tab_howto:
    st.subheader("How to run an analysis")
    st.markdown(
        """
        1. Go to **New Fruit Analysis** in the sidebar.
        2. Upload a photo of the fruit (PNG or JPG).
        3. Optionally adjust the detection confidence threshold in the
           sidebar controls — lower values catch more fruits but may
           include false positives; higher values are stricter.
        4. Review the detected fruits, their ripeness, and any blemish
           percentage shown.
        5. Click **Save this run to history** to record the results.
        6. View past results anytime under **Analysis History**, or see
           trends over time on the **Dashboard**.
        """
    )
    st.info("If a photo looks blurry, the app will warn you before showing results — a clearer photo usually gives more accurate detections.")

with tab_roles:
    st.caption("Ask an admin if you need your role changed.")

with tab_faq:
    st.subheader("Frequently asked questions")

    with st.expander("Why didn't it detect my fruit?"):
        st.write(
            "Try lowering the confidence threshold in the sidebar, or use a "
            "clearer, well-lit photo with the fruit clearly visible against "
            "the background."
        )

    with st.expander("Why does it show more than one ripeness guess?"):
        st.write(
            "Ripeness is calculated by combining multiple models' opinions "
            "into a single result, so you may sometimes see close scores "
            "between two ripeness categories rather than one very confident "
            "answer."
        )

    with st.expander("What does the blemish percentage mean?"):
        st.write(
            "It's the estimated percentage of the fruit's visible surface "
            "that shows discoloration, bruising, or other defects, based on "
            "color analysis of the segmented fruit region."
        )

    with st.expander("Can I export my results?"):
        st.write(
            "Admins can export analysis history as CSV or a PDF summary "
            "from the Reports page."
        )

with tab_contact:
    st.subheader("Contact / Support")
    st.markdown(
        """
        Questions, bugs, or feedback? Reach out:

        - Email: [SUPPORT EMAIL HERE]
        - Team: [TEAM/PROJECT CONTACT HERE]
        """
    )