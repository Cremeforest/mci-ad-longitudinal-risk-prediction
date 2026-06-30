"""
Polished Streamlit demo for the MCI-to-AD research model.

Research use only.
Not clinically validated.
"""

from __future__ import annotations

import streamlit as st

from src.mci_ad.inference import predict_risk_scores


st.set_page_config(
    page_title="MCI-to-AD Risk Research Demo",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1080px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .risk-card {
        padding: 1.1rem 1.3rem;
        border: 1px solid #e6e6e6;
        border-radius: 14px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    .small-muted {
        color: #666;
        font-size: 0.92rem;
    }
    div.stButton > button:first-child {
        background-color: #c81e3a;
        color: white;
        border-radius: 10px;
        border: 0;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        background-color: #a9152d;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MCI-to-AD Conversion Risk Research Demo")

st.warning(
    "Research demonstration only. This model is not clinically validated and must not be used "
    "for diagnosis, treatment, triage, prognosis communication, or patient-level medical decisions."
)

with st.expander("What this demo uses", expanded=False):
    st.markdown(
        """
        The default public demo model uses five routine clinical variables:

        - age at current visit
        - sex
        - years of education
        - MMSE
        - FAQ total score

        It does **not** use ADAS13, CDR, CDR-SB, APOE, PET, CSF, MRI,
        diagnosis labels, or future outcome information.
        """
    )

st.header("Patient-like input values")
st.caption("Enter values for a single current visit. These are used only for a research demonstration.")

left, right = st.columns(2, gap="large")

with left:
    age = st.number_input(
        "Age at current visit (years)",
        min_value=40.0,
        max_value=100.0,
        value=72.0,
        step=1.0,
    )
    sex = st.selectbox("Sex", ["Female", "Male"], index=0)
    education = st.number_input(
        "Education (years)",
        min_value=0.0,
        max_value=30.0,
        value=16.0,
        step=1.0,
    )

with right:
    mmse = st.number_input(
        "MMSE",
        min_value=0.0,
        max_value=30.0,
        value=26.0,
        step=1.0,
    )
    faq = st.number_input(
        "FAQ total",
        min_value=0.0,
        max_value=30.0,
        value=6.0,
        step=1.0,
    )

submitted = st.button("Estimate research-demo risk")

if submitted:
    sex_male = 1 if sex == "Male" else 0

    try:
        scores = predict_risk_scores(
            age_at_visit=age,
            sex_male=sex_male,
            pteducat=education,
            mmse=mmse,
            faqtotal=faq,
        )

        st.divider()
        st.header("Exploratory model scores")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1 year", f"{scores['1y']:.1f}%")
        c2.metric("2 years", f"{scores['2y']:.1f}%")
        c3.metric("3 years", f"{scores['3y']:.1f}%")
        c4.metric("5 years", f"{scores['5y']:.1f}%")

        st.caption(
            "Displayed values are exploratory retrospective model scores, not clinically calibrated probabilities."
        )

    except Exception as exc:
        st.error("Inference failed.")
        st.exception(exc)

st.divider()

st.header("Limitations")
st.markdown(
    """
    - This is an **ADNI-trained retrospective research demonstration**.
    - NACC evaluation remains a **preliminary dry-run / transportability analysis**, not final external validation.
    - The original project is longitudinal; this page uses a **single current visit** as a simplified public-demo input.
    - Risk estimates may be miscalibrated, especially outside ADNI-like data distributions.
    - This demo must not be used for diagnosis, treatment, triage, prognosis communication, or medical decision-making.
    """
)
