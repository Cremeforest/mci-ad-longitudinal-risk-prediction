"""
Clean Streamlit demo for the MCI-to-AD risk research model.

This app calls src/mci_ad/inference.py.

Research use only.
Not clinically validated.
Not for diagnosis, treatment, triage, prognosis communication, or patient-level decisions.
"""

from __future__ import annotations

import streamlit as st

from src.mci_ad.inference import predict_risk_scores


st.set_page_config(
    page_title="MCI-to-AD Risk Research Demo",
    page_icon="🧠",
    layout="centered",
)

st.title("MCI-to-AD Conversion Risk Research Demo")

st.warning(
    "Research demonstration only. This model is not clinically validated and must not be used "
    "for diagnosis, treatment, triage, prognosis communication, or patient-level medical decisions."
)

st.markdown(
    """
This demo uses a compact routine-clinical-variable model to estimate exploratory
1-, 2-, 3-, and 5-year model scores for conversion from mild cognitive impairment
(MCI) to Alzheimer's disease dementia.

The default demo model uses only:

- age at visit
- sex
- years of education
- MMSE
- FAQ total score
"""
)

with st.form("input_form"):
    st.subheader("Input variables")

    age = st.number_input("Age at visit", min_value=40.0, max_value=100.0, value=72.0, step=1.0)
    sex = st.selectbox("Sex", ["Female", "Male"], index=0)
    education = st.number_input("Years of education", min_value=0.0, max_value=30.0, value=16.0, step=1.0)
    mmse = st.number_input("MMSE", min_value=0.0, max_value=30.0, value=26.0, step=1.0)
    faq = st.number_input("FAQ total score", min_value=0.0, max_value=30.0, value=6.0, step=1.0)

    submitted = st.form_submit_button("Run research demo inference")

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

        st.subheader("Exploratory model scores")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1 year", f"{scores['1y']:.1f}%")
        col2.metric("2 years", f"{scores['2y']:.1f}%")
        col3.metric("3 years", f"{scores['3y']:.1f}%")
        col4.metric("5 years", f"{scores['5y']:.1f}%")

        st.caption(
            "Displayed values are exploratory retrospective model scores, not clinically calibrated probabilities."
        )

    except Exception as exc:
        st.error("Inference failed.")
        st.exception(exc)

st.divider()

st.markdown(
    """
### Model boundary

The default demo does **not** use ADAS13, CDR, CDR-SB, APOE, PET, CSF, MRI,
diagnosis labels, or future outcome information.

The NACC analysis associated with this project is a preliminary transportability dry-run,
not final external validation.
"""
)
