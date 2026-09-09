from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Liver Disease Screening",
    page_icon="🧬",
    layout="wide",
)


# Locate the project root when this file is stored in app/pages
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "liver_disease_model.joblib"


@st.cache_resource
def load_model_artifact(model_path: Path):
    """Load and validate the saved liver model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    artifact = joblib.load(model_path)
    required_keys = {"model", "threshold", "features"}
    missing_keys = required_keys.difference(artifact)

    if missing_keys:
        raise ValueError(
            f"The saved model is missing required fields: {sorted(missing_keys)}"
        )

    return artifact


try:
    model_artifact = load_model_artifact(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the liver model: {error}")
    st.stop()


model = model_artifact["model"]
decision_threshold = float(model_artifact["threshold"])
feature_order = list(model_artifact["features"])

st.title("🧬 Liver Disease Clinical Screening")
st.write(
    "Enter the demographic information and laboratory measurements below to "
    "generate a machine-learning screening result."
)

st.warning(
    "This educational model is not a medical diagnosis. The laboratory values "
    "must come from appropriate clinical tests, and the result must be interpreted "
    "by a qualified healthcare professional."
)

with st.form("liver_screening_form"):
    st.subheader("Patient Information")
    column_1, column_2 = st.columns(2)

    with column_1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45,
            step=1,
        )

    with column_2:
        gender_display = st.selectbox("Gender", ["Male", "Female"])
        gender = gender_display.lower()

    st.divider()
    st.subheader("Bilirubin and Liver Enzymes")
    column_1, column_2 = st.columns(2)

    with column_1:
        total_bilirubin = st.number_input(
            "Total bilirubin (mg/dL)",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.1,
            format="%.2f",
        )
        alkaline_phosphatase = st.number_input(
            "Alkaline phosphatase — ALP (IU/L)",
            min_value=1.0,
            max_value=3000.0,
            value=208.0,
            step=1.0,
        )
        aspartate_aminotransferase = st.number_input(
            "Aspartate aminotransferase — AST/SGOT (IU/L)",
            min_value=1.0,
            max_value=6000.0,
            value=42.0,
            step=1.0,
        )

    with column_2:
        direct_bilirubin = st.number_input(
            "Direct bilirubin (mg/dL)",
            min_value=0.0,
            max_value=30.0,
            value=0.3,
            step=0.1,
            format="%.2f",
        )
        alanine_aminotransferase = st.number_input(
            "Alanine aminotransferase — ALT/SGPT (IU/L)",
            min_value=1.0,
            max_value=3000.0,
            value=35.0,
            step=1.0,
        )

    st.divider()
    st.subheader("Protein Measurements")
    column_1, column_2 = st.columns(2)

    with column_1:
        total_proteins = st.number_input(
            "Total proteins (g/dL)",
            min_value=1.0,
            max_value=15.0,
            value=6.6,
            step=0.1,
            format="%.2f",
        )
        albumin_globulin_ratio = st.number_input(
            "Albumin/Globulin ratio",
            min_value=0.1,
            max_value=5.0,
            value=0.95,
            step=0.01,
            format="%.2f",
        )

    with column_2:
        albumin = st.number_input(
            "Albumin (g/dL)",
            min_value=0.1,
            max_value=10.0,
            value=3.1,
            step=0.1,
            format="%.2f",
        )

    submitted = st.form_submit_button(
        "Generate Screening Result",
        type="primary",
        use_container_width=True,
    )


if submitted:
    validation_errors = []

    if direct_bilirubin > total_bilirubin:
        validation_errors.append(
            "Direct bilirubin should not be greater than total bilirubin."
        )

    if albumin > total_proteins:
        validation_errors.append(
            "Albumin should not be greater than total proteins."
        )

    if validation_errors:
        for message in validation_errors:
            st.error(message)
        st.stop()

    submitted_values = {
        "age": float(age),
        "gender": gender,
        "total_bilirubin": float(total_bilirubin),
        "direct_bilirubin": float(direct_bilirubin),
        "alkaline_phosphatase": float(alkaline_phosphatase),
        "alanine_aminotransferase": float(alanine_aminotransferase),
        "aspartate_aminotransferase": float(aspartate_aminotransferase),
        "total_proteins": float(total_proteins),
        "albumin": float(albumin),
        "albumin_globulin_ratio": float(albumin_globulin_ratio),
    }

    missing_features = [
        feature for feature in feature_order if feature not in submitted_values
    ]
    if missing_features:
        st.error(f"Missing model inputs: {missing_features}")
        st.stop()

    input_data = pd.DataFrame(
        [[submitted_values[feature] for feature in feature_order]],
        columns=feature_order,
    )

    model_score = float(model.predict_proba(input_data)[0, 1])
    elevated_indicators = model_score >= decision_threshold

    st.divider()
    st.header("Screening Result")

    result_column, threshold_column = st.columns(2)
    result_column.metric("Model risk score", f"{model_score:.1%}")
    threshold_column.metric("Decision threshold", f"{decision_threshold:.1%}")

    st.progress(float(np.clip(model_score, 0.0, 1.0)))

    if elevated_indicators:
        st.error(
            "The entered profile was classified as having elevated indicators "
            "associated with liver disease."
        )
        st.info(
            "Discuss this result with a qualified healthcare professional. "
            "Interpretation may require additional liver-function tests, medical "
            "history, imaging, and clinical examination."
        )
    else:
        st.success(
            "The entered profile was not classified as having elevated indicators "
            "associated with liver disease."
        )
        st.info(
            "A lower model result does not rule out liver disease. Seek professional "
            "evaluation when symptoms or abnormal laboratory results are present."
        )

    st.caption(
        "The displayed score is a machine-learning model score, not a calibrated "
        "clinical probability."
    )

    with st.expander("View submitted model inputs"):
        st.dataframe(input_data, use_container_width=True, hide_index=True)
