from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Respiratory Disease Screening",
    page_icon="🫁",
    layout="wide",
)


# Locate the project root when this file is stored in app/pages
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "respiratory_disease_model.joblib"


@st.cache_resource
def load_model_artifact(model_path: Path):
    """Load and validate the saved respiratory model artifact."""
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
    st.error(f"Unable to load the respiratory model: {error}")
    st.stop()


model = model_artifact["model"]
decision_threshold = float(model_artifact["threshold"])
feature_order = list(model_artifact["features"])

st.title("🫁 Respiratory Disease Clinical Screening")
st.write(
    "Enter the patient information below to generate a machine-learning "
    "screening result for the Respiratory Disease / Asthma module."
)

st.warning(
    "This tool is for educational screening only. It is not a medical diagnosis, "
    "and its output must be interpreted by a qualified healthcare professional."
)

with st.form("respiratory_screening_form"):
    st.subheader("Patient Information")
    column_1, column_2 = st.columns(2)

    with column_1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=35,
            step=1,
        )

        smoking_status_display = st.selectbox(
            "Smoking Status",
            [
                "Non Smoker",
                "Current Smoker",
                "Ex Smoker",
            ],
        )

    with column_2:
        gender_display = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
            ],
        )

        peak_flow = st.number_input(
            "Peak Flow",
            min_value=1.0,
            max_value=1000.0,
            value=320.0,
            step=1.0,
            format="%.1f",
            help=(
                "Peak expiratory flow measurement. Use the same unit convention "
                "that was used in your source dataset."
            ),
        )

    submitted = st.form_submit_button(
        "Generate Screening Result",
        type="primary",
        use_container_width=True,
    )


if submitted:
    gender = gender_display.lower()

    smoking_status_mapping = {
        "Non Smoker": "non_smoker",
        "Current Smoker": "current_smoker",
        "Ex Smoker": "ex_smoker",
    }
    smoking_status = smoking_status_mapping[smoking_status_display]

    submitted_values = {
        "age": float(age),
        "gender": gender,
        "smoking_status": smoking_status,
        "peak_flow": float(peak_flow),
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
            "associated with respiratory disease / asthma."
        )
        st.info(
            "Discuss this result with a qualified healthcare professional. "
            "Proper interpretation may require medical history, symptom review, "
            "spirometry, peak-flow follow-up, and clinical examination."
        )
    else:
        st.success(
            "The entered profile was not classified as having elevated indicators "
            "associated with respiratory disease / asthma."
        )
        st.info(
            "A lower model result does not rule out respiratory disease. "
            "Seek professional evaluation if symptoms are present."
        )

    st.caption(
        "The displayed score is a machine-learning model score, not a calibrated "
        "clinical probability."
    )

    with st.expander("View submitted model inputs"):
        st.dataframe(input_data, use_container_width=True, hide_index=True)
