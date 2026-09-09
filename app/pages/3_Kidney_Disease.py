from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Kidney Disease Screening",
    page_icon="🫘",
    layout="wide",
)


# Locate the project root when this file is stored in app/pages
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "kidney_disease_model.joblib"


@st.cache_resource
def load_model_artifact(model_path: Path):
    """Load and validate the saved kidney model artifact."""
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


def yes_no_input(label: str, key: str, default: str = "No") -> str:
    """Create a Yes/No selector and return the dataset-compatible value."""
    options = ["No", "Yes"] if default == "No" else ["Yes", "No"]
    selected = st.selectbox(label, options, key=key)
    return selected.lower()


def normal_abnormal_input(label: str, key: str) -> str:
    """Create a Normal/Abnormal selector."""
    selected = st.selectbox(label, ["Normal", "Abnormal"], key=key)
    return selected.lower()


try:
    model_artifact = load_model_artifact(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the kidney model: {error}")
    st.stop()


model = model_artifact["model"]
decision_threshold = float(model_artifact["threshold"])
feature_order = list(model_artifact["features"])

st.title("🫘 Chronic Kidney Disease Clinical Screening")
st.write(
    "Enter the clinical and laboratory measurements below to generate a "
    "machine-learning screening result."
)

st.warning(
    "This educational model is not a medical diagnosis. Several inputs require "
    "clinical laboratory testing, and the result must be interpreted by a "
    "qualified healthcare professional."
)

with st.form("kidney_screening_form"):
    st.subheader("Patient Information and Basic Measurements")
    column_1, column_2 = st.columns(2)

    with column_1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=54, step=1
        )
        bp = st.number_input(
            "Blood pressure (mm Hg)",
            min_value=40.0,
            max_value=250.0,
            value=80.0,
            step=1.0,
        )
        sg = st.selectbox(
            "Urine specific gravity",
            ["1.020", "1.025", "1.015", "1.010", "1.005"],
        )

    with column_2:
        al = st.selectbox(
            "Urine albumin level (0–5)", ["0", "1", "2", "3", "4", "5"]
        )
        su = st.selectbox(
            "Urine sugar level (0–5)", ["0", "1", "2", "3", "4", "5"]
        )
        appet_display = st.selectbox("Appetite", ["Good", "Poor"])
        appet = appet_display.lower()

    st.divider()
    st.subheader("Urine Microscopy")
    column_1, column_2 = st.columns(2)

    with column_1:
        rbc = normal_abnormal_input("Red blood cells", "rbc")
        pc = normal_abnormal_input("Pus cells", "pc")

    with column_2:
        pcc_display = st.selectbox(
            "Pus cell clumps", ["Not present", "Present"]
        )
        pcc = "notpresent" if pcc_display == "Not present" else "present"

        ba_display = st.selectbox("Bacteria", ["Not present", "Present"])
        ba = "notpresent" if ba_display == "Not present" else "present"

    st.divider()
    st.subheader("Blood Laboratory Measurements")
    column_1, column_2 = st.columns(2)

    with column_1:
        bgr = st.number_input(
            "Random blood glucose (mg/dL)",
            min_value=20.0,
            max_value=600.0,
            value=121.0,
            step=1.0,
        )
        bu = st.number_input(
            "Blood urea (mg/dL)",
            min_value=1.0,
            max_value=400.0,
            value=36.0,
            step=1.0,
        )
        sc = st.number_input(
            "Serum creatinine (mg/dL)",
            min_value=0.1,
            max_value=80.0,
            value=1.2,
            step=0.1,
        )
        sod = st.number_input(
            "Sodium (mEq/L)",
            min_value=80.0,
            max_value=180.0,
            value=138.0,
            step=1.0,
        )
        pot = st.number_input(
            "Potassium (mEq/L)",
            min_value=1.0,
            max_value=15.0,
            value=4.4,
            step=0.1,
        )
        hemo = st.number_input(
            "Hemoglobin (g/dL)",
            min_value=1.0,
            max_value=25.0,
            value=15.0,
            step=0.1,
        )

    with column_2:
        pcv = st.number_input(
            "Packed cell volume (%)",
            min_value=5.0,
            max_value=70.0,
            value=44.0,
            step=1.0,
        )
        wbcc = st.number_input(
            "White blood cell count (cells/cumm)",
            min_value=1000.0,
            max_value=30000.0,
            value=7800.0,
            step=100.0,
        )
        rbcc = st.number_input(
            "Red blood cell count (millions/cmm)",
            min_value=1.0,
            max_value=10.0,
            value=5.2,
            step=0.1,
        )
        htn = yes_no_input("Hypertension", "htn")
        dm = yes_no_input("Diabetes mellitus", "dm")
        cad = yes_no_input("Coronary artery disease", "cad")

    st.divider()
    st.subheader("Additional Clinical Indicators")
    column_1, column_2 = st.columns(2)

    with column_1:
        pe = yes_no_input("Pedal edema", "pe")

    with column_2:
        ane = yes_no_input("Anemia", "ane")

    submitted = st.form_submit_button(
        "Generate Screening Result",
        type="primary",
        use_container_width=True,
    )


if submitted:
    submitted_values = {
        "age": float(age),
        "bp": float(bp),
        "sg": sg,
        "al": al,
        "su": su,
        "rbc": rbc,
        "pc": pc,
        "pcc": pcc,
        "ba": ba,
        "bgr": float(bgr),
        "bu": float(bu),
        "sc": float(sc),
        "sod": float(sod),
        "pot": float(pot),
        "hemo": float(hemo),
        "pcv": float(pcv),
        "wbcc": float(wbcc),
        "rbcc": float(rbcc),
        "htn": htn,
        "dm": dm,
        "cad": cad,
        "appet": appet,
        "pe": pe,
        "ane": ane,
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
            "The entered clinical profile was classified as having elevated "
            "indicators associated with chronic kidney disease."
        )
        st.info(
            "This result should be reviewed by a qualified healthcare professional "
            "together with kidney-function tests and an appropriate clinical examination."
        )
    else:
        st.success(
            "The entered clinical profile was not classified as having elevated "
            "indicators associated with chronic kidney disease."
        )
        st.info(
            "A lower model result does not rule out kidney disease. Follow professional "
            "screening advice when symptoms or abnormal laboratory results are present."
        )

    st.caption(
        "The displayed score is a machine-learning model score, not a calibrated "
        "clinical probability."
    )

    with st.expander("View submitted model inputs"):
        st.dataframe(input_data, use_container_width=True, hide_index=True)
