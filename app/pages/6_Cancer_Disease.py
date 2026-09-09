from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Cancer Disease Screening",
    page_icon="🎗️",
    layout="wide",
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "cancer_disease_model.joblib"


# ============================================================
# Model loader
# ============================================================

@st.cache_resource
def load_model_artifact(model_path: Path):
    """Load and validate the saved cancer model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    artifact = joblib.load(model_path)

    required_keys = {
        "model",
        "threshold",
        "features",
    }

    missing_keys = required_keys.difference(
        artifact
    )

    if missing_keys:
        raise ValueError(
            "The saved model artifact is missing required fields: "
            f"{sorted(missing_keys)}"
        )

    return artifact


try:
    model_artifact = load_model_artifact(
        MODEL_PATH
    )
except Exception as error:
    st.error(
        f"Unable to load the cancer model: {error}"
    )
    st.stop()


model = model_artifact["model"]
decision_threshold = float(
    model_artifact["threshold"]
)
feature_order = list(
    model_artifact["features"]
)


# ============================================================
# Feature metadata
# ============================================================

FEATURE_METADATA = {
    # Mean measurements
    "radius_mean": ("Radius Mean", 14.0, 0.1),
    "texture_mean": ("Texture Mean", 19.0, 0.1),
    "perimeter_mean": ("Perimeter Mean", 92.0, 0.5),
    "area_mean": ("Area Mean", 655.0, 1.0),
    "smoothness_mean": ("Smoothness Mean", 0.096, 0.001),
    "compactness_mean": ("Compactness Mean", 0.104, 0.001),
    "concavity_mean": ("Concavity Mean", 0.089, 0.001),
    "concave_points_mean": ("Concave Points Mean", 0.049, 0.001),
    "symmetry_mean": ("Symmetry Mean", 0.181, 0.001),
    "fractal_dimension_mean": ("Fractal Dimension Mean", 0.063, 0.001),

    # Standard-error measurements
    "radius_se": ("Radius SE", 0.405, 0.01),
    "texture_se": ("Texture SE", 1.22, 0.01),
    "perimeter_se": ("Perimeter SE", 2.87, 0.05),
    "area_se": ("Area SE", 40.3, 0.5),
    "smoothness_se": ("Smoothness SE", 0.007, 0.0001),
    "compactness_se": ("Compactness SE", 0.025, 0.001),
    "concavity_se": ("Concavity SE", 0.032, 0.001),
    "concave_points_se": ("Concave Points SE", 0.012, 0.001),
    "symmetry_se": ("Symmetry SE", 0.021, 0.001),
    "fractal_dimension_se": ("Fractal Dimension SE", 0.004, 0.0001),

    # Worst / largest measurements
    "radius_worst": ("Radius Worst", 16.3, 0.1),
    "texture_worst": ("Texture Worst", 25.7, 0.1),
    "perimeter_worst": ("Perimeter Worst", 107.0, 0.5),
    "area_worst": ("Area Worst", 880.0, 1.0),
    "smoothness_worst": ("Smoothness Worst", 0.132, 0.001),
    "compactness_worst": ("Compactness Worst", 0.254, 0.001),
    "concavity_worst": ("Concavity Worst", 0.272, 0.001),
    "concave_points_worst": ("Concave Points Worst", 0.115, 0.001),
    "symmetry_worst": ("Symmetry Worst", 0.290, 0.001),
    "fractal_dimension_worst": ("Fractal Dimension Worst", 0.084, 0.001),
}


MEAN_FEATURES = [
    feature
    for feature in feature_order
    if feature.endswith("_mean")
]

SE_FEATURES = [
    feature
    for feature in feature_order
    if feature.endswith("_se")
]

WORST_FEATURES = [
    feature
    for feature in feature_order
    if feature.endswith("_worst")
]


def render_feature_inputs(
    features: list[str],
    prefix: str,
):
    """
    Render numerical inputs in two columns and return
    the entered values as a dictionary.
    """
    values = {}

    left_column, right_column = st.columns(2)

    for index, feature in enumerate(features):
        label, default_value, step = FEATURE_METADATA.get(
            feature,
            (
                feature.replace("_", " ").title(),
                0.0,
                0.01,
            ),
        )

        target_column = (
            left_column
            if index % 2 == 0
            else right_column
        )

        with target_column:
            values[feature] = st.number_input(
                label,
                min_value=0.0,
                value=float(default_value),
                step=float(step),
                format="%.6f",
                key=f"{prefix}_{feature}",
            )

    return values


# ============================================================
# Header
# ============================================================

st.title(
    "🎗️ Breast Cancer Diagnostic Screening"
)

st.write(
    "Enter the 30 diagnostic measurements used by the "
    "Breast Cancer Wisconsin (Diagnostic) dataset. "
    "The model estimates whether the measurement pattern is "
    "more consistent with a benign or malignant class."
)

st.warning(
    "Educational use only. These measurements are obtained from "
    "clinical analysis of a breast mass. This application is not "
    "a medical diagnosis and must not replace professional clinical evaluation."
)

with st.expander(
    "What do these measurements represent?"
):
    st.markdown(
        """
The dataset summarizes characteristics of cell nuclei from a breast mass.

The same ten characteristics are measured in three forms:

- **Mean** measurements
- **Standard Error (SE)** measurements
- **Worst / largest** measurements

The characteristics include radius, texture, perimeter, area,
smoothness, compactness, concavity, concave points, symmetry,
and fractal dimension.
"""
    )


# ============================================================
# Input form
# ============================================================

with st.form(
    "cancer_screening_form"
):

    st.subheader(
        "Diagnostic Measurements"
    )

    mean_tab, se_tab, worst_tab = st.tabs(
        [
            "Mean Measurements",
            "Standard Error Measurements",
            "Worst Measurements",
        ]
    )

    with mean_tab:
        st.caption(
            "Enter the ten mean measurement values."
        )
        mean_values = render_feature_inputs(
            MEAN_FEATURES,
            "mean",
        )

    with se_tab:
        st.caption(
            "Enter the ten standard-error measurement values."
        )
        se_values = render_feature_inputs(
            SE_FEATURES,
            "se",
        )

    with worst_tab:
        st.caption(
            "Enter the ten worst/largest measurement values."
        )
        worst_values = render_feature_inputs(
            WORST_FEATURES,
            "worst",
        )

    submitted = st.form_submit_button(
        "Generate Screening Result",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# Prediction
# ============================================================

if submitted:

    submitted_values = {
        **mean_values,
        **se_values,
        **worst_values,
    }

    missing_features = [
        feature
        for feature in feature_order
        if feature not in submitted_values
    ]

    if missing_features:
        st.error(
            "The application is missing model inputs: "
            f"{missing_features}"
        )
        st.stop()

    input_data = pd.DataFrame(
        [
            [
                float(
                    submitted_values[feature]
                )
                for feature in feature_order
            ]
        ],
        columns=feature_order,
    )

    malignant_score = float(
        model.predict_proba(
            input_data
        )[0, 1]
    )

    predicted_malignant = (
        malignant_score
        >= decision_threshold
    )

    st.divider()
    st.header(
        "Screening Result"
    )

    result_column, threshold_column = (
        st.columns(2)
    )

    result_column.metric(
        "Model malignant-class score",
        f"{malignant_score:.1%}",
    )

    threshold_column.metric(
        "Decision threshold",
        f"{decision_threshold:.1%}",
    )

    st.progress(
        float(
            np.clip(
                malignant_score,
                0.0,
                1.0,
            )
        )
    )

    if predicted_malignant:
        st.error(
            "The entered measurement pattern was classified "
            "as more consistent with the malignant class."
        )

        st.info(
            "This result requires professional medical interpretation. "
            "Diagnosis of breast cancer involves appropriate pathology, "
            "imaging, clinical examination, and other medical evidence."
        )

    else:
        st.success(
            "The entered measurement pattern was classified "
            "as more consistent with the benign class."
        )

        st.info(
            "A benign model classification does not rule out cancer. "
            "Clinical decisions must be made by qualified healthcare professionals."
        )

    st.caption(
        "The displayed score is a machine-learning model score, "
        "not a calibrated clinical probability."
    )

    with st.expander(
        "View submitted model inputs"
    ):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )
