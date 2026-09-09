from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Configure the Streamlit page
st.set_page_config(
    page_title="Heart Disease Screening",
    page_icon="❤️",
    layout="wide"
)


# Locate the project root and saved model
page_file = Path(__file__).resolve()
project_root = page_file.parents[2]

model_path = (
    project_root
    / "models"
    / "heart_disease_model.joblib"
)


@st.cache_resource
def load_model_package():
    """Load and cache the trained heart disease model package."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    return joblib.load(model_path)


# Define the exact category values used during training
sex_options = {
    "Female": "Female",
    "Male": "Male"
}

chest_pain_options = {
    "Typical angina": "typical angina",
    "Atypical angina": "atypical angina",
    "Non-anginal pain": "non-anginal",
    "Asymptomatic": "asymptomatic"
}

yes_no_unknown_options = {
    "No": "No",
    "Yes": "Yes",
    "Not available": np.nan
}

resting_ecg_options = {
    "Normal": "normal",
    "Left ventricular hypertrophy": "lv hypertrophy",
    "ST-T wave abnormality": "st-t abnormality",
    "Not available": np.nan
}

slope_options = {
    "Upsloping": "upsloping",
    "Flat": "flat",
    "Downsloping": "downsloping",
    "Not available": np.nan
}

major_vessel_options = {
    "0 vessels": 0.0,
    "1 vessel": 1.0,
    "2 vessels": 2.0,
    "3 vessels": 3.0,
    "Not available": np.nan
}

thalassemia_options = {
    "Normal": "normal",
    "Fixed defect": "fixed defect",
    "Reversible defect": "reversable defect",
    "Not available": np.nan
}


# Load the trained model package
try:
    model_package = load_model_package()

    heart_model = model_package[
        "model"
    ]

    decision_threshold = float(
        model_package[
            "threshold"
        ]
    )

    feature_names = model_package[
        "feature_names"
    ]

except Exception as error:
    st.error(
        f"Unable to load the heart disease model: {error}"
    )
    st.stop()


# Display the application header
st.title("Heart Disease Clinical Screening")

st.write(
    "Enter the clinical measurements below to generate "
    "a machine-learning screening result."
)

st.warning(
    "This educational model is not a medical diagnosis. "
    "Several inputs require clinical tests, and the output "
    "must be interpreted by a qualified healthcare professional."
)


# Create the clinical input form
with st.form(
    "heart_disease_screening_form"
):

    st.subheader(
        "Patient Information"
    )

    patient_column_1, patient_column_2 = (
        st.columns(2)
    )

    with patient_column_1:
        age = st.number_input(
            "Age (years)",
            min_value=28,
            max_value=77,
            value=54,
            step=1
        )

    with patient_column_2:
        sex_label = st.selectbox(
            "Sex",
            options=list(
                sex_options.keys()
            )
        )

    st.divider()
    st.subheader(
        "Symptoms and Basic Measurements"
    )

    measurement_column_1, measurement_column_2 = (
        st.columns(2)
    )

    with measurement_column_1:
        chest_pain_label = st.selectbox(
            "Chest pain type",
            options=list(
                chest_pain_options.keys()
            ),
            index=3
        )

        resting_blood_pressure = (
            st.number_input(
                "Resting blood pressure (mm Hg)",
                min_value=50.0,
                max_value=200.0,
                value=130.0,
                step=1.0
            )
        )

        cholesterol = st.number_input(
            "Serum cholesterol (mg/dL)",
            min_value=50.0,
            max_value=603.0,
            value=240.0,
            step=1.0
        )

        fasting_blood_sugar_label = (
            st.selectbox(
                "Fasting blood sugar above 120 mg/dL",
                options=list(
                    yes_no_unknown_options.keys()
                )
            )
        )

    with measurement_column_2:
        resting_ecg_label = st.selectbox(
            "Resting ECG result",
            options=list(
                resting_ecg_options.keys()
            )
        )

        maximum_heart_rate = st.number_input(
            "Maximum heart rate achieved",
            min_value=60.0,
            max_value=202.0,
            value=140.0,
            step=1.0
        )

        exercise_angina_label = st.selectbox(
            "Exercise-induced angina",
            options=list(
                yes_no_unknown_options.keys()
            )
        )

        oldpeak = st.number_input(
            "ST depression induced by exercise (oldpeak)",
            min_value=-2.6,
            max_value=6.2,
            value=1.0,
            step=0.1
        )

    st.divider()
    st.subheader(
        "Advanced Clinical Test Information"
    )

    st.info(
        "Select 'Not available' when an advanced test "
        "result is unavailable. The saved pipeline will "
        "handle the missing value automatically."
    )

    advanced_column_1, advanced_column_2 = (
        st.columns(2)
    )

    with advanced_column_1:
        slope_label = st.selectbox(
            "Slope of the peak exercise ST segment",
            options=list(
                slope_options.keys()
            ),
            index=3
        )

        major_vessels_label = st.selectbox(
            "Major vessels colored by fluoroscopy",
            options=list(
                major_vessel_options.keys()
            ),
            index=4
        )

    with advanced_column_2:
        thalassemia_label = st.selectbox(
            "Thalassemia test result",
            options=list(
                thalassemia_options.keys()
            ),
            index=3
        )

    submit_prediction = st.form_submit_button(
        "Analyze Heart Disease Risk",
        type="primary",
        use_container_width=True
    )


# Process the submitted clinical information
if submit_prediction:

    input_values = {
        "age": int(age),
        "trestbps": float(
            resting_blood_pressure
        ),
        "chol": float(
            cholesterol
        ),
        "thalch": float(
            maximum_heart_rate
        ),
        "oldpeak": float(
            oldpeak
        ),
        "sex": sex_options[
            sex_label
        ],
        "cp": chest_pain_options[
            chest_pain_label
        ],
        "fbs": yes_no_unknown_options[
            fasting_blood_sugar_label
        ],
        "restecg": resting_ecg_options[
            resting_ecg_label
        ],
        "exang": yes_no_unknown_options[
            exercise_angina_label
        ],
        "slope": slope_options[
            slope_label
        ],
        "ca": major_vessel_options[
            major_vessels_label
        ],
        "thal": thalassemia_options[
            thalassemia_label
        ]
    }

    # Preserve the exact feature order used during training
    input_dataframe = pd.DataFrame(
        [
            [
                input_values[feature]
                for feature in feature_names
            ]
        ],
        columns=feature_names
    )

    # Generate the model score and classification
    model_score = float(
        heart_model.predict_proba(
            input_dataframe
        )[0, 1]
    )

    predicted_class = int(
        model_score
        >= decision_threshold
    )

    st.divider()
    st.subheader(
        "Screening Result"
    )

    result_column_1, result_column_2 = (
        st.columns(2)
    )

    with result_column_1:
        st.metric(
            "Model risk score",
            f"{model_score * 100:.1f}%"
        )

    with result_column_2:
        st.metric(
            "Decision threshold",
            f"{decision_threshold * 100:.1f}%"
        )

    st.progress(
        min(
            int(
                model_score * 100
            ),
            100
        )
    )

    if predicted_class == 1:
        st.error(
            "The entered clinical profile was classified "
            "as having elevated indicators associated with "
            "heart disease."
        )

        st.info(
            "This result should be reviewed by a qualified "
            "healthcare professional together with the "
            "appropriate clinical examinations."
        )

    else:
        st.success(
            "The entered clinical profile was not classified "
            "as having elevated indicators associated with "
            "heart disease."
        )

        st.info(
            "A lower model result does not rule out heart "
            "disease, especially when symptoms or important "
            "test results are present."
        )

    st.caption(
        "The displayed score is a machine-learning model "
        "score, not a calibrated clinical probability."
    )

    with st.expander(
        "View submitted model inputs"
    ):
        st.dataframe(
            input_dataframe,
            use_container_width=True,
            hide_index=True
        )


# Display model information
with st.sidebar:
    st.header(
        "Heart Model Information"
    )

    st.write(
        "**Algorithm:** Logistic Regression"
    )

    st.write(
        "**Dataset:** UCI Heart Disease"
    )

    st.write(
        "**Dataset records:** 920"
    )

    st.write(
        "**Input features:** 13"
    )

    st.write(
        f"**Decision threshold:** "
        f"{decision_threshold:.3f}"
    )

    st.write(
        "**Test ROC-AUC:** 0.909"
    )

    st.write(
        "**Test Recall:** 92.16%"
    )

    st.write(
        "**Test F1 Score:** 87.04%"
    )

    st.divider()

    st.caption(
        "The dataset combines Cleveland, Hungary, "
        "Switzerland, and VA Long Beach records."
    )