from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Configure the Streamlit page
st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide"
)


# Locate the project root from the multipage application file
page_file = Path(__file__).resolve()
project_root = page_file.parents[2]

# Build the path to the saved diabetes model
model_path = (
    project_root
    / "models"
    / "diabetes_model.joblib"
)

@st.cache_resource
def load_model_package():
    """Load and cache the trained diabetes model package."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    return joblib.load(model_path)


# Define category mappings used by the BRFSS dataset
age_options = {
    "18–24 years": 1,
    "25–29 years": 2,
    "30–34 years": 3,
    "35–39 years": 4,
    "40–44 years": 5,
    "45–49 years": 6,
    "50–54 years": 7,
    "55–59 years": 8,
    "60–64 years": 9,
    "65–69 years": 10,
    "70–74 years": 11,
    "75–79 years": 12,
    "80 years or older": 13
}

general_health_options = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5
}

education_options = {
    "Never attended school or kindergarten only": 1,
    "Elementary school": 2,
    "Some high school": 3,
    "High school graduate": 4,
    "Some college or technical school": 5,
    "College graduate": 6
}

income_options = {
    "Less than $10,000": 1,
    "$10,000–$14,999": 2,
    "$15,000–$19,999": 3,
    "$20,000–$24,999": 4,
    "$25,000–$34,999": 5,
    "$35,000–$49,999": 6,
    "$50,000–$74,999": 7,
    "$75,000 or more": 8
}

yes_no_options = {
    "No": 0,
    "Yes": 1
}

sex_options = {
    "Female": 0,
    "Male": 1
}


# Load the trained model package
try:
    model_package = load_model_package()
    diabetes_model = model_package["model"]
    decision_threshold = float(
        model_package["threshold"]
    )
    feature_names = model_package["feature_names"]

except Exception as error:
    st.error(
        f"Unable to load the diabetes model: {error}"
    )
    st.stop()


# Display the application header
st.title("Diabetes Risk Screening System")

st.write(
    "Enter the health information below to generate a "
    "machine-learning screening result."
)

st.warning(
    "This educational system is not a medical diagnosis. "
    "Its output must not replace laboratory testing or "
    "consultation with a qualified healthcare professional."
)


# Create the patient input form
with st.form("diabetes_screening_form"):

    st.subheader("Basic Information")

    basic_column_1, basic_column_2 = st.columns(2)

    with basic_column_1:
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

        age_label = st.selectbox(
            "Age group",
            options=list(age_options.keys()),
            index=4
        )

        sex_label = st.selectbox(
            "Sex",
            options=list(sex_options.keys())
        )

        education_label = st.selectbox(
            "Education level",
            options=list(education_options.keys()),
            index=3
        )

    with basic_column_2:
        income_label = st.selectbox(
            "Annual income category",
            options=list(income_options.keys()),
            index=4
        )

        general_health_label = st.selectbox(
            "General health",
            options=list(
                general_health_options.keys()
            ),
            index=2
        )

        mental_health_days = st.slider(
            "Poor mental health days during the past 30 days",
            min_value=0,
            max_value=30,
            value=0
        )

        physical_health_days = st.slider(
            "Poor physical health days during the past 30 days",
            min_value=0,
            max_value=30,
            value=0
        )

    st.divider()
    st.subheader("Medical History")

    medical_column_1, medical_column_2 = st.columns(2)

    with medical_column_1:
        high_blood_pressure = st.selectbox(
            "High blood pressure",
            options=list(yes_no_options.keys())
        )

        high_cholesterol = st.selectbox(
            "High cholesterol",
            options=list(yes_no_options.keys())
        )

        cholesterol_check = st.selectbox(
            "Cholesterol checked within the past five years",
            options=list(yes_no_options.keys()),
            index=1
        )

        stroke = st.selectbox(
            "History of stroke",
            options=list(yes_no_options.keys())
        )

        heart_disease = st.selectbox(
            "Coronary heart disease or heart attack",
            options=list(yes_no_options.keys())
        )

    with medical_column_2:
        difficulty_walking = st.selectbox(
            "Serious difficulty walking or climbing stairs",
            options=list(yes_no_options.keys())
        )

        healthcare_coverage = st.selectbox(
            "Healthcare coverage",
            options=list(yes_no_options.keys()),
            index=1
        )

        unable_to_see_doctor = st.selectbox(
            "Unable to see a doctor because of cost",
            options=list(yes_no_options.keys())
        )

        smoker = st.selectbox(
            "Smoked at least 100 cigarettes during lifetime",
            options=list(yes_no_options.keys())
        )

    st.divider()
    st.subheader("Lifestyle Information")

    lifestyle_column_1, lifestyle_column_2 = (
        st.columns(2)
    )

    with lifestyle_column_1:
        physical_activity = st.selectbox(
            "Physical activity during the past 30 days",
            options=list(yes_no_options.keys()),
            index=1
        )

        fruit_consumption = st.selectbox(
            "Consumes fruit one or more times per day",
            options=list(yes_no_options.keys()),
            index=1
        )

        vegetable_consumption = st.selectbox(
            "Consumes vegetables one or more times per day",
            options=list(yes_no_options.keys()),
            index=1
        )

    with lifestyle_column_2:
        heavy_alcohol_consumption = st.selectbox(
            "Heavy alcohol consumption",
            options=list(yes_no_options.keys())
        )

    submit_prediction = st.form_submit_button(
        "Analyze Diabetes Risk",
        type="primary",
        use_container_width=True
    )


# Process the submitted health information
if submit_prediction:

    input_values = {
        "HighBP": yes_no_options[
            high_blood_pressure
        ],
        "HighChol": yes_no_options[
            high_cholesterol
        ],
        "CholCheck": yes_no_options[
            cholesterol_check
        ],
        "BMI": float(bmi),
        "Smoker": yes_no_options[
            smoker
        ],
        "Stroke": yes_no_options[
            stroke
        ],
        "HeartDiseaseorAttack": yes_no_options[
            heart_disease
        ],
        "PhysActivity": yes_no_options[
            physical_activity
        ],
        "Fruits": yes_no_options[
            fruit_consumption
        ],
        "Veggies": yes_no_options[
            vegetable_consumption
        ],
        "HvyAlcoholConsump": yes_no_options[
            heavy_alcohol_consumption
        ],
        "AnyHealthcare": yes_no_options[
            healthcare_coverage
        ],
        "NoDocbcCost": yes_no_options[
            unable_to_see_doctor
        ],
        "GenHlth": general_health_options[
            general_health_label
        ],
        "MentHlth": int(
            mental_health_days
        ),
        "PhysHlth": int(
            physical_health_days
        ),
        "DiffWalk": yes_no_options[
            difficulty_walking
        ],
        "Sex": sex_options[
            sex_label
        ],
        "Age": age_options[
            age_label
        ],
        "Education": education_options[
            education_label
        ],
        "Income": income_options[
            income_label
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
        diabetes_model.predict_proba(
            input_dataframe
        )[0, 1]
    )

    predicted_class = int(
        model_score >= decision_threshold
    )

    st.divider()
    st.subheader("Screening Result")

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
            int(model_score * 100),
            100
        )
    )

    if predicted_class == 1:
        st.error(
            "The entered profile was classified as having "
            "elevated indicators associated with prediabetes "
            "or diabetes."
        )

        st.info(
            "Consider discussing this result with a qualified "
            "healthcare professional and obtaining appropriate "
            "clinical or laboratory testing."
        )

    else:
        st.success(
            "The entered profile was not classified as having "
            "elevated indicators associated with prediabetes "
            "or diabetes."
        )

        st.info(
            "A lower model result does not rule out diabetes. "
            "Continue following professional screening advice."
        )

    st.caption(
        "The displayed score is a machine-learning model score, "
        "not a calibrated clinical probability."
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
    st.header("Model Information")

    st.write(
        "**Algorithm:** HistGradientBoosting"
    )

    st.write(
        "**Dataset:** CDC BRFSS 2015"
    )

    st.write(
        f"**Decision threshold:** "
        f"{decision_threshold:.3f}"
    )

    st.write(
        "**Test ROC-AUC:** 0.831"
    )

    st.write(
        "**Test Recall:** 75.3%"
    )

    st.write(
        "**Features:** 21"
    )

    st.divider()

    st.caption(
        "Developed as an educational machine-learning project."
    )