from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Multi-Disease ML System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Project paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
PAGES_DIR = APP_DIR / "pages"


# ============================================================
# Application modules
# ============================================================

DISEASE_MODULES = [
    {
        "name": "Diabetes",
        "icon": "🩸",
        "description": (
            "Machine-learning screening using patient health measurements "
            "associated with diabetes risk."
        ),
        "page": "pages/1_Diabetes.py",
        "model": "diabetes_model.joblib",
        "status": "completed",
    },
    {
        "name": "Heart Disease",
        "icon": "❤️",
        "description": (
            "Clinical screening using cardiovascular measurements and "
            "patient characteristics."
        ),
        "page": "pages/2_Heart_Disease.py",
        "model": "heart_disease_model.joblib",
        "status": "completed",
    },
    {
        "name": "Chronic Kidney Disease",
        "icon": "🫘",
        "description": (
            "Screening using kidney-related laboratory measurements, "
            "urinalysis findings, and clinical indicators."
        ),
        "page": "pages/3_Kidney_Disease.py",
        "model": "kidney_disease_model.joblib",
        "status": "completed",
    },
    {
        "name": "Liver Disease",
        "icon": "🧬",
        "description": (
            "Screening using demographic information, bilirubin, liver "
            "enzymes, proteins, and related laboratory measurements."
        ),
        "page": "pages/4_Liver_Disease.py",
        "model": "liver_disease_model.joblib",
        "status": "completed",
    },
    {
        "name": "Respiratory Disease",
        "icon": "🫁",
        "description": (
            "Respiratory-disease screening module. Dataset preparation "
            "and model development are the next project stage."
        ),
        "page": "pages/5_Respiratory_Disease.py",
        "model": "respiratory_disease_model.joblib",
        "status": "completed",
    },
    {
        "name": "Cancer",
        "icon": "🎗️",
        "description": (
            "Cancer screening module planned after completion of the "
            "respiratory-disease workflow."
        ),
        "page": "pages/6_Cancer_Disease.py",
        "model": "cancer_disease_model.joblib",
        "status": "completed",
    },
]


# ============================================================
# Helpers
# ============================================================

def module_is_ready(module: dict) -> bool:
    """Return True only when both the Streamlit page and model exist."""
    if module["status"] != "completed":
        return False

    page_path = APP_DIR / module["page"]
    model_path = MODELS_DIR / module["model"]

    return page_path.exists() and model_path.exists()


def render_module_card(module: dict) -> None:
    """Render one disease module card."""
    ready = module_is_ready(module)

    with st.container(border=True):
        st.subheader(f'{module["icon"]} {module["name"]}')
        st.write(module["description"])

        if module["status"] == "planned":
            st.info("Planned module")
            st.caption("This module will be added in a later development stage.")
            return

        if ready:
            st.success("Model and Streamlit page ready")

            st.page_link(
                module["page"],
                label=f'Open {module["name"]} Screening',
                icon="➡️",
                use_container_width=True,
            )
        else:
            st.warning("Module files are incomplete")

            page_path = APP_DIR / module["page"]
            model_path = MODELS_DIR / module["model"]

            if not page_path.exists():
                st.caption(f'Missing page: {module["page"]}')

            if not model_path.exists():
                st.caption(f'Missing model: models/{module["model"]}')


# ============================================================
# Header
# ============================================================

st.title("🩺 Multi-Disease Machine Learning System")

st.write(
    "A multi-model educational screening application that combines separate "
    "machine-learning pipelines for several diseases in one Streamlit interface."
)

st.warning(
    "Educational use only. The outputs of these machine-learning models are "
    "screening results and must not be interpreted as medical diagnoses."
)


# ============================================================
# Project status
# ============================================================

completed_modules = [
    module
    for module in DISEASE_MODULES
    if module["status"] == "completed"
]

ready_modules = [
    module
    for module in completed_modules
    if module_is_ready(module)
]

planned_modules = [
    module
    for module in DISEASE_MODULES
    if module["status"] == "planned"
]

metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric(
    "Completed Disease Modules",
    f"{len(completed_modules)} / {len(DISEASE_MODULES)}",
)

metric_2.metric(
    "Ready on This Computer",
    len(ready_modules),
)

metric_3.metric(
    "Remaining Modules",
    len(planned_modules),
)

st.progress(
    len(completed_modules) / len(DISEASE_MODULES),
    text=(
        f"Project progress: {len(completed_modules)} of "
        f"{len(DISEASE_MODULES)} disease modules completed"
    ),
)


# ============================================================
# Disease screening modules
# ============================================================

st.divider()
st.header("Disease Screening Modules")

for index in range(0, len(DISEASE_MODULES), 2):
    left_column, right_column = st.columns(2)

    with left_column:
        render_module_card(DISEASE_MODULES[index])

    if index + 1 < len(DISEASE_MODULES):
        with right_column:
            render_module_card(DISEASE_MODULES[index + 1])


# ============================================================
# Workflow
# ============================================================

st.divider()
st.header("Project Workflow")

st.markdown(
    """
Each disease module follows the same reproducible machine-learning workflow:

**Raw Dataset → Data Exploration & Cleaning → Processed Dataset → Model Training
& Evaluation → Saved Model Artifact → Streamlit Screening Page**

Current completed modules:

1. Diabetes
2. Heart Disease
3. Chronic Kidney Disease
4. Liver Disease
5. Respiratory Disease
6. Cancer
"""
)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Multi-Disease Machine Learning System | Artificial Intelligence Project"
)
