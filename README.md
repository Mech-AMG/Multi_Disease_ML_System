# Multi-Disease Machine Learning System

An academic Artificial Intelligence course project that integrates six independently trained supervised-classification models into one Streamlit application.

## Project Modules

1. Diabetes
2. Heart Disease
3. Chronic Kidney Disease
4. Liver Disease
5. Respiratory Disease / Asthma
6. Breast Cancer

Each module has its own:

- raw dataset
- exploratory data analysis and cleaning notebook
- processed dataset
- model-training and evaluation notebook
- saved `joblib` model artifact
- Streamlit prediction page

## System Architecture

```text
Raw Dataset
    ↓
Exploration & Cleaning
    ↓
Processed Dataset
    ↓
Train / Validation / Test Split
    ↓
Candidate Model Training
    ↓
Model Evaluation
    ↓
Threshold Selection
    ↓
Saved Model Artifact
    ↓
Streamlit Disease Module
```

## Project Structure

```text
Multi_Disease_ML_System/
│
├── app/
│   ├── main.py
│   └── pages/
│       ├── 1_Diabetes.py
│       ├── 2_Heart_Disease.py
│       ├── 3_Kidney_Disease.py
│       ├── 4_Liver_Disease.py
│       ├── 5_Respiratory_Disease.py
│       └── 6_Cancer_Disease.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── diabetes_model.joblib
│   ├── heart_disease_model.joblib
│   ├── kidney_disease_model.joblib
│   ├── liver_disease_model.joblib
│   ├── respiratory_disease_model.joblib
│   └── cancer_disease_model.joblib
│
├── notebooks/
│   ├── 01_diabetes_data_exploration.ipynb
│   ├── 02_diabetes_model_training.ipynb
│   ├── 03_heart_disease_data_exploration.ipynb
│   ├── 04_heart_disease_model_training.ipynb
│   ├── 05_kidney_disease_data_exploration.ipynb
│   ├── 06_kidney_disease_model_training.ipynb
│   ├── 07_liver_disease_data_exploration.ipynb
│   ├── 08_liver_disease_model_training.ipynb
│   ├── 09_respiratory_disease_data_exploration.ipynb
│   ├── 10_respiratory_disease_model_training.ipynb
│   ├── 11_cancer_data_exploration.ipynb
│   └── 12_cancer_model_training.ipynb
│
├── reports/
│   ├── figures/
│   ├── final_model_comparison.csv
│   └── final_model_comparison.md
│
├── requirements.txt
├── Run_Project.bat
├── final_quality_check.py
└── README.md
```

## Installation

Create or activate the project virtual environment and install the dependencies:

```bash
".venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Running the Application

Preferred method on Windows:

```text
Double-click Run_Project.bat
```

Or run manually:

```bash
".venv\Scripts\python.exe" -m streamlit run app\main.py
```

Then open:

```text
http://localhost:8501
```

## Machine Learning Methodology

The disease modules follow the same general experimental methodology:

1. Dataset inspection and exploratory analysis.
2. Data cleaning and target preparation.
3. Removal of identifiers and leakage-prone variables.
4. Separation into training, validation, and held-out test data.
5. Preprocessing inside scikit-learn pipelines where applicable.
6. Comparison of multiple classification algorithms.
7. Model selection using validation performance.
8. Decision-threshold selection using validation data where applicable.
9. Final evaluation on the untouched test set.
10. Export of the trained pipeline and metadata using `joblib`.

## Final Test Results

| Disease | Selected Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Diabetes | HistGradientBoostingClassifier | 0.7510 | 0.7519 | 0.3284 | 0.7530 | 0.4574 | 0.8313 |
| Heart Disease | Logistic Regression | 0.8478 | 0.8388 | 0.8246 | 0.9216 | 0.8704 | 0.9085 |
| Chronic Kidney Disease | Logistic Regression | 0.9750 | 0.9667 | 0.9615 | 1.0000 | 0.9804 | 1.0000 |
| Liver Disease | Logistic Regression | 0.7368 | 0.6263 | 0.7742 | 0.8889 | 0.8276 | 0.8047 |
| Respiratory Disease | Support Vector Machine | 0.6000 | 0.5929 | 0.5814 | 0.8065 | 0.6757 | 0.5717 |
| Breast Cancer | Logistic Regression | 0.9737 | 0.9692 | 0.9756 | 0.9524 | 0.9639 | 0.9927 |

### Interpretation

The models are independent experiments trained on different datasets, with different sample sizes, features, class distributions, and problem difficulty. Their numerical scores should therefore not be interpreted as a direct comparison of clinical disease-diagnosis difficulty.

The Chronic Kidney Disease and Breast Cancer experiments achieved the strongest held-out test performance. The Respiratory Disease model was intentionally evaluated without the `Medication` field because that field created target leakage; consequently, its weaker performance is a more defensible academic result than an artificially inflated score.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- SciPy
- Matplotlib
- Seaborn
- joblib
- Streamlit
- Jupyter Notebook

## Important Limitation

This application is an educational machine-learning project. Its outputs are model classifications or screening indicators and must not be interpreted as medical diagnoses or clinical probabilities.

## Reproducibility

Before final submission, run:

```bash
".venv\Scripts\python.exe" final_quality_check.py
```

The core release target is:

```text
FINAL RESULT: PASS
```

Then manually verify all six Streamlit pages end-to-end.
