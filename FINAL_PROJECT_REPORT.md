# Multi-Disease Machine Learning System
## Final Project Report — Artificial Intelligence Course

## Abstract

This project presents a multi-disease machine-learning system that integrates six independent supervised-classification models into a unified Streamlit application. The system covers diabetes, heart disease, chronic kidney disease, liver disease, respiratory disease/asthma, and breast cancer. Each disease was treated as a separate machine-learning problem with its own dataset, exploratory analysis, preprocessing strategy, model-training workflow, evaluation process, and saved model artifact. Multiple candidate classifiers were evaluated, and the selected models were deployed through interactive Streamlit pages. The final held-out test results show particularly strong performance for the chronic kidney disease and breast cancer modules, while the respiratory module demonstrates the importance of avoiding target leakage even when that choice reduces apparent predictive performance. The project is intended as an academic demonstration of end-to-end machine-learning development rather than a clinical diagnostic system.

## 1. Introduction

Machine learning can be applied to structured healthcare data to identify patterns associated with disease classes. However, an academically sound implementation requires more than obtaining high accuracy. Data quality, leakage prevention, reproducible train/validation/test separation, appropriate evaluation metrics, and transparent limitations are all essential.

The purpose of this project is to implement a complete machine-learning workflow for six disease-classification problems and combine the resulting trained models into a single user-facing application.

## 2. Project Objectives

The project objectives are:

- Build six independent supervised-classification pipelines.
- Perform exploratory data analysis and cleaning for each dataset.
- Prevent the use of identifiers and target-leakage variables as model inputs.
- Compare candidate machine-learning algorithms.
- Evaluate selected models using held-out test data.
- Save trained models and metadata in reusable `joblib` artifacts.
- Integrate the models into a Streamlit application.
- Provide a unified academic interface for demonstration and testing.
- Document model limitations and avoid presenting outputs as medical diagnosis.

## 3. System Architecture

The overall workflow is:

```text
Raw Dataset
    ↓
Data Exploration
    ↓
Cleaning & Feature Preparation
    ↓
Processed Dataset
    ↓
Train / Validation / Test Split
    ↓
Candidate Model Training
    ↓
Validation-Based Model Selection
    ↓
Threshold Selection
    ↓
Held-Out Test Evaluation
    ↓
Saved Model Artifact
    ↓
Streamlit Application
```

The six disease modules are kept independent so that each model uses only the features and preprocessing appropriate for its own dataset.

## 4. Disease Modules

### 4.1 Diabetes

The diabetes module uses structured health indicators to classify diabetes risk. The final selected model was `HistGradientBoostingClassifier`.

### 4.2 Heart Disease

The heart disease module uses cardiovascular and patient-characteristic variables. Logistic Regression was selected as the final model.

### 4.3 Chronic Kidney Disease

The chronic kidney disease module uses 24 predictors after cleaning and preprocessing. The model combines numerical and categorical variables through a preprocessing pipeline, and Logistic Regression was selected as the final classifier.

### 4.4 Liver Disease

The liver disease module uses 10 predictors including age, gender, bilirubin measurements, liver enzymes, proteins, albumin, and albumin/globulin ratio. Logistic Regression was selected as the final classifier.

### 4.5 Respiratory Disease / Asthma

The respiratory module uses four non-leaking predictors:

- age
- gender
- smoking status
- peak flow

The `Medication` field was intentionally excluded because its relationship with the target created target leakage. The final selected model was Support Vector Machine.

### 4.6 Breast Cancer

The breast cancer module uses 30 numerical diagnostic measurements from the Wisconsin Diagnostic dataset. The source CSV identifier and empty column were excluded during preprocessing. Logistic Regression was selected as the final classifier.

## 5. Data Preprocessing

The preprocessing strategy depended on the dataset, but the project consistently applied the following principles:

- remove patient/sample identifiers
- standardize feature names
- convert targets to binary classes
- normalize categorical values
- convert documented numerical features to numeric types
- handle missing values inside model pipelines where applicable
- avoid imputation before the train/validation/test split
- remove features that expose or strongly encode the target
- preserve a held-out test set for final evaluation only

A particularly important example is the respiratory dataset, where `Medication` was removed to avoid target leakage.

## 6. Model Training Strategy

Candidate algorithms used across the project include:

- Logistic Regression
- Random Forest
- Extra Trees
- Support Vector Machine
- HistGradientBoostingClassifier

The exact candidate set varied by disease module.

For the later disease modules, the data was divided into training, validation, and test subsets. Candidate models were trained on the training set, compared using validation metrics, and then evaluated once on the untouched test set.

## 7. Decision Thresholds

Binary classifiers often use a default threshold of 0.50, but this is not always optimal. Several project modules selected a decision threshold using validation data rather than the test set.

The final saved thresholds were approximately:

- Diabetes: 0.547
- Heart Disease: 0.313
- Chronic Kidney Disease: 0.091
- Liver Disease: 0.295
- Respiratory Disease: 0.495
- Breast Cancer: 0.330

This approach keeps threshold selection separate from final test evaluation.

## 8. Evaluation Metrics

The project uses several classification metrics:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC where stored in the model artifact

Accuracy alone is not sufficient, especially when classes are imbalanced. Recall, precision, F1 score, and ROC-AUC provide additional information about model behavior.

## 9. Final Model Results

| Disease | Selected Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Diabetes | HistGradientBoostingClassifier | 0.7510 | 0.7519 | 0.3284 | 0.7530 | 0.4574 | 0.8313 |
| Heart Disease | Logistic Regression | 0.8478 | 0.8388 | 0.8246 | 0.9216 | 0.8704 | 0.9085 |
| Chronic Kidney Disease | Logistic Regression | 0.9750 | 0.9667 | 0.9615 | 1.0000 | 0.9804 | 1.0000 |
| Liver Disease | Logistic Regression | 0.7368 | 0.6263 | 0.7742 | 0.8889 | 0.8276 | 0.8047 |
| Respiratory Disease | Support Vector Machine | 0.6000 | 0.5929 | 0.5814 | 0.8065 | 0.6757 | 0.5717 |
| Breast Cancer | Logistic Regression | 0.9737 | 0.9692 | 0.9756 | 0.9524 | 0.9639 | 0.9927 |

## 10. Results Discussion

### 10.1 Chronic Kidney Disease

The chronic kidney disease model achieved an accuracy of 0.9750, recall of 1.0000, F1 score of 0.9804, and ROC-AUC of 1.0000 on the held-out test set. This was the strongest overall test performance among the six independent experiments.

The result should not be interpreted as perfect clinical diagnosis. It describes performance on the specific held-out dataset used in this experiment.

### 10.2 Breast Cancer

The breast cancer model also achieved very strong performance, with accuracy of 0.9737, precision of 0.9756, recall of 0.9524, F1 score of 0.9639, and ROC-AUC of 0.9927. This indicates strong discrimination between benign and malignant classes within the selected dataset.

### 10.3 Heart Disease

The heart disease model achieved accuracy of 0.8478 and ROC-AUC of 0.9085. Recall was high at 0.9216, while precision remained strong at 0.8246. This represents a comparatively balanced result.

### 10.4 Liver Disease

The liver model achieved accuracy of 0.7368 and ROC-AUC of 0.8047. Recall was 0.8889, while balanced accuracy was lower at 0.6263. The model therefore showed stronger sensitivity to the positive class than balanced class discrimination.

### 10.5 Diabetes

The diabetes model achieved ROC-AUC of 0.8313 and recall of 0.7530, but precision was 0.3284. This indicates that the model captures a meaningful proportion of positive cases but also produces more false-positive classifications than the stronger modules.

### 10.6 Respiratory Disease

The respiratory model achieved accuracy of 0.6000 and ROC-AUC of 0.5717, making it the weakest discrimination result in the project. Recall remained 0.8065.

This result is academically important because the leakage-prone `Medication` feature was intentionally excluded. The lower result therefore reflects the limited predictive signal of the remaining four valid predictors rather than an artificially high score caused by leakage.

## 11. Why the Six Models Should Not Be Compared as Clinical Difficulty

The disease models use different datasets, sample sizes, feature sets, class distributions, measurement processes, and levels of predictive signal. Therefore, the statement that one disease is "easier to diagnose" because its model obtained a higher score would be invalid.

The correct interpretation is:

> The Chronic Kidney Disease model achieved the strongest test performance among the six independent machine-learning experiments.

## 12. Streamlit Application

The final application provides:

- a central academic dashboard
- six disease-specific modules
- input controls matched to each trained model
- loading of saved model artifacts using `joblib`
- model score display
- threshold-based classification
- educational result messaging

The application checks the feature ordering saved inside the model artifact to reduce the risk of mismatched inference inputs.

## 13. Quality Assurance

The project includes a final-quality workflow that checks:

- presence of the six Streamlit pages
- presence of the six saved models
- Python syntax
- required runtime packages
- successful loading of model artifacts
- release-folder hygiene

All six application modules were also tested end-to-end through the Streamlit interface.

## 14. Limitations

The project has several limitations:

- The datasets are independent and have different characteristics.
- Some datasets are relatively small.
- Some models use screening-style variables, while others use diagnostic measurements.
- Model scores are not calibrated clinical probabilities unless calibration is explicitly performed.
- The respiratory dataset contains limited non-leaking predictive information.
- The project does not perform prospective clinical validation.
- The application is intended for academic demonstration only.

## 15. Ethical and Academic Use

The system must not be presented as a replacement for medical professionals. Predictions should be described as machine-learning classifications, risk indicators, or educational screening outputs.

No result from the application should be treated as a medical diagnosis.

## 16. Conclusion

The project successfully demonstrates an end-to-end machine-learning workflow across six disease-classification tasks. It includes data exploration, preprocessing, leakage prevention, model comparison, held-out evaluation, artifact serialization, application integration, and final quality assurance.

The strongest held-out results were observed for chronic kidney disease and breast cancer, while the respiratory module demonstrated the importance of methodological integrity by preserving leakage-free inputs even at the cost of lower performance.

Overall, the system meets the academic objective of demonstrating how multiple independent machine-learning models can be developed and integrated into a unified AI application.

## 17. Recommended Figures for the Final Report

Insert the following figures from `reports/figures/` where available:

1. Diabetes model evaluation figure
2. Heart disease model evaluation figure
3. Kidney disease model evaluation figure
4. Liver disease model evaluation figure
5. Respiratory model evaluation figure
6. Cancer model evaluation figure
7. Relevant target-distribution charts
8. Feature-importance or coefficient charts
9. Main Streamlit dashboard screenshot
10. One representative prediction-page screenshot

## 18. References

Use the exact dataset source URLs and citations associated with the downloaded datasets. Do not invent missing references. At minimum, include the official UCI citation for the Chronic Kidney Disease dataset and the documented source for the Wisconsin Diagnostic Breast Cancer dataset, together with the exact sources used for the diabetes, heart, liver, and respiratory datasets.
