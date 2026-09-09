# Final Model Comparison

The values below are extracted directly from the saved model artifacts in the `models/` directory. Metrics marked `N/A` were not stored in that artifact and should not be invented.

| Disease | Best Model | Features | Threshold | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetes | HistGradientBoostingClassifier | N/A | 0.547 | 0.7510 | 0.7519 | 0.3284 | 0.7530 | 0.4574 | 0.8313 | 0.4317 |
| Heart Disease | Logistic Regression | N/A | 0.313 | 0.8478 | 0.8388 | 0.8246 | 0.9216 | 0.8704 | 0.9085 | 0.9188 |
| Chronic Kidney Disease | Logistic Regression | 24 | 0.091 | 0.9750 | 0.9667 | 0.9615 | 1.0000 | 0.9804 | 1.0000 | 1.0000 |
| Liver Disease | Logistic Regression | 10 | 0.295 | 0.7368 | 0.6263 | 0.7742 | 0.8889 | 0.8276 | 0.8047 | 0.9245 |
| Respiratory Disease | Support Vector Machine | 4 | 0.495 | 0.6000 | 0.5929 | 0.5814 | 0.8065 | 0.6757 | 0.5717 | N/A |
| Breast Cancer | Logistic Regression | 30 | 0.330 | 0.9737 | 0.9692 | 0.9756 | 0.9524 | 0.9639 | 0.9927 | N/A |

## Interpretation rule

- Use held-out **test-set metrics** in the final report.
- Do not compare diseases as though they share the same dataset difficulty.
- A higher score for one disease model does not imply that disease is easier to diagnose clinically.
- Report the respiratory model cautiously if its discrimination is weak after leakage-prone features are excluded.
