# Final Quality Checklist

## Release Gate 1 — Structure
- [ ] app/main.py is the only release main file.
- [ ] Six Streamlit page files exist in app/pages/.
- [ ] Six trained .joblib files exist in models/.
- [ ] No old "planned" Respiratory/Cancer logic remains in the release main.py.

## Release Gate 2 — Automated QA
Run:

```bash
python final_quality_check.py
```

Required result:

```text
FINAL RESULT: PASS
```

## Release Gate 3 — Manual End-to-End Test
Open the application:

```bash
streamlit run app/main.py
```

For each module:
- [ ] Page opens with no traceback.
- [ ] Inputs can be changed.
- [ ] Generate Screening Result works.
- [ ] Model score is displayed.
- [ ] Decision threshold is displayed where applicable.
- [ ] No impossible or missing feature-name error appears.
- [ ] Result language is educational and does not claim clinical diagnosis.

Modules:
- [ ] Diabetes
- [ ] Heart Disease
- [ ] Chronic Kidney Disease
- [ ] Liver Disease
- [ ] Respiratory Disease
- [ ] Breast Cancer

## Release Gate 4 — Academic ML Review
- [ ] Identifiers are excluded from training.
- [ ] Target leakage is explicitly checked.
- [ ] Respiratory Medication is excluded from the model.
- [ ] Train/validation/test separation is documented.
- [ ] Test data is not used for model or threshold selection.
- [ ] Final metrics are copied from held-out test evaluation.
- [ ] Dataset limitations are documented.

## Release Gate 5 — Submission Package
- [ ] README.md
- [ ] requirements.txt
- [ ] Final report
- [ ] Final results comparison table
- [ ] Application screenshots
- [ ] Dataset references/citations
- [ ] Final ZIP created from the clean release folder
