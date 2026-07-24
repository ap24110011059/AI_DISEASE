# Adaptive AI Models for Proactive Chronic Kidney Disease Detection

An 8-week Summer Research Internship project exploring whether machine learning systems can adapt to changing patient data and proactively identify disease risk before conventional classification methods.

The project began with the Pima Indians Diabetes Dataset and was later extended to the Chronic Kidney Disease (CKD) dataset to evaluate cross-dataset robustness, reproducibility, and generalization.

**Mentor:** Dr. Ch. Anil Carie, SRM University–AP

**Team:**
- Venkata Ajay Odugu (AP24110011016)
- Mohanasritha Eerla (AP24110011024)
- Vijay Perla (AP24110011059)
- Neelima Bojanapu (AP24110011111)

---

## Why this project exists

Most healthcare machine learning systems are static—they are trained once, evaluated once, and rarely revisited. In practice, however, patient populations, clinical measurements, and data distributions change over time.

This project explores three questions:

1. Can machine learning models adapt to new patient data over time?
2. Can adaptive learning improve robustness across datasets?
3. Can proactive detection identify high-risk cases earlier, even if it increases false alarms?

Rather than optimizing for a single metric, this repository focuses on understanding the trade-offs between accuracy, recall, reproducibility, and clinical usefulness.

---

## What was built

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- Adaptive MLP
- Proactive Detection Layer
- CKD Cross-Dataset Validation
- Error Analysis Pipeline
- Publication-Ready Figures
- Conference Paper Artifacts

---

## Datasets

### Primary Dataset

**Pima Indians Diabetes Dataset**

- 768 patients
- 8 clinical features
- Binary diabetes outcome

### Validation Dataset

**Chronic Kidney Disease Dataset**

- 400 patients
- 24+ clinical attributes
- Used for cross-dataset validation

---

## Key Results

| Model | Accuracy | F1 | AUC |
|------|------|------|------|
| Logistic Regression | 0.71 | 0.54 | 0.81 |
| Decision Tree | 0.69 | 0.58 | 0.76 |
| Random Forest | 0.74 | 0.63 | 0.84 |
| SVM | 0.73 | 0.61 | 0.82 |
| Adaptive MLP | 0.76 | 0.66 | 0.86 |

### Adaptive Framework

| Configuration | Accuracy | Recall | F1 |
|---------------|----------|--------|----|
| Static MLP | 0.75 | 0.56 | 0.61 |
| Adaptive MLP | 0.73 | 0.47 | 0.52 |
| Adaptive + Proactive | 0.65 | 0.80 | 0.60 |

The Adaptive + Proactive framework significantly improved recall, suggesting that adaptive healthcare systems may be more useful for early screening applications where missing a positive case is more costly than a false alarm.

---

## Cross-Dataset Validation

The CKD evaluation demonstrated that the proposed framework generalizes beyond a single disease domain.

Validation included:

- Train/Test Evaluation
- Multi-seed Validation
- Error Analysis
- Publication Figure Generation
- Reproducibility Checks

These experiments showed that adaptive learning approaches remain competitive when evaluated on independent healthcare datasets.

---

## Repository Structure

```text
AI_DISEASE/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── CKD/
│
├── md/
├── paper/
├── results/
├── scripts/
│
├── adaptive_model.py
├── pipeline.py
├── requirements.txt
└── README.md
```

---

## Publication Artifacts

This repository contains:

- 6 publication-ready figures
- Error analysis reports
- Related work summaries
- Conference paper drafts
- Weekly research reports
- CKD notebooks
- Canonical results

---

## Honest Limitations

- Adaptive MLP does not consistently outperform all baseline models.
- Evaluation was performed on relatively small healthcare datasets.
- Real-time deployment was not explored.
- Clinical validation was outside the scope of this internship.

These limitations are intentionally documented because reproducibility and transparency are important aspects of machine learning research.

---

## Internship Timeline

| Week | Deliverable |
|------|-------------|
| Week 1 | Environment Setup |
| Week 2 | EDA |
| Week 3 | Classical ML |
| Week 4 | MLP |
| Week 5 | Adaptive Learning |
| Week 6 | CKD Evaluation |
| Week 7 | Paper Writing |
| Week 8 | Final Demonstration |

---

## Running the Project

```bash
git clone https://github.com/ap24110011059/AI_DISEASE.git

cd AI_DISEASE

pip install -r requirements.txt

jupyter notebook
```

---

## Future Work

- Evaluate on larger healthcare datasets.
- Explore transformer-based approaches.
- Investigate real-time adaptation.
- Study deployment in clinical environments.

---

## Author

### Vijay Perla

B.Tech CSE  
SRM University AP  
Summer Research Intern

---

> "A machine learning project is only complete when it is reproducible, understandable, and useful to others."