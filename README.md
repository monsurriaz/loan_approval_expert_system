# Loan Approval Expert System

**CSE 4383 — Artificial Intelligence and Expert Systems | Spring 2026**
IUBAT School of Computer Science and Engineering (ISCSE)
Course Instructor: Jubair Ahmed Nabin | Section: F | Group: 9

---

## Project Overview

This is an AI-based expert system for loan approval decision support. Given an applicant's financial and personal details, the system predicts whether the application should be **Approved**, **Rejected**, or sent for **Manual Review**.

The system integrates four AI components working together:

| Component | Description |
|-----------|-------------|
| Knowledge Base | Data-driven expert rules with weighted scoring |
| Bayesian Reasoning | Naive Bayes-style probabilistic inference with Laplace smoothing, built from scratch |
| Search Algorithm | Greedy Best-First Search ranking over Approve, Reject, and Manual Review |
| Machine Learning | RandomForestClassifier trained on historical applicant data |

---

## Assignment Requirements Covered

| Requirement | Implementation |
|-------------|---------------|
| Data handling and preprocessing | `data_handler.py`, `preprocessing.py` |
| Knowledge base / expert rules | `knowledge_base.py` |
| Probabilistic reasoning | `bayesian_reasoning.py` |
| Search algorithm | `search.py` |
| Machine learning model | `ml_model.py` |
| System integration | `expert_system.py` |
| Evaluation (accuracy, precision, recall, F1) | `evaluation.py` |
| Bias and ethical analysis | `evaluation.py` -> `results/bias_analysis.json` |

---

## System Architecture Overview

```
Applicant Data
     |
     ├──> Preprocessing (imputation, encoding, splitting)
     |
     ├──> Knowledge Base      ---> kb_score (0.0 – 1.0)    -┐
     ├──> Bayesian Reasoner   ---> bayes_prob (0.0 – 1.0)   ├──> Best-First Search ---> Final Decision
     └──> ML Model (RF)       ---> ml_prob (0.0 – 1.0)     -┘         |
                                                                  + Confidence Score
                                                                  + Explanation
```

---

## Dataset Information

**Dataset:** Loan Prediction Problem Dataset
**Source:** Kaggle
**URL:** https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
**License:** CC0 Public Domain
**Size:** 614 records, 12 features + 1 target

After downloading, place the file at: `data/raw/loan_data.csv`

| Feature | Type | Notes |
|---------|------|-------|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0 / 1 / 2 / 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income (dataset original unit) |
| CoapplicantIncome | Numerical | Monthly income of co-applicant (0 if none) |
| LoanAmount | Numerical | Loan amount in thousands (120 = 120,000) |
| Loan_Amount_Term | Numerical | Repayment term in months |
| Credit_History | Categorical | 1 = Good, 0 = Poor |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| Loan_Status | **Target** | Y = Approved, N = Rejected |

---

## Project Structure

```
loan_approval_expert_system/
|
├── data/
|   ├── raw/loan_data.csv          # Place Kaggle dataset here
|   ├── processed/train.csv
|   └── processed/test.csv
|
├── src/
|   ├── config.py                  # All paths, constants, weights
|   ├── data_handler.py            # Load data, EDA summary
|   ├── preprocessing.py           # Imputation, encoding, splitting
|   ├── knowledge_base.py          # Expert rules and rule scoring
|   ├── bayesian_reasoning.py      # Naive Bayes from scratch
|   ├── search.py                  # Greedy Best-First Search
|   ├── ml_model.py                # RandomForest training and prediction
|   ├── expert_system.py           # Integration of all components
|   └── evaluation.py             # Metrics, plots, bias analysis
|
├── models/
|   └── best_model.pkl             # Saved trained model
|
├── results/                       # Generated after running main.py
├── notebooks/
|   └── final_demo.ipynb           # Demo notebook for viva/presentation
|
├── main.py                        # Entry point — runs full pipeline
├── demo.py                        # Interactive demo for evaluation day
├── requirements.txt
├── README.md
├── CONTRIBUTORS.md
└── .gitignore
```

---

## Installation and Setup

### Option 1 — Virtual Environment (Recommended)

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2 — Global Install
```bash
pip install -r requirements.txt
```

---

## How to Run

> Make sure `data/raw/loan_data.csv` exists before running.

```bash
# Stage 1: Run core pipeline (required first)
python main.py

# Stage 1 + Stage 2: Full pipeline with all plots and analysis
python main.py --stage2

# Demo: Run with a predefined sample applicant
python demo.py --sample

# Demo: Interactive mode (enter applicant details manually)
python demo.py
```

| Command | What it does |
|---------|-------------|
| `python main.py` | Loads data, preprocesses, trains all components, saves model, outputs core results |
| `python main.py --stage2` | Everything above + generates feature importance, component comparison, and bias analysis |
| `python demo.py --sample` | Runs a predefined sample applicant through the expert system |
| `python demo.py` | Interactive console — enter any applicant's details and get a decision |

---

## Output Files

| File | Stage | Description |
|------|-------|-------------|
| `data/processed/train.csv` | Stage 1 | Processed training split |
| `data/processed/test.csv` | Stage 1 | Processed test split |
| `models/best_model.pkl` | Stage 1 | Saved RandomForest model |
| `results/metrics.json` | Stage 1 | Accuracy, Precision, Recall, F1-score |
| `results/predictions.csv` | Stage 1 | Per-applicant decisions with component scores |
| `results/confusion_matrix.png` | Stage 1 | Confusion matrix heatmap |
| `results/system_meta.pkl` | Stage 1 | Saved KB, Bayesian, and Search metadata |
| `results/feature_importance.png` | Stage 2 | RandomForest feature importances |
| `results/component_comparison.png` | Stage 2 | KB vs Bayesian vs ML vs Integrated accuracy |
| `results/bias_analysis.json` | Stage 2 | Approval rates by gender and marital status |

---

## Methodology Summary

1. Raw applicant data is loaded and a basic EDA summary is printed.
2. Missing values are imputed (median for numerical, mode for categorical).
3. Categorical features are label-encoded; Loan_Status Y/N is converted to 1/0.
4. Data is split into train (80%) and test (20%) using stratified sampling.
5. The Knowledge Base extracts thresholds from training data and applies weighted expert rules to produce a `kb_score`.
6. The Bayesian Reasoner bins numerical features into low/medium/high, computes conditional likelihoods with Laplace smoothing, and returns `bayes_prob`.
7. The RandomForest classifier is trained and returns `ml_prob` (approval probability).
8. The Search Algorithm computes a heuristic score for each candidate decision:
   - `Approve score = final_approval_score`
   - `Reject score = 1 - final_approval_score`
   - `Manual Review score = 1 - |final_approval_score - 0.5| × 2`
9. The highest-scoring candidate is selected as the final decision (Greedy Best-First).
10. The system outputs the decision, confidence score, component scores, rules fired, and a natural-language explanation.
11. Performance is evaluated using accuracy, precision, recall, F1-score, confusion matrix, and component-wise comparison.

**Manual Review** is triggered when the integrated score is near the decision boundary (approximately 0.40–0.60), routing uncertain cases to a human officer rather than forcing an automated binary decision.

---

## Evaluation Metrics

Evaluated on 123 test applicants (20% stratified split from 614 total records):

| Metric | Value |
|--------|-------|
| Accuracy | 78.86% |
| Precision | 85.54% |
| Recall | 83.53% |
| F1-Score | 84.52% |

Component-wise accuracy (standalone):

| Component | Accuracy |
|-----------|---------|
| Knowledge Base | 81.30% |
| Bayesian Reasoning | 71.54% |
| ML Model (RandomForest) | 82.93% |
| Integrated System | 78.86% |

---

## Demo Instructions

After running `python main.py` at least once:

```bash
python demo.py
```

You will be prompted to enter applicant details one by one. The system will output:

- Final decision (Approve / Manual Review / Reject)
- Confidence score
- Knowledge Base score
- Bayesian probability
- ML model probability
- Rules fired
- Ranked candidate decisions

To run a quick test with a predefined applicant:

```bash
python demo.py --sample
```

---

## Ethical Considerations

- The training dataset may encode historical bias related to gender and marital status.
- Bias analysis (`results/bias_analysis.json`) inspects approval rates across demographic groups.
- The system is designed as a **decision support tool**, not a replacement for human loan officers.
- The Manual Review category ensures borderline cases are escalated to human judgment.
- Transparency is maintained through explicit output of rules fired, KB score, Bayesian probability, ML probability, and ranked decision candidates for every prediction.

---

## Limitations

- The dataset contains only 614 records, which limits generalisability.
- Real banking systems require more features, larger datasets, and stricter validation.
- The Bayesian component uses a simplified Naive Bayes approach, not a full Bayesian Network.
- This system is built for academic demonstration purposes only.
- All real loan decisions still require human review and institutional accountability.

---

## Contributors

| Member | Student ID | Responsibility |
|--------|------------|---------------|
| Md. Monsur Rahman | 23103157 | Data handling, preprocessing, knowledge base, Bayesian reasoning |
| Suraiya Jahan Shanta | 23103091 | Search algorithm, ML model, expert system integration, evaluation, report |

---

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- scikit-learn: https://scikit-learn.org/stable/documentation.html
- pandas: https://pandas.pydata.org/docs/
- NumPy: https://numpy.org/doc/
- matplotlib: https://matplotlib.org/stable/contents.html
