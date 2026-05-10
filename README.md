# Loan Approval Expert System
**CSE 4383 — Artificial Intelligence and Expert Systems | Spring 2026**
IUBAT School of Computer Science and Engineering

---

## Project Overview
An AI-based expert system that predicts loan approval outcomes using:
- **Knowledge Base** — data-driven expert rules
- **Probabilistic Reasoning** — Naive Bayes from scratch with Laplace smoothing
- **Search Algorithm** — Greedy Best-First Search for decision ranking
- **Machine Learning** — RandomForestClassifier

## Dataset
Download from Kaggle: [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
Place the file at: `data/raw/loan_data.csv`

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
# Stage 1 (core pipeline)
python main.py

# Stage 1 + Stage 2 (all plots and analysis)
python main.py --stage2

# Interactive demo (run after main.py)
python demo.py
```

## Output Files
| File | Description |
|------|-------------|
| `results/metrics.json` | Accuracy, Precision, Recall, F1 |
| `results/predictions.csv` | Per-applicant decisions |
| `results/confusion_matrix.png` | Confusion matrix |
| `results/feature_importance.png` | RandomForest feature importances |
| `results/component_comparison.png` | KB vs Bayesian vs ML vs Integrated |
| `results/bias_analysis.json` | Approval rate by gender and marital status |

## Project Structure
```
src/
├── config.py             # All settings and constants
├── data_handler.py       # Data loading and EDA
├── preprocessing.py      # Cleaning, encoding, splitting
├── knowledge_base.py     # Expert rules
├── bayesian_reasoning.py # Naive Bayes from scratch
├── search.py             # Greedy Best-First Search
├── ml_model.py           # RandomForest
├── expert_system.py      # Integration layer
└── evaluation.py         # Metrics and plots
```

## Contributors
| Member | Responsibility |
|--------|---------------|
| [Name 1] | data_handler.py, preprocessing.py, EDA |
| [Name 2] | knowledge_base.py, bayesian_reasoning.py |
| [Name 3] | search.py, ml_model.py |
| [Name 4] | expert_system.py, evaluation.py, report |
