# Dataset

**Source:** Kaggle — Loan Prediction Problem Dataset
**URL:** https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
**License:** CC0 Public Domain

## Columns
| Column | Type | Description |
|--------|------|-------------|
| Loan_ID | str | Unique identifier (dropped before training) |
| Gender | categorical | Male / Female |
| Married | categorical | Yes / No |
| Dependents | categorical | 0 / 1 / 2 / 3+ |
| Education | categorical | Graduate / Not Graduate |
| Self_Employed | categorical | Yes / No |
| ApplicantIncome | numerical | Monthly income of applicant |
| CoapplicantIncome | numerical | Monthly income of co-applicant |
| LoanAmount | numerical | Loan amount in thousands |
| Loan_Amount_Term | numerical | Term of loan in months |
| Credit_History | categorical | 1 = Good, 0 = Bad |
| Property_Area | categorical | Urban / Semiurban / Rural |
| Loan_Status | target | Y = Approved, N = Rejected |

## Instructions
Place the downloaded CSV at: `data/raw/loan_data.csv`
