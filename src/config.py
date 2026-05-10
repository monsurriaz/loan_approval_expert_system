# =============================================================================
# config.py
# Project-wide configuration: paths, constants, feature lists, thresholds
# Component: Data Handling (shared across all modules)
# =============================================================================

import os

# -- Paths ----------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw", "loan_data.csv")
DATA_TRAIN  = os.path.join(BASE_DIR, "data", "processed", "train.csv")
DATA_TEST   = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
PREP_PATH   = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# -- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# -- Dataset columns -------------------------------------------------------
TARGET_COL = "Loan_Status"
DROP_COLS  = ["Loan_ID"]

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "Property_Area", "Credit_History"
]

NUMERICAL_COLS = [
    "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term"
]

# Columns that need binning before Bayesian reasoning
COLS_TO_BIN = NUMERICAL_COLS

# -- Bayesian binning -------------------------------------------------------
# Each numerical column will be cut into 3 equal-frequency bins: low/medium/high
N_BINS      = 3
BIN_LABELS  = ["low", "medium", "high"]
LAPLACE_K   = 1          # Laplace smoothing constant

# -- Search weights -------------------------------------------------------
# How much each component contributes to final decision score
WEIGHT_ML      = 0.50
WEIGHT_BAYES   = 0.30
WEIGHT_KB      = 0.20

# Confidence thresholds for three-way decision
APPROVE_THRESHOLD       = 0.60   # score >= 0.60 = Approve
MANUAL_REVIEW_THRESHOLD = 0.40   # 0.40 <= score < 0.60 = Manual Review
                                 # score < 0.40 = Reject

# -- ML model ---------------------------------------------------------------
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = None
