# =============================================================================
# bayesian_reasoning.py
# Component 3: Probabilistic Reasoning
# Naive Bayes classifier built from scratch (no sklearn)
# Numerical features are binned (low/medium/high) before computing likelihoods
# Laplace smoothing applied to avoid zero-probability issues
# =============================================================================

import pandas as pd
import numpy as np
from src.config import (
    NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL,
    N_BINS, BIN_LABELS, LAPLACE_K
)


# -- Step 1: Binning -------------------------------------------------------

def bin_numerical_columns(
    df: pd.DataFrame,
    bin_edges: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Convert numerical columns into ordinal bins: low / medium / high.
    Bin edges are computed from the data on first call, then reused for test data.

    Args:
        df       : DataFrame with numerical columns as floats.
        bin_edges: Pre-computed edges dict (pass when binning test data).

    Returns:
        (binned_df, bin_edges_dict)
    """
    df = df.copy()
    computed_edges = {}

    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue

        if bin_edges and col in bin_edges:
            edges = list(bin_edges[col])
            # Set infinite bounds to safely handle out-of-range values
            edges[0] = -float('inf')
            edges[-1] = float('inf')
            df[col] = pd.cut(
                df[col],
                bins=edges,
                labels=False,
                include_lowest=True
            ).astype(str)
        else:
            binned, edges = pd.qcut(
                df[col],
                q=N_BINS,
                labels=False,
                retbins=True,
                duplicates="drop"
            )
            df[col] = binned.astype(str)
            computed_edges[col] = edges

    return df, (bin_edges if bin_edges else computed_edges)


# -- Step 2: Priors -------------------------------------------------------

def compute_priors(y_train: pd.Series) -> dict:
    """
    Compute prior probabilities P(Approved) and P(Rejected) from training labels.

    Args:
        y_train: Series of 0/1 target values.

    Returns:
        Dict {"approved": P(Y=1), "rejected": P(Y=0)}
    """
    n = len(y_train)
    n_approved = (y_train == 1).sum()
    priors = {
        "approved": n_approved / n,
        "rejected": (n - n_approved) / n,
    }
    print(f"[bayesian] Prior P(Approved)={priors['approved']:.3f}  P(Rejected)={priors['rejected']:.3f}")
    return priors


# -- Step 3: Likelihoods ---------------------------------------------------

def compute_likelihoods(
    X_train_binned: pd.DataFrame,
    y_train: pd.Series
) -> dict:
    """
    Compute P(feature_value | class) for every feature and every unique value,
    with Laplace smoothing to prevent zero probabilities.

    Formula:
        P(x_i = v | C) = (count(x_i=v AND class=C) + k) /
                         (count(class=C) + k * |vocab_i|)
    where k = LAPLACE_K (smoothing constant), |vocab_i| = number of unique values.

    Args:
        X_train_binned: Training features after binning.
        y_train       : Training labels (0/1).

    Returns:
        Nested dict: likelihoods[feature][class_label][feature_value] = probability
    """
    likelihoods = {}
    all_cols = list(X_train_binned.columns)
    classes  = {1: "approved", 0: "rejected"}

    df = X_train_binned.copy()
    df["_label"] = y_train.values

    for col in all_cols:
        likelihoods[col] = {}
        vocab = df[col].unique().tolist()
        vocab_size = len(vocab)

        for class_code, class_name in classes.items():
            subset = df[df["_label"] == class_code]
            n_class = len(subset)
            likelihoods[col][class_name] = {}

            for val in vocab:
                count = (subset[col] == val).sum()
                # Laplace-smoothed probability
                prob = (count + LAPLACE_K) / (n_class + LAPLACE_K * vocab_size)
                likelihoods[col][class_name][str(val)] = prob

    return likelihoods


# -- Step 4: Inference -----------------------------------------------------

def bayesian_score(
    applicant_binned: dict,
    priors: dict,
    likelihoods: dict
) -> float:
    """
    Compute P(Approved | features) using the Naive Bayes formula.

    P(Approved | x) ~ P(Approved) x product P(x_i | Approved)
    P(Rejected | x) ~ P(Rejected) x product P(x_i | Rejected)

    Returns the normalised probability of approval.

    Args:
        applicant_binned : Dict of feature -> binned string value.
        priors           : Output of compute_priors().
        likelihoods      : Output of compute_likelihoods().

    Returns:
        Float in [0, 1] representing P(Approved | features).
    """
    log_approved = np.log(priors["approved"])
    log_rejected = np.log(priors["rejected"])

    for feature, value in applicant_binned.items():
        value = str(value)
        if feature not in likelihoods:
            continue

        # Use Laplace-smoothed fallback if unseen value
        p_approved = likelihoods[feature]["approved"].get(value, LAPLACE_K / (1 + LAPLACE_K * 10))
        p_rejected = likelihoods[feature]["rejected"].get(value, LAPLACE_K / (1 + LAPLACE_K * 10))

        log_approved += np.log(p_approved)
        log_rejected += np.log(p_rejected)

    # Convert log-probabilities back to probabilities and normalise
    max_log = max(log_approved, log_rejected)          # numerical stability
    exp_approved = np.exp(log_approved - max_log)
    exp_rejected = np.exp(log_rejected - max_log)

    prob_approved = exp_approved / (exp_approved + exp_rejected)
    return round(float(prob_approved), 4)


# -- Public training wrapper -----------------------------------------------

def train_bayesian(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> tuple[dict, dict, dict]:
    """
    Full Bayesian training pipeline:
    bin -> compute priors -> compute likelihoods.

    Args:
        X_train: Encoded feature DataFrame (numerical columns still as floats).
        y_train: Target Series (0/1).

    Returns:
        (priors, likelihoods, bin_edges)
    """
    print("[bayesian] Training Bayesian reasoner...")
    X_binned, bin_edges = bin_numerical_columns(X_train)
    priors              = compute_priors(y_train)
    likelihoods         = compute_likelihoods(X_binned, y_train)
    print(f"[bayesian] Likelihoods computed for {len(likelihoods)} features.")
    return priors, likelihoods, bin_edges
