# =============================================================================
# ml_model.py
# Component 5: Machine Learning - RandomForestClassifier
# Single model, first version. Cross-validation and model selection
# will be added in Stage 2 if time permits.
# =============================================================================

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    RANDOM_SEED, MODEL_PATH
)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the training data.

    RandomForest was chosen because:
    - It handles mixed feature types without scaling.
    - It provides feature importance scores useful for the report.
    - It is robust to outliers and small datasets.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training labels (0 = Rejected, 1 = Approved).

    Returns:
        Fitted RandomForestClassifier.
    """
    print(f"[ml_model] Training RandomForestClassifier (n_estimators={RF_N_ESTIMATORS})...")
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_SEED,
        class_weight="balanced"   # handles class imbalance (~70/30 split)
    )
    model.fit(X_train, y_train)
    print("[ml_model] Training complete.")
    return model


def predict(
    model: RandomForestClassifier,
    X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate class predictions and approval probabilities.

    Args:
        model : Fitted RandomForestClassifier.
        X_test: Test feature DataFrame.

    Returns:
        (predictions, probabilities)
        predictions  : Array of 0/1 class labels.
        probabilities: Array of P(Approved) floats in [0, 1].
    """
    predictions   = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # P(class=1)
    return predictions, probabilities


def save_model(model: RandomForestClassifier, path: str = MODEL_PATH) -> None:
    """
    Persist the trained model to disk using pickle.

    Args:
        model: Fitted RandomForestClassifier.
        path : File path for the .pkl file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[ml_model] Model saved -> {path}")


def load_model(path: str = MODEL_PATH) -> RandomForestClassifier:
    """
    Load a previously saved model from disk.

    Args:
        path: File path of the .pkl file.

    Returns:
        Fitted RandomForestClassifier.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[ml_model] Model loaded <- {path}")
    return model


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract and sort feature importances from the trained RandomForest.
    Used by evaluation.py for the feature_importance.png plot (Stage 2).

    Args:
        model        : Fitted RandomForestClassifier.
        feature_names: List of column names in the same order as X_train.

    Returns:
        DataFrame sorted by importance descending.
    """
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df
