# =============================================================================
# preprocessing.py
# Component 1b: Preprocessing — clean, encode, split, and save processed data
# No scaling applied (RandomForest does not require it)
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL,
    DROP_COLS, TEST_SIZE, RANDOM_SEED,
    DATA_TRAIN, DATA_TEST
)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numerical columns → median imputation
    - Categorical columns → mode imputation

    Args:
        df: Raw DataFrame (may contain NaN).

    Returns:
        DataFrame with no missing values.
    """
    df = df.copy()

    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [preprocess] {col}: filled {df[col].isnull().sum()} NaN → median={median_val:.1f}")

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [preprocess] {col}: filled NaN → mode='{mode_val}'")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features as integer codes.
    Convert Target column Loan_Status Y/N → 1/0.

    Args:
        df: DataFrame after missing value handling.

    Returns:
        DataFrame with all columns as numeric types.
    """
    df = df.copy()

    # Encode target
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Y": 1, "N": 0})

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes  # -1 for any residual NaN

    return df


def drop_unnecessary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful for modelling (e.g. Loan_ID).

    Args:
        df: Encoded DataFrame.

    Returns:
        DataFrame without dropped columns.
    """
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the processed DataFrame into train/test feature matrices and targets.

    Args:
        df: Fully cleaned and encoded DataFrame.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"[preprocess] Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save the processed train and test DataFrames to disk.

    Args:
        train_df: Training split (X + y combined).
        test_df:  Testing split  (X + y combined).
    """
    train_df.to_csv(DATA_TRAIN, index=False)
    test_df.to_csv(DATA_TEST, index=False)
    print(f"[preprocess] Saved → {DATA_TRAIN}")
    print(f"[preprocess] Saved → {DATA_TEST}")


def run_preprocessing(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline in one call.
    Called by main.py.

    Args:
        df: Raw DataFrame from data_handler.load_data()

    Returns:
        X_train, X_test, y_train, y_test, processed_df
    """
    print("\n[preprocess] Starting preprocessing pipeline...")
    df = handle_missing(df)
    df = encode_features(df)
    df = drop_unnecessary(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # Reconstruct full train/test DataFrames for saving
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    save_processed(train_df, test_df)
    print("[preprocess] Preprocessing complete.\n")

    return X_train, X_test, y_train, y_test, df
