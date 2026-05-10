# =============================================================================
# data_handler.py
# Component 1a: Data Handling — load dataset, print basic EDA summary
# =============================================================================

import pandas as pd
from src.config import DATA_RAW, TARGET_COL


def load_data(path: str = DATA_RAW) -> pd.DataFrame:
    """
    Load the raw loan dataset from a CSV file.

    Args:
        path: File path to the CSV dataset.

    Returns:
        Raw DataFrame as loaded from disk.
    """
    df = pd.read_csv(path)
    print(f"[data_handler] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_eda(df: pd.DataFrame) -> None:
    """
    Print a basic Exploratory Data Analysis summary:
    column types, missing value counts, and target distribution.

    Args:
        df: Raw DataFrame.
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n-- Column types ------------------------------------------")
    print(df.dtypes.to_string())

    print("\n-- Missing values ----------------------------------------")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values found.")
    else:
        print(missing.to_string())

    print(f"\n-- Target distribution ({TARGET_COL}) --------------------")
    print(df[TARGET_COL].value_counts().to_string())
    print(f"  Approval rate: {(df[TARGET_COL] == 'Y').mean():.1%}")

    print("\n-- Sample rows -------------------------------------------")
    print(df.head(3).to_string())
    print("=" * 60 + "\n")
