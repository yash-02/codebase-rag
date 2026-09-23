"""Data loading and preparation utilities for the training pipeline."""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(path: str) -> pd.DataFrame:
    """Read the raw customer dataset from a CSV file on disk."""
    return pd.read_csv(path)


def undersample(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Randomly drop rows from the majority class so both classes are balanced.

    This is how class imbalance is handled before training -- the majority
    class is downsampled to match the minority class count.
    """
    counts = df[target_col].value_counts()
    minority_count = counts.min()
    balanced_parts = [
        group.sample(minority_count, random_state=42)
        for _, group in df.groupby(target_col)
    ]
    return pd.concat(balanced_parts).sample(frac=1, random_state=42)


def prepare_training_data(path: str, target_col: str):
    """Load the dataset, balance it, and split into train/test sets."""
    df = load_raw_data(path)
    if df[target_col].value_counts(normalize=True).min() < 0.4:
        df = undersample(df, target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)
