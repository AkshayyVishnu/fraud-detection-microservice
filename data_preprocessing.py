"""Data preprocessing utilities for fraud detection."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import config


def load_dataset(file_path=None):
    """
    Load the credit card fraud dataset.
    
    Returns:
        DataFrame with all columns including Time, V1-V28, Amount, Class
    """
    if file_path is None:
        # Try multiple possible paths
        possible_paths = [
            config.DATASET_PATH,
            Path(__file__).parent / "creditcard.csv" / "creditcard.csv",
            Path(__file__).parent / "data" / "creditcard.csv",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                file_path = path
                break
        
        if file_path is None or not Path(file_path).exists():
            raise FileNotFoundError(
                f"Dataset not found. Please ensure creditcard.csv is available. "
                f"Tried: {possible_paths}"
            )
    
    df = pd.read_csv(file_path)
    return df


def create_time_based_split(df, test_size=0.2):
    """
    Create time-based train-test split.
    Assumes transactions are ordered by time.
    """
    # Sort by time if not already sorted
    df = df.sort_values("Time").reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df)} transactions ({len(train_df[train_df.Class==1])} frauds)")
    print(f"Test set: {len(test_df)} transactions ({len(test_df[test_df.Class==1])} frauds)")
    
    return train_df, test_df


def prepare_features(df, scaler=None, fit_scaler=False):
    """
    Prepare features for model training.
    
    Args:
        df: DataFrame with V1-V28, Time, Amount, Class columns
        scaler: Optional StandardScaler (for test set)
        fit_scaler: Whether to fit the scaler (for train set)
    
    Returns:
        X: Feature array
        y: Target array (if Class column exists)
        scaler: Fitted or used scaler
    """
    # Feature columns
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    
    X = df[feature_cols].values
    
    # Scale Time and Amount (V1-V28 are already PCA-transformed)
    # We'll scale both Time and Amount for consistency
    if scaler is None:
        scaler = StandardScaler()
    
    # Scale Time and Amount columns (last 2 columns)
    if fit_scaler:
        X[:, -2:] = scaler.fit_transform(X[:, -2:])
    else:
        X[:, -2:] = scaler.transform(X[:, -2:])
    
    # Extract target if it exists
    y = None
    if "Class" in df.columns:
        y = df["Class"].values
    
    return X, y, scaler


def get_class_weights(y):
    """Calculate scale_pos_weight for XGBoost."""
    fraud_count = np.sum(y == 1)
    non_fraud_count = np.sum(y == 0)
    return non_fraud_count / fraud_count if fraud_count > 0 else 1.0

