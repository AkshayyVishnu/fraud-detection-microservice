"""Configuration settings for Merchant Shield Fraud Detection Model."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration (defaults, will be overridden by Optuna)
MODEL_CONFIG = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 580,  # Approximate ratio: non_fraud / fraud
    "random_state": 42,
    "eval_metric": "logloss",
    "use_label_encoder": False,
}

# Dataset paths (will be checked in data_preprocessing.py)
DATASET_PATH = BASE_DIR / "dataset" / "creditcard.csv"  # Primary location
ADDITIONAL_DATASET_PATH = BASE_DIR / "data" /"dataset" / "AdditionalCreditcard.csv"  # Extended dataset for training
TRAINED_MODEL_PATH = MODELS_DIR / "xgb_fraud_model.pkl"
CALIBRATED_MODEL_PATH = MODELS_DIR / "xgb_fraud_model_calibrated.pkl"
