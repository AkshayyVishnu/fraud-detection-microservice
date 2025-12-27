"""
Train XGBoost Fraud Detection Model
Trains on creditcard.csv with class imbalance handling
Target: >95% AUC-ROC
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path(__file__).parent / 'data' / 'dataset' / 'creditcard.csv'
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_PATH = MODEL_DIR / 'fraud_detector.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'

def load_and_prepare_data():
    """Load dataset and prepare features"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].sum() / len(df) * 100:.2f}%)")
    
    # Features: V1-V28, Amount, Time
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    X = df[feature_cols].copy()
    y = df['Class'].copy()
    
    return X, y, feature_cols

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model():
    """Train XGBoost model with SMOTE for class imbalance"""
    print("\n" + "="*60)
    print("  TRAINING FRAUD DETECTION MODEL")
    print("="*60 + "\n")
    
    # Load data
    X, y, feature_cols = load_and_prepare_data()
    
    # Train/test split with stratification
    print("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Train set: {len(X_train)} samples ({y_train.sum()} fraud)")
    print(f"Test set: {len(X_test)} samples ({y_test.sum()} fraud)")
    
    # Scale features
    print("\nScaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42, sampling_strategy=0.3)  # Balance to 30% fraud
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE - Train set: {len(X_train_balanced)} samples ({y_train_balanced.sum()} fraud)")
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # SMOTE handles imbalance, so we can use 1
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )
    
    model.fit(
        X_train_balanced, 
        y_train_balanced,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC-ROC Score: {auc_score:.4f}")
    
    if auc_score < 0.95:
        print(f"⚠ Warning: AUC-ROC ({auc_score:.4f}) is below target (0.95)")
    else:
        print(f"✓ Model meets target AUC-ROC (>{auc_score:.4f})")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    print(f"\nSaving model to {MODEL_PATH}...")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save feature names for reference
    feature_info = {
        'feature_names': feature_cols,
        'n_features': len(feature_cols),
        'auc_score': float(auc_score)
    }
    joblib.dump(feature_info, MODEL_DIR / 'feature_info.pkl')
    
    print(f"✓ Model saved successfully")
    print(f"✓ Scaler saved to {SCALER_PATH}")
    print(f"✓ Feature info saved")
    
    return model, scaler, feature_cols, auc_score

if __name__ == '__main__':
    try:
        model, scaler, features, auc = train_model()
        print("\n" + "="*60)
        print("  TRAINING COMPLETE")
        print("="*60)
        print(f"\nModel AUC-ROC: {auc:.4f}")
        print(f"Model location: {MODEL_PATH}")
        print("\nNext steps:")
        print("1. Model will be loaded automatically in app.py")
        print("2. Use model_explainer.py for SHAP explanations")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

