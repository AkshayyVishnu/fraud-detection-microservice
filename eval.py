"""Evaluation script for fraud detection model.
Loads the trained XGBoost model and evaluates it on the test set with multiple thresholds.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
import config
from data_preprocessing import load_dataset


def evaluate_model_at_thresholds(model, X_test, y_test, thresholds=None):
    """
    Evaluate model at multiple thresholds and output confusion matrices with metrics.
    
    Args:
        model: Trained model
        X_test: Test features (DataFrame)
        y_test: Test labels (Series or array)
        thresholds: List of thresholds to evaluate. If None, uses default range.
    
    Returns:
        Dictionary with threshold as key and metrics as value
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {}
    
    print("\n" + "=" * 80)
    print("Evaluation at Multiple Thresholds")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'TN':<8} {'FP':<8} {'FN':<8} {'TP':<8}")
    print("-" * 80)
    
    for threshold in thresholds:
        # Binary predictions at threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results
        results[threshold] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        
        # Print formatted results
        print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {accuracy:<12.4f} {int(tn):<8} {int(fp):<8} {int(fn):<8} {int(tp):<8}")
    
    print("=" * 80)
    
    # Print Precision and Recall Summary Table
    print("\n" + "=" * 80)
    print("Precision and Recall Summary")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 80)
    for threshold in thresholds:
        metrics = results[threshold]
        print(f"{threshold:<12.2f} {metrics['precision']:<15.4f} {metrics['recall']:<15.4f} {metrics['f1_score']:<15.4f}")
    print("=" * 80)
    
    # Print detailed confusion matrices
    print("\n" + "=" * 80)
    print("Detailed Confusion Matrices for Each Threshold")
    print("=" * 80)
    for threshold in thresholds:
        metrics = results[threshold]
        print(f"\nThreshold: {threshold:.2f}")
        print("-" * 80)
        print(f"Confusion Matrix:")
        print(f"  {metrics['confusion_matrix']}")
        print(f"\n  True Negatives (TN):  {metrics['tn']}")
        print(f"  False Positives (FP): {metrics['fp']}")
        print(f"  False Negatives (FN): {metrics['fn']}")
        print(f"  True Positives (TP):  {metrics['tp']}")
        print(f"\n  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    
    return results


def main():
    """Main evaluation pipeline."""
    print("=" * 80)
    print("Fraud Detection Model Evaluation")
    print("=" * 80)
    
    # Load the trained model
    print("\n1. Loading trained model...")
    try:
        model = joblib.load(config.TRAINED_MODEL_PATH)
        print(f"   ✓ Model loaded from: {config.TRAINED_MODEL_PATH}")
    except FileNotFoundError:
        print(f"   ✗ Error: Model file not found at {config.TRAINED_MODEL_PATH}")
        print("   Please run model_training.py first to train the model.")
        return
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Load best parameters (for reference)
    try:
        best_params = joblib.load(config.MODELS_DIR / "best_params.pkl")
        print(f"   ✓ Best parameters loaded")
        print(f"   Key parameters:")
        print(f"     - n_estimators: {best_params.get('n_estimators', 'N/A')}")
        print(f"     - max_depth: {best_params.get('max_depth', 'N/A')}")
        print(f"     - learning_rate: {best_params.get('learning_rate', 'N/A')}")
    except FileNotFoundError:
        print(f"   ⚠ Warning: Best parameters file not found")
    except Exception as e:
        print(f"   ⚠ Warning: Could not load best parameters: {e}")
    
    # Load dataset
    print("\n2. Loading dataset...")
    try:
        df = load_dataset()
        print(f"   ✓ Dataset loaded: {len(df)} transactions")
        print(f"   Frauds: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return
    
    # Sort by Time (same as training)
    print("\n3. Sorting data by Time (for consistent train-test split)...")
    df = df.sort_values("Time").reset_index(drop=True)
    print("   ✓ Data sorted by Time")
    
    # Prepare features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # Create time-based train-test split (same as training: 80/20)
    print("\n4. Creating time-based train-test split (80/20)...")
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Train set: {len(X_train)} transactions ({y_train.sum()} frauds)")
    print(f"   Test set:  {len(X_test)} transactions ({y_test.sum()} frauds)")
    
    # Calculate PR-AUC (overall model performance)
    print("\n5. Calculating overall model performance (PR-AUC)...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"   PR-AUC: {pr_auc:.4f}")
    
    # Evaluate at multiple thresholds
    print("\n6. Evaluating model at multiple thresholds...")
    results = evaluate_model_at_thresholds(model, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Model: {config.TRAINED_MODEL_PATH}")
    print(f"Test Set Size: {len(X_test)} transactions")
    print(f"Fraud Cases in Test Set: {y_test.sum()}")
    print(f"Overall PR-AUC: {pr_auc:.4f}")
    print("\nFor detailed metrics at each threshold, see the tables above.")
    print("=" * 80)


if __name__ == "__main__":
    main()

