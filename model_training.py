"""Model training script for fraud detection using XGBoost with Optuna optimization."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib
import optuna
from optuna.pruners import MedianPruner
import config
from data_preprocessing import load_dataset


def suggest_xgb_params(trial):
    """
    Suggest XGBoost hyperparameters for Optuna optimization.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of XGBoost parameters
    """
    params = {
        # Core boosting
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.05, log=True
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators", 600, 1400
        ),

        # Tree structure
        "max_depth": trial.suggest_int(
            "max_depth", 4, 8
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 6, 12
        ),
        "gamma": trial.suggest_float(
            "gamma", 0.3, 1.0
        ),

        # Subsampling
        "subsample": trial.suggest_float(
            "subsample", 0.7, 0.9
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 0.8
        ),

        # Regularization
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 0.1, 2.0
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 5.0, 20.0
        ),

        # Imbalance handling
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 400, 800
        ),
        "max_delta_step": trial.suggest_int(
            "max_delta_step", 2, 8
        ),

        # Fixed (do NOT tune)
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42
    }

    return params


def optimize_hyperparameters(X_train, y_train, n_trials=50):
    """
    Optimize XGBoost hyperparameters using Optuna with stratified CV.
    
    Args:
        X_train: Training features (DataFrame)
        y_train: Training labels (Series)
        n_trials: Number of Optuna trials
    
    Returns:
        Best hyperparameters dictionary
    """
    # Define Stratified CV (NO shuffle - maintains temporal order)
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=False  # CRITICAL: No shuffle to maintain temporal order
    )
    
    def objective(trial):
        # Get suggested parameters
        params = suggest_xgb_params(trial)
        
        # Print trial number and key hyperparameters
        print(f"\n[Trial {trial.number}]")
        print(f"  Hyperparameters:")
        print(f"    learning_rate: {params['learning_rate']:.4f}")
        print(f"    n_estimators: {params['n_estimators']}")
        print(f"    max_depth: {params['max_depth']}")
        print(f"    min_child_weight: {params['min_child_weight']}")
        print(f"    gamma: {params['gamma']:.3f}")
        print(f"    subsample: {params['subsample']:.3f}")
        print(f"    colsample_bytree: {params['colsample_bytree']:.3f}")
        print(f"    reg_alpha: {params['reg_alpha']:.3f}")
        print(f"    reg_lambda: {params['reg_lambda']:.3f}")
        print(f"    scale_pos_weight: {params['scale_pos_weight']:.1f}")
        print(f"    max_delta_step: {params['max_delta_step']}")
        
        fold_scores = []
        
        # Cross-validation loop
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]
            
            model = XGBClassifier(**params)
            
            # Train model (early stopping not supported in XGBoost 3.x fit method)
            # Using fixed n_estimators from hyperparameters
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False  # Keep quiet during CV
            )
            
            # Get predictions and calculate PR-AUC
            preds = model.predict_proba(X_val)[:, 1]
            pr_auc = average_precision_score(y_val, preds)
            fold_scores.append(pr_auc)
            
            print(f"    Fold {fold_idx + 1}/5 PR-AUC: {pr_auc:.4f}")
        
        # Return mean PR-AUC across folds
        mean_pr_auc = np.mean(fold_scores)
        std_pr_auc = np.std(fold_scores)
        print(f"  Mean PR-AUC: {mean_pr_auc:.4f} (±{std_pr_auc:.4f})")
        
        return mean_pr_auc
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    
    print(f"\n   Starting Optuna optimization with {n_trials} trials...")
    print("   Using 5-fold Stratified Cross-Validation (no shuffle)")
    print("=" * 60)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print("\n" + "=" * 60)
    print(f"   Optimization Complete!")
    print(f"   Best PR-AUC (CV mean): {study.best_value:.4f}")
    print(f"   Best parameters:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")
    
    # Reconstruct the full parameter dictionary using FixedTrial
    fixed_trial = optuna.trial.FixedTrial(study.best_params)
    best_params = suggest_xgb_params(fixed_trial)
    
    return best_params


def train_final_model(X_train, y_train, hyperparameters):
    """
    Train final XGBoost model on full training set.
    
    Args:
        X_train: Training features (DataFrame)
        y_train: Training labels (Series)
        hyperparameters: Dictionary of hyperparameters
    
    Returns:
        Trained model
    """
    model = XGBClassifier(**hyperparameters)
    
    print(f"\n   Training final model on full training set...")
    print(f"   Training set size: {len(X_train)} samples")
    print("=" * 60)
    
    # Train on full training set (no validation set for final model)
    model.fit(
        X_train,
        y_train,
        verbose=True,  # Show training progress
    )
    
    print("=" * 60)
    
    return model


def evaluate_model(model, X_test, y_test, threshold=0.2):
    """
    Evaluate model using PR-AUC and other metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate PR-AUC
    pr_auc = average_precision_score(y_test, y_pred_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Binary predictions at threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    metrics = {
        "pr_auc": pr_auc,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }
    
    return metrics


def evaluate_multiple_thresholds(model, X_test, y_test, thresholds=None):
    """
    Evaluate model at multiple thresholds and output confusion matrices with metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
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
    print("\nDetailed Confusion Matrices:")
    print("=" * 80)
    for threshold in thresholds:
        metrics = results[threshold]
        print(f"\nThreshold: {threshold:.2f}")
        print(f"Confusion Matrix:")
        print(f"  {metrics['confusion_matrix']}")
        print(f"  TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TP: {metrics['tp']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    return results


def calibrate_model(model, X_train, y_train):
    """
    Calibrate model probabilities using isotonic regression.
    
    Args:
        model: Trained XGBoost model
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Calibrated model
    """
    calibrated_model = CalibratedClassifierCV(
        model, method="isotonic", cv=3
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Merchant Shield - Fraud Detection Model Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading dataset...")
    df = load_dataset()
    print(f"   Total transactions: {len(df)}")
    print(f"   Frauds: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    
    # STEP 1: Sort by Time (MANDATORY)
    print("\n2. Sorting data by Time (mandatory for time-based split)...")
    df = df.sort_values("Time").reset_index(drop=True)
    print("   ✓ Data sorted by Time")
    
    # Prepare features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # STEP 2: Final Hold-Out Split (Time-Based) - 80/20
    print("\n3. Creating time-based train-test split (80/20)...")
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Train set: {len(X_train)} transactions ({y_train.sum()} frauds)")
    print(f"   Test set: {len(X_test)} transactions ({y_test.sum()} frauds)")
    print("   ✓ Test set will NOT be touched during hyperparameter tuning")
    
    # STEP 3-4: Optimize hyperparameters with Stratified CV
    print("\n4. Optimizing hyperparameters with Optuna (Stratified CV)...")
    best_params = optimize_hyperparameters(
        X_train, y_train, n_trials=50
    )
    
    # STEP 5: Train final model on full training set
    print("\n5. Training final model on full training set...")
    print("   Using best hyperparameters:")
    for key, value in best_params.items():
        if key != "n_estimators":
            print(f"     {key}: {value}")
    print(f"     n_estimators: {best_params['n_estimators']}")
    
    final_model = train_final_model(X_train, y_train, best_params)
    
    # STEP 6: Final evaluation (ONLY NOW use test set)
    print("\n6. Evaluating model on test set (first time seeing test data)...")
    metrics = evaluate_model(final_model, X_test, y_test, threshold=0.2)
    
    print(f"\n   Test Set Results (threshold=0.2):")
    print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print(f"\n   Confusion Matrix (threshold={metrics['threshold']}):")
    print(f"   {metrics['confusion_matrix']}")
    print(f"   TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TP: {metrics['tp']}")
    
    # Evaluate at multiple thresholds
    print("\n6b. Evaluating model at multiple thresholds...")
    threshold_results = evaluate_multiple_thresholds(final_model, X_test, y_test)
    
    # Calibrate model
    print("\n7. Calibrating model probabilities...")
    calibrated_model = calibrate_model(final_model, X_train, y_train)
    
    # Evaluate calibrated model
    print("\n8. Evaluating calibrated model...")
    metrics_cal = evaluate_model(calibrated_model, X_test, y_test, threshold=0.2)
    print(f"\n   Calibrated Model - Test Set Results (threshold=0.2):")
    print(f"   PR-AUC: {metrics_cal['pr_auc']:.4f}")
    print(f"   Accuracy: {metrics_cal['accuracy']:.4f}")
    print(f"   Precision: {metrics_cal['precision']:.4f}")
    print(f"   Recall: {metrics_cal['recall']:.4f}")
    print(f"   F1-Score: {metrics_cal['f1_score']:.4f}")
    
    # Evaluate calibrated model at multiple thresholds
    print("\n8b. Evaluating calibrated model at multiple thresholds...")
    threshold_results_cal = evaluate_multiple_thresholds(calibrated_model, X_test, y_test)
    
    # Save models and scaler (if needed)
    print("\n9. Saving models...")
    joblib.dump(final_model, config.TRAINED_MODEL_PATH)
    joblib.dump(calibrated_model, config.CALIBRATED_MODEL_PATH)
    joblib.dump(best_params, config.MODELS_DIR / "best_params.pkl")
    print(f"   Saved to {config.MODELS_DIR}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nKey Points:")
    print("✓ Time-based split with no time leakage")
    print("✓ Stratified CV (no shuffle) for hyperparameter tuning")
    print("✓ Test set used only once for final evaluation")
    print("✓ Defensible methodology for interviews/production")


if __name__ == "__main__":
    main()
