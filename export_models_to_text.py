"""Export trained models to human-readable text format."""

import joblib
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import config
import numpy as np


def export_xgb_model_to_text(model, output_file, model_name="XGBoost Model"):
    """
    Export XGBoost model information to readable text format.
    
    Args:
        model: XGBoost model object
        output_file: Path to output text file
        model_name: Name of the model for header
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Model Parameters
        f.write("MODEL PARAMETERS\n")
        f.write("-" * 80 + "\n")
        params = model.get_params()
        for key, value in sorted(params.items()):
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Feature Names (if available)
        try:
            feature_names = model.feature_names_in_
            f.write("FEATURE NAMES\n")
            f.write("-" * 80 + "\n")
            for i, name in enumerate(feature_names):
                f.write(f"Feature {i}: {name}\n")
            f.write(f"\nTotal Features: {len(feature_names)}\n\n")
        except AttributeError:
            f.write("FEATURE NAMES: Not available\n\n")
        
        # Feature Importance
        f.write("FEATURE IMPORTANCE\n")
        f.write("-" * 80 + "\n")
        try:
            importance = model.feature_importances_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            
            # Create list of (importance, feature_name) tuples
            if feature_names is not None:
                importance_list = [(imp, name) for imp, name in zip(importance, feature_names)]
            else:
                importance_list = [(imp, f"Feature_{i}") for i, imp in enumerate(importance)]
            
            # Sort by importance (descending)
            importance_list.sort(reverse=True, key=lambda x: x[0])
            
            f.write(f"{'Rank':<6} {'Feature Name':<20} {'Importance':<15}\n")
            f.write("-" * 80 + "\n")
            for rank, (imp, name) in enumerate(importance_list, 1):
                f.write(f"{rank:<6} {name:<20} {imp:<15.6f}\n")
            f.write("\n")
        except Exception as e:
            f.write(f"Error extracting feature importance: {e}\n\n")
        
        # Model Booster Information
        f.write("BOOSTER INFORMATION\n")
        f.write("-" * 80 + "\n")
        try:
            booster = model.get_booster()
            f.write(f"Number of trees: {booster.num_boosted_rounds()}\n")
            f.write(f"Number of features: {booster.num_feature()}\n")
            f.write(f"Number of classes: {booster.num_class()}\n")
            f.write("\n")
        except Exception as e:
            f.write(f"Error extracting booster info: {e}\n\n")
        
        # Tree Structures (first 5 trees as example)
        f.write("TREE STRUCTURES (First 5 trees as example)\n")
        f.write("-" * 80 + "\n")
        try:
            trees = model.get_booster().get_dump(with_stats=True)
            num_trees_to_show = min(5, len(trees))
            f.write(f"Showing first {num_trees_to_show} of {len(trees)} trees:\n\n")
            
            for i in range(num_trees_to_show):
                f.write(f"\n{'='*80}\n")
                f.write(f"TREE {i+1} of {len(trees)}\n")
                f.write(f"{'='*80}\n")
                f.write(trees[i])
                f.write("\n")
            
            f.write(f"\n... ({len(trees) - num_trees_to_show} more trees)\n")
            f.write("(To see all trees, check the full tree dump file)\n\n")
        except Exception as e:
            f.write(f"Error extracting tree structures: {e}\n\n")
        
        # Save all trees to separate file
        try:
            tree_file = output_file.replace('.txt', '_all_trees.txt')
            with open(tree_file, 'w', encoding='utf-8') as tf:
                tf.write(f"Complete Tree Dump for {model_name}\n")
                tf.write("=" * 80 + "\n\n")
                trees = model.get_booster().get_dump(with_stats=True)
                for i, tree in enumerate(trees):
                    tf.write(f"\n{'='*80}\n")
                    tf.write(f"TREE {i+1} of {len(trees)}\n")
                    tf.write(f"{'='*80}\n")
                    tf.write(tree)
                    tf.write("\n")
            f.write(f"Complete tree dump saved to: {tree_file}\n")
        except Exception as e:
            f.write(f"Error saving complete tree dump: {e}\n")


def export_calibrated_model_to_text(calibrated_model, output_file, model_name="Calibrated Model"):
    """
    Export calibrated model information to readable text format.
    
    Args:
        calibrated_model: CalibratedClassifierCV model object
        output_file: Path to output text file
        model_name: Name of the model for header
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Calibration Information
        f.write("CALIBRATION INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Calibration Method: {calibrated_model.method}\n")
        f.write(f"Number of CV Folds: {calibrated_model.cv}\n")
        f.write("\n")
        
        # Base Estimator Information
        f.write("BASE ESTIMATOR (XGBoost Model)\n")
        f.write("-" * 80 + "\n")
        # Get base estimator from calibrated classifiers
        try:
            if hasattr(calibrated_model, 'calibrated_classifiers_') and len(calibrated_model.calibrated_classifiers_) > 0:
                base_estimator = calibrated_model.calibrated_classifiers_[0].base_estimator
            elif hasattr(calibrated_model, 'base_estimator'):
                base_estimator = calibrated_model.base_estimator
            elif hasattr(calibrated_model, 'estimator'):
                base_estimator = calibrated_model.estimator
            else:
                raise AttributeError("Cannot find base estimator")
        except Exception as e:
            f.write(f"Error accessing base estimator: {e}\n\n")
            return
        
        # Model Parameters
        params = base_estimator.get_params()
        for key, value in sorted(params.items()):
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Feature Names
        try:
            feature_names = base_estimator.feature_names_in_
            f.write("FEATURE NAMES\n")
            f.write("-" * 80 + "\n")
            for i, name in enumerate(feature_names):
                f.write(f"Feature {i}: {name}\n")
            f.write(f"\nTotal Features: {len(feature_names)}\n\n")
        except AttributeError:
            f.write("FEATURE NAMES: Not available\n\n")
        
        # Feature Importance
        f.write("FEATURE IMPORTANCE\n")
        f.write("-" * 80 + "\n")
        try:
            importance = base_estimator.feature_importances_
            feature_names = base_estimator.feature_names_in_ if hasattr(base_estimator, 'feature_names_in_') else None
            
            if feature_names is not None:
                importance_list = [(imp, name) for imp, name in zip(importance, feature_names)]
            else:
                importance_list = [(imp, f"Feature_{i}") for i, imp in enumerate(importance)]
            
            importance_list.sort(reverse=True, key=lambda x: x[0])
            
            f.write(f"{'Rank':<6} {'Feature Name':<20} {'Importance':<15}\n")
            f.write("-" * 80 + "\n")
            for rank, (imp, name) in enumerate(importance_list, 1):
                f.write(f"{rank:<6} {name:<20} {imp:<15.6f}\n")
            f.write("\n")
        except Exception as e:
            f.write(f"Error extracting feature importance: {e}\n\n")
        
        # Calibrated Classifiers Info
        f.write("CALIBRATED CLASSIFIERS\n")
        f.write("-" * 80 + "\n")
        try:
            if hasattr(calibrated_model, 'calibrated_classifiers_'):
                f.write(f"Number of calibrated estimators: {len(calibrated_model.calibrated_classifiers_)}\n")
                for i, cal_clf in enumerate(calibrated_model.calibrated_classifiers_):
                    f.write(f"\nCalibrated Classifier {i+1}:\n")
                    if hasattr(cal_clf, 'base_estimator'):
                        f.write(f"  Base Estimator Type: {type(cal_clf.base_estimator).__name__}\n")
                    if hasattr(cal_clf, 'calibrator'):
                        f.write(f"  Calibrator Type: {type(cal_clf.calibrator).__name__}\n")
                        if hasattr(cal_clf.calibrator, 'X_'):
                            f.write(f"  Calibration Samples: {len(cal_clf.calibrator.X_)}\n")
            else:
                f.write("Calibrated classifiers information not available\n")
        except Exception as e:
            f.write(f"Error extracting calibrated classifiers info: {e}\n\n")
        
        # Tree Structures (first 5 trees)
        f.write("\nTREE STRUCTURES (First 5 trees as example)\n")
        f.write("-" * 80 + "\n")
        try:
            trees = base_estimator.get_booster().get_dump(with_stats=True)
            num_trees_to_show = min(5, len(trees))
            f.write(f"Showing first {num_trees_to_show} of {len(trees)} trees:\n\n")
            
            for i in range(num_trees_to_show):
                f.write(f"\n{'='*80}\n")
                f.write(f"TREE {i+1} of {len(trees)}\n")
                f.write(f"{'='*80}\n")
                f.write(trees[i])
                f.write("\n")
            
            f.write(f"\n... ({len(trees) - num_trees_to_show} more trees)\n")
            f.write("(To see all trees, check the full tree dump file)\n\n")
        except Exception as e:
            f.write(f"Error extracting tree structures: {e}\n\n")
        
        # Save all trees to separate file
        try:
            tree_file = output_file.replace('.txt', '_all_trees.txt')
            with open(tree_file, 'w', encoding='utf-8') as tf:
                tf.write(f"Complete Tree Dump for {model_name}\n")
                tf.write("=" * 80 + "\n\n")
                trees = base_estimator.get_booster().get_dump(with_stats=True)
                for i, tree in enumerate(trees):
                    tf.write(f"\n{'='*80}\n")
                    tf.write(f"TREE {i+1} of {len(trees)}\n")
                    tf.write(f"{'='*80}\n")
                    tf.write(tree)
                    tf.write("\n")
            f.write(f"Complete tree dump saved to: {tree_file}\n")
        except Exception as e:
            f.write(f"Error saving complete tree dump: {e}\n")


def export_best_params_to_text(best_params, output_file):
    """Export best hyperparameters to text format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BEST HYPERPARAMETERS (from Optuna Optimization)\n")
        f.write("=" * 80 + "\n\n")
        
        for key, value in sorted(best_params.items()):
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("JSON Format:\n")
        f.write("=" * 80 + "\n")
        f.write(json.dumps(best_params, indent=2, default=str))


def main():
    """Main function to export all models to text format."""
    print("=" * 80)
    print("Exporting Models to Text Format")
    print("=" * 80)
    
    output_dir = config.MODELS_DIR / "text_exports"
    output_dir.mkdir(exist_ok=True)
    
    # Export base XGBoost model
    print("\n1. Exporting base XGBoost model...")
    try:
        model = joblib.load(config.TRAINED_MODEL_PATH)
        output_file = output_dir / "xgb_fraud_model.txt"
        export_xgb_model_to_text(model, str(output_file), "XGBoost Fraud Detection Model")
        print(f"   ✓ Saved to: {output_file}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Export calibrated model
    print("\n2. Exporting calibrated model...")
    try:
        calibrated_model = joblib.load(config.CALIBRATED_MODEL_PATH)
        output_file = output_dir / "xgb_fraud_model_calibrated.txt"
        export_calibrated_model_to_text(calibrated_model, str(output_file), "Calibrated XGBoost Fraud Detection Model")
        print(f"   ✓ Saved to: {output_file}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Export best parameters
    print("\n3. Exporting best hyperparameters...")
    try:
        best_params = joblib.load(config.MODELS_DIR / "best_params.pkl")
        output_file = output_dir / "best_params.txt"
        export_best_params_to_text(best_params, str(output_file))
        print(f"   ✓ Saved to: {output_file}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print(f"\nAll text exports saved to: {output_dir}")
    print("\nFiles created:")
    print("  - xgb_fraud_model.txt (model summary)")
    print("  - xgb_fraud_model_all_trees.txt (complete tree dump)")
    print("  - xgb_fraud_model_calibrated.txt (calibrated model summary)")
    print("  - xgb_fraud_model_calibrated_all_trees.txt (complete tree dump)")
    print("  - best_params.txt (hyperparameters)")


if __name__ == "__main__":
    main()

