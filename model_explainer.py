"""
SHAP Model Explainer
Loads trained model and provides SHAP explanations for transactions
"""

import joblib
import numpy as np
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
import config
MODEL_DIR = config.MODELS_DIR
MODEL_PATH = config.TRAINED_MODEL_PATH  # Use the trained model with best parameters
CALIBRATED_MODEL_PATH = config.CALIBRATED_MODEL_PATH  # Calibrated model (better probabilities)
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
FEATURE_INFO_PATH = MODEL_DIR / 'feature_info.pkl'
BEST_PARAMS_PATH = MODEL_DIR / 'best_params.pkl'

# Global cache for model and explainer
_model_cache = None
_scaler_cache = None
_explainer_cache = None
_feature_names_cache = None
_model_info_cache = None  # Store model metadata

def load_model(use_calibrated=True):
    """Load model, scaler, and feature info (cached)
    The model should be trained with best parameters from model_training.py
    
    Args:
        use_calibrated: If True, prefer calibrated model (better probability estimates)
    
    Returns:
        model, scaler, feature_names, model_info
    """
    global _model_cache, _scaler_cache, _feature_names_cache, _model_info_cache
    
    if _model_cache is not None:
        return _model_cache, _scaler_cache, _feature_names_cache, _model_info_cache
    
    # Try to load calibrated model first (better for probability estimates)
    model_path = None
    model_type = "standard"
    
    if use_calibrated and CALIBRATED_MODEL_PATH.exists():
        model_path = CALIBRATED_MODEL_PATH
        model_type = "calibrated"
        print(f"Loading calibrated fraud detection model from {CALIBRATED_MODEL_PATH}...")
    elif MODEL_PATH.exists():
        model_path = MODEL_PATH
        model_type = "standard"
        print(f"Loading fraud detection model from {MODEL_PATH}...")
    else:
        raise FileNotFoundError(f"Model not found. Tried: {CALIBRATED_MODEL_PATH}, {MODEL_PATH}. Run model_training.py first.")
    
    _model_cache = joblib.load(model_path)
    
    # Load and store model information
    model_info = {
        "model_type": model_type,
        "model_path": str(model_path),
        "model_class": type(_model_cache).__name__
    }
    
    # Check if best parameters file exists and log them
    if BEST_PARAMS_PATH.exists():
        best_params = joblib.load(BEST_PARAMS_PATH)
        model_info["best_params"] = best_params
        print(f"✓ Model trained with best parameters:")
        print(f"  - Learning Rate: {best_params.get('learning_rate', 'N/A')}")
        print(f"  - N Estimators: {best_params.get('n_estimators', 'N/A')}")
        print(f"  - Max Depth: {best_params.get('max_depth', 'N/A')}")
    else:
        print("⚠ Best parameters file not found, but model loaded")
        model_info["best_params"] = None
    
    _model_info_cache = model_info
    
    # Load scaler if available
    if SCALER_PATH.exists():
        _scaler_cache = joblib.load(SCALER_PATH)
        print("✓ Scaler loaded")
    else:
        print("⚠ Scaler not found, features may not be scaled correctly")
        _scaler_cache = None
    
    # Load feature names
    if FEATURE_INFO_PATH.exists():
        feature_info = joblib.load(FEATURE_INFO_PATH)
        _feature_names_cache = feature_info.get('feature_names', [])
    else:
        # Fallback: assume standard feature order (V1-V28, Amount, Time)
        _feature_names_cache = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        print("⚠ Feature info not found, using default feature order")
    
    print(f"✓ Model loaded successfully ({len(_feature_names_cache)} features)")
    print(f"  Model type: {type(_model_cache).__name__} ({model_type})")
    return _model_cache, _scaler_cache, _feature_names_cache, _model_info_cache

def create_explainer(model, sample_data=None):
    """Create SHAP TreeExplainer (cached)"""
    global _explainer_cache
    
    if _explainer_cache is not None:
        return _explainer_cache
    
    print("Initializing SHAP TreeExplainer...")
    
    # For TreeExplainer, we can use the model directly
    # Optionally provide background data for better explanations
    if sample_data is not None and len(sample_data) > 0:
        # Use a sample of background data (up to 100 samples for speed)
        background_size = min(100, len(sample_data))
        background = sample_data[:background_size]
        _explainer_cache = shap.TreeExplainer(model, background)
    else:
        # Use model's expected value
        _explainer_cache = shap.TreeExplainer(model)
    
    print("✓ SHAP explainer ready")
    return _explainer_cache

def extract_features_from_request(data):
    """Extract and order features from API request"""
    _, _, feature_names, _ = load_model()
    
    # Build feature array in correct order
    features = []
    missing_features = []
    zero_features = []
    
    for feat_name in feature_names:
        # Try both exact match and case-insensitive match (v1 vs V1)
        value = None
        if feat_name in data:
            value = float(data[feat_name])
        elif feat_name.lower() in data:
            value = float(data[feat_name.lower()])
        else:
            # Default to 0 if missing (shouldn't happen in production)
            missing_features.append(feat_name)
            value = 0.0
        
        if value == 0.0 and feat_name not in ['Time']:  # Time can legitimately be 0
            zero_features.append(feat_name)
        
        features.append(value)
    
    # Warn if many features are missing or zero
    if len(missing_features) > 5:
        print(f"[WARNING] {len(missing_features)} features missing from request: {missing_features[:5]}...")
    if len(zero_features) > 20:
        print(f"[WARNING] {len(zero_features)} features are zero - this may cause incorrect predictions!")
        print(f"[WARNING] Zero features include: {zero_features[:10]}")
    
    print(f"[DEBUG] Extracted {len(features)} features")
    print(f"[DEBUG] First 10 feature values: {features[:10]}")
    print(f"[DEBUG] V14 (important): {features[13] if len(features) > 13 else 'N/A'}, V17: {features[16] if len(features) > 16 else 'N/A'}, Amount: {features[28] if len(features) > 28 else 'N/A'}")
    
    return np.array(features).reshape(1, -1)

def explain_transaction(features_array, top_n=5):
    """
    Generate SHAP explanation for a transaction
    
    Args:
        features_array: numpy array of shape (1, n_features) - already scaled
        top_n: Number of top features to return
    
    Returns:
        List of dicts with feature, value, shap_value, importance
    """
    model, scaler, feature_names, _ = load_model()
    explainer = create_explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(features_array)
    
    # Handle binary classification (SHAP returns array for each class)
    if isinstance(shap_values, list):
        # Use fraud class (index 1) SHAP values
        shap_values = shap_values[1]
    
    # Get feature contributions
    shap_array = shap_values[0]  # First (and only) sample
    feature_values = features_array[0]  # Original feature values
    
    # Create explanation list
    explanations = []
    for i, (feat_name, shap_val, feat_val) in enumerate(zip(feature_names, shap_array, feature_values)):
        explanations.append({
            'feature': feat_name,
            'value': float(feat_val),
            'shap_value': float(shap_val),
            'importance': float(abs(shap_val)),  # Absolute value for importance
            'contribution': 'positive' if shap_val > 0 else 'negative',
            'description': _get_feature_description(feat_name)
        })
    
    # Sort by absolute SHAP value (importance) and return top N
    explanations.sort(key=lambda x: x['importance'], reverse=True)
    return explanations[:top_n]

def _get_feature_description(feature_name):
    """Get human-readable description for feature"""
    descriptions = {
        'V14': 'Transaction behavior pattern (negative = suspicious)',
        'V17': 'Identity verification signal (negative = anomaly)',
        'V12': 'Transaction velocity indicator',
        'V10': 'Amount pattern analysis',
        'V4': 'Transaction type classification',
        'Amount': 'Transaction amount',
        'Time': 'Time since first transaction (seconds)'
    }
    
    if feature_name in descriptions:
        return descriptions[feature_name]
    elif feature_name.startswith('V'):
        return f'PCA component {feature_name} (anonymized feature)'
    else:
        return f'{feature_name} feature'

def predict_fraud_probability(features_array, use_calibrated=True):
    """
    Predict fraud probability for a transaction using the trained model with best parameters
    
    Args:
        features_array: numpy array of shape (1, n_features) - already scaled
        use_calibrated: If True, use calibrated model (better probability estimates)
    
    Returns:
        fraud_probability (float), is_fraud (bool), model_info (dict)
    """
    model, scaler, _, model_info = load_model(use_calibrated=use_calibrated)
    
    # Ensure features are properly shaped
    if len(features_array.shape) == 1:
        features_array = features_array.reshape(1, -1)
    
    print(f"[DEBUG] Predicting with model - input shape: {features_array.shape}")
    
    # Predict probability using the trained model
    # This model was trained with best parameters from Optuna optimization
    try:
        proba = model.predict_proba(features_array)[0]
        
        # Ensure we have valid probabilities
        if len(proba) < 2:
            print(f"[ERROR] Model returned invalid probability array: {proba}")
            raise ValueError("Model prediction returned invalid format")
        
        fraud_prob = float(proba[1])  # Probability of fraud class (class 1)
        legit_prob = float(proba[0])  # Probability of legitimate class (class 0)
        
        # Validate probability is reasonable
        if fraud_prob < 0 or fraud_prob > 1:
            print(f"[ERROR] Invalid fraud probability: {fraud_prob}, clamping to [0, 1]")
            fraud_prob = max(0.0, min(1.0, fraud_prob))
        
        # Check if probabilities sum to ~1.0 (with small tolerance)
        prob_sum = fraud_prob + legit_prob
        if abs(prob_sum - 1.0) > 0.01:
            print(f"[WARNING] Probabilities don't sum to 1.0: {prob_sum}")
        
        print(f"[DEBUG] Raw model output - probabilities: [legit={legit_prob:.6f}, fraud={fraud_prob:.6f}], sum={prob_sum:.6f}")
        
        # Predict class (using default threshold of 0.5)
        prediction = model.predict(features_array)[0]
        is_fraud = bool(prediction == 1)
        
        print(f"[DEBUG] Model prediction - class: {prediction}, is_fraud: {is_fraud}, fraud_prob: {fraud_prob:.6f}")
        
        # Additional validation: if fraud_prob is exactly 0, check if features are all zeros
        if fraud_prob == 0.0:
            non_zero_count = np.count_nonzero(features_array)
            if non_zero_count == 0:
                print(f"[WARNING] All features are zero! This will always predict 0% fraud.")
            else:
                print(f"[INFO] Model predicted 0% fraud with {non_zero_count} non-zero features")
        
        # Add prediction details to model_info
        model_info["prediction_details"] = {
            "fraud_probability": fraud_prob,
            "legitimate_probability": legit_prob,
            "predicted_class": int(prediction),
            "is_fraud": is_fraud,
            "model_type": model_info.get("model_type", "unknown")
        }
        
        return fraud_prob, is_fraud, model_info
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        # Return a safe default instead of crashing
        model_info["prediction_details"] = {
            "error": str(e),
            "fraud_probability": 0.0,
            "is_fraud": False
        }
        return 0.0, False, model_info

def explain_transaction_from_request(data):
    """
    Complete pipeline: extract features, scale, predict, explain
    Uses the trained model with best parameters from model_training.py
    
    Args:
        data: Dict with feature values (from API request)
            Should contain: amount, time, v1-v28 (or V1-V28)
    
    Returns:
        dict with fraud_probability, is_fraud, shap_explanation
        - fraud_probability: The actual model prediction probability (0-1)
        - is_fraud: Boolean prediction based on 0.5 threshold
        - shap_explanation: List of top contributing features
    """
    # Load model (trained with best parameters from Optuna)
    # Use calibrated model if available (better probability estimates)
    model, scaler, feature_names, model_info = load_model(use_calibrated=True)
    
    # Extract features from request (V1-V28, Amount, Time)
    features_raw = extract_features_from_request(data)
    
    print(f"[DEBUG] Extracted {len(features_raw[0])} features from request")
    print(f"[DEBUG] Sample feature values - V1: {features_raw[0][0]:.4f}, V14: {features_raw[0][13]:.4f}, Amount: {features_raw[0][28]:.2f}")
    
    # Scale features using the same scaler used during training
    if scaler is not None:
        features_scaled = scaler.transform(features_raw)
        print(f"[DEBUG] Features scaled using training scaler")
    else:
        # If no scaler, use raw features (not recommended but handle gracefully)
        print("[WARNING] No scaler found, using raw features (may affect prediction accuracy)")
        features_scaled = features_raw
    
    print(f"[DEBUG] Scaled sample values - V1: {features_scaled[0][0]:.4f}, V14: {features_scaled[0][13]:.4f}, Amount: {features_scaled[0][28]:.4f}")
    
    # Predict using the trained model with best parameters (calibrated if available)
    # This is the actual model output from the XGBoost model trained with Optuna-optimized parameters
    fraud_prob, is_fraud, prediction_info = predict_fraud_probability(features_scaled, use_calibrated=True)
    
    print(f"[DEBUG] Final prediction - fraud_probability: {fraud_prob:.6f}, is_fraud: {is_fraud}")
    print(f"[DEBUG] Model used: {model_info.get('model_type', 'unknown')} ({model_info.get('model_class', 'unknown')})")
    
    # Generate SHAP explanation
    try:
        shap_explanation = explain_transaction(features_scaled, top_n=5)
    except Exception as e:
        print(f"[WARNING] SHAP explanation failed: {e}")
        shap_explanation = []
    
    return {
        'fraud_probability': fraud_prob,  # This is the actual model prediction probability
        'is_fraud': is_fraud,
        'shap_explanation': shap_explanation,
        'model_info': {
            'model_type': model_info.get('model_type', 'unknown'),
            'model_class': model_info.get('model_class', 'unknown'),
            'best_params': model_info.get('best_params'),
            'prediction_details': prediction_info.get('prediction_details', {})
        }
    }

# Initialize on import (optional - can be lazy loaded)
try:
    load_model()
except FileNotFoundError:
    print("⚠ Model not found. Run train_model.py first.")
except Exception as e:
    print(f"⚠ Error loading model: {e}")

