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
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_PATH = MODEL_DIR / 'fraud_detector.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
FEATURE_INFO_PATH = MODEL_DIR / 'feature_info.pkl'

# Global cache for model and explainer
_model_cache = None
_scaler_cache = None
_explainer_cache = None
_feature_names_cache = None

def load_model():
    """Load model, scaler, and feature info (cached)"""
    global _model_cache, _scaler_cache, _feature_names_cache
    
    if _model_cache is not None:
        return _model_cache, _scaler_cache, _feature_names_cache
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
    
    print("Loading fraud detection model...")
    _model_cache = joblib.load(MODEL_PATH)
    _scaler_cache = joblib.load(SCALER_PATH)
    
    if FEATURE_INFO_PATH.exists():
        feature_info = joblib.load(FEATURE_INFO_PATH)
        _feature_names_cache = feature_info.get('feature_names', [])
    else:
        # Fallback: assume standard feature order
        _feature_names_cache = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    
    print(f"✓ Model loaded ({len(_feature_names_cache)} features)")
    return _model_cache, _scaler_cache, _feature_names_cache

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
    feature_names = load_model()[2]
    
    # Build feature array in correct order
    features = []
    for feat_name in feature_names:
        if feat_name in data:
            features.append(float(data[feat_name]))
        else:
            # Default to 0 if missing (shouldn't happen in production)
            features.append(0.0)
    
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
    model, scaler, feature_names = load_model()
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

def predict_fraud_probability(features_array):
    """
    Predict fraud probability for a transaction
    
    Args:
        features_array: numpy array of shape (1, n_features) - already scaled
    
    Returns:
        fraud_probability (float), is_fraud (bool)
    """
    model, scaler, _ = load_model()
    
    # Predict probability
    proba = model.predict_proba(features_array)[0]
    fraud_prob = proba[1]  # Probability of fraud class
    
    # Predict class
    prediction = model.predict(features_array)[0]
    is_fraud = bool(prediction == 1)
    
    return float(fraud_prob), is_fraud

def explain_transaction_from_request(data):
    """
    Complete pipeline: extract features, scale, predict, explain
    
    Args:
        data: Dict with feature values (from API request)
    
    Returns:
        dict with fraud_probability, is_fraud, shap_explanation
    """
    # Extract and scale features
    model, scaler, feature_names = load_model()
    features_raw = extract_features_from_request(data)
    features_scaled = scaler.transform(features_raw)
    
    # Predict
    fraud_prob, is_fraud = predict_fraud_probability(features_scaled)
    
    # Explain
    shap_explanation = explain_transaction(features_scaled, top_n=5)
    
    return {
        'fraud_probability': fraud_prob,
        'is_fraud': is_fraud,
        'shap_explanation': shap_explanation
    }

# Initialize on import (optional - can be lazy loaded)
try:
    load_model()
except FileNotFoundError:
    print("⚠ Model not found. Run train_model.py first.")
except Exception as e:
    print(f"⚠ Error loading model: {e}")

