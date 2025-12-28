"""
The Merchant Shield - Fraud Detection API
A lightweight fraud prevention microservice for small e-commerce businesses
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from datetime import datetime
import random
import json
import threading
import time

# Try to import data processor for real dataset
try:
    from data_processor import (
        build_fraud_network, 
        get_temporal_stats, 
        get_sample_transactions,
        load_dataset
    )
    DATA_AVAILABLE = True
    print("✓ Data processor loaded successfully")
except ImportError as e:
    DATA_AVAILABLE = False
    print(f"⚠ Data processor not available: {e}")

# Try to import ML model
try:
    from model_explainer import (
        explain_transaction_from_request,
        load_model,
        predict_fraud_probability,
        extract_features_from_request
    )
    MODEL_AVAILABLE = True
    print("✓ ML model loaded successfully")
    # Pre-load model on startup
    load_model()
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"⚠ ML model not available: {e}")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"⚠ Error loading ML model: {e}")

# Try to import evaluation and loss calculation modules
try:
    import joblib
    import config
    from eval import evaluate_model_at_thresholds
    from data_preprocessing import load_dataset as load_dataset_preprocessing
    from loss import compute_optimal_threshold
    EVAL_MODULES_AVAILABLE = True
    print("✓ Model evaluation modules loaded successfully")
except ImportError as e:
    EVAL_MODULES_AVAILABLE = False
    print(f"⚠ Model evaluation modules not available: {e}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fraud-detection-secret-key-change-in-production'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ============================================================================
# MOCK DATA - Will be replaced with actual ML model predictions
# ============================================================================

def generate_mock_transaction():
    """Generate a mock transaction for demo purposes"""
    amounts = [49.99, 129.00, 88.50, 203.45, 450.00, 67.30, 1250.00, 9999.00, 15.99, 89.00]
    statuses = ["approved", "approved", "approved", "flagged", "blocked"]
    
    fraud_prob = random.uniform(0.05, 0.95)
    if fraud_prob > 0.7:
        status = "blocked"
        risk_level = "HIGH"
    elif fraud_prob > 0.4:
        status = "flagged"
        risk_level = "MEDIUM"
    else:
        status = "approved"
        risk_level = "LOW"
    
    return {
        "id": f"TXN_{random.randint(100000, 999999)}",
        "amount": random.choice(amounts),
        "timestamp": datetime.now().isoformat(),
        "fraud_probability": round(fraud_prob, 3),
        "risk_level": risk_level,
        "status": status,
        "customer_id": f"CUS_{random.randint(10000, 99999)}",
        "merchant_id": f"MER_{random.randint(1000, 9999)}"
    }

# In-memory transaction store (replace with DB in production)
TRANSACTIONS = []

# Pre-populate with some transactions
for _ in range(25):
    TRANSACTIONS.append(generate_mock_transaction())

# ============================================================================
# ROUTES - Dashboard Views
# ============================================================================

@app.route('/')
def dashboard():
    """Main dashboard view"""
    return render_template('dashboard.html')

@app.route('/audit')
def audit_log():
    """Admin audit log for reviewing flagged transactions"""
    return render_template('audit.html')

@app.route('/analyze')
def analyze():
    """Transaction analysis interface"""
    return render_template('analyze.html')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/analyze-risk', methods=['POST'])
def analyze_risk():
    """
    POST /api/analyze-risk
    
    Analyzes a transaction and returns fraud risk assessment using ML model.
    
    Request body:
    {
        "amount": float,
        "time": int (seconds from start of period),
        "v1": float, "v2": float, ..., "v28": float
    }
    
    Response:
    {
        "fraud_probability": float,
        "is_fraud": bool,
        "risk_level": "LOW" | "MEDIUM" | "HIGH",
        "temporal_context": {...},
        "explanation": {...},
        "recommendation": string,
        "shap_explanation": [...]
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Use ML model if available, otherwise fallback to mock
    if MODEL_AVAILABLE:
        try:
            # Get real prediction and SHAP explanation
            result = explain_transaction_from_request(data)
            fraud_prob = result['fraud_probability']
            is_fraud = result['is_fraud']
            shap_explanation = result['shap_explanation']
            
            # Format SHAP explanation for response
            explanation_list = [
                {
                    "feature": exp['feature'],
                    "value": exp['value'],
                    "importance": exp['importance'],
                    "contribution": exp['contribution'],
                    "description": exp['description']
                }
                for exp in shap_explanation
            ]
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            # Fallback to mock
            fraud_prob = random.uniform(0.05, 0.95)
            is_fraud = fraud_prob > 0.5
            explanation_list = [
                {
                    "feature": "Error",
                    "value": 0,
                    "importance": 0,
                    "contribution": "neutral",
                    "description": f"Model error: {str(e)}"
                }
            ]
    else:
        # Fallback to mock prediction
        fraud_prob = random.uniform(0.05, 0.95)
        is_fraud = fraud_prob > 0.5
        explanation_list = [
            {
                "feature": "Mock",
                "value": 0,
                "importance": 0,
                "contribution": "neutral",
                "description": "ML model not available - using mock data"
            }
        ]
    
    # Determine risk level
    if fraud_prob > 0.70:
        risk_level = "HIGH"
        recommendation = "BLOCK - High confidence fraud detected"
    elif fraud_prob > 0.40:
        risk_level = "MEDIUM"
        recommendation = "REVIEW - Transaction requires additional verification"
    else:
        risk_level = "LOW"
        recommendation = "APPROVE - Transaction appears legitimate"
    
    # Build response with temporal context (placeholder)
    response = {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "temporal_context": {
            "recent_fraud_count": random.randint(0, 10),
            "time_since_last_fraud": random.randint(60, 7200),
            "fraud_contagion_score": round(random.uniform(0, 1), 3),
            "transaction_intensity_ratio": round(random.uniform(0.5, 5), 2),
            "current_time_bucket_risk": random.choice(["LOW", "MODERATE", "HIGH", "CRITICAL"])
        },
        "explanation": {
            "primary_factors": [exp['feature'] for exp in explanation_list[:3]]
        },
        "shap_explanation": explanation_list,
        "recommendation": recommendation
    }
    
    # Log transaction
    transaction = {
        "id": f"TXN_{random.randint(100000, 999999)}",
        "amount": data.get("amount", 0),
        "timestamp": datetime.now().isoformat(),
        **response
    }
    TRANSACTIONS.insert(0, transaction)
    
    # Emit via WebSocket if fraud detected
    if is_fraud and fraud_prob > 0.7:
        socketio.emit('fraud_alert', {
            'transaction_id': transaction['id'],
            'amount': transaction['amount'],
            'fraud_probability': fraud_prob,
            'timestamp': transaction['timestamp']
        })
    
    # Emit new transaction via WebSocket
    socketio.emit('new_transaction', transaction)
    
    return jsonify(response)

@app.route('/api/transactions')
def get_transactions():
    """Get recent transactions for the dashboard"""
    limit = request.args.get('limit', 20, type=int)
    status_filter = request.args.get('status', None)
    
    result = TRANSACTIONS[:limit]
    
    if status_filter:
        result = [t for t in result if t.get('status') == status_filter]
    
    return jsonify({
        "transactions": result,
        "total": len(TRANSACTIONS),
        "filtered": len(result)
    })

@app.route('/api/stats')
def get_stats():
    """Get dashboard statistics"""
    total = len(TRANSACTIONS)
    flagged = len([t for t in TRANSACTIONS if t.get('status') == 'flagged'])
    blocked = len([t for t in TRANSACTIONS if t.get('status') == 'blocked'])
    approved = len([t for t in TRANSACTIONS if t.get('status') == 'approved'])
    
    # Calculate average fraud probability
    avg_fraud_prob = sum(t.get('fraud_probability', 0) for t in TRANSACTIONS) / max(total, 1)
    
    # Calculate total amount at risk
    amount_at_risk = sum(
        t.get('amount', 0) 
        for t in TRANSACTIONS 
        if t.get('risk_level') in ['HIGH', 'MEDIUM']
    )
    
    return jsonify({
        "total_transactions": total,
        "flagged_count": flagged,
        "blocked_count": blocked,
        "approved_count": approved,
        "avg_fraud_probability": round(avg_fraud_prob, 4),
        "amount_at_risk": round(amount_at_risk, 2),
        "threat_level": "ELEVATED" if avg_fraud_prob > 0.3 else "NORMAL"
    })

@app.route('/api/temporal-data')
def get_temporal_data():
    """Get temporal fraud pattern data for heatmap visualization"""
    # Generate mock temporal data for 48 hours (30-minute buckets = 96 buckets)
    buckets = []
    for i in range(96):
        hour = (i * 30) // 60
        minute = (i * 30) % 60
        
        # Simulate higher fraud during night hours (2-6 AM)
        if 2 <= hour <= 6:
            fraud_rate = random.uniform(0.02, 0.08)
        else:
            fraud_rate = random.uniform(0.001, 0.02)
        
        buckets.append({
            "bucket_index": i,
            "hour": hour,
            "minute": minute,
            "time_label": f"{hour:02d}:{minute:02d}",
            "transaction_count": random.randint(50, 500),
            "fraud_count": random.randint(0, 15),
            "fraud_rate": round(fraud_rate, 4),
            "risk_level": "HIGH" if fraud_rate > 0.04 else "MEDIUM" if fraud_rate > 0.015 else "LOW"
        })
    
    return jsonify({
        "buckets": buckets,
        "period_start": "2024-01-01T00:00:00",
        "period_end": "2024-01-02T23:59:59",
        "bucket_size_minutes": 30
    })

@app.route('/api/fraud-network')
def get_fraud_network():
    """Get fraud network graph data for D3.js visualization"""
    if not DATA_AVAILABLE:
        return jsonify({
            "error": "Dataset not loaded",
            "nodes": [],
            "edges": [],
            "sessions": [],
            "stats": {}
        })
    
    try:
        network = build_fraud_network(
            time_window_seconds=1800,  # 30 minutes
            similarity_threshold=0.6,
            max_nodes=100
        )
        return jsonify(network)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-transactions')
def get_real_transactions():
    """Get real transactions from the dataset"""
    if not DATA_AVAILABLE:
        # Fallback to mock
        return get_transactions()
    
    try:
        limit = request.args.get('limit', 20, type=int)
        transactions = get_sample_transactions(n=limit)
        return jsonify({
            "transactions": transactions,
            "total": len(transactions),
            "source": "dataset"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/real-temporal')
def get_real_temporal():
    """Get real temporal stats from the dataset"""
    if not DATA_AVAILABLE:
        return get_temporal_data()
    
    try:
        stats = get_temporal_stats()
        return jsonify({
            "buckets": stats[:96],  # Limit to 48 hours
            "source": "dataset"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimize-threshold', methods=['POST'])
def optimize_threshold():
    """
    POST /api/optimize-threshold
    
    Finds the optimal probability threshold that minimizes total cost
    based on false positive and false negative costs.
    
    Request body:
    {
        "cost_fp": float,  // Cost per false positive
        "cost_fn": float   // Cost per false negative
    }
    
    Response:
    {
        "all_thresholds": [
            {
                "probability": float,
                "fp": int,
                "fn": int,
                "cost": float,
                "precision": float,
                "recall": float,
                "f1_score": float,
                "accuracy": float,
                "tp": int,
                "tn": int
            },
            ...
        ],
        "optimal": {
            "probability": float,
            "fp": int,
            "fn": int,
            "cost": float,
            ...
        }
    }
    """
    if not EVAL_MODULES_AVAILABLE:
        return jsonify({"error": "Model evaluation modules not available"}), 503
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    cost_fp = data.get('cost_fp')
    cost_fn = data.get('cost_fn')
    
    if cost_fp is None or cost_fn is None:
        return jsonify({"error": "Both cost_fp and cost_fn are required"}), 400
    
    try:
        cost_fp = float(cost_fp)
        cost_fn = float(cost_fn)
    except (ValueError, TypeError):
        return jsonify({"error": "cost_fp and cost_fn must be valid numbers"}), 400
    
    if cost_fp < 0 or cost_fn < 0:
        return jsonify({"error": "Costs must be non-negative"}), 400
    
    try:
        # Load the trained model
        model = joblib.load(config.TRAINED_MODEL_PATH)
        
        # Load dataset
        df = load_dataset_preprocessing()
        
        # Sort by Time (same as training)
        df = df.sort_values("Time").reset_index(drop=True)
        
        # Prepare features and target
        X = df.drop("Class", axis=1)
        y = df["Class"]
        
        # Create time-based train-test split (same as training: 80/20)
        split_idx = int(0.8 * len(df))
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Evaluate model at multiple thresholds
        eval_results = evaluate_model_at_thresholds(model, X_test, y_test)
        
        # Calculate costs and find optimal threshold
        result = compute_optimal_threshold(eval_results, cost_fp, cost_fn)
        
        return jsonify(result)
        
    except FileNotFoundError as e:
        return jsonify({"error": f"Model file not found: {str(e)}"}), 404
    except Exception as e:
        return jsonify({"error": f"Error during optimization: {str(e)}"}), 500

# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

# Training state - tracks ongoing training
TRAINING_STATE = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 100,
    "loss_history": [],
    "metrics": {},
    "status": "idle"
}

# Feature name mapping - V1-V28 to human-readable names
FEATURE_NAME_MAPPING = {
    'Amount': 'Transaction Amount',
    'Time': 'Time of Day',
    'V1': 'Transaction Velocity',
    'V2': 'Card Usage Pattern',
    'V3': 'Geographic Risk Score',
    'V4': 'Merchant Category Risk',
    'V5': 'Purchase Frequency',
    'V6': 'Amount Deviation',
    'V7': 'Time Since Last Purchase',
    'V8': 'Cross-Border Indicator',
    'V9': 'Device Trust Score',
    'V10': 'IP Risk Level',
    'V11': 'Session Duration',
    'V12': 'Failed Attempts',
    'V13': 'Account Age',
    'V14': 'Anomaly Score',
    'V15': 'Velocity Spike',
    'V16': 'Weekend Activity',
    'V17': 'Night Activity',
    'V18': 'High-Value Flag',
    'V19': 'New Merchant Flag',
    'V20': 'Payment Method Risk',
    'V21': 'Shipping Address Risk',
    'V22': 'Email Domain Risk',
    'V23': 'Phone Verification',
    'V24': 'Address Match Score',
    'V25': 'CVV Match',
    'V26': 'AVS Response',
    'V27': '3DS Authentication',
    'V28': 'Historical Fraud Rate'
}

@app.route('/api/train-model', methods=['POST'])
def train_model_endpoint():
    """
    POST /api/train-model
    
    Starts model training. Progress is streamed via WebSocket.
    
    Request body:
    {
        "dataset_path": string (optional, defaults to creditcard.csv),
        "epochs": int (optional, default 100)
    }
    """
    global TRAINING_STATE
    
    if TRAINING_STATE["is_training"]:
        return jsonify({"error": "Training already in progress"}), 409
    
    data = request.get_json() or {}
    epochs = data.get('epochs', 100)
    
    # Start training in background thread
    def run_training():
        global TRAINING_STATE
        TRAINING_STATE = {
            "is_training": True,
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": epochs,
            "loss_history": [],
            "metrics": {},
            "status": "initializing"
        }
        
        try:
            socketio.emit('training_started', TRAINING_STATE)
            
            # Import training modules
            from train_model import load_and_prepare_data, scale_features
            from sklearn.model_selection import train_test_split
            from imblearn.over_sampling import SMOTE
            import xgboost as xgb
            from sklearn.metrics import roc_auc_score
            
            TRAINING_STATE["status"] = "loading_data"
            socketio.emit('training_progress', TRAINING_STATE)
            
            # Load data
            X, y, feature_cols = load_and_prepare_data()
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale
            X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
            
            TRAINING_STATE["status"] = "applying_smote"
            socketio.emit('training_progress', TRAINING_STATE)
            
            # SMOTE
            smote = SMOTE(random_state=42, sampling_strategy=0.3)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            TRAINING_STATE["status"] = "training"
            socketio.emit('training_progress', TRAINING_STATE)
            
            # Train with progress callbacks
            n_estimators = min(epochs, 200)
            
            for i in range(1, n_estimators + 1):
                # Train partial model
                model = xgb.XGBClassifier(
                    n_estimators=i,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='auc',
                    use_label_encoder=False
                )
                
                model.fit(X_train_balanced, y_train_balanced, verbose=False)
                
                # Calculate metrics
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Simulate loss (1 - AUC for visualization)
                loss = 1 - auc
                
                TRAINING_STATE["current_epoch"] = i
                TRAINING_STATE["progress"] = int((i / n_estimators) * 100)
                TRAINING_STATE["loss_history"].append({
                    "epoch": i,
                    "loss": round(loss, 4),
                    "auc": round(auc, 4)
                })
                TRAINING_STATE["metrics"] = {
                    "auc_roc": round(auc, 4),
                    "loss": round(loss, 4),
                    "trees": i
                }
                
                socketio.emit('training_progress', TRAINING_STATE)
                time.sleep(0.05)  # Small delay for visual effect
            
            # Save final model
            TRAINING_STATE["status"] = "saving"
            socketio.emit('training_progress', TRAINING_STATE)
            
            from pathlib import Path
            MODEL_DIR = Path(__file__).parent / 'models'
            MODEL_DIR.mkdir(exist_ok=True)
            
            joblib.dump(model, MODEL_DIR / 'fraud_detector.pkl')
            joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
            joblib.dump({
                'feature_names': feature_cols,
                'n_features': len(feature_cols),
                'auc_score': float(auc)
            }, MODEL_DIR / 'feature_info.pkl')
            
            # Get feature importance
            importance = model.feature_importances_
            feature_importance = sorted(
                zip(feature_cols, importance),
                key=lambda x: x[1],
                reverse=True
            )
            
            TRAINING_STATE["status"] = "completed"
            TRAINING_STATE["progress"] = 100
            TRAINING_STATE["feature_importance"] = [
                {
                    "feature": f,
                    "feature_name": FEATURE_NAME_MAPPING.get(f, f),
                    "importance": round(float(imp), 4)
                }
                for f, imp in feature_importance[:15]
            ]
            
            socketio.emit('training_completed', TRAINING_STATE)
            
        except Exception as e:
            TRAINING_STATE["status"] = "error"
            TRAINING_STATE["error"] = str(e)
            socketio.emit('training_error', TRAINING_STATE)
        finally:
            TRAINING_STATE["is_training"] = False
    
    # Start training thread
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Training started", "status": TRAINING_STATE})

@app.route('/api/training-status')
def get_training_status():
    """Get current training status and progress"""
    return jsonify(TRAINING_STATE)

@app.route('/api/feature-importance')
def get_feature_importance():
    """
    GET /api/feature-importance
    
    Returns feature importance with human-readable names.
    """
    try:
        from pathlib import Path
        MODEL_DIR = Path(__file__).parent / 'models'
        
        # Try to load existing model
        model = joblib.load(MODEL_DIR / 'fraud_detector.pkl')
        feature_info = joblib.load(MODEL_DIR / 'feature_info.pkl')
        
        feature_cols = feature_info['feature_names']
        importance = model.feature_importances_
        
        feature_importance = sorted(
            zip(feature_cols, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return jsonify({
            "features": [
                {
                    "feature": f,
                    "feature_name": FEATURE_NAME_MAPPING.get(f, f),
                    "importance": round(float(imp), 4),
                    "importance_pct": round(float(imp) * 100 / sum(importance), 2)
                }
                for f, imp in feature_importance
            ],
            "model_auc": feature_info.get('auc_score', 0)
        })
        
    except FileNotFoundError:
        # Return mock data if no model exists
        mock_features = [
            ("Amount", 0.15),
            ("V14", 0.12),
            ("V10", 0.10),
            ("V12", 0.09),
            ("V17", 0.08),
            ("Time", 0.07),
            ("V4", 0.06),
            ("V11", 0.05),
            ("V3", 0.04),
            ("V7", 0.04)
        ]
        
        return jsonify({
            "features": [
                {
                    "feature": f,
                    "feature_name": FEATURE_NAME_MAPPING.get(f, f),
                    "importance": round(imp, 4),
                    "importance_pct": round(imp * 100, 2)
                }
                for f, imp in mock_features
            ],
            "model_auc": 0.95,
            "is_mock": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🛡️  THE MERCHANT SHIELD - Fraud Detection API")
    print("="*60)
    print(f"\n  📊 Data: {'LOADED' if DATA_AVAILABLE else 'MOCK MODE'}")
    print(f"  🤖 ML Model: {'LOADED' if MODEL_AVAILABLE else 'MOCK MODE'}")
    print("\n  📊 Dashboard:    http://127.0.0.1:5000/")
    print("  📋 Audit Log:    http://127.0.0.1:5000/audit")
    print("  🔍 Analyze:      http://127.0.0.1:5000/analyze")
    print("\n  API Endpoints:")
    print("  • POST /api/analyze-risk")
    print("  • GET  /api/transactions")
    print("  • GET  /api/stats")
    print("  • GET  /api/temporal-data")
    print("  • GET  /api/fraud-network    [NEW]")
    print("  • GET  /api/real-transactions [NEW]")
    print("  • POST /api/optimize-threshold [NEW]")
    print("\n" + "="*60 + "\n")
    
    # Run with SocketIO (supports WebSocket)
    socketio.run(app, debug=True, port=5000, host='127.0.0.1')

