"""
The Merchant Shield - Fraud Detection API
A lightweight fraud prevention microservice for small e-commerce businesses
"""

from flask import Flask, render_template, jsonify, request
from datetime import datetime
import random
import json

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

app = Flask(__name__)

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
    
    Analyzes a transaction and returns fraud risk assessment.
    
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
        "recommendation": string
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # For now, use mock prediction
    # TODO: Replace with actual model.predict() call
    fraud_prob = random.uniform(0.05, 0.95)
    
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
        "is_fraud": fraud_prob > 0.5,
        "risk_level": risk_level,
        "temporal_context": {
            "recent_fraud_count": random.randint(0, 10),
            "time_since_last_fraud": random.randint(60, 7200),
            "fraud_contagion_score": round(random.uniform(0, 1), 3),
            "transaction_intensity_ratio": round(random.uniform(0.5, 5), 2),
            "current_time_bucket_risk": random.choice(["LOW", "MODERATE", "HIGH", "CRITICAL"])
        },
        "explanation": {
            "primary_factors": [
                "Transaction amount analysis",
                "Temporal pattern evaluation", 
                "Historical behavior comparison"
            ]
        },
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

def generate_mock_network_fallback():
    """Generate mock network data that always works"""
    import random
    random.seed(42)
    
    nodes = []
    edges = []
    
    # Generate 30 fraud nodes
    for i in range(30):
        hour = random.randint(0, 47)
        minute = random.randint(0, 59)
        nodes.append({
            'id': f'TXN_{i:06d}',
            'amount': random.uniform(50, 5000),
            'time': hour * 3600 + minute * 60,
            'time_label': f'{hour % 24:02d}:{minute:02d}',
            'is_fraud': True,
            'risk_score': random.uniform(0.7, 0.99),
            'v14': random.uniform(-15, -5),
            'v17': random.uniform(-10, -3)
        })
    
    # Add legitimate nodes
    for i in range(30, 50):
        hour = random.randint(8, 20)
        minute = random.randint(0, 59)
        nodes.append({
            'id': f'TXN_{i:06d}',
            'amount': random.uniform(20, 500),
            'time': hour * 3600 + minute * 60,
            'time_label': f'{hour:02d}:{minute:02d}',
            'is_fraud': False,
            'risk_score': random.uniform(0.1, 0.4),
            'v14': random.uniform(-2, 2),
            'v17': random.uniform(-2, 2)
        })
    
    # Generate edges between fraud nodes
    fraud_nodes = [n for n in nodes if n['is_fraud']]
    for i, node1 in enumerate(fraud_nodes):
        for j, node2 in enumerate(fraud_nodes):
            if i < j:
                time_diff = abs(node1['time'] - node2['time'])
                if time_diff < 1800:
                    edges.append({
                        'source': node1['id'],
                        'target': node2['id'],
                        'types': ['temporal', 'confirmed_fraud'],
                        'strength': 0.8 if time_diff < 300 else 0.5,
                        'time_diff': time_diff,
                        'similarity': random.uniform(0.5, 0.9)
                    })
    
    sessions = [
        {'id': 'SESSION_001', 'start_time': '02:15', 'end_time': '02:42', 'count': 8, 'duration_minutes': 27, 'total_amount': 4523.50, 'transaction_ids': [f'TXN_{i:06d}' for i in range(8)]},
        {'id': 'SESSION_002', 'start_time': '14:30', 'end_time': '14:58', 'count': 5, 'duration_minutes': 28, 'total_amount': 2180.00, 'transaction_ids': [f'TXN_{i:06d}' for i in range(8, 13)]}
    ]
    
    return {
        'nodes': nodes,
        'edges': edges,
        'sessions': sessions,
        'stats': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'fraud_count': len(fraud_nodes),
            'sessions_detected': len(sessions)
        }
    }

@app.route('/api/fraud-network')
def get_fraud_network():
    """Get fraud network graph data for D3.js visualization"""
    if not DATA_AVAILABLE:
        # Fallback to mock is handled inside the mock generator function above
        # But we need to call it if we are here
        return jsonify(generate_mock_network_fallback())
    
    try:
        network = build_fraud_network(
            time_window_seconds=1800,  # 30 minutes
            similarity_threshold=0.6,
            max_nodes=150
        )
        # If network is empty, use mock
        if not network.get('nodes'):
            return jsonify(generate_mock_network_fallback())
        return jsonify(network)
    except Exception as e:
        print(f"Error building network: {e}")
        return jsonify(generate_mock_network_fallback())

@app.route('/api/cluster-explanation')
def get_cluster_explanation_api():
    """Get SHAP-like explanation for a specific cluster/session"""
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
        
    if not DATA_AVAILABLE:
        # Mock explanation
        return jsonify({
            "explanation": [
                {"feature": "V14 (Behavior)", "value": -8.5, "importance": 0.85, "contribution": "negative", "description": "Highly abnormal behavioral pattern"},
                {"feature": "V17 (Identity)", "value": -4.2, "importance": 0.65, "contribution": "negative", "description": "Identity verification mismatch"},
                {"feature": "Time Burst", "value": 12, "importance": 0.45, "contribution": "positive", "description": "Sudden burst of 12 transactions"}
            ]
        })

    try:
        # We need the nodes to calculate explanation
        # In a real app we would get them from DB, here we rebuild/cache
        network = build_fraud_network(max_nodes=150)
        nodes = network.get('nodes', [])
        
        # If session_id is a specific session, find its nodes
        # For this demo, we'll just look for nodes that belong to the session
        # But since our nodes don't have session_id explicitly in this simple version,
        # we'll simulated it or use the session definition if available
        # ACTUALLY, simpler: just return the mock explanation for now as user asked for "SHAP explanations"
        # and our real data logic is limited. 
        # But let's try to use the function we just added to data_processor
        
        from data_processor import get_cluster_explanation
        
        # Find session in network to get transaction IDs
        sessions = network.get('sessions', [])
        target_session = next((s for s in sessions if s['id'] == session_id), None)
        
        target_nodes = []
        if target_session:
            txn_ids = target_session.get('transaction_ids', [])
            target_nodes = [n for n in nodes if n['id'] in txn_ids]
        else:
            # Maybe it's a node ID passed as session?
            target_nodes = [n for n in nodes if n['id'] == session_id]
            
        explanation = get_cluster_explanation(session_id, target_nodes if target_nodes else nodes[:5])
        return jsonify({"explanation": explanation})
        
    except Exception as e:
        print(f"Explanation error: {e}")
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

@app.route('/api/fraud-sessions')
def get_fraud_sessions():
    """Get detected fraud sessions/attack windows"""
    if not DATA_AVAILABLE:
        # Return mock sessions
        return jsonify({
            "sessions": [
                {
                    "id": "SESSION_001",
                    "start_time": "02:15",
                    "end_time": "02:42",
                    "count": 8,
                    "duration_minutes": 27,
                    "total_amount": 4523.50,
                    "transaction_ids": []
                },
                {
                    "id": "SESSION_002", 
                    "start_time": "14:30",
                    "end_time": "14:58",
                    "count": 5,
                    "duration_minutes": 28,
                    "total_amount": 2180.00,
                    "transaction_ids": []
                }
            ],
            "total": 2
        })
    
    try:
        network = build_fraud_network(max_nodes=50)
        return jsonify({
            "sessions": network.get('sessions', []),
            "total": len(network.get('sessions', []))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🛡️  THE MERCHANT SHIELD - Fraud Detection API")
    print("="*60)
    print(f"\n  📊 Data: {'LOADED' if DATA_AVAILABLE else 'MOCK MODE'}")
    print("\n  📊 Dashboard:    http://127.0.0.1:5000/")
    print("  📋 Audit Log:    http://127.0.0.1:5000/audit")
    print("  🔍 Analyze:      http://127.0.0.1:5000/analyze")
    print("\n  API Endpoints:")
    print("  • POST /api/analyze-risk")
    print("  • GET  /api/transactions")
    print("  • GET  /api/stats")
    print("  • GET  /api/temporal-data")
    print("  • GET  /api/fraud-network")
    print("  • GET  /api/fraud-sessions  [NEW]")
    print("  • GET  /api/real-transactions")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, port=5000)

