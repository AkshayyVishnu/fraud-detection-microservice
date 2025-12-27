# The Merchant Shield
## Fraud Detection API for E-commerce Merchants

A lightweight machine learning microservice that provides instant fraud risk assessment for transactions.

## Quick Start

### 1. Install Dependencies

```bash
pip install flask
```

### 2. Run the Server

```bash
python app.py
```

### 3. Open the Dashboard

Navigate to http://127.0.0.1:5000/ in your browser.

## Features

- **Real-time Dashboard**: Monitor transactions with live updates
- **Fraud Time Machine**: Temporal heatmap visualization of fraud patterns
- **Transaction Analysis**: Submit transactions for instant risk scoring
- **Admin Audit Log**: Review flagged and blocked transactions

## API Endpoints

### POST /api/analyze-risk

Analyze a transaction for fraud risk.

**Request:**
```json
{
    "amount": 9999.00,
    "time": 13620,
    "v1": -18.5,
    "v2": 8.23,
    ...
    "v28": -0.87
}
```

**Response:**
```json
{
    "fraud_probability": 0.87,
    "is_fraud": true,
    "risk_level": "HIGH",
    "temporal_context": {
        "recent_fraud_count": 5,
        "time_since_last_fraud": 180,
        "fraud_contagion_score": 0.73,
        "transaction_intensity_ratio": 4.2,
        "current_time_bucket_risk": "CRITICAL"
    },
    "explanation": {
        "primary_factors": [...]
    },
    "recommendation": "BLOCK - High confidence fraud detected"
}
```

### GET /api/transactions

Get recent transactions.

**Query Parameters:**
- `limit` (int): Number of transactions to return (default: 20)
- `status` (string): Filter by status (approved/flagged/blocked)

### GET /api/stats

Get dashboard statistics.

### GET /api/temporal-data

Get temporal fraud pattern data for heatmap visualization.

## Project Structure

```
merchant-shield/
├── app.py              # Flask application
├── static/
│   ├── styles.css      # Premium dark theme CSS
│   └── app.js          # Dashboard JavaScript
├── templates/
│   ├── dashboard.html  # Main dashboard
│   ├── analyze.html    # Transaction analysis form
│   └── audit.html      # Audit log
├── models/             # ML model files (placeholder)
│   └── README.md
└── README.md
```

## ML Model Integration

The ML model is currently a placeholder. Your coworker's trained model should be saved as `models/model.pkl` using joblib or pickle.

To integrate:

1. Save your trained model:
```python
import joblib
joblib.dump(model, 'models/model.pkl')
```

2. Update `app.py` to load and use the model:
```python
import joblib
model = joblib.load('models/model.pkl')

# In analyze_risk():
prediction = model.predict_proba(features)[0][1]
```

## License

MIT License
