# The Merchant Shield 🛡️
## Fraud Detection API for E-commerce Merchants

A lightweight machine learning microservice that provides instant fraud risk assessment for transactions with real-time training capabilities.

---

## 🚀 Quick Start

```bash
git clone https://github.com/AkshayyVishnu/fraud-detection-microservice.git
cd fraud-detection-microservice

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

Navigate to **http://127.0.0.1:5000/** in your browser.

---

## ✨ Features

### Dashboard
- **Real-time Metrics**: Live transaction monitoring with animated counters
- **Fraud Network Graph**: D3.js force-directed visualization of transaction relationships
- **Temporal Heatmap**: Time-based fraud pattern analysis
- **Risk Distribution**: Interactive donut chart

### Analyze Page
- **Transaction Analysis**: Submit transactions for instant fraud scoring
- **Train New Model**: Upload datasets and train models with real-time loss visualization
- **Feature Importance**: Human-readable explanations (not V1-V28 labels)
- **Live Metrics**: Training progress with performance charts

### Audit Log
- **Transaction History**: Review all flagged and blocked transactions
- **Filtering**: Filter by status (approved/flagged/blocked)
- **Statistics**: Detection rate and fraud prevention metrics

---

## 🔌 API Endpoints

### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze-risk` | Analyze transaction for fraud risk |
| GET | `/api/transactions` | Get recent transactions |
| GET | `/api/stats` | Get dashboard statistics |
| GET | `/api/temporal-data` | Get temporal fraud patterns |

### Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/train-model` | Start model training |
| GET | `/api/training-status` | Get training progress |
| GET | `/api/feature-importance` | Get feature importance with readable names |

### Example: Analyze Risk
```json
POST /api/analyze-risk
{
    "amount": 9999.00,
    "time": 13620,
    "v1": -18.5,
    ...
    "v28": -0.87
}
```

---

## 📁 Project Structure

```
merchant-shield/
├── app.py                  # Flask + SocketIO application
├── model_training.py       # Optuna-optimized training (time-based split, PR-AUC)
├── eval.py                 # Evaluates the saved model at multiple thresholds
├── data_preprocessing.py   # Data loading utilities
├── static/
│   ├── styles.css          # Premium fintech design system
│   ├── app.js              # Dashboard JavaScript
│   ├── network.js          # D3.js fraud network graph
│   ├── temporal.js         # Temporal heatmap
│   └── realtime.js         # Live transaction feed (SocketIO)
├── templates/
│   ├── dashboard.html      # Main dashboard
│   ├── analyze.html        # Analysis + training
│   └── audit.html          # Audit log
├── models/                 # Saved ML models
└── data/                   # Dataset directory
```

---

## 🧠 ML Model

XGBoost classifier trained on the [Kaggle credit card fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (284,807 transactions, 0.173% fraud).

```bash
# Train (Optuna-tuned XGBoost)
python model_training.py

# Evaluate the saved model at multiple thresholds
python eval.py
```

**Methodology:**
- Time-based train/test split (80/20) — the test set is never touched during tuning
- Hyperparameters tuned with Optuna over 5-fold Stratified CV (`shuffle=False`, so folds stay in temporal order)
- Optimized for **PR-AUC**, not ROC-AUC — with 0.173% fraud, ROC-AUC is misleadingly high (~0.99) regardless of model quality; PR-AUC reflects real precision/recall tradeoffs
- Test set evaluated exactly once, after tuning is finalized

**Results** (`python eval.py`, on the held-out time-based test set — 56,962 transactions, 75 frauds):

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.20 | 85.3% | 77.3% | 81.1% |
| 0.50 | 90.5% | 76.0% | 82.6% |

Overall **PR-AUC: 0.8034**

Model files saved to `models/xgb_fraud_model.pkl` (and a calibrated variant, `models/xgb_fraud_model_calibrated.pkl`).

---

## 📝 License

MIT License
