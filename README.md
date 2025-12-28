# The Merchant Shield 🛡️
## Fraud Detection API for E-commerce Merchants

A lightweight machine learning microservice that provides instant fraud risk assessment for transactions with real-time training capabilities.

---

## 🚀 Quick Start

### Using the Enhanced UI Branch

This project has a modernized UI on the `redesign/calm-financial-ui` branch:

```bash
# Clone and checkout the enhanced UI branch
git clone https://github.com/AkshayyVishnu/fraud-detection-microservice.git
cd fraud-detection-microservice
git checkout redesign/calm-financial-ui

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

Navigate to **http://127.0.0.1:5000/** in your browser.

### Using Main Branch

```bash
git checkout main
pip install flask
python app.py
```

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

### Training (New in UI Branch)
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
├── train_model.py          # Standalone training script
├── model_training.py       # Optuna-optimized training
├── data_preprocessing.py   # Data loading utilities
├── static/
│   ├── styles.css          # Premium fintech design system
│   ├── app.js              # Dashboard JavaScript
│   ├── network.js          # D3.js fraud network graph
│   ├── temporal.js         # Temporal heatmap
│   └── training.js         # Training UI (live charts)
├── templates/
│   ├── dashboard.html      # Main dashboard
│   ├── analyze.html        # Analysis + training
│   └── audit.html          # Audit log
├── models/                 # Saved ML models
└── data/                   # Dataset directory
```

---

## 🧠 ML Model

Train a model from the UI or command line:

```bash
# Command line training
python train_model.py

# Model files saved to:
# - models/fraud_detector.pkl
# - models/scaler.pkl
# - models/feature_info.pkl
```

Target: **>95% AUC-ROC** with XGBoost + SMOTE for class imbalance.

---

## 📝 License

MIT License
