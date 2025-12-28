# 🛡️ Merchant Shield - Fraud Detection Microservice

A comprehensive machine learning-powered fraud detection system for credit card transactions, featuring real-time risk assessment, cost-optimized threshold selection, and interactive visualizations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Approach & Methodology](#-approach--methodology)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [Screenshots & Demos](#-screenshots--demos)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Statement

Credit card fraud is a significant challenge in e-commerce, with billions of dollars lost annually. Traditional rule-based fraud detection systems are:
- **Rigid**: Cannot adapt to evolving fraud patterns
- **High false positive rates**: Lead to customer friction and lost revenue
- **Inefficient**: Cannot balance the cost of false positives vs. false negatives optimally
- **Limited scalability**: Struggle with high transaction volumes

**Merchant Shield** addresses these challenges by providing:
- Machine learning-based fraud detection with high accuracy
- Real-time risk assessment with sub-second response times
- Cost-optimized threshold selection based on business requirements
- Interactive dashboards for monitoring and analysis
- Model explainability using SHAP values

---

## 🔬 Approach & Methodology

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Time-based train-test split (80/20) to prevent data leakage
   - Feature scaling using StandardScaler
   - Handles highly imbalanced dataset (~0.17% fraud rate)

2. **Model Training**
   - **Algorithm**: XGBoost Classifier with gradient boosting
   - **Hyperparameter Optimization**: Optuna with TPE sampler (50 trials)
   - **Cross-Validation**: 5-fold Stratified K-Fold (no shuffle to maintain temporal order)
   - **Optimization Metric**: PR-AUC (Precision-Recall Area Under Curve)
   - **Model Calibration**: Isotonic regression for probability calibration

3. **Evaluation Strategy**
   - Time-based split ensures no temporal leakage
   - Test set used only once for final evaluation
   - Comprehensive evaluation at multiple probability thresholds (0.1 to 0.9)
   - Metrics: Precision, Recall, F1-Score, Accuracy, PR-AUC

4. **Cost Optimization**
   - Calculates total cost: `Cost = FP × cost_fp + FN × cost_fn`
   - Finds optimal threshold that minimizes total cost
   - Supports business-specific cost structures

### Key Design Decisions

- **Time-based splitting**: Critical for fraud detection to avoid future information leakage
- **PR-AUC optimization**: Better than ROC-AUC for imbalanced datasets
- **Stratified CV without shuffle**: Maintains temporal order while ensuring class balance
- **Model calibration**: Ensures probability scores are well-calibrated for threshold selection
- **Multi-threshold evaluation**: Allows business to choose threshold based on precision/recall trade-offs

---

## ✨ Features

### Core Features

- 🔍 **Real-time Fraud Detection**: Instant risk assessment for incoming transactions
- 📊 **Interactive Dashboard**: Real-time transaction monitoring with live updates via WebSocket
- 📈 **Fraud Time Machine**: Temporal heatmap visualization of fraud patterns
- 🎯 **Cost-Optimized Thresholds**: Find optimal decision threshold based on FP/FN costs
- 📉 **Model Evaluation**: Comprehensive evaluation at multiple thresholds with confusion matrices
- 🔬 **Model Explainability**: SHAP values for understanding model predictions
- 📝 **Admin Audit Log**: Review flagged and blocked transactions
- 🧪 **Transaction Analysis**: Submit transactions for instant risk scoring

### Advanced Features

- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Model Calibration**: Probability calibration for accurate risk assessment
- **Feature Importance Analysis**: Identify most important features for fraud detection
- **Export Model to Text**: Human-readable model representation
- **Multiple Model Versions**: Support for both calibrated and uncalibrated models

---

## 🛠 Tech Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask 3.0.0**: Web framework for API and web interface
- **Flask-SocketIO**: Real-time bidirectional communication
- **XGBoost 2.0.3**: Gradient boosting framework for ML model
- **scikit-learn**: ML utilities (metrics, calibration, preprocessing)
- **Optuna**: Hyperparameter optimization framework
- **SHAP**: Model explainability and feature importance
- **Pandas & NumPy**: Data processing and manipulation
- **Joblib**: Model serialization

### Frontend
- **HTML5/CSS3**: User interface
- **JavaScript**: Interactive dashboard functionality
- **D3.js**: Data visualization (fraud network graph)
- **WebSocket**: Real-time updates

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AkshayyVishnu/fraud-detection-microservice.git
cd fraud-detection-microservice
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

The project uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

1. Download the dataset from Kaggle
2. Extract `creditcard.csv` to `creditcard.csv/creditcard.csv` (or update `config.py` with your path)

**Note**: Due to dataset size, it's not included in the repository. You'll need to download it separately.

---

## 🚀 Getting Started

### Option 1: Run with Pre-trained Model

If you have a pre-trained model in the `models/` directory:

```bash
python app.py
```

Then open your browser and navigate to:
- **Dashboard**: http://127.0.0.1:5000/
- **Transaction Analysis**: http://127.0.0.1:5000/analyze
- **Audit Log**: http://127.0.0.1:5000/audit

### Option 2: Train Your Own Model

1. **Train the model**:
   ```bash
   python model_training.py
   ```
   This will:
   - Load and preprocess the dataset
   - Optimize hyperparameters using Optuna (takes 30-60 minutes)
   - Train the final model
   - Evaluate at multiple thresholds
   - Save models to `models/` directory

2. **Evaluate the model**:
   ```bash
   python eval.py
   ```
   This evaluates the trained model and outputs precision/recall for different thresholds.

3. **Export model to text** (optional):
   ```bash
   python export_models_to_text.py
   ```
   Creates human-readable text files in `models/text_exports/`.

4. **Run the application**:
   ```bash
   python app.py
   ```

---

## 📁 Project Structure

```
fraud-detection-microservice/
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── model_training.py           # Model training script with Optuna optimization
├── eval.py                     # Model evaluation script
├── loss.py                     # Cost optimization module
├── export_models_to_text.py    # Export models to readable format
├── data_preprocessing.py       # Data preprocessing utilities
├── data_processor.py           # Data processing and visualization
├── model_explainer.py          # SHAP-based model explainability
├── train_model.py              # Alternative training script
├── transaction_simulator.py    # Transaction simulation utilities
│
├── models/                     # Trained models directory
│   ├── xgb_fraud_model.pkl           # Main trained model
│   ├── xgb_fraud_model_calibrated.pkl # Calibrated model
│   ├── best_params.pkl               # Best hyperparameters
│   └── text_exports/                 # Human-readable model exports
│
├── templates/                  # HTML templates
│   ├── dashboard.html          # Main dashboard
│   ├── analyze.html            # Transaction analysis form
│   └── audit.html              # Admin audit log
│
├── static/                     # Static assets
│   ├── styles.css              # CSS styles
│   ├── app.js                  # Dashboard JavaScript
│   ├── network.js              # Network visualization
│   └── realtime.js             # Real-time updates
│
├── data/                       # Data directory (empty, add dataset here)
├── creditcard.csv/             # Dataset location
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📡 API Documentation

### Base URL

```
http://127.0.0.1:5000
```

### Endpoints

#### 1. POST `/api/analyze-risk`

Analyze a transaction for fraud risk.

**Request Body:**
```json
{
    "amount": 9999.00,
    "time": 13620,
    "v1": -18.5,
    "v2": 8.23,
    "v3": 12.45,
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
    "shap_explanation": [
        {
            "feature": "V14",
            "value": -19.214,
            "impact": 0.15
        },
        ...
    ],
    "recommendation": "BLOCK - High confidence fraud detected"
}
```

#### 2. POST `/api/optimize-threshold`

Find the optimal probability threshold that minimizes total cost.

**Request Body:**
```json
{
    "cost_fp": 10.0,  // Cost per false positive
    "cost_fn": 100.0  // Cost per false negative
}
```

**Response:**
```json
{
    "all_thresholds": [
        {
            "probability": 0.1,
            "fp": 150,
            "fn": 5,
            "cost": 2000.0,
            "precision": 0.85,
            "recall": 0.95,
            "f1_score": 0.90,
            "accuracy": 0.998,
            "tp": 95,
            "tn": 56800
        },
        ...
    ],
    "optimal": {
        "probability": 0.25,
        "fp": 80,
        "fn": 12,
        "cost": 2000.0,
        ...
    }
}
```

#### 3. GET `/api/transactions`

Get recent transactions.

**Query Parameters:**
- `limit` (int, optional): Number of transactions to return (default: 20)
- `status` (string, optional): Filter by status (`approved`, `flagged`, `blocked`)

**Example:**
```
GET /api/transactions?limit=50&status=flagged
```

#### 4. GET `/api/stats`

Get dashboard statistics.

**Response:**
```json
{
    "total_transactions": 1250,
    "flagged_count": 45,
    "blocked_count": 12,
    "approved_count": 1193,
    "avg_fraud_probability": 0.15,
    "amount_at_risk": 125000.50,
    "threat_level": "ELEVATED"
}
```

#### 5. GET `/api/temporal-data`

Get temporal fraud pattern data for heatmap visualization.

#### 6. GET `/api/fraud-network`

Get fraud network graph data for D3.js visualization.

---

## 🎓 Model Training

### Training Process

The model training follows a rigorous methodology to prevent data leakage:

1. **Data Loading**: Load credit card transaction dataset
2. **Time-based Sorting**: Sort all transactions by time
3. **Train-Test Split**: 80/20 split maintaining temporal order
4. **Hyperparameter Optimization**: Optuna with 5-fold Stratified CV
   - Optimizes: learning_rate, n_estimators, max_depth, regularization, etc.
   - Uses PR-AUC as optimization metric
5. **Final Model Training**: Train on full training set with best hyperparameters
6. **Model Calibration**: Apply isotonic regression for probability calibration
7. **Evaluation**: Evaluate on test set (first time seeing test data)

### Running Training

```bash
python model_training.py
```

### Expected Output

- Model files saved to `models/`:
  - `xgb_fraud_model.pkl`: Main trained model
  - `xgb_fraud_model_calibrated.pkl`: Calibrated version
  - `best_params.pkl`: Best hyperparameters
- Training metrics and evaluation results printed to console

---

## 📊 Model Evaluation

### Evaluation Script

Run comprehensive evaluation:

```bash
python eval.py
```

### What It Does

1. Loads trained model
2. Loads test dataset
3. Creates time-based split (same as training)
4. Evaluates at multiple thresholds (0.1, 0.2, 0.25, ..., 0.9)
5. Outputs:
   - Precision, Recall, F1-Score, Accuracy for each threshold
   - Confusion matrices for each threshold
   - PR-AUC (overall performance metric)

### Example Output

```
Evaluation at Multiple Thresholds
================================================================================
Threshold    Precision    Recall       F1-Score     Accuracy     TN      FP      FN      TP
--------------------------------------------------------------------------------
0.10         0.8234       0.9567       0.8852       0.9987       56800   150     5       95
0.20         0.8756       0.9123       0.8936       0.9989       56920   80      9       91
...
```

---

## 📸 Screenshots & Demos

### Dashboard View
![Dashboard](docs/screenshots/dashboard.png)
*Real-time transaction monitoring dashboard with live statistics and transaction list*

### Fraud Analysis
![Analysis](docs/screenshots/analysis.png)
*Transaction analysis interface with SHAP explanations and risk assessment*

### Audit Log
![Audit](docs/screenshots/audit.png)
*Admin audit log for reviewing flagged and blocked transactions*

### Fraud Time Machine
![Time Machine](docs/screenshots/time-machine.png)
*Temporal heatmap showing fraud patterns over time*

### Model Evaluation
![Evaluation](docs/screenshots/evaluation.png)
*Model evaluation results showing precision/recall for different thresholds*

**Note**: Screenshots should be added to `docs/screenshots/` directory. Create this directory and add your screenshots, then update the paths above.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (when available)
pytest

# Format code
black .

# Lint code
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- XGBoost development team
- Optuna developers
- SHAP library creators

---

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for fraud prevention**
