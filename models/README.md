# ML Model Directory

This directory is for storing trained ML models.

## Expected Files

- `model.pkl` - Trained Logistic Regression or Random Forest model
- `scaler.pkl` - Feature scaler (if used during training)

## Training Instructions

Your coworker should save the trained model here:

```python
import joblib
from sklearn.linear_model import LogisticRegression
# or
from sklearn.ensemble import RandomForestClassifier

# After training
joblib.dump(model, 'models/model.pkl')
```

## Model Input Features

The model expects the following features in this order:

1. `Time` - Seconds elapsed between this transaction and the first transaction
2. `V1` through `V28` - PCA transformed features (anonymized)
3. `Amount` - Transaction amount

Total: 30 features

## Usage

The Flask API will load and use this model automatically when present.
