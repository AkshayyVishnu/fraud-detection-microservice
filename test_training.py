"""
Test training logic without Flask server
"""
import sys
import time

print("=" * 60)
print("  TESTING TRAINING LOGIC")
print("=" * 60)

# Test 1: Import dependencies
print("\n[1] Testing imports...")
try:
    from train_model import load_and_prepare_data, scale_features
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    import joblib
    print("    ✓ All imports successful")
except ImportError as e:
    print(f"    ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load data
print("\n[2] Loading dataset...")
start = time.time()
try:
    X, y, feature_cols = load_and_prepare_data()
    print(f"    ✓ Loaded {len(X)} samples, {len(feature_cols)} features in {time.time()-start:.2f}s")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Split and scale
print("\n[3] Splitting and scaling...")
start = time.time()
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(f"    ✓ Train: {len(X_train)}, Test: {len(X_test)} in {time.time()-start:.2f}s")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

# Test 4: SMOTE
print("\n[4] Applying SMOTE...")
start = time.time()
try:
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"    ✓ Balanced: {len(X_train_balanced)} samples in {time.time()-start:.2f}s")
except Exception as e:
    print(f"    ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Train XGBoost with progress (the key test!)
print("\n[5] Training XGBoost with batched progress...")
total_trees = 50
batch_size = 5
model = None

start = time.time()
for batch_num in range(total_trees // batch_size):
    trees_so_far = (batch_num + 1) * batch_size
    
    model = xgb.XGBClassifier(
        n_estimators=trees_so_far,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False,
        tree_method='hist',
        n_jobs=-1
    )
    
    model.fit(X_train_balanced, y_train_balanced, verbose=False)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    progress = int((trees_so_far / total_trees) * 100)
    elapsed = time.time() - start
    print(f"    Batch {batch_num+1}/10: Trees={trees_so_far}, AUC={auc:.4f}, Progress={progress}%, Time={elapsed:.1f}s")

print(f"\n    ✓ Training completed in {time.time()-start:.1f}s")
print(f"    ✓ Final AUC-ROC: {auc:.4f}")

# Test 6: Feature importance
print("\n[6] Feature importance (top 5)...")
importance = model.feature_importances_
top_5 = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:5]
for feat, imp in top_5:
    print(f"    {feat}: {imp:.4f}")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED ✓")
print("=" * 60)
