# Purpose: Train an XGBoost churn model on processed CSVs, print metrics, top features, and threshold sweep.
# Notes:
# - Expects FeaturePipeline.py to have created: data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
# - model artifacts are saved 

import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ----------------- Config -----------------
PROCESSED_DIR = os.path.join("data", "processed")
X_TRAIN_CSV = os.path.join(PROCESSED_DIR, "X_train.csv")
X_TEST_CSV  = os.path.join(PROCESSED_DIR, "X_test.csv")
Y_TRAIN_CSV = os.path.join(PROCESSED_DIR, "y_train.csv")
Y_TEST_CSV  = os.path.join(PROCESSED_DIR, "y_test.csv")
TARGET = "Churned"
RANDOM_STATE = 42
# ------------------------------------------

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def threshold_sweep(y_true, y_proba, thresholds=None):
    """Return a DataFrame with metrics across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    rows = []
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_true, y_hat, zero_division=0),
            "recall":    recall_score(y_true, y_hat, zero_division=0),
            "f1":        f1_score(y_true, y_hat, zero_division=0),
            "accuracy":  accuracy_score(y_true, y_hat),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("threshold").reset_index(drop=True)

def main():
    # 1) Load processed datasets
    X_train = pd.read_csv(X_TRAIN_CSV)
    X_test  = pd.read_csv(X_TEST_CSV)
    y_train = pd.read_csv(Y_TRAIN_CSV)[TARGET].astype(int)
    y_test  = pd.read_csv(Y_TEST_CSV)[TARGET].astype(int)

    print_header("Dataset Shapes")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")
    print(f"Train positives (churn=1) rate: {y_train.mean():.3f}")

    # 2) Compute imbalance weight (helps recall on minority class)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # 3) Define a solid baseline XGBoost model
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=6,
        min_child_weight=1.0,
        gamma=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,  # handle class imbalance
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False
    )

    # 4) Train
    print_header("Training")
    model.fit(X_train, y_train)
    print("Model trained.")

# 5) Evaluate at threshold 0.5
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_proba >= 0.50).astype(int)

    print("\nConfusion Matrix @ 0.50:")
    print(confusion_matrix(y_test, y_pred_05))

    print("\nClassification Report @ 0.50:")
    print(classification_report(y_test, y_pred_05, digits=4))

    # Threshold sweep (to choose a better cutoff than 0.5 if needed)
    print_header("Threshold Sweep (Precision/Recall/F1 across thresholds)")
    sweep = threshold_sweep(y_test, y_proba, thresholds=np.linspace(0.2, 0.8, 7))
    print(sweep.to_string(index=False))


    #  Save model artifact 
    import joblib, pathlib
    pathlib.Path("models").mkdir(exist_ok=True, parents=True)
    joblib.dump(model, "models/xgb_model.pkl")
    print("\n Saved model to models/xgb_model.pkl")

    # print("\n✅ Training complete (no model file saved).")

if __name__ == "__main__":
    main()
