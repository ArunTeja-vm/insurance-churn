# FeaturePipeline.py
# Purpose: Read raw CSV -> clean/encode/split -> add derived features -> write processed CSVs to data/processed/
# No joblib/pickle artifacts are created.

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ----------------- Config -----------------
INPUT_CSV = "synthetic_auto_retention_full.csv"
OUTPUT_DIR = os.path.join("data", "processed")
TARGET = "Churned"

# Columns that can cause leakage or are IDs (drop them)
LEAKAGE_OR_ID_COLS = [
    "Policy_Status",
    "Policy_Cancellation_Date",
    "Policy_Expiry_Date",
    "Policy_Effective_Date",
    "Claim_Closed_Date",
    "Customer_ID",
    "Policy_Number",
]

# Base numeric columns from raw data 
NUMERIC_COLS = [
    "Customer_Age", "Income",
    "Address_Change_Flag", "VIN_Validated", "Policy_Tenure_Months",
    "Deductibles", "Has_Multi_Policy", "Loyalty_Program_Enrollment",
    "Discount_Count", "Premium_Amount", "Premium_Change_Percent_Last_Renewal",
    "Late_Payment_Count", "Auto_Renew_Enabled",
    "Claims_Count_Lifetime", "Claims_Count_Last_3_Years", "At_Fault_Accident_Count",
    "Time_Since_Last_Claim", "Total_Claim_Payout_Amount",
    "Customer_Satisfaction_Score", "Interaction_Score", "NPS",
    "Complaint_Count", "Sentiment_Score",
    "Vehicle_Value", "Annual_Mileage_Estimate",
]

CAT_COLS = [
    "Coverage_Type", "Billing_Method", "Billing_Frequency",
    "Payment_Method", "Claim_Type", "Claim_Outcome",
    "State", "Vehicle_Make",

]

# Engineered (derived) numeric columns we will create below
ENGINEERED_COLS = [
    "Complaints_per_Year",
    "Premium_Hike_x_Tenure",
    "Recent_Claim_Flag",
    "Income_to_Premium_Ratio",
    "Age_x_Tenure",
    "Engagement_Index",
    "Renewal_Due_Flag",
    "Claim_Severity_Proxy",
]
# ------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived/interaction/time-based features safely."""
    X = df.copy()

    # Safe helpers
    tenure_m = X["Policy_Tenure_Months"].fillna(0).clip(lower=0)
    tenure_y = (tenure_m / 12.0).replace(0, 1e-6)  # avoid divide-by-zero

    premium_amt = X["Premium_Amount"].replace(0, np.nan)
    payout = X["Total_Claim_Payout_Amount"].fillna(0)

    # 1) Complaints per year
    X["Complaints_per_Year"] = (
        (X["Complaint_Count"].fillna(0) / tenure_y)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # 2) Premium hike x tenure (interaction)
    X["Premium_Hike_x_Tenure"] = X["Premium_Change_Percent_Last_Renewal"].fillna(0) * tenure_m

    # 3) Recent claim flag (last 12 months)
    X["Recent_Claim_Flag"] = (X["Time_Since_Last_Claim"].fillna(999) <= 12).astype(int)

    # 4) Income to premium ratio (affordability)
    X["Income_to_Premium_Ratio"] = (
        (X["Income"].fillna(0) / premium_amt)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .clip(0, 1e3)  # guard against extreme ratios
    )

    # 5) Age x Tenure (interaction)
    X["Age_x_Tenure"] = X["Customer_Age"].fillna(0) * tenure_m

    # 6) Engagement Index (composite)
    X["Engagement_Index"] = (
        0.4 * X["Interaction_Score"].fillna(0) +
        0.3 * (X["NPS"].fillna(0) / 10.0) +
        0.3 * (X["Customer_Satisfaction_Score"].fillna(0) / 100.0)
    ).clip(0, 1.0)

    # 7) Renewal due flag (within ~2 months of annual cycle)
    X["Renewal_Due_Flag"] = ((tenure_m % 12) >= 10).astype(int)

    # 8) Claim severity proxy (claim payout relative to premium)
    X["Claim_Severity_Proxy"] = (
        (payout / premium_amt)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .clip(0, 10)
    )

    return X

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load raw data
    df = pd.read_csv(INPUT_CSV)

    # 2) Drop leakage / ID columns if they exist
    df = df.drop(columns=[c for c in LEAKAGE_OR_ID_COLS if c in df.columns], errors="ignore")

    # 3) Keep only the columns we want + target (ignore extras safely)
    need_cols = NUMERIC_COLS + CAT_COLS + [TARGET]
    keep_cols = [c for c in need_cols if c in df.columns]
    missing = set(need_cols) - set(keep_cols)
    if missing:
        print(f"[Info] These expected raw columns were not found and will be skipped: {sorted(list(missing))}")

    df = df[keep_cols].copy()

    # Safety: ensure target is int/binary
    if df[TARGET].dtype not in (np.int64, np.int32, np.int16, np.int8):
        df[TARGET] = df[TARGET].astype(int)

    # 4) Add engineered features ON THE FULL DATA before splitting
    df_eng = add_engineered_features(df)

    # 5) Split X / y
    X = df_eng.drop(columns=[TARGET])
    y = df_eng[TARGET]

    # 6) Train/test split (stratify keeps label distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 7) Build preprocessing pipelines
    numeric_cols_available = [c for c in NUMERIC_COLS + ENGINEERED_COLS if c in X_train.columns]
    cat_cols_available = [c for c in CAT_COLS if c in X_train.columns]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # Scaling not required for tree models, but left here for portability to linear models if needed
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # scikit-learn 1.2.1 supports 'sparse_output' arg
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols_available),
            ("cat", cat_pipeline, cat_cols_available),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # 8) Fit ONLY on train, transform both train and test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 9) Get the expanded feature names (after one-hot, etc.)
    feature_names = preprocessor.get_feature_names_out()

    # 10) Convert to DataFrames for saving as CSV
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    y_train_df = pd.DataFrame({TARGET: y_train.values})
    y_test_df = pd.DataFrame({TARGET: y_test.values})

    # 11) Save CSVs
    X_train_path = os.path.join(OUTPUT_DIR, "X_train.csv")
    X_test_path  = os.path.join(OUTPUT_DIR, "X_test.csv")
    y_train_path = os.path.join(OUTPUT_DIR, "y_train.csv")
    y_test_path  = os.path.join(OUTPUT_DIR, "y_test.csv")
    feat_names_path = os.path.join(OUTPUT_DIR, "feature_names.txt")

    X_train_df.to_csv(X_train_path, index=False)
    X_test_df.to_csv(X_test_path, index=False)
    y_train_df.to_csv(y_train_path, index=False)
    y_test_df.to_csv(y_test_path, index=False)

    # Save feature names as plain text for reference
    with open(feat_names_path, "w", encoding="utf-8") as f:
        for n in feature_names:
            f.write(f"{n}\n")

    print("✅ Feature pipeline complete (with advanced features).")
    print(f"Saved to {OUTPUT_DIR}:")
    print(f"  - X_train.csv  (shape={X_train_df.shape})")
    print(f"  - X_test.csv   (shape={X_test_df.shape})")
    print(f"  - y_train.csv  (shape={y_train_df.shape})")
    print(f"  - y_test.csv   (shape={y_test_df.shape})")
    print(f"  - feature_names.txt (len={len(feature_names)})")
    print("\nEngineered columns added:", ENGINEERED_COLS)

if __name__ == "__main__":
    main()
