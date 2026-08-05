"""
Material Property Predictor v1
Day 5 Project - AI Materials Engineer Journey

Dataset: NIMS Steel Fatigue Database (437 samples)
Goal: Predict Fatigue Strength from composition and heat treatment parameters.

Workflow:
1. Load real data and define Features (X) / Target (y)
2. Train/Test split
3. Train two models: Linear Regression and Random Forest
4. Evaluate both with R2 Score and MAE
5. Compare performance and print feature importance from Random Forest
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================================
# CONFIG
# ============================================================
DATA_FILE = "fatigue.csv"
TARGET_COLUMN = "Fatigue"

# Features: composition + key heat treatment parameters
FEATURE_COLUMNS = [
    "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo",   # composition (%)
    "NT", "THT", "THt", "CT", "Ct", "DT", "Dt", "QmT", "TT", "Tt"  # heat treatment
]
# ============================================================


def main():
    # ---- Step 1: Load data ----
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # ---- Step 2: Train/Test split ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ---- Step 3: Train both models ----
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # ---- Step 4: Evaluate ----
    print(f"\n{'=' * 50}")
    print("MODEL COMPARISON")
    print(f"{'=' * 50}")

    lr_r2 = r2_score(y_test, lr_predictions)
    lr_mae = mean_absolute_error(y_test, lr_predictions)
    print(f"\nLinear Regression:")
    print(f"  R2 Score: {lr_r2:.3f}")
    print(f"  MAE: {lr_mae:.2f} MPa")

    rf_r2 = r2_score(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    print(f"\nRandom Forest:")
    print(f"  R2 Score: {rf_r2:.3f}")
    print(f"  MAE: {rf_mae:.2f} MPa")

    better_model = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
    print(f"\nBetter performing model: {better_model}")

    # ---- Step 5: Feature importance (Random Forest) ----
    print(f"\n{'=' * 50}")
    print("TOP 5 MOST IMPORTANT FEATURES (Random Forest)")
    print(f"{'=' * 50}")
    importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print(importances.head(5).round(3))


if __name__ == "__main__":
    main()