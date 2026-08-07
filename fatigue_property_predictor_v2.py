"""
Material Property Predictor v2
Day 6 Project - AI Materials Engineer Journey

Improvements over v1 (Day 5):
1. Cross-Validation (5-Fold) instead of a single Train/Test Split -
   gives a more reliable performance estimate.
2. Feature Selection using Random Forest feature importance - compares
   performance using all features vs only the top 5 most important ones.
3. Reports standard deviation across folds to check model consistency.

Dataset: NIMS Steel Fatigue Database (437 samples)
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

# ============================================================
# CONFIG
# ============================================================
DATA_FILE = "fatigue.csv"
TARGET_COLUMN = "Fatigue"
ALL_FEATURES = [
    "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo",
    "NT", "THT", "THt", "CT", "Ct", "DT", "Dt", "QmT", "TT", "Tt"
]
CV_FOLDS = 5
TOP_N_FEATURES = 5
# ============================================================


def main():
    df = pd.read_csv(DATA_FILE)
    y = df[TARGET_COLUMN]

    # IMPORTANT: this dataset is sorted (low-fatigue samples first,
    # high-fatigue samples last). Plain cv=5 does NOT shuffle by default,
    # which would put entirely different fatigue ranges in each fold and
    # produce catastrophically wrong results. Always shuffle explicitly
    # unless you know the data is already in random order.
    cv_strategy = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    print(f"{'=' * 55}")
    print("STEP 1: BASELINE - ALL FEATURES, CROSS-VALIDATED")
    print(f"{'=' * 55}")
    X_all = df[ALL_FEATURES]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }


    baseline_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_all, y, cv=cv_strategy, scoring="r2")
        baseline_scores[name] = scores
        print(f"{name}: Mean R2 = {scores.mean():.3f}, Std = {scores.std():.3f}")

    # ---- Step 2: Feature importance from Random Forest ----
    print(f"\n{'=' * 55}")
    print("STEP 2: FEATURE IMPORTANCE")
    print(f"{'=' * 55}")
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_full.fit(X_all, y)
    importances = pd.Series(rf_full.feature_importances_, index=ALL_FEATURES)
    importances = importances.sort_values(ascending=False)
    print(importances.head(TOP_N_FEATURES).round(3))

    top_features = importances.head(TOP_N_FEATURES).index.tolist()

    # ---- Step 3: Re-evaluate using only top features ----
    print(f"\n{'=' * 55}")
    print(f"STEP 3: TOP {TOP_N_FEATURES} FEATURES ONLY - CROSS-VALIDATED")
    print(f"{'=' * 55}")
    X_top = df[top_features]

    for name, model in models.items():
        scores = cross_val_score(model, X_top, y, cv=cv_strategy, scoring="r2")
        baseline_mean = baseline_scores[name].mean()
        print(f"{name}: Mean R2 = {scores.mean():.3f}, Std = {scores.std():.3f} "
              f"(baseline was {baseline_mean:.3f})")

    print(f"\nUsed {len(top_features)} features instead of {len(ALL_FEATURES)} "
          f"({len(top_features)/len(ALL_FEATURES)*100:.0f}% of original feature set).")


if __name__ == "__main__":
    main()