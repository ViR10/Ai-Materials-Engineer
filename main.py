"""
AI Materials Property Predictor - Unified Platform (v2)
Week 1 Capstone Project - AI Materials Engineer Journey
Run with: streamlit run app.py
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="AI Materials Property Predictor", layout="wide")

MISSING_WARNING_THRESHOLD = 0.5
SMALL_DATASET_THRESHOLD = 30
HIGH_CARDINALITY_THRESHOLD = 20


def clean_mixed_column(series):
    extracted = series.astype(str).str.extract(r"(-?\d+\.?\d*)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def clean_missing(df, columns, strategy="mean"):
    log = []
    usable_columns = []
    total_rows = len(df)
    for col in columns:
        missing_count = df[col].isnull().sum()
        if missing_count == total_rows:
            log.append(f"DROPPED '{col}' — entirely missing, no data to learn from.")
            continue
        if missing_count > 0:
            missing_frac = missing_count / total_rows
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)
            flag = " (WARNING: >50% missing)" if missing_frac > MISSING_WARNING_THRESHOLD else ""
            log.append(f"Filled {missing_count} missing values in '{col}' "
                        f"with {strategy} {fill_value:.2f}{flag}")
        usable_columns.append(col)
    return df, usable_columns, log


def check_target_leakage(X, y, feature_cols, threshold=0.98):
    suspects = []
    for col in feature_cols:
        corr = X[col].corr(y)
        if pd.notnull(corr) and abs(corr) > threshold:
            suspects.append((col, corr))
    return suspects


def build_dashboard(importances, results_df, target_col, full_target_series, top_n):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    top_importances = importances.head(top_n)
    axes[0, 0].bar(top_importances.index, top_importances.values, color="steelblue")
    axes[0, 0].set_title(f"Top {top_n} Feature Importances")
    axes[0, 0].tick_params(axis="x", rotation=30)

    axes[0, 1].scatter(results_df["Actual"], results_df["Predicted"], color="darkred", alpha=0.6)
    lims = [
        min(results_df["Actual"].min(), results_df["Predicted"].min()),
        max(results_df["Actual"].max(), results_df["Predicted"].max())
    ]
    axes[0, 1].plot(lims, lims, "k--", label="Perfect Prediction")
    axes[0, 1].set_xlabel(f"Actual {target_col}")
    axes[0, 1].set_ylabel(f"Predicted {target_col}")
    axes[0, 1].set_title("Actual vs Predicted (single split, for visualization)")
    axes[0, 1].legend()

    axes[1, 0].hist(results_df["Error"], bins=15, color="coral", edgecolor="black")
    axes[1, 0].set_xlabel("Absolute Error")
    axes[1, 0].set_title("Prediction Error Distribution (single split)")

    axes[1, 1].hist(full_target_series, bins=15, color="mediumseagreen", edgecolor="black")
    axes[1, 1].set_xlabel(target_col)
    axes[1, 1].set_title(f"{target_col} Distribution (Full Dataset)")

    plt.tight_layout()
    return fig


def generate_colab_script(target_col, feature_cols, group_col, cv_folds, top_n,
                           impute_strategy, best_model_name):
    group_line = f'"{group_col}"' if group_col and group_col != "None" else "None"
    features_formatted = ",\n    ".join([f'"{f}"' for f in feature_cols])

    return f'''"""
AI Materials Property Predictor - Colab Training Script
Auto-generated from your Tab 1 analysis choices.
"""

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from google.colab import files

# ---- CONFIG (matches your Tab 1 selections) ----
TARGET_COLUMN = "{target_col}"
FEATURE_COLUMNS = [
    {features_formatted}
]
GROUP_COLUMN = {group_line}
CV_FOLDS = {cv_folds}
TOP_N_FEATURES = {top_n}
IMPUTE_STRATEGY = "{impute_strategy}"

print("Please upload your materials CSV file...")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename, encoding="latin1")
print(f"Loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns")

work_df = df[df[TARGET_COLUMN].notnull()].copy()
feature_cols = []
for col in FEATURE_COLUMNS:
    missing = work_df[col].isnull().sum()
    if missing == len(work_df):
        print(f"Dropped {{col}} - entirely missing")
        continue
    if missing > 0:
        fill_val = work_df[col].median() if IMPUTE_STRATEGY == "median" else work_df[col].mean()
        work_df[col] = work_df[col].fillna(fill_val)
    feature_cols.append(col)

X = work_df[feature_cols]
y = work_df[TARGET_COLUMN]

if len(work_df) < {SMALL_DATASET_THRESHOLD}:
    print(f"WARNING: Only {{len(work_df)}} samples - cross-validation results may be unreliable.")

if GROUP_COLUMN:
    cv = GroupKFold(n_splits=CV_FOLDS)
    cv_kwargs = {{"groups": work_df[GROUP_COLUMN]}}
else:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_kwargs = {{}}

models = {{
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}}
scores_summary = {{}}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", **cv_kwargs)
    scores_summary[name] = scores.mean()
    print(f"{{name}}: Mean R2 = {{scores.mean():.3f}}")

best_model_name = max(scores_summary, key=scores_summary.get)
print(f"Best model: {{best_model_name}}")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
top_features = importances.head(TOP_N_FEATURES).index.tolist()
print("Top features:", top_features)

X_top = work_df[top_features]
final_model = models[best_model_name]
cv_r2_scores = cross_val_score(final_model, X_top, y, cv=cv, scoring="r2", **cv_kwargs)
cv_mae_scores = -cross_val_score(final_model, X_top, y, cv=cv, scoring="neg_mean_absolute_error", **cv_kwargs)
print(f"Cross-validated R2 = {{cv_r2_scores.mean():.3f}} (Std = {{cv_r2_scores.std():.3f}})")
print(f"Cross-validated MAE = {{cv_mae_scores.mean():.2f}}")

final_model.fit(X_top, y)

bundle = {{
    "model": final_model,
    "model_name": best_model_name,
    "feature_columns": top_features,
    "target_column": TARGET_COLUMN,
    "cv_r2_mean": cv_r2_scores.mean(),
    "cv_r2_std": cv_r2_scores.std(),
    "cv_mae_mean": cv_mae_scores.mean()
}}
joblib.dump(bundle, "trained_model.joblib")
files.download("trained_model.joblib")
print("Done! trained_model.joblib has been downloaded.")
'''


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }


st.title("AI Materials Property Predictor")
st.caption("Week 1 Capstone — one platform: analyze your data, get a model "
           "recommendation, train it in Colab, and predict with it.")

tab1, tab2, tab3 = st.tabs(["1. Analyze & Recommend", "2. Train in Colab", "3. Predict"])

with tab1:
    with st.sidebar:
        st.header("Settings")
        cv_folds = st.slider("Cross-Validation folds", min_value=3, max_value=10, value=5)
        top_n = st.slider("Number of top features to use", min_value=2, max_value=15, value=5)
        impute_strategy = st.selectbox(
            "Missing value fill strategy", ["mean", "median"],
            help="Median is more robust if your data has outliers."
        )
        encoding_strategy = st.selectbox(
            "Categorical encoding strategy", ["One-Hot (safe default)", "Label Encoding"],
            help="Only use Label Encoding if your categories have a genuine order."
        )

    uploaded_file = st.file_uploader("Upload a materials CSV file", type=["csv"], key="analyze_upload")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.subheader("1. Dataset Preview")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())

        if df.shape[0] < SMALL_DATASET_THRESHOLD:
            st.warning(
                f"This dataset has only {df.shape[0]} samples. With fewer than "
                f"{SMALL_DATASET_THRESHOLD} samples, cross-validation results can be "
                f"unstable. Treat results with caution — see Week 1, Day 6 for why."
            )

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        st.subheader("2. Choose Target and Features")
        all_cols = df.columns.tolist()
        target_col = st.selectbox("Target column (what to predict)", all_cols)

        default_features = [c for c in numeric_cols if c != target_col]
        feature_cols = st.multiselect(
            "Feature columns (inputs)", [c for c in all_cols if c != target_col],
            default=default_features
        )

        clean_text_cols = st.multiselect(
            "Any feature columns with mixed text/numbers (e.g. '269 HRB')? Select to auto-clean:",
            feature_cols
        )

        group_col = st.selectbox(
            "Optional: column identifying samples sharing the same group "
            "(e.g. composition ID) — uses GroupKFold to prevent Group Leakage",
            ["None"] + all_cols
        )

        if st.button("Run Analysis", type="primary"):
            if not feature_cols:
                st.error("Select at least one feature column.")
                st.stop()

            work_df = df.copy()
            for col in clean_text_cols:
                work_df[col] = clean_mixed_column(work_df[col])

            work_df = work_df[work_df[target_col].notnull()].copy()
            if len(work_df) == 0:
                st.error("No rows have a valid target value. Cannot proceed.")
                st.stop()

            non_numeric_features = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(work_df[c])]
            high_card_cols = [c for c in non_numeric_features
                               if work_df[c].nunique() > HIGH_CARDINALITY_THRESHOLD]
            if high_card_cols:
                st.warning(
                    f"These categorical columns have more than {HIGH_CARDINALITY_THRESHOLD} "
                    f"unique values: {high_card_cols}. Encoding them will create many new "
                    f"columns. Consider removing them or grouping categories first."
                )

            if non_numeric_features:
                if encoding_strategy == "Label Encoding":
                    for c in non_numeric_features:
                        le = LabelEncoder()
                        work_df[c] = le.fit_transform(work_df[c].astype(str))
                    st.info(f"Label-encoded categorical columns: {non_numeric_features}")
                else:
                    work_df = pd.get_dummies(work_df, columns=non_numeric_features)
                    expanded_features = []
                    for c in feature_cols:
                        if c in non_numeric_features:
                            expanded_features += [col for col in work_df.columns if col.startswith(f"{c}_")]
                        else:
                            expanded_features.append(c)
                    feature_cols = expanded_features
                    st.info(f"One-hot encoded categorical columns: {non_numeric_features}")

            st.subheader("3. Data Cleaning")
            strategy_key = "median" if impute_strategy == "median" else "mean"
            work_df, feature_cols, log = clean_missing(work_df, feature_cols, strategy=strategy_key)
            if log:
                for line in log:
                    st.write("- " + line)
            else:
                st.write("No missing values found.")

            if not feature_cols:
                st.error("All feature columns were dropped due to missing data.")
                st.stop()

            X = work_df[feature_cols]
            y = work_df[target_col]

            if X.isnull().values.any() or y.isnull().values.any():
                st.error("Data still contains missing values after cleaning.")
                st.stop()

            leakage_suspects = check_target_leakage(X, y, feature_cols)
            if leakage_suspects:
                st.warning(
                    "Possible Target Leakage detected:\n\n" +
                    "\n".join([f"- **{c}**: correlation = {corr:.3f}" for c, corr in leakage_suspects])
                )

            if group_col != "None":
                groups = work_df[group_col]
                cv_strategy = GroupKFold(n_splits=cv_folds)
                cv_kwargs = {"groups": groups}
                st.info(f"Using GroupKFold on '{group_col}' to prevent Group Leakage.")
            else:
                cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_kwargs = {}

            samples_per_fold = len(X) // cv_folds
            if samples_per_fold < 5:
                st.warning(
                    f"Only ~{samples_per_fold} samples per fold with {cv_folds}-fold CV. "
                    f"Consider lowering the number of folds in the sidebar."
                )

            st.subheader("4. Model Comparison (Cross-Validated)")
            models = build_models()

            comparison_rows = []
            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring="r2", **cv_kwargs)
                    comparison_rows.append({"Model": name, "Mean R2": scores.mean(), "Std R2": scores.std()})
                except Exception as e:
                    st.warning(f"{name} failed during cross-validation: {e}")

            if not comparison_rows:
                st.error("All models failed. Check your feature/target selection.")
                st.stop()

            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df.style.format({"Mean R2": "{:.3f}", "Std R2": "{:.3f}"}))

            best_model_name = comparison_df.loc[comparison_df["Mean R2"].idxmax(), "Model"]
            best_std = comparison_df.loc[comparison_df["Mean R2"].idxmax(), "Std R2"]
            st.success(f"Recommended model: **{best_model_name}**")
            if best_std > 0.15:
                st.warning(f"Std across folds is {best_std:.3f} — fairly high. Treat with caution.")

            st.subheader("5. Feature Importance")
            rf_for_importance = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_for_importance.fit(X, y)
            importances = pd.Series(rf_for_importance.feature_importances_, index=feature_cols)
            importances = importances.sort_values(ascending=False)
            actual_top_n = min(top_n, len(importances))
            st.bar_chart(importances.head(actual_top_n))
            top_features = importances.head(actual_top_n).index.tolist()

            st.subheader("6. Final Model Performance (Cross-Validated)")
            X_top = work_df[top_features]
            final_model = models[best_model_name]

            final_cv_r2 = cross_val_score(final_model, X_top, y, cv=cv_strategy, scoring="r2", **cv_kwargs)
            final_cv_mae = -cross_val_score(final_model, X_top, y, cv=cv_strategy,
                                             scoring="neg_mean_absolute_error", **cv_kwargs)

            col1, col2 = st.columns(2)
            col1.metric("Cross-Validated R2 (mean)", f"{final_cv_r2.mean():.3f}")
            col2.metric("Cross-Validated MAE (mean)", f"{final_cv_mae.mean():.2f}")
            st.caption(f"R2 Std across folds: {final_cv_r2.std():.3f}")

            X_train, X_test, y_train, y_test = train_test_split(
                X_top, y, test_size=0.2, random_state=42
            )
            final_model.fit(X_train, y_train)
            predictions = final_model.predict(X_test)
            results_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": predictions,
                "Error": abs(y_test.values - predictions)
            })

            st.subheader("7. Visualization Dashboard")
            fig = build_dashboard(importances, results_df, target_col, y, actual_top_n)
            st.pyplot(fig)

            st.subheader("8. Export Results")
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="prediction_results.csv",
                mime="text/csv"
            )

            final_model.fit(X_top, y)
            st.session_state["target_col"] = target_col
            st.session_state["top_features"] = top_features
            st.session_state["group_col"] = group_col
            st.session_state["cv_folds"] = cv_folds
            st.session_state["top_n"] = actual_top_n
            st.session_state["best_model_name"] = best_model_name
            st.session_state["impute_strategy"] = strategy_key
            st.session_state["analysis_done"] = True

            st.success("Analysis saved. Go to **Tab 2: Train in Colab**.")
    else:
        st.info("Upload a CSV file to get started.")

with tab2:
    st.header("Step-by-Step: Train Your Model in Google Colab")

    if not st.session_state.get("analysis_done"):
        st.warning("Run an analysis in **Tab 1** first.")
    else:
        target_col = st.session_state["target_col"]
        top_features = st.session_state["top_features"]
        group_col = st.session_state["group_col"]
        cv_folds = st.session_state["cv_folds"]
        top_n = st.session_state["top_n"]
        best_model_name = st.session_state["best_model_name"]
        impute_strategy = st.session_state["impute_strategy"]

        st.success(f"Recommended model: **{best_model_name}** | Target: **{target_col}**")

        st.markdown("""
### Follow these steps:

**Step 1** — Open [colab.research.google.com](https://colab.research.google.com), new notebook.
**Step 2** — Copy the script below into a single cell.
**Step 3** — Run it, upload your CSV when prompted.
**Step 4** — Wait for training to finish.
**Step 5** — `trained_model.joblib` downloads automatically.
**Step 6** — Go to Tab 3, upload that file, start predicting.
""")

        script_text = generate_colab_script(
            target_col, top_features, group_col, cv_folds, top_n,
            impute_strategy, best_model_name
        )

        st.subheader("Your Customized Colab Script")
        st.code(script_text, language="python")

        st.download_button(
            label="Download this script as a .py file",
            data=script_text,
            file_name="train_model_colab.py",
            mime="text/plain"
        )

with tab3:
    st.header("Make Predictions with a Trained Model")

    model_file = st.file_uploader("Upload trained_model.joblib", type=["joblib"], key="predict_upload")

    if model_file is not None:
        bundle = joblib.load(model_file)
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        target_column = bundle["target_column"]
        model_name = bundle.get("model_name", "Unknown")

        st.success(f"Loaded model: **{model_name}** — predicts **{target_column}**")

        if "cv_r2_mean" in bundle:
            col1, col2, col3 = st.columns(3)
            col1.metric("CV R2 (mean)", f"{bundle['cv_r2_mean']:.3f}")
            col2.metric("CV R2 (std)", f"{bundle['cv_r2_std']:.3f}")
            col3.metric("CV MAE (mean)", f"{bundle['cv_mae_mean']:.2f}")

        pred_tab1, pred_tab2 = st.tabs(["Single Sample", "Batch CSV"])

        with pred_tab1:
            input_values = {}
            cols = st.columns(min(3, len(feature_columns)))
            for i, feature in enumerate(feature_columns):
                with cols[i % len(cols)]:
                    input_values[feature] = st.number_input(feature, value=0.0, format="%.4f")

            if st.button("Predict", type="primary"):
                input_df = pd.DataFrame([input_values])[feature_columns]
                prediction = model.predict(input_df)[0]
                st.metric(f"Predicted {target_column}", f"{prediction:.2f}")

        with pred_tab2:
            batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_upload")

            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file, encoding="latin1")
                except Exception as e:
                    st.error(f"Could not read file: {e}")
                    st.stop()

                missing_cols = [c for c in feature_columns if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"Uploaded file is missing required columns: {missing_cols}")
                else:
                    X_batch = batch_df[feature_columns]
                    if X_batch.isnull().values.any():
                        X_batch = X_batch.fillna(X_batch.mean())

                    predictions = model.predict(X_batch)
                    result_df = batch_df.copy()
                    result_df[f"Predicted_{target_column}"] = predictions
                    st.dataframe(result_df)

                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download predictions as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
    else:
        st.info("Upload a trained_model.joblib file to get started.")