# AI Materials Property Predictor — Project Documentation

**Week 1 Capstone Project — AI Materials Engineer Journey**

This document describes the design, features, benefits, and limitations of the
unified `app.py` tool in detail. For a day-by-day learning log, see the main
[`README.md`](../README.md).

---

## 1. What This Project Is

A single Streamlit application that takes **any** materials CSV dataset and walks a
user through the full applied machine learning workflow used throughout Week 1:

```
Upload CSV -> Clean -> Check for Leakage -> Compare Models -> Select Features
-> Recommend a Model -> Generate a Colab Training Script -> Train -> Predict
```

It is not built around one specific dataset. It has been tested against three
structurally different real datasets:
- NIMS Steel Fatigue Database (437 samples, 19 numeric features, no missing values)
- SAE Carbon Steel grade reference table (mixed text/number values, e.g. "269 HRB")
- SAE Stainless Steel grade reference table (category labels embedded as data rows,
  columns that are 100% missing, near-duplicate unit-converted columns)

---

## 2. Architecture

One file, three tabs, connected by `st.session_state`:

| Tab | Purpose |
|---|---|
| **1. Analyze & Recommend** | Full pipeline: load, clean, encode, leakage-check, cross-validate 3 models, select top features, recommend the best model |
| **2. Train in Colab** | Auto-generates a Google Colab script, pre-filled with the exact target/features/settings chosen in Tab 1, plus numbered instructions |
| **3. Predict** | Loads a `trained_model.joblib` file (produced by the Tab 2 script) and makes single-sample or batch predictions without retraining |

**Why this separation matters:** training (compute-heavy, done once) and inference
(lightweight, done often) are kept apart - the same pattern used in real ML deployment.
Tab 3 can be used completely independently of Tabs 1-2, by anyone who just has a
`.joblib` file.

---

## 3. Pipeline Steps (What Happens Automatically)

1. **Load** - reads the CSV with `latin1` encoding (handles special characters that
   break plain UTF-8 reads).
2. **Mixed text/number cleaning** - user marks any columns like `"269 HRB"`; a regex
   extracts the numeric part.
3. **Categorical encoding** - One-Hot (default) or Label Encoding (user's choice).
   A warning is shown if a categorical column has more than 20 unique values, since
   One-Hot Encoding would explode the feature count.
4. **Missing value handling** -
   - Columns that are **100% missing** are dropped (a mean/median cannot be computed
     from zero values).
   - Columns with **some** missing values are filled with the column mean or median
     (user's choice).
   - Columns more than 50% missing are filled but flagged with a warning.
5. **Target Leakage check** - any feature with >0.98 correlation to the target is
   flagged (commonly caused by unit-converted duplicate columns).
6. **Cross-validation strategy** - shuffled `KFold` by default (prevents Distribution
   Mismatch on sorted data); switches to `GroupKFold` if the user specifies a group
   column (prevents Group Leakage when samples share an underlying group, e.g. same
   composition tested under different heat treatments).
7. **Model comparison** - Linear Regression, Random Forest, and Gradient Boosting are
   each cross-validated; the one with the highest mean R2 is recommended.
8. **Feature Selection** - Random Forest feature importance ranks all features; the
   top N (configurable) are used for the final model.
9. **Final reported metric** - cross-validated R2/MAE on the top features (not a
   single train/test split). A single held-out split is used only to draw the
   Actual-vs-Predicted plot, and is explicitly labeled as being for visualization only.

---

## 4. Benefits

- **Dataset-agnostic** - works on any materials CSV with numeric properties and a
  target column, verified on 3 structurally different real datasets.
- **Handles real-world messiness automatically** - entirely-missing columns, mixed
  text/number values, embedded category labels, encoding issues.
- **Leakage-aware by default** - Target Leakage detection and Group Leakage
  prevention (GroupKFold) are built in, not something the user has to remember.
- **Shuffle-safe cross-validation** - avoids the Distribution Mismatch failure mode
  found during Day 6 (sorted data + unshuffled `cv=5` produced catastrophically wrong
  negative R2 scores).
- **Reports a statistically honest metric** - cross-validated, not a single lucky (or
  unlucky) split.
- **Training/inference separation** - a genuine deployment pattern, not just a demo
  script.
- **Transparent, not a black box** - every cleaning decision (what was dropped, what
  was filled, and with what value) is printed to the user.

---

## 5. Known Limitations

These are honest, current gaps, not hidden:

| # | Limitation | Why It Matters | Possible Future Fix |
|---|---|---|---|
| 1 | Only 3 models available (Linear Regression, Random Forest, Gradient Boosting) | No SVR, XGBoost, neural networks | Add more models to the comparison dictionary |
| 2 | No hyperparameter tuning | Random Forest always uses `n_estimators=100`; may not be optimal | Add `GridSearchCV` or `RandomizedSearchCV` |
| 3 | Regression only | Cannot handle classification targets (e.g. Pass/Fail) | Add a target-type detector and classification models |
| 4 | Target Leakage check is correlation-based only (threshold 0.98) | Won't catch leakage that isn't near-perfectly linear, or leakage below the threshold | Add more sophisticated leakage heuristics |
| 5 | Feature importance always computed via Random Forest | Even when Linear Regression is the recommended model, feature ranking may not reflect what matters most to a linear model | Use model-appropriate importance (e.g. standardized coefficients for Linear Regression) |
| 6 | No feature scaling applied automatically | Not an issue for the current 3 models (tree-based models are scale-invariant; Linear Regression's R2 is unaffected by scaling), but would matter if SVR/KNN/Neural Networks are added later | Add optional `StandardScaler` step |
| 7 | GroupKFold requires the user to manually identify the correct grouping column | No automatic detection of likely group structure | Add a heuristic to suggest candidate group columns |
| 8 | No outlier detection | A few extreme values can skew mean-based imputation and Linear Regression | Add a simple z-score or IQR-based outlier flag |
| 9 | Session state resets if the Streamlit app restarts | Tab 2/3 depend on Tab 1 having been run in the same session | Persist the last analysis to disk (e.g. a local JSON/pickle) |
| 10 | No model interpretability beyond feature importance | No SHAP values or partial dependence plots | Add SHAP as an optional deeper-analysis section |

**Most important limitation to know before trusting a result:** with small datasets
(under ~30 samples), cross-validation itself can be unstable - the app warns about
this, but a small sample size is a limitation of the data, not something the app can
fix. This was directly observed during Day 6: a 12-sample toy dataset produced negative
R2 scores under 5-fold CV even though the underlying relationship was genuine.

---

## 6. How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser:
1. **Tab 1** - upload a CSV, pick target/features, click "Run Analysis"
2. **Tab 2** - copy the generated script into a new Google Colab notebook, run it,
   upload the same CSV when prompted, download the resulting `trained_model.joblib`
3. **Tab 3** - upload that `.joblib` file, get predictions instantly

---

## 7. Datasets Used for Testing

| Dataset | Samples | Notable Characteristics |
|---|---|---|
| NIMS Steel Fatigue Database | 437 | Clean, but sorted by target (triggered the shuffle bug in Day 6) |
| SAE Carbon Steel | 190 | Hardness values mixed with unit text ("269 HRB") |
| SAE Stainless Steel | 65 | Category labels embedded as rows; two 100%-missing columns; a near-duplicate unit-converted column (Target Leakage) |

---

## 8. Version History

- **v1** - initial unified 3-tab tool (Analyze, Train Guide, Predict)
- **v2** (current) - added Gradient Boosting, cross-validated final metric (replacing
  a single-split metric), Mean/Median imputation choice, One-Hot/Label Encoding choice,
  small-dataset warning, high-cardinality warning