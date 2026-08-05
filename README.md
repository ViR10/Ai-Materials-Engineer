# AI Materials Engineer Journey

My transition from Materials Engineering to AI Engineering, specializing in
Materials Informatics — learning Python, Data Science, and Machine Learning
and applying every concept directly to real materials engineering problems.

---

## Day 1: Python Fundamentals + Material Database

### What I Learned
- Variables and Data Types (int, float, str, bool, list)
- Functions (reusable calculations, e.g. density, stress)
- Loops (batch processing multiple samples)
- Classes (Material and MaterialDatabase objects)
- Exception Handling (crash-proof code)
- File Handling (permanent data storage)

### Project: Material Database
A command-line application that manages materials data:
- Add / Delete / Search materials
- Stores Density, Hardness, and UTS (Ultimate Tensile Strength)
- Data is permanently saved to a `.txt` file

### How to Run
```bash
python materials_database.py
```

### Files
- `materials_database.py` — main program
- `materials_data.txt` — saved data (auto-generated)

---

## Day 2: NumPy for Materials Data

### What I Learned
- NumPy arrays (1D and 2D) and why they outperform plain Python lists for numerical work
- Vectorization — applying one operation across an entire dataset without loops
- Indexing, slicing, and boolean filtering (e.g. selecting samples above a strength threshold)
- 2D arrays (matrices) — representing multiple samples with multiple properties, like a spreadsheet
- Broadcasting — applying different operations across columns in a single step
- Statistical functions: mean, median, standard deviation, min, max, argmax, argmin

### Project: Mechanical Properties Analyzer
A command-line tool that analyzes a dataset of material samples (Steel, Aluminum,
Titanium, Copper, Brass, Nickel) across three properties: Density, Hardness, and
Ultimate Tensile Strength (UTS).

Features:
- Column-wise statistics (mean, median, std dev, min, max)
- Identifies the densest, hardest, and lightest material
- Filters high-strength materials using boolean masking
- Converts units (Density g/cm3 → kg/m3, Hardness HB → HV) using broadcasting

### How to Run
```bash
python mechanical_properties_analyzer.py
```

### Files
- `mechanical_properties_analyzer.py` — main program

### Research Referenced
- Maqsood et al. (2024), "The Future of Material Scientists in an Age of Artificial Intelligence", *Advanced Science*
- Goswami, Deka, Roy (2023), "Artificial intelligence in material engineering: A review", *Advanced Engineering Materials*

---

## Day 3: Pandas for Materials Data

### What I Learned
- DataFrames — labeled, structured tables (vs. NumPy's plain numeric arrays)
- Dataset inspection: `.head()`, `.shape()`, `.info()`, `.describe()`
- Column and row selection, including boolean filtering
- Missing value detection and handling (`isnull()`, `fillna()`, `dropna()`)
- Merge vs. Concatenate — joining tables on a common key vs. stacking tables with matching columns
- GroupBy aggregation — category-wise statistics (mean, max, min, count) in a single line
- Cleaning real-world messy data: values embedded as text (e.g. "269 HRB"), category
  labels embedded as data rows instead of columns, and file encoding issues

### Project: Steel Grade Comparator
A command-line tool built on real SAE grade reference data for Carbon Steel and
Stainless Steel (Ferritic, Martensitic, Austenitic).

Features:
- Combines two real datasets with matching columns using `pd.concat()`
- Extracts hidden sub-category labels from the Stainless Steel file using forward-fill
- Cleans mixed text/number values (e.g. "269 HRB" → 269) using regex extraction
- Category-wise statistics on Ultimate Tensile Strength (UTS)
- Identifies the best and worst performing grade in each category
- Generates a Quality Control (QC) pass/fail report against a strength threshold

Result sanity check: Martensitic Stainless Steel showed the highest average UTS
(consistent with it being heat-treatable for strength), while Ferritic Stainless
showed the lowest (consistent with it being valued for ductility and corrosion
resistance rather than peak strength) — confirming the cleaned data produced
metallurgically sensible results.

### How to Run
```bash
python steel_grade_comparator.py
```

### Files
- `steel_grade_comparator.py` — main program
- `Carbon Steel.csv`, `Stainless Steel.csv` — source data (SAE grade reference tables)

---

## Day 4: Matplotlib for Materials Data Visualization

### What I Learned
- Line Plots — visualizing trends between two continuous, ordered variables
  (e.g. Carbon % vs Hardness)
- Bar Charts — comparing values across categories (e.g. average UTS by steel category)
- Scatter Plots — checking whether two properties are related (e.g. Hardness vs UTS),
  and the distinction between correlation and causation
- Histograms — visualizing the distribution of a single variable (e.g. spread of UTS
  values across many samples)
- Subplots — arranging multiple charts in a single figure/grid for a dashboard-style view

### Project: Materials Visualization Dashboard
A 2x2 dashboard built on the same real Carbon Steel + Stainless Steel dataset from Day 3:
- Bar Chart — average UTS by category (Carbon Steel, Ferritic, Martensitic, Austenitic)
- Scatter Plot — Hardness vs UTS across all samples, to check the relationship
- Histogram — distribution of UTS values across all samples
- Line Plot — Carbon Steel only: average Carbon content (%) vs Hardness

### How to Run
```bash
python materials_visualization_dashboard.py
```

### Files
- `materials_visualization_dashboard.py` — main program
- `materials_dashboard.png` — output chart (auto-generated)

---

## Day 5: Machine Learning Basics (Scikit-learn)

### What I Learned
- Features (X) vs Target (y) — framing a materials problem in ML terms
- Train/Test Split — why a model must be evaluated on unseen data to avoid overfitting
- Linear Regression — fitting a best-fit line/plane and interpreting slope (`coef_`)
  and intercept
- Evaluation Metrics — R² Score (variation explained) and MAE (average error in real units),
  and why both are needed together
- Random Forest Regressor — ensemble learning with decision trees, and when it outperforms
  Linear Regression (complex, non-linear relationships)
- Feature Importance — identifying which inputs actually drive the prediction
- Multicollinearity — the risk of using two features that carry the same information

### Project: Fatigue Property Predictor
A model comparison tool built on the real NIMS Steel Fatigue Database (437 samples),
predicting Fatigue Strength from composition (C, Si, Mn, P, S, Ni, Cr, Cu, Mo) and heat
treatment parameters (Normalizing, Carburizing, Diffusion, Quenching, Tempering).

Features:
- Trains and compares two models: Linear Regression and Random Forest Regressor
- Evaluates both using R² Score and MAE
- Reports the top 5 most important features driving the prediction

Results: Linear Regression achieved R² = 0.973 (MAE = 24.93 MPa); Random Forest achieved
R² = 0.986 (MAE = 18.80 MPa), confirming Random Forest as the stronger model. The top
predictive features were Normalizing Temperature, Carburizing Temperature, Quenching
Media Temperature, Chromium %, and Carburizing Time — showing that heat treatment
parameters influenced Fatigue Strength more than raw composition in this dataset.

### How to Run
```bash
python fatigue_property_predictor.py
```

### Files
- `fatigue_property_predictor.py` — main program
- `fatigue.csv` — source data (NIMS Steel Fatigue Database)

---

## Next Steps
Day 6: Feature Engineering — scaling, encoding, cross-validation, and improving model reliability.