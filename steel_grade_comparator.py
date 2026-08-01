"""
Steel Grade Comparator
Day 3 Project - AI Materials Engineer Journey

Datasets: Carbon Steel and Stainless Steel property/composition sheets
(SAE grade reference tables).

These two files share the same column structure, so they are COMBINED
using concatenation (pd.concat) - not merge. Merge joins tables on a
common key column (adds columns); concat stacks tables with the same
columns on top of each other (adds rows). Here we have two files with
matching columns but different materials, so concat is the correct tool.

Data quality notes handled in this script:
1. The Stainless Steel file has "section header" rows embedded in the
   data (e.g. "Ferritic stainless steels(b)") that mark sub-categories
   but aren't real data rows. These are extracted using forward-fill
   and then removed.
2. Composition (Ni, Cr, Mo, etc.) Min/Max columns are heavily incomplete
   because many grades simply don't specify those elements - this is
   NOT random missing data, so those columns are excluded from the
   cleaned analysis rather than being blindly imputed.

Workflow:
1. Load both files
2. Extract clean categories from the Stainless Steel file
3. Concatenate into one unified dataset
4. Clean missing values in the core property columns only
5. Category-wise statistics on UTS
6. Best/worst performer per category
7. QC pass/fail report
"""

import pandas as pd

# ============================================================
# CONFIG
# ============================================================
CARBON_FILE = "Carbon Steel.csv"
STAINLESS_FILE = "Stainless Steel.csv"
FILE_ENCODING = "latin1"  # these files contain special characters

# Core columns we trust and will actually analyze
CORE_COLUMNS = ["SAE Grade", "Conditions", "Category",
                "UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"]

PERFORMANCE_COLUMN = "UTS (MPa)"
QC_THRESHOLD = 500  # minimum acceptable UTS (MPa)
# ============================================================


def load_carbon_steel():
    """Load the Carbon Steel file. All rows belong to one category."""
    df = pd.read_csv(CARBON_FILE, encoding=FILE_ENCODING)
    df["Category"] = "Carbon Steel"
    return df


def load_stainless_steel():
    """
    Load the Stainless Steel file and extract sub-categories
    (Ferritic / Martensitic / Austenitic) from embedded header rows.
    """
    df = pd.read_csv(STAINLESS_FILE, encoding=FILE_ENCODING)

    # Header rows have a SAE Grade value but no UTS value
    is_header_row = df["UTS (MPa)"].isnull() & df["SAE Grade"].notnull()

    if not is_header_row.any():
        print("  WARNING: No section-header rows detected in this file.")
        print("  Column names found:", df.columns.tolist())
        print("  UTS missing count:", df["UTS (MPa)"].isnull().sum())
        print("  Assigning a single generic category instead.")
        df["Category"] = "Stainless Steel"
        df = df[df["UTS (MPa)"].notnull()].copy()
        return df

    # Mark category only on header rows, then forward-fill downward
    df["Category"] = df["SAE Grade"].where(is_header_row)
    df["Category"] = df["Category"].ffill()

    # Clean up the category text: "Ferritic stainless steels(b)" -> "Ferritic Stainless"
    df["Category"] = (
        df["Category"]
        .str.replace(r"\(b\)", "", regex=True)
        .str.replace("stainless steels", "Stainless", case=False, regex=False)
        .str.strip()
    )

    # Now remove header rows and blank trailing rows (no UTS = not real data)
    df = df[df["UTS (MPa)"].notnull()].copy()
    return df


def clean_numeric_column(series):
    """
    Some values in this dataset mix numbers with text, e.g. '269 HRB'
    instead of a plain number. This extracts just the numeric part
    and converts the whole column to a proper numeric type.
    Anything that still can't be converted becomes NaN (missing).
    """
    extracted = series.astype(str).str.extract(r"(\d+\.?\d*)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def clean_missing(df, columns):
    """Fill missing numeric values with the column mean, for given columns only."""
    for col in columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
            print(f"  Filled {missing_count} missing '{col}' values with mean: {mean_value:.2f}")
    return df


def main():
    # ---- Step 1: Load both files ----
    print(f"\n{'=' * 60}")
    print("LOADING DATA")
    print(f"{'=' * 60}")
    carbon = load_carbon_steel()
    stainless = load_stainless_steel()
    print(f"Carbon Steel: {carbon.shape[0]} rows")
    print(f"Stainless Steel: {stainless.shape[0]} rows (after removing header/blank rows)")
    print("Stainless categories found:", stainless["Category"].unique())

    # ---- Step 2: Concatenate into one dataset ----
    df = pd.concat([carbon[CORE_COLUMNS], stainless[CORE_COLUMNS]], ignore_index=True)
    print(f"\nCombined dataset shape: {df.shape}")

    # Some columns (like Hardness) mix numbers with text (e.g. '269 HRB').
    # Clean them into proper numeric columns before doing any math on them.
    for col in ["UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"]:
        df[col] = clean_numeric_column(df[col])

    # ---- Step 3: Clean missing values (core columns only) ----
    print(f"\n{'=' * 60}")
    print("CLEANING MISSING VALUES (core property columns only)")
    print(f"{'=' * 60}")
    df = clean_missing(df, ["UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"])

    # ---- Step 4: Category-wise statistics ----
    print(f"\n{'=' * 60}")
    print(f"CATEGORY-WISE STATISTICS ({PERFORMANCE_COLUMN})")
    print(f"{'=' * 60}")
    stats = df.groupby("Category")[PERFORMANCE_COLUMN].agg(["mean", "max", "min", "count"])
    print(stats.round(1))

    # ---- Step 5: Best and worst performer per category ----
    print(f"\n{'=' * 60}")
    print(f"BEST PERFORMER PER CATEGORY (by {PERFORMANCE_COLUMN})")
    print(f"{'=' * 60}")
    best_idx = df.groupby("Category")[PERFORMANCE_COLUMN].idxmax()
    print(df.loc[best_idx, ["SAE Grade", "Category", PERFORMANCE_COLUMN]])

    print(f"\n{'=' * 60}")
    print(f"WORST PERFORMER PER CATEGORY (by {PERFORMANCE_COLUMN})")
    print(f"{'=' * 60}")
    worst_idx = df.groupby("Category")[PERFORMANCE_COLUMN].idxmin()
    print(df.loc[worst_idx, ["SAE Grade", "Category", PERFORMANCE_COLUMN]])

    # ---- Step 6: QC pass/fail report ----
    print(f"\n{'=' * 60}")
    print(f"QUALITY CONTROL REPORT (Threshold: {PERFORMANCE_COLUMN} >= {QC_THRESHOLD})")
    print(f"{'=' * 60}")
    df["QC_Status"] = df[PERFORMANCE_COLUMN].apply(lambda x: "Pass" if x >= QC_THRESHOLD else "Fail")

    pass_rate = df.groupby("Category")["QC_Status"].apply(
        lambda x: (x == "Pass").sum() / len(x) * 100
    )
    print("Pass Rate by Category (%):")
    print(pass_rate.round(1))


if __name__ == "__main__":
    main()