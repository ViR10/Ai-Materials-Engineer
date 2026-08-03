"""
Materials Visualization Dashboard
Day 4 Project - AI Materials Engineer Journey

Builds a 2x2 dashboard of charts from the real Carbon Steel + Stainless Steel
datasets used in Day 3:
  1. Bar Chart    - Average UTS by Category
  2. Scatter Plot - Hardness vs UTS (relationship check)
  3. Histogram    - Distribution of UTS across all samples
  4. Line Plot    - Carbon Steel only: average Carbon% vs Hardness (sorted trend)

This reuses the same data loading/cleaning logic from Day 3
(pd.concat, forward-fill category extraction, regex numeric cleaning).
"""

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
CARBON_FILE = "Carbon Steel.csv"
STAINLESS_FILE = "Stainless Steel.csv"
FILE_ENCODING = "latin1"
CORE_COLUMNS = ["SAE Grade", "Conditions", "Category",
                "UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"]
# ============================================================


def load_carbon_steel():
    df = pd.read_csv(CARBON_FILE, encoding=FILE_ENCODING)
    df["Category"] = "Carbon Steel"
    return df


def load_stainless_steel():
    df = pd.read_csv(STAINLESS_FILE, encoding=FILE_ENCODING)
    is_header_row = df["UTS (MPa)"].isnull() & df["SAE Grade"].notnull()

    if not is_header_row.any():
        df["Category"] = "Stainless Steel"
        df = df[df["UTS (MPa)"].notnull()].copy()
        return df

    df["Category"] = df["SAE Grade"].where(is_header_row)
    df["Category"] = df["Category"].ffill()
    df["Category"] = (
        df["Category"]
        .str.replace(r"\(b\)", "", regex=True)
        .str.replace("stainless steels", "Stainless", case=False, regex=False)
        .str.strip()
    )
    df = df[df["UTS (MPa)"].notnull()].copy()
    return df


def clean_numeric_column(series):
    extracted = series.astype(str).str.extract(r"(\d+\.?\d*)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def clean_missing(df, columns):
    for col in columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    return df


def main():
    # ---- Load and combine data (same as Day 3) ----
    carbon_raw = load_carbon_steel()
    stainless = load_stainless_steel()
    df = pd.concat([carbon_raw[CORE_COLUMNS], stainless[CORE_COLUMNS]], ignore_index=True)

    for col in ["UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"]:
        df[col] = clean_numeric_column(df[col])
    df = clean_missing(df, ["UTS (MPa)", "YS (MPa)", "Elongation (%)", "Hardness (HB)"])

    # ---- Build the 2x2 dashboard ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Materials Visualization Dashboard - Steel Grade Data", fontsize=14)

    # Chart 1: Bar chart - average UTS by category
    category_avg = df.groupby("Category")["UTS (MPa)"].mean().sort_values()
    axes[0, 0].bar(category_avg.index, category_avg.values, color="steelblue")
    axes[0, 0].set_xlabel("Category")
    axes[0, 0].set_ylabel("Average UTS (MPa)")
    axes[0, 0].set_title("Average UTS by Category")
    axes[0, 0].tick_params(axis="x", rotation=15)

    # Chart 2: Scatter plot - Hardness vs UTS
    axes[0, 1].scatter(df["Hardness (HB)"], df["UTS (MPa)"], color="darkred", alpha=0.6)
    axes[0, 1].set_xlabel("Hardness (HB)")
    axes[0, 1].set_ylabel("UTS (MPa)")
    axes[0, 1].set_title("Hardness vs UTS (All Samples)")

    # Chart 3: Histogram - UTS distribution
    axes[1, 0].hist(df["UTS (MPa)"], bins=15, color="coral", edgecolor="black")
    axes[1, 0].set_xlabel("UTS (MPa)")
    axes[1, 0].set_ylabel("Number of Samples")
    axes[1, 0].set_title("Distribution of UTS (All Samples)")

    # Chart 4: Line plot - Carbon Steel only, avg Carbon% vs Hardness (sorted trend)
    carbon_raw["Avg_Carbon"] = (carbon_raw["C (Min)"] + carbon_raw["C (Max)"]) / 2
    carbon_raw["Hardness_Clean"] = clean_numeric_column(carbon_raw["Hardness (HB)"])
    trend = carbon_raw.dropna(subset=["Avg_Carbon", "Hardness_Clean"])
    trend = trend.groupby("Avg_Carbon")["Hardness_Clean"].mean().sort_index()

    axes[1, 1].plot(trend.index, trend.values, marker="o", color="green")
    axes[1, 1].set_xlabel("Average Carbon Content (%)")
    axes[1, 1].set_ylabel("Hardness (HB)")
    axes[1, 1].set_title("Carbon Steel: Carbon% vs Hardness")

    plt.tight_layout()
    plt.savefig("materials_dashboard.png")
    plt.show()

    print("Dashboard saved as materials_dashboard.png")


if __name__ == "__main__":
    main()