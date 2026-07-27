"""
Mechanical Properties Analyzer
Day 2 Project - AI Materials Engineer Journey

This program analyzes mechanical properties of multiple material samples
using NumPy for fast, vectorized numerical operations.

Concepts used: NumPy arrays, 2D arrays (matrices), indexing, slicing,
boolean filtering, broadcasting, and statistical functions.
"""

import numpy as np


def main():
    # Sample materials and their properties
    # Columns: Density (g/cm3), Hardness (HB), UTS (MPa)
    names = ["Steel", "Aluminum", "Titanium", "Copper", "Brass", "Nickel"]

    data = np.array([
        [7.85, 180, 400],   # Steel
        [2.70, 60, 95],     # Aluminum
        [4.50, 349, 950],   # Titanium
        [8.96, 35, 210],    # Copper
        [8.40, 130, 550],   # Brass
        [8.90, 110, 462]    # Nickel
    ])

    properties = ["Density (g/cm3)", "Hardness (HB)", "UTS (MPa)"]

    print("=" * 50)
    print("MECHANICAL PROPERTIES ANALYZER")
    print("=" * 50)

    print(f"\nTotal Samples: {data.shape[0]}")
    print(f"Properties per Sample: {data.shape[1]}")

    # ---- Column-wise statistics ----
    print("\n--- Statistics per Property ---")
    for i in range(len(properties)):
        column = data[:, i]
        print(f"\n{properties[i]}:")
        print(f"  Mean   : {np.mean(column):.2f}")
        print(f"  Median : {np.median(column):.2f}")
        print(f"  Std Dev: {np.std(column):.2f}")
        print(f"  Min    : {np.min(column)} | Max: {np.max(column)}")

    # ---- Find extremes ----
    max_density_idx = np.argmax(data[:, 0])
    max_hardness_idx = np.argmax(data[:, 1])
    min_density_idx = np.argmin(data[:, 0])

    print("\n--- Key Findings ---")
    print(f"Densest Material   : {names[max_density_idx]}")
    print(f"Hardest Material   : {names[max_hardness_idx]}")
    print(f"Lightest Material  : {names[min_density_idx]}")

    # ---- Boolean filtering: high-strength materials ----
    high_strength_mask = data[:, 2] >= 400
    high_strength_names = [names[i] for i in range(len(names)) if high_strength_mask[i]]
    print(f"\nHigh-Strength Materials (UTS >= 400 MPa): {high_strength_names}")

    # ---- Broadcasting example: unit conversion ----
    # Convert Density (g/cm3 -> kg/m3) and Hardness (HB -> HV) in one operation
    conversion_factors = np.array([1000, 1.05, 1])   # applies per column
    converted_data = data * conversion_factors

    print("\n--- Converted Units (Density in kg/m3, Hardness in HV) ---")
    for i in range(len(names)):
        print(f"{names[i]}: Density={converted_data[i, 0]:.1f} kg/m3, "
              f"Hardness={converted_data[i, 1]:.1f} HV")


if __name__ == "__main__":
    main()