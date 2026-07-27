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

## Next Steps
Day 3: Pandas — structured, labeled datasets for cleaning and analyzing materials data.