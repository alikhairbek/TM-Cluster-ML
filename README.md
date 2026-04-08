# TM-Cluster-ML
Machine Learning pipeline for transition metal nanoclusters (N ≤ 55) using the Quantum Cluster Database (QCD). Features: structural phase transition detection, magnetic-geometric coupling, d-band center analysis, and virtual nano-alloy screening with realistic XYZ generation. Produces 25 publication-ready figures, 8 tables, and a full  Word report.
# TM-Cluster-ML

**Machine Learning Suite for Transition Metal Nanoclusters (N ≤ 55)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

---

## Overview

**TM-Cluster-ML** is a complete, reproducible, and publication-ready machine learning pipeline for analyzing transition metal nanoclusters using the Quantum Cluster Database (QCD).

**Key Capabilities:**
- High-accuracy binding energy prediction (LightGBM: MAE = 0.02776 eV/atom, R² = 0.9989)
- Automatic detection of 2D → 3D structural phase transitions
- Discovery of magnetic-geometric and electronic-geometric coupling
- d-band center analysis for catalytic relevance
- Virtual screening of binary nano-alloys with realistic XYZ structure generation
- Professional Word report with 25 figures and 8 tables

---

## Installation

### Step 1: Install Requirements
```bash
python Step_1_install_requirements.py

**Machine Learning Suite for Transition Metal Nanoclusters (N ≤ 55)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

---

**## Overview**

**TM-Cluster-ML** is a complete, reproducible, and publication-ready machine learning pipeline for analyzing transition metal nanoclusters using the Quantum Cluster Database (QCD).

**Key Capabilities:**
- High-accuracy binding energy prediction (LightGBM: MAE = 0.02776 eV/atom, R² = 0.9989)
- Automatic detection of 2D → 3D structural phase transitions
- Discovery of magnetic-geometric and electronic-geometric coupling
- d-band center analysis for catalytic relevance
- Virtual screening of binary nano-alloys with realistic XYZ structure generation
- Professional Word report with 25 figures and 8 tables

---

## Installation

### Step 1: Install Requirements
```bash
python Step_1_install_requirements.py
Step 2: Download the Dataset
Download Data9M.xlsx from the official Quantum Cluster Database (QCD):

Source: Quantum Cluster Database (QCD) – Manna et al., Scientific Data (2023)
Link: http://muellergroup.jhu.edu/qcd
DOI: https://doi.org/10.1038/s41597-023-02200-4

Place the file in the root folder of the project.

How to Run
Bashpython main_analysis.py
The script will automatically:

Load Data9M.xlsx
Train and evaluate 7 ML models with 5-fold CV
Generate 25 high-resolution figures
Detect phase transitions
Perform virtual nano-alloy screening
Generate realistic XYZ files for alloys
Create a full Word report

Expected runtime: 4–7 minutes.

Project Structure
textTM-Cluster-ML/
├── Figures/                    # 25 publication-ready figures
├── Tables/                     # 8 professional tables
├── Predicted_Nano_Alloys/      # 27 generated XYZ files
├── Full_Report_*.docx          # Complete report
├── Supporting_Information/     # Supplementary figures & tables
├── Step_1_install_requirements.py
├── main_analysis.py
└── Data9M.xlsx                 # (must be downloaded)

Citation
If you use this code or generated data in your research, please cite:
TM-Cluster-ML: Machine Learning Suite for Transition Metal Nanoclusters, GitHub (2026).
Original Dataset Citation:
Manna, S. et al. A database of low-energy atomically precise nanoclusters. Sci. Data 10, 308 (2023). https://doi.org/10.1038/s41597-023-02200-4

License
This project is licensed under the MIT License.
