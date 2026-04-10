# TM-Cluster-ML

**Machine Learning Insights into Structural Phase Transitions and Magnetic-Geometric Coupling in Transition Metal Nanoclusters**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

---

## Overview

TM-Cluster-ML is a reproducible machine learning framework for analyzing **transition-metal nanoclusters (N ≤ 55)** based on data derived from the Quantum Cluster Database (QCD).

The workflow integrates **DFT-derived data**, geometric descriptor engineering, and multiple ML models to uncover structure–property relationships in nanoclusters.

---

## Key Features

- Multi-model ML benchmark (7 models with 5-fold cross-validation)
- Binding energy prediction at DFT level accuracy
- Detection of **structural phase transitions (2D → 3D)** using unsupervised learning
- Analysis of **magnetic–geometric coupling**
- Analysis of **electronic–geometric interactions (PDP)**
- Identification of **magic numbers** from global minimum structures
- Generation of **24 high-resolution publication-quality figures**
- Automated **Word report with tables and insights**
- Exploratory **feature-based nano-alloy interpolation** (hypothesis generation only)
- Automatic generation of **idealized XYZ structures** for alloy candidates

---

## Important Scientific Notes

- All predictions are based on **machine learning models trained on monometallic clusters only**
- Nano-alloy results are **exploratory extrapolations**, not validated DFT results
- Generated XYZ files are **initial geometries** and require **DFT optimization**
- No new DFT calculations are performed in this workflow

---

## Installation

Install required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost shap statsmodels python-docx openpyxl 
````
(Also included in **Step 1 install_requirements.py** file)


---

## Dataset

* Main dataset: `Data9M.xlsx` (included)
* Source: Quantum Cluster Database (QCD) https://muellergroup.jhu.edu/qcd/

Reference:

> Manna, S. et al. *Scientific Data* **10**, 308 (2023)
> [https://doi.org/10.1038/s41597-023-02200-4](https://doi.org/10.1038/s41597-023-02200-4)

---

## How to Run

```bash
python main_analysis.py
```

---

## What the Code Does

The pipeline performs the following steps:

1. Data preprocessing and metal identification
2. Extraction of atomic coordinates
3. Geometric feature engineering (radius of gyration, asphericity, compactness, etc.)
4. Train/test split with stratification by metal
5. Training of 7 ML models:

   * ExtraTrees
   * RandomForest
   * XGBoost
   * LightGBM
   * CatBoost
   * GradientBoosting
   * Neural Network
6. Model evaluation using:

   * Test MAE
   * R² score
   * 5-fold cross-validation
7. Selection of best-performing model
8. Scientific analysis:

   * Magic number identification
   * Phase transition detection (KMeans)
   * Feature importance and SHAP analysis
9. Generation of figures and tables
10. Exploratory nano-alloy interpolation
11. Automatic Word report generation

---

## Output Structure

```
TRANSITION_METALS_2026/
├── Figures/                         # 24 figures (600 dpi)
├── Tables/                          # Excel tables
├── Exploratory_Nano_Alloy_Predictions/
│   └── *.xyz                        # Generated structures
├── Full_Report_*.docx               # Final report
└── *.zip                            # Complete archive
```

---

## Figures

The script generates **24 publication-quality figures**, including:

* Model validation (parity plot, learning curve)
* Feature importance and SHAP analysis
* Stability vs cluster size
* Magic numbers
* PCA and t-SNE projections
* Structural descriptors distributions
* Magnetic moment vs stability
* Phase transition visualization
* Partial dependence plots
* 3D stability landscape
* Nano-alloy screening results

---

## Limitations

* Alloy predictions are **not physically validated**
* No explicit electronic structure calculations beyond input dataset
* Phase transition detection is **data-driven (unsupervised)**

---

## Citation

If you use this code, please cite:

> TM-Cluster-ML: Machine Learning Analysis of Transition-Metal Nanoclusters (2026)

Dataset citation:

> Manna, S. et al. *A database of low-energy atomically precise nanoclusters*.
> Scientific Data **10**, 308 (2023).
> [https://doi.org/10.1038/s41597-023-02200-4](https://doi.org/10.1038/s41597-023-02200-4) and https://muellergroup.jhu.edu/qcd/

---

## License

This project is licensed under the **MIT License**.

---

## Author

Postdoctoral Researcher in Computational Chemistry
Specializing in DFT, catalysis, and ML-driven materials design

```
