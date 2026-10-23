# Tabar (D): Optimal Outlier & Contamination Rate Selection for GNSS Time Series

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python framework for determining the optimal outlier contamination rate and benchmarking anomaly detection algorithms in 3D GNSS coordinate time series using the multi-criteria **Tabar ($D$)** ideal signal distance metric.

---

## 🚀 Key Features

- **Optimal Contamination Sweep ($c_{\text{opt}}$):**
  - Performs an automated grid search across contamination rates $c \in [0.01, 0.35]$ ($\Delta c = 0.01$) for parametric and machine learning anomaly detectors.
  - Identifies the optimal hyperparameter $c_{\text{opt}}$ that minimizes the composite **Tabar ($D$)** distance to the ideal noise floor.

---

## 📦 Requirements & Installation

Ensure you have **Python 3.8+** installed. Install the required scientific and machine learning dependencies via `pip`:

```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn pyod openpyxl
```

---

## 📂 Input Data Format

Place an Excel workbook named `GNSS.xlsx` in the root directory. Each worksheet represents an individual **GNSS station** (the sheet name is used as the station identifier) and must contain the following columns:

| Column Name | Description | Unit |
| :--- | :--- | :--- |
| `Date` | Observation timestamp (supports `DD.MM.YYYY` or `YYYY-MM-DD`) | Date |
| `e_estimate` | East coordinate component | Meters ($\text{m}$) |
| `n_estimate` | North coordinate component | Meters ($\text{m}$) |
| `u_estimate` | Up coordinate component | Meters ($\text{m}$) |

---

## ⚙️ Usage & Output Structure

Run the main script to process all stations in `GNSS.xlsx`:

```bash
python Tabar.py
```

All statistical tables and publication-grade vector graphics (`.pdf`) are automatically exported to the `results/` directory:

```text
results/
├── all_stations_summary.xlsx                # Master workbook (ALL_STATIONS, per-station tabs, DECOMPOSITION)
└── Station_<Name>/
    ├── methods_summary.xlsx                 # Detailed metrics (RMS, STD, MAD, velocity, sigma, M1-M3, D)
    ├── 01_time_series_overview.pdf          # Raw mean-centered series vs. Huber detrended residuals
    ├── 02_contamination_sweep.pdf           # Tabar (D) optimization curves across c in [0.01, 0.35]
    ├── 03_distance_comparison.pdf           # Horizontal bar ranking of methods by Tabar (D)
    ├── 04_decomposition_stacked.pdf         # Stacked bar breakdown of D² into M1², M2², and M3² (%)
    ├── 05_methods_landscape.pdf             # Trade-off scatter: % Data Removed vs. Tabar (D)
    ├── methods_diagnostics/
    │   └── <Method>_diagnostic.pdf          # 4-page PDF: Direct/Collateral outliers, 3D/2D scatters & histograms
    └── methods_cleaned_timeseries/
        └── <Method>_cleaned.pdf             # 2-page PDF: 3D and 2D spatial projections of inliers
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
