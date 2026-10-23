# Tabar (D): Optimal Outlier Rate Selection for GNSS Time Series

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python framework for determining the optimal outlier rate and benchmarking anomaly detection algorithms in 3D GNSS coordinate time series using the multi-criteria **Tabar ($D$)** ideal signal distance metric.

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
├── all_stations_summary.xlsx                # Master workbook
└── Station_<Name>/
    ├── methods_summary.xlsx                 # Detailed metrics
    ├── 01_time_series_overview.pdf          # Raw mean-centered series vs. Huber detrended residuals
    ├── 02_contamination_sweep.pdf           # Tabar (D) optimization curves across c in [0.01, 0.35]
    ├── 03_distance_comparison.pdf           # Horizontal bar ranking of methods by Tabar (D)
    ├── 04_decomposition_stacked.pdf         # Stacked bar breakdown of D² into M1², M2², and M3² (%)
    ├── 05_methods_landscape.pdf             # Trade-off scatter: % Data Removed vs. Tabar (D)
    ├── methods_diagnostics/
    │   └── <Method>_diagnostic.pdf          # Direct/Collateral outliers, 3D/2D scatters & histograms
    └── methods_cleaned_timeseries/
        └── <Method>_cleaned.pdf             # 3D and 2D spatial projections of inliers
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
