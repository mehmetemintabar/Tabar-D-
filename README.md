# Tabar (D): Optimal Outlier & Contamination Rate Selection for GNSS Time Series

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python framework for determining the optimal outlier contamination rate ($c_{\text{opt}}$) and benchmarking anomaly detection algorithms in 3D GNSS coordinate time series ($\text{East, North, Up}$) using the multi-criteria **Tabar ($D$)** ideal signal distance metric.

---

## 🚀 Key Features

- **Harmonic Detrending & Dual-Regression Estimation:**
  - Converts raw station coordinates into mean-centered millimeter deviations ($\text{mm}$) and relative decimal years ($t$).
  - Models linear tectonic velocity alongside **annual** ($1\text{ cpy}$) and **semi-annual** ($2\text{ cpy}$) harmonic signals:
    $$y(t) = a + v t + \sum_{k=1}^{2} \left[ c_k \sin(2\pi k t) + d_k \cos(2\pi k t) \right] + \varepsilon(t)$$
  - **OLS with HAC (Newey–West, `maxlags=14`)** covariance estimation is employed to obtain autocorrelation-consistent station velocities ($v$) and formal uncertainties ($\sigma_v$).
  - **Huber M-Estimator (`epsilon=1.35`)** is used to extract robust detrended residuals ($\varepsilon$) uncorrupted by gross outliers.

- **3D Joint Robust Standardization & Union Masking:**
  - Residuals across $\text{E, N, U}$ components are jointly standardized using the component-wise median and normal-consistent Median Absolute Deviation ($\text{MAD}_{\text{norm}} = 1.4826 \times \text{MAD}$).
  - Applies **epoch-level union masking** ($\mathcal{M}_{\text{union}} = \mathcal{M}_E \cup \mathcal{M}_N \cup \mathcal{M}_U$) to preserve 3D temporal alignment across all coordinate components, while explicitly tracking **Direct Outliers** vs. **Collateral Outliers** in diagnostic reports.

- **Optimal Contamination Sweep ($c_{\text{opt}}$):**
  - Performs an automated grid search across contamination rates $c \in [0.01, 0.35]$ ($\Delta c = 0.01$) for parametric and machine learning anomaly detectors.
  - Identifies the optimal hyperparameter $c_{\text{opt}}$ that minimizes the composite **Tabar ($D$)** distance to the ideal noise floor.

---

## 📐 Mathematical Formulation of Tabar ($D$)

To avoid both under-cleaning (leaving gross outliers that inflate variance) and over-cleaning (truncating the core Gaussian signal and depleting sample size), **Tabar ($D$)** evaluates cleaned time series in a three-dimensional criterion space $(M_1, M_2, M_3)$:

1. **$M_1$ — Gaussian Consistency (Tail-to-Core Ratio):**
   Evaluates the convergence between the root-mean-square ($\text{RMS}$) and the normal-scaled $\text{MAD}$ of the cleaned residuals across all three components. For an ideal outlier-free Gaussian process, $\text{RMS} \approx \text{MAD}_{\text{norm}}$, yielding $M_1 \to 0$:
   $$M_1 = \sqrt{\frac{1}{3} \sum_{i \in \{E,N,U\}} \left[ \ln \left( \frac{\text{RMS}_{\text{clean}, i}}{\text{MAD}_{\text{clean}, i}} \right) \right]^2}$$

2. **$M_2$ — Scale Preservation (Core Dispersion Stability):**
   Penalizes distortion of the underlying noise scale by comparing the cleaned robust dispersion against the baseline robust dispersion:
   $$M_2 = \sqrt{\frac{1}{3} \sum_{i \in \{E,N,U\}} \left[ \ln \left( \frac{\text{MAD}_{\text{clean}, i}}{\text{MAD}_{\text{base}, i}} \right) \right]^2}$$

3. **$M_3$ — Data Retention Barrier:**
   Acts as a non-linear penalty against excessive epoch removal, parameterized by the retention exponent $\gamma = 1.5$:
   $$M_3 = \left( \frac{N_{\text{total}}}{N_{\text{kept}}} \right)^{\gamma} - 1$$

4. **Composite Tabar ($D$) Metric & Variance Decomposition:**
   The overall distance to the ideal signal state $(0, 0, 0)$ is defined as the Euclidean norm:
   $$D = \sqrt{M_1^2 + M_2^2 + M_3^2}$$
   where the relative contribution of each criterion to $D^2$ is decomposed as $\%M_k = \frac{M_k^2}{D^2} \times 100$ for $k \in \{1, 2, 3\}$.

---

## 🔍 Evaluated Outlier Detection Methods

The framework benchmarks **8 outlier detection algorithms** grouped into two methodological categories:

| Category | Method | Parameterization / Threshold Rule |
| :--- | :--- | :--- |
| **Automatic (Fixed-Threshold)** | **IQR** | Component-wise $[Q_1 - 1.5\,\text{IQR},\; Q_3 + 1.5\,\text{IQR}]$ |
| **Automatic (Fixed-Threshold)** | **Hampel Identifier** | Component-wise $\vert{}\varepsilon_i - \text{med}(\varepsilon_i)\vert{} > 3.0\,\text{MAD}_{\text{norm}}$ |
| **Automatic (Fixed-Threshold)** | **Z-Score** | Component-wise $\vert{}z_i\vert{} > 3.0$ |
| **Automatic (Fixed-Threshold)** | **Mahalanobis (Classic)** | 3D multivariate distance $d_M^2 > \chi^2_{0.975, df=3}$ |
| **Contamination-Driven (Sweep)** | **Elliptic Envelope** | Minimum Covariance Determinant (MCD), swept over $c \in [0.01, 0.35]$ |
| **Contamination-Driven (Sweep)** | **Isolation Forest** | Tree-based ensemble isolation (`random_state=42`), swept over $c \in [0.01, 0.35]$ |
| **Contamination-Driven (Sweep)** | **kNN Distance** | $k$-Nearest Neighbors distance (PyOD), swept over $c \in [0.01, 0.35]$ |
| **Contamination-Driven (Sweep)** | **COPOD** | Copula-Based Outlier Detection (PyOD), swept over $c \in [0.01, 0.35]$ |

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
| `u_estimate` | Up (vertical) coordinate component | Meters ($\text{m}$) |

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
        └── <Method>_cleaned.pdf             # 2-page PDF: 3D and 2D spatial projections of cleaned inliers
```

---

## 📖 Citation

If you use **Tabar (D)** or this codebase in your research, please cite:

```bibtex
@software{tabar2026tabard,
  author       = {Tabar, Mehmet Emin},
  title        = {{Tabar (D): Optimal Outlier and Contamination Rate Selection for GNSS Coordinate Time Series}},
  year         = {2026},
  url          = {[https://github.com/](https://github.com/)<your-username>/Tabar-D},
  license      = {MIT}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
