import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import HuberRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN
from pyod.models.copod import COPOD

input = "GNSS.xlsx"
output = Path("results")
components = ["E", "N", "U"]
contamgrid = np.round(np.arange(0.01, 0.36, 0.01), 2)
random_state = 42
min_valid_sweep_points = 4
retention_power = 1.5
hac_maxlags = 14
comp_to_col = {'E': 'e_estimate', 'N': 'n_estimate', 'U': 'u_estimate'}

methodcolors = {
    'IQR': ('#008080', '#DC143C'),
    'Hampel': ('#1E90FF', '#FF4500'),
    'Mahalanobis (Classic)': ('#3CB371', '#9400D3'),
    'Elliptic Envelope': ('#708090', '#FF1493'),
    'Isolation Forest': ('#4682B4', '#FF8C00'),
    'kNN Distance': ('#4169E1', '#FF0000'),
    'COPOD': ('#20B2AA', '#4B0082'),
    'Z-Score': ('#5F9EA0', '#800000'),
}

methodlinecolor = {m: c[1] for m, c in methodcolors.items()}
colorraw = '#708090'
colorresidual = '#FF1493'

decompcolors = {
    'M1': '#00BFFF',
    'M2': '#FFA500',
    'M3': '#FF3131',
}

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False,
})

def load_stations(filepath):
    xls = pd.ExcelFile(filepath)
    stations = {}
    needed = ['Date'] + list(comp_to_col.values())
    for sheet in xls.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Sheet '{sheet}' missing columns: {missing}")
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        except Exception:
            df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna(subset=needed).sort_values('Date').reset_index(drop=True)
        stations[sheet] = df
    return stations

def to_decimal_year(dates):
    years = dates.dt.year
    start = pd.to_datetime(years.astype(str) + '-01-01')
    end = pd.to_datetime((years + 1).astype(str) + '-01-01')
    return years + (dates - start) / (end - start)

def preprocess(df):
    out = df.copy()
    for comp, col in comp_to_col.items():
        out[f'{comp}_mm'] = (out[col] - out[col].mean()) * 1000.0
    out['decyear'] = to_decimal_year(out['Date'])
    out['t'] = out['decyear'] - out['decyear'].iloc[0]
    return out

def design_matrix(t):
    return np.column_stack([
        np.ones_like(t), t,
        np.sin(2 * np.pi * t), np.cos(2 * np.pi * t),
        np.sin(4 * np.pi * t), np.cos(4 * np.pi * t),
    ])

def ols_fit(t, y):
    X = design_matrix(t)
    n, p = X.shape
    if n - p <= 0:
        return np.nan, np.nan, y - np.mean(y)

    try:
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_maxlags})
        v = float(results.params[1])
        vsig = float(results.bse[1])
        res = np.asarray(results.resid)
        return v, vsig, res
    except Exception as e:
        return np.nan, np.nan, y - np.mean(y)

def robust_fit(t, y, epsilon=1.35, max_iter=200):
    X = design_matrix(t)
    try:
        m = HuberRegressor(epsilon=epsilon, max_iter=max_iter,
                           fit_intercept=False).fit(X, y)
        fitted = m.predict(X)
        res = y - fitted
        return float(m.coef_[1]), res
    except Exception:
        v, _, res = ols_fit(t, y)
        return v, res

def m_iqr(X, k=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    return (X < q1 - k * iqr) | (X > q3 + k * iqr)

def m_hampel(X, threshold=3.0):
    med = np.median(X, axis=0)
    mad = stats.median_abs_deviation(X, axis=0, scale='normal')
    mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[1]):
        if mad[i] != 0:
            mask[:, i] = np.abs(X[:, i] - med[i]) > threshold * mad[i]
    return mask

def m_zscore(X, threshold=3.0):
    z = stats.zscore(X, axis=0)
    return np.abs(z) > threshold

def m_mahalanobis_classic(X):
    mu = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
        diff = X - mu
        d2 = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        return d2 > stats.chi2.ppf(0.975, df=X.shape[1])
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[0], dtype=bool)

def m_elliptic(X, c):
    return EllipticEnvelope(contamination=c, random_state=random_state).fit_predict(X) == -1

def m_iforest(X, c):
    return IsolationForest(contamination=c, random_state=random_state).fit_predict(X) == -1

def m_knn(X, c):
    mdl = KNN(contamination=c)
    mdl.fit(X)
    return mdl.labels_.astype(bool)

def m_copod(X, c):
    mdl = COPOD(contamination=c)
    mdl.fit(X)
    return mdl.labels_.astype(bool)

automethods = {
    'IQR': m_iqr,
    'Hampel': m_hampel,
    'Mahalanobis (Classic)': m_mahalanobis_classic,
    'Z-Score': m_zscore,
}

contammethods = {
    'Elliptic Envelope': m_elliptic,
    'Isolation Forest': m_iforest,
    'kNN Distance': m_knn,
    'COPOD': m_copod,
}

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

def mean_metric(d):
    return float(np.mean(list(d.values())))

def core_stats(res):
    return dict(
        rms=rms(res),
        std=float(np.std(res, ddof=1)),
        mad=float(stats.median_abs_deviation(res, scale='normal')),
    )

def baseline_panel(t, components_y):
    out = {}
    for comp, y in components_y.items():
        v, vsig, _ = ols_fit(t, y)
        _, res = robust_fit(t, y)
        cs = core_stats(res)
        cs['velocity_sigma'] = vsig
        cs['velocity'] = v
        cs['residuals'] = res
        out[comp] = cs
    return out

def cleaned_panel(t, components_y, mask):
    keep = ~mask
    if keep.sum() < 10:
        return None
    out = {}
    t_cln = t[keep]
    for comp, y in components_y.items():
        y_cln = y[keep]
        v, vsig, _ = ols_fit(t_cln, y_cln)
        _, res = robust_fit(t_cln, y_cln)
        cs = core_stats(res)
        cs['velocity_sigma'] = vsig
        cs['velocity'] = v
        cs['residuals'] = res
        out[comp] = cs
    return out

def split_mask(mask):
    per_comp = {}
    if mask.ndim == 2:
        per_comp['E'] = mask[:, 0]
        per_comp['N'] = mask[:, 1]
        per_comp['U'] = mask[:, 2]
        union_mask = mask.any(axis=1)
    else:
        per_comp['E'] = mask
        per_comp['N'] = mask
        per_comp['U'] = mask
        union_mask = mask
    per_comp['_union'] = union_mask
    return union_mask, per_comp

def robust_scale_3d(residuals_dict):
    X = np.column_stack([residuals_dict['E'], residuals_dict['N'], residuals_dict['U']])
    med = np.median(X, axis=0)
    mad = stats.median_abs_deviation(X, axis=0, scale='normal')
    mad[mad == 0] = 1e-6
    return (X - med) / mad

def joint_mask_auto(detect_fn, residuals_dict):
    X_scaled = robust_scale_3d(residuals_dict)
    return split_mask(detect_fn(X_scaled))

def joint_mask_contam(detect_fn, residuals_dict, c):
    X_scaled = robust_scale_3d(residuals_dict)
    return split_mask(detect_fn(X_scaled, c))

def safe_log_ratio(num, den):
    if num is None or den is None:
        return np.nan
    if np.isnan(num) or np.isnan(den) or num <= 0 or den <= 0:
        return np.nan
    return float(np.log(num / den))

def compute_M1(cleaned):
    sq_logs = []
    for comp in components:
        lr = safe_log_ratio(cleaned[comp]['rms'], cleaned[comp]['mad'])
        if np.isnan(lr):
            return np.nan
        sq_logs.append(lr ** 2)
    return float(np.sqrt(np.mean(sq_logs)))

def compute_M2(cleaned, baseline):
    sq_logs = []
    for comp in components:
        lr = safe_log_ratio(cleaned[comp]['mad'], baseline[comp]['mad'])
        if np.isnan(lr):
            return np.nan
        sq_logs.append(lr ** 2)
    return float(np.sqrt(np.mean(sq_logs)))

def ideal_signal_distance(cleaned, baseline, n_cln, n_total, return_components=False):
    if (cleaned is None or baseline is None or
            n_cln is None or n_total is None or
            n_cln <= 0 or n_total <= 0):
        return (np.nan, np.nan, np.nan, np.nan) if return_components else np.nan

    M1 = compute_M1(cleaned)
    M2 = compute_M2(cleaned, baseline)
    if np.isnan(M1) or np.isnan(M2):
        return (np.nan, M1, M2, np.nan) if return_components else np.nan

    M3 = float(((n_total / n_cln) ** retention_power) - 1.0)
    D = float(np.sqrt(M1 ** 2 + M2 ** 2 + M3 ** 2))
    return (D, M1, M2, M3) if return_components else D

def find_optimal_c(c_values, dist_values):
    valid = ~np.isnan(dist_values)
    if valid.sum() < min_valid_sweep_points:
        return np.nan, 'insufficient_data'

    cv = np.asarray(c_values)[valid]
    dv = np.asarray(dist_values)[valid]
    best_idx = int(np.argmin(dv))
    return float(cv[best_idx]), 'optimal_found'

def empty_sweep_row(c, pct_removed=np.nan):
    return dict(c=c, rms_m=np.nan, pct_removed=pct_removed,
                ideal_distance=np.nan,
                M1=np.nan, M2=np.nan, M3=np.nan,
                rms_E=np.nan, rms_N=np.nan, rms_U=np.nan)

def parameter_sweep(t, components_y, residuals, baseline,
                    detect_fn, method_name, c_grid=contamgrid):
    n_total = len(t)
    rows = []

    for c in c_grid:
        try:
            mask, _ = joint_mask_contam(detect_fn, residuals, c)
        except Exception as e:
            rows.append(empty_sweep_row(c))
            continue

        keep = ~mask
        n_cln = int(keep.sum())

        if n_cln < 10:
            rows.append(empty_sweep_row(c, pct_removed=100 * mask.mean()))
            continue

        cleaned = cleaned_panel(t, components_y, mask)
        if cleaned is None:
            rows.append(empty_sweep_row(c, pct_removed=100 * mask.mean()))
            continue

        rms_clean = mean_metric({k: cleaned[k]['rms'] for k in components})
        dist, M1_s, M2_s, M3_s = ideal_signal_distance(
            cleaned, baseline, n_cln, n_total, return_components=True)

        rows.append(dict(c=c, rms_m=rms_clean, pct_removed=100 * mask.mean(),
                         ideal_distance=dist, M1=M1_s, M2=M2_s, M3=M3_s,
                         rms_E=cleaned['E']['rms'], rms_N=cleaned['N']['rms'],
                         rms_U=cleaned['U']['rms']))

    df = pd.DataFrame(rows)
    c_opt, status = find_optimal_c(df['c'].values, df['ideal_distance'].values)
    return df, c_opt, status

def imp_pct(orig, clean):
    if orig is None or np.isnan(orig) or orig == 0 or np.isnan(clean):
        return np.nan
    return 100.0 * (orig - clean) / orig

def build_rows(station, method, category, c_opt, opt_status,
               baseline, cleaned, mask, n_total,
               distance=np.nan, M1=np.nan, M2=np.nan, M3=np.nan):
    rows = []

    c1 = M1 ** 2 if not np.isnan(M1) else np.nan
    c2 = M2 ** 2 if not np.isnan(M2) else np.nan
    c3 = M3 ** 2 if not np.isnan(M3) else np.nan
    D2 = (c1 + c2 + c3) if not any(np.isnan([c1, c2, c3])) else np.nan

    if D2 is not None and not np.isnan(D2) and D2 > 0:
        p1, p2, p3 = 100 * c1 / D2, 100 * c2 / D2, 100 * c3 / D2
    else:
        p1 = p2 = p3 = np.nan

    for comp in components:
        b = baseline[comp]
        if cleaned is None:
            row = {
                'station': station, 'component': comp,
                'method': method, 'category': category,
                'optimal_c': c_opt, 'opt_status': opt_status,
                'ideal_distance': distance,
                'M1': M1, 'M2': M2, 'M3': M3,
                'M1sq': c1, 'M2sq': c2, 'M3sq': c3,
                'M1_pct': p1, 'M2_pct': p2, 'M3_pct': p3,
                'rms_orig': b['rms'], 'rms_clean': np.nan,
                'std_orig': b['std'], 'std_clean': np.nan,
                'mad_orig': b['mad'], 'mad_clean': np.nan,
                'rms_improvement_pct': np.nan,
                'std_improvement_pct': np.nan,
                'mad_improvement_pct': np.nan,
                'velocity_orig': b['velocity'], 'velocity': np.nan,
                'velocity_sigma_orig': b['velocity_sigma'],
                'velocity_sigma': np.nan,
                'velocity_uncertainty_reduction_pct': np.nan,
                'n_total': n_total,
                'n_kept': np.nan, 'n_removed': np.nan,
                'pct_removed': np.nan,
            }
        else:
            cl = cleaned[comp]
            keep = ~mask
            row = {
                'station': station, 'component': comp,
                'method': method, 'category': category,
                'optimal_c': c_opt, 'opt_status': opt_status,
                'ideal_distance': distance,
                'M1': M1, 'M2': M2, 'M3': M3,
                'M1sq': c1, 'M2sq': c2, 'M3sq': c3,
                'M1_pct': p1, 'M2_pct': p2, 'M3_pct': p3,
                'rms_orig': b['rms'], 'rms_clean': cl['rms'],
                'std_orig': b['std'], 'std_clean': cl['std'],
                'mad_orig': b['mad'], 'mad_clean': cl['mad'],
                'rms_improvement_pct': imp_pct(b['rms'], cl['rms']),
                'std_improvement_pct': imp_pct(b['std'], cl['std']),
                'mad_improvement_pct': imp_pct(b['mad'], cl['mad']),
                'velocity_orig': b['velocity'], 'velocity': cl['velocity'],
                'velocity_sigma_orig': b['velocity_sigma'],
                'velocity_sigma': cl['velocity_sigma'],
                'velocity_uncertainty_reduction_pct':
                    imp_pct(b['velocity_sigma'], cl['velocity_sigma']),
                'n_total': n_total, 'n_kept': int(keep.sum()),
                'n_removed': int(mask.sum()),
                'pct_removed': 100.0 * mask.sum() / n_total,
            }
        rows.append(row)
    return rows

def process_station(name, df_raw):
    pp = preprocess(df_raw)
    t = pp['t'].values
    n_total = len(pp)
    components_y = {c: pp[f'{c}_mm'].values for c in components}

    baseline = baseline_panel(t, components_y)
    residuals = {c: baseline[c]['residuals'] for c in components}
    rows, sweep_curves, method_masks = [], {}, {}

    rows.extend(build_rows(name, 'Baseline', 'baseline',
                           0.0, 'none', baseline, baseline,
                           np.zeros(n_total, dtype=bool), n_total,
                           distance=np.nan, M1=np.nan, M2=np.nan, M3=np.nan))

    for mname, fn in automethods.items():
        try:
            mask, per_comp = joint_mask_auto(fn, residuals)
        except Exception as e:
            rows.extend(build_rows(name, mname, 'automatic',
                                   np.nan, 'apply_failed',
                                   baseline, None,
                                   np.zeros(n_total, dtype=bool), n_total,
                                   M1=np.nan, M2=np.nan, M3=np.nan))
            continue

        cleaned = cleaned_panel(t, components_y, mask)
        dist_auto, M1_a, M2_a, M3_a = np.nan, np.nan, np.nan, np.nan
        if cleaned is not None:
            n_cln = int((~mask).sum())
            dist_auto, M1_a, M2_a, M3_a = ideal_signal_distance(
                cleaned, baseline, n_cln, n_total, return_components=True)

        rows.extend(build_rows(name, mname, 'automatic',
                               np.nan, 'fixed_threshold',
                               baseline, cleaned, mask, n_total,
                               distance=dist_auto,
                               M1=M1_a, M2=M2_a, M3=M3_a))
        per_comp['_union'] = mask
        method_masks[mname] = per_comp

    for mname, fn in contammethods.items():
        try:
            sweep_df, c_opt, status = parameter_sweep(
                t, components_y, residuals, baseline, fn, method_name=mname)
        except Exception as e:
            rows.extend(build_rows(name, mname, 'contamination',
                                   np.nan, 'sweep_failed',
                                   baseline, None,
                                   np.zeros(n_total, dtype=bool), n_total,
                                   M1=np.nan, M2=np.nan, M3=np.nan))
            continue

        sweep_curves[mname] = sweep_df

        if np.isnan(c_opt):
            rows.extend(build_rows(name, mname, 'contamination',
                                   np.nan, status,
                                   baseline, None,
                                   np.zeros(n_total, dtype=bool), n_total,
                                   M1=np.nan, M2=np.nan, M3=np.nan))
            continue

        match = sweep_df.loc[np.isclose(sweep_df['c'], c_opt, atol=1e-6)]
        if len(match):
            dist_at_opt = float(match['ideal_distance'].iloc[0])
            M1_c = float(match['M1'].iloc[0])
            M2_c = float(match['M2'].iloc[0])
            M3_c = float(match['M3'].iloc[0])
        else:
            dist_at_opt = M1_c = M2_c = M3_c = np.nan

        try:
            mask, per_comp = joint_mask_contam(fn, residuals, c_opt)
        except Exception as e:
            rows.extend(build_rows(name, mname, 'contamination',
                                   c_opt, 'apply_failed',
                                   baseline, None,
                                   np.zeros(n_total, dtype=bool), n_total,
                                   distance=dist_at_opt,
                                   M1=M1_c, M2=M2_c, M3=M3_c))
            continue

        cleaned = cleaned_panel(t, components_y, mask)
        rows.extend(build_rows(name, mname, 'contamination',
                               c_opt, status,
                               baseline, cleaned, mask, n_total,
                               distance=dist_at_opt,
                               M1=M1_c, M2=M2_c, M3=M3_c))
        per_comp['_union'] = mask
        method_masks[mname] = per_comp

    res_df = pd.DataFrame(rows)
    return res_df, residuals, sweep_curves, pp, method_masks

def plot_time_series_overview(pp, residuals, outpath, station_name):
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for i, comp in enumerate(components):
        ax_r, ax_d = axes[i, 0], axes[i, 1]
        ax_r.plot(pp['Date'], pp[f'{comp}_mm'], '.', ms=2, alpha=0.6, c=colorraw)
        ax_r.set_ylabel(f'{comp} (mm)')
        ax_r.grid(alpha=0.3)
        ax_r.set_title(f'{comp} - mean-centered (mm)')
        ax_d.plot(pp['Date'], residuals[comp], '.', ms=2, alpha=0.6, c=colorresidual)
        ax_d.axhline(0, color='k', lw=0.5)
        ax_d.set_ylabel(f'{comp} residual (mm)')
        ax_d.grid(alpha=0.3)
        ax_d.set_title(f'{comp} - Huber detrended residual')
    axes[-1, 0].set_xlabel('Date')
    axes[-1, 1].set_xlabel('Date')
    fig.suptitle(f'Station {station_name} - Time series overview', fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_sweep_curves(sweep_curves, optimal_cs, opt_statuses, outpath, station_name):
    if not sweep_curves:
        return
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    for m, g in sorted(sweep_curves.items()):
        col = methodlinecolor.get(m, '#FF1493')
        ax.plot(g['c'], g['ideal_distance'], '-', label=m,
                color=col, lw=1.6, alpha=0.9)
        c_opt = optimal_cs.get(m, np.nan)
        status = opt_statuses.get(m, 'unknown')
        if not np.isnan(c_opt) and status == 'optimal_found':
            row = g[np.isclose(g['c'], c_opt, atol=1e-6)]
            if len(row):
                ax.plot(c_opt, row['ideal_distance'].iloc[0], 'o', color=col,
                        ms=9, mec='black', mew=0.9)
    ax.axhline(0, color='k', lw=0.5)
    ax.set(xlabel='Contamination c',
           ylabel='Ideal Signal Distance',
           title=f'Station {station_name} - Ideal Distance sweep')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='best')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_score_comparison(df, outpath, station_name):
    df_plot = df[df['category'] != 'baseline'].drop_duplicates(subset=['method'])
    df_plot = df_plot.dropna(subset=['ideal_distance'])
    df_plot = df_plot.sort_values(by='ideal_distance', ascending=False)
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [methodcolors.get(m, ('#000000', '#9400D3'))[1]
              for m in df_plot['method']]
    bars = ax.barh(df_plot['method'], df_plot['ideal_distance'],
                   color=colors, alpha=0.85)
    ax.set_xlabel('Ideal Signal Distance')
    ax.set_title(f'Station {station_name} - Method Distance Comparison')
    ax.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        align = 'left' if width >= 0 else 'right'
        offset = 0.002 if width >= 0 else -0.002
        ax.text(width + offset, bar.get_y() + bar.get_height() / 2,
                f'{width:.4f}', va='center', ha=align,
                fontsize=9, fontweight='bold', color='#333333')

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_methods_landscape(df, outpath, station_name):
    df_plot = df[df['component'] == 'E'].drop_duplicates(subset=['method'])
    df_plot = df_plot[df_plot['category'] != 'baseline'].dropna(subset=['ideal_distance'])
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for _, row in df_plot.iterrows():
        method = row['method']
        dist = row['ideal_distance']
        pct = row['pct_removed']
        color = methodcolors.get(method, ('#000', '#FF0000'))[1]
        ax.scatter(pct, dist, color=color, s=160, edgecolor='black',
                   linewidth=1, label=method, alpha=0.85)

    ax.set_xlabel('Percentage of Data Removed (%)', fontweight='bold')
    ax.set_ylabel('Ideal Signal Distance', fontweight='bold')
    ax.set_title(f'Station {station_name} - Method Performance', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Methods")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_decomposition_stacked(df, outpath, station_name):
    df_plot = df[df['component'] == 'E'].drop_duplicates(subset=['method'])
    df_plot = df_plot[df_plot['category'] != 'baseline'].dropna(subset=['ideal_distance'])
    df_plot = df_plot.sort_values(by='ideal_distance')
    if df_plot.empty:
        return

    methods = df_plot['method'].tolist()
    c1 = df_plot['M1sq'].values
    c2 = df_plot['M2sq'].values
    c3 = df_plot['M3sq'].values

    p1 = df_plot['M1_pct'].values
    p2 = df_plot['M2_pct'].values
    p3 = df_plot['M3_pct'].values

    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(methods))

    color_m1 = decompcolors['M1']
    color_m2 = decompcolors['M2']
    color_m3 = decompcolors['M3']

    ax.barh(y, c1, color=color_m1, label='M1² (Consistency)')
    ax.barh(y, c2, left=c1, color=color_m2, label='M2² (Scale Preservation)')
    ax.barh(y, c3, left=c1 + c2, color=color_m3, label=f'M3² (Retention Barrier)')

    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel('D² components')
    ax.set_title(f'Station {station_name} - Decomposition of D²')

    ax.legend(loc='upper right')
    ax.grid(axis='x', alpha=0.3)

    for i, (total_val, m1_p, m2_p, m3_p) in enumerate(zip(c1 + c2 + c3, p1, p2, p3)):
        ax.text(total_val + 0.005, i, f"({m1_p:.1f}%, {m2_p:.1f}%, {m3_p:.1f}%)",
                va='center', fontsize=9, fontweight='bold', color='#333333')

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_method_diagnostic(pp, masks, results_df, outpath, method_name, station_name):
    if '_union' not in masks:
        return
    union_mask = masks['_union']
    n_total = len(pp)
    n_out = int(union_mask.sum())
    pct = 100.0 * n_out / n_total

    df = results_df[(results_df['station'] == station_name) &
                    (results_df['method'] == method_name)]
    c_opt = df['optimal_c'].iloc[0] if len(df) else np.nan
    c_label = (f"c={c_opt:.2f}"
               if isinstance(c_opt, (int, float)) and not np.isnan(c_opt)
               else "auto threshold")
    header = (f'{method_name} - {c_label}\n'
              f'Station {station_name} | n={n_total} | '
              f'epochs removed: {n_out} ({pct:.2f}%)')

    color_in, color_out = methodcolors.get(method_name, ('#008080', '#DC143C'))
    color_collat = '#A9A9A9'
    MS = 4

    with PdfPages(outpath) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        for i, comp in enumerate(components):
            ax = axes[i]
            vals = pp[f'{comp}_mm'].values
            comp_mask = masks[comp]
            pure_inlier = ~union_mask
            direct_outlier = comp_mask
            collateral_outlier = union_mask & ~comp_mask

            ax.plot(pp.loc[pure_inlier, 'Date'], vals[pure_inlier], 'o',
                    ms=MS, alpha=0.7, c=color_in, mec='none', label='Inliers')
            if collateral_outlier.sum() > 0:
                ax.plot(pp.loc[collateral_outlier, 'Date'],
                        vals[collateral_outlier], 'o', ms=MS, alpha=0.85,
                        c=color_collat, mec='none',
                        label='Collateral Outliers')
            if direct_outlier.sum() > 0:
                ax.plot(pp.loc[direct_outlier, 'Date'], vals[direct_outlier],
                        'o', ms=MS, alpha=0.85, c=color_out, mec='none',
                        label='Direct Outliers')
            ax.set_ylabel(f'{comp} (mm)')
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        axes[-1].set_xlabel('Date')
        fig.suptitle(f'Time series with outliers - {header}', fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        E = pp['E_mm'].values
        N = pp['N_mm'].values
        U = pp['U_mm'].values
        m_in = ~union_mask
        ax.scatter(E[m_in], N[m_in], U[m_in], c=color_in, s=14, alpha=0.6,
                   edgecolors='none', label=f'Inliers (n={m_in.sum()})')
        ax.scatter(E[union_mask], N[union_mask], U[union_mask], c=color_out,
                   s=14, alpha=0.85, edgecolors='none',
                   label=f'Outliers Total (n={union_mask.sum()})')
        ax.set_xlabel('E (mm)')
        ax.set_ylabel('N (mm)')
        ax.set_zlabel('U (mm)')
        ax.legend(loc='best')
        ax.set_title(f'3D scatter - {header}', fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        for ax, (a, b) in zip(axes, [('E', 'N'), ('E', 'U'), ('N', 'U')]):
            xa = pp[f'{a}_mm'].values
            yb = pp[f'{b}_mm'].values
            m_in = ~union_mask
            ax.scatter(xa[m_in], yb[m_in], c=color_in, s=14, alpha=0.6,
                       edgecolors='none', label='Inliers')
            ax.scatter(xa[union_mask], yb[union_mask], c=color_out, s=14,
                       alpha=0.85, edgecolors='none', label='Outliers Total')
            ax.set_xlabel(f'{a} (mm)')
            ax.set_ylabel(f'{b} (mm)')
            ax.set_title(f'{a} vs {b}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(f'2D projections - {header}', fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        for ax, comp in zip(axes, components):
            vals = pp[f'{comp}_mm'].values
            comp_mask = masks[comp]
            pure_inlier = ~union_mask
            direct_outlier = comp_mask
            collateral_outlier = union_mask & ~comp_mask

            bins = np.linspace(vals.min(), vals.max(), 50)
            ax.hist(vals[pure_inlier], bins=bins, color=color_in, alpha=0.85,
                    label='Inliers', edgecolor='white', linewidth=0.3)
            if collateral_outlier.sum() > 0:
                ax.hist(vals[collateral_outlier], bins=bins,
                        color=color_collat, alpha=0.85,
                        label='Collateral Outliers',
                        edgecolor='white', linewidth=0.3)
            if direct_outlier.sum() > 0:
                ax.hist(vals[direct_outlier], bins=bins, color=color_out,
                        alpha=0.85, label='Direct Outliers',
                        edgecolor='white', linewidth=0.3)
            ax.set_xlabel(f'{comp} (mm)')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {comp}')
            ax.legend(fontsize=9)
        fig.suptitle(f'Component distributions - {header}', fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

def plot_cleaned_timeseries(pp, masks, outpath, method_name, station_name):
    if '_union' not in masks:
        return
    union_mask = masks['_union']
    pure_inlier = ~union_mask
    if pure_inlier.sum() == 0:
        return

    color_in, _ = methodcolors.get(method_name, ('#008080', '#DC143C'))

    E_cln = pp.loc[pure_inlier, 'E_mm'].values
    N_cln = pp.loc[pure_inlier, 'N_mm'].values
    U_cln = pp.loc[pure_inlier, 'U_mm'].values

    with PdfPages(outpath) as pdf:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(E_cln, N_cln, U_cln, c=color_in, s=14, alpha=0.85,
                   edgecolors='none', label='Cleaned Data (Inliers)')
        ax.set_xlabel('E (mm)')
        ax.set_ylabel('N (mm)')
        ax.set_zlabel('U (mm)')
        ax.legend(loc='best')
        ax.set_title(f'Cleaned Data 3D Scatter - {method_name}\nStation {station_name}', fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        projections = [('E', 'N', E_cln, N_cln), ('E', 'U', E_cln, U_cln), ('N', 'U', N_cln, U_cln)]

        for ax_sub, (lbl_x, lbl_y, data_x, data_y) in zip(axes, projections):
            ax_sub.scatter(data_x, data_y, c=color_in, s=14, alpha=0.85,
                           edgecolors='none', label='Cleaned Data')
            ax_sub.set_xlabel(f'{lbl_x} (mm)')
            ax_sub.set_ylabel(f'{lbl_y} (mm)')
            ax_sub.set_title(f'{lbl_x} vs {lbl_y}')
            ax_sub.legend(fontsize=8)
            ax_sub.grid(alpha=0.3)

        fig.suptitle(f'Cleaned Data 2D Projections - {method_name}\nStation {station_name}', fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

def main(input=input, output=output):
    np.random.seed(random_state)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    try:
        stations = load_stations(input)
    except FileNotFoundError:
        return

    station_names = list(stations.keys())

    all_results = []

    for sname in station_names:
        df = stations[sname]
        res_df, residuals, sweep_curves, pp, method_masks = process_station(
            sname, df)
        all_results.append(res_df)

        st_dir = output / f'Station_{sname}'.replace(' ', '_').replace('/', '_')
        st_dir.mkdir(exist_ok=True)
        res_df.to_excel(st_dir / 'methods_summary.xlsx', index=False)

        contam = res_df[res_df['category'] == 'contamination'].drop_duplicates('method')
        optimal_cs = dict(zip(contam['method'], contam['optimal_c']))
        opt_statuses = dict(zip(contam['method'], contam['opt_status']))

        plot_time_series_overview(pp, residuals,
                                  st_dir / '01_time_series_overview.pdf', sname)
        plot_sweep_curves(sweep_curves, optimal_cs, opt_statuses,
                          st_dir / '02_contamination_sweep.pdf', sname)
        plot_score_comparison(res_df, st_dir / '03_distance_comparison.pdf', sname)
        plot_decomposition_stacked(res_df, st_dir / '04_decomposition_stacked.pdf', sname)
        plot_methods_landscape(res_df, st_dir / '05_methods_landscape.pdf', sname)

        methods_dir = st_dir / 'methods_diagnostics'
        cleaned_dir = st_dir / 'methods_cleaned_timeseries'
        methods_dir.mkdir(exist_ok=True)
        cleaned_dir.mkdir(exist_ok=True)

        for mname, masks in method_masks.items():
            safe = (mname.replace(' ', '_').replace('/', '_').replace("'", '')
                    .replace('(', '').replace(')', '').replace('.', ''))
            plot_method_diagnostic(pp, masks, res_df,
                                   methods_dir / f'{safe}_diagnostic.pdf',
                                   mname, sname)
            plot_cleaned_timeseries(pp, masks,
                                    cleaned_dir / f'{safe}_cleaned.pdf',
                                    mname, sname)

    master = pd.concat(all_results, ignore_index=True)

    with pd.ExcelWriter(output / 'all_stations_summary.xlsx',
                        engine='openpyxl') as w:
        master.to_excel(w, sheet_name='ALL_STATIONS', index=False)
        for sname in station_names:
            sub = master[master['station'] == sname]
            sub.to_excel(w, sheet_name=str(sname)[:31], index=False)

        decomp_cols = ['station', 'method', 'category', 'optimal_c',
                       'opt_status', 'ideal_distance',
                       'M1', 'M2', 'M3',
                       'M1sq', 'M2sq', 'M3sq',
                       'M1_pct', 'M2_pct', 'M3_pct',
                       'n_total', 'n_kept', 'n_removed', 'pct_removed']
        decomp = master[master['component'] == 'E'][decomp_cols].copy()
        decomp = decomp[decomp['category'] != 'baseline'].sort_values(
            ['station', 'ideal_distance'])
        decomp.to_excel(w, sheet_name='DECOMPOSITION', index=False)

if __name__ == '__main__':
    main()