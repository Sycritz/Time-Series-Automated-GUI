import pandas as pd
import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller

def load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV or TXT file using pandas. Handles clean parsing.
    """
    try:
        df = pd.read_csv(path, sep=None, engine='python')
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV/TXT file: {e}")

def handle_missing(series: pd.Series, method: str) -> tuple[pd.Series, float]:
    """
    Imputes missing values in the series using the specified method:
    'Forward Fill', 'Linear Interpolation', or 'Mean Imputation'.
    Returns the imputed series and the percentage of missing values that were imputed.
    """
    n_missing = series.isna().sum()
    pct_missing = (n_missing / len(series)) * 100.0 if len(series) > 0 else 0.0
    
    imputed = series.copy()
    if n_missing > 0:
        if method == 'Forward Fill':
            imputed = imputed.ffill().bfill()
        elif method == 'Linear Interpolation':
            imputed = imputed.interpolate(method='linear').ffill().bfill()
        elif method == 'Mean Imputation':
            mean_val = imputed.mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            imputed = imputed.fillna(mean_val)
        else:
            raise ValueError(f"Unknown imputation method: {method}")
    return imputed, pct_missing

def run_adf_test(series: pd.Series) -> dict:
    """
    Runs the Augmented Dickey-Fuller test using statsmodels.
    """
    clean_series = series.dropna()
    if len(clean_series) < 10:
        raise ValueError("Insufficient data points (at least 10 required) to run ADF test.")
    try:
        res = adfuller(clean_series)
        adf_stat = float(res[0])
        p_value = float(res[1])
        critical_vals = {k: float(v) for k, v in res[4].items()}
        verdict = 'Stationary' if p_value < 0.05 else 'Non-Stationary'
        return {
            'adf_stat': adf_stat,
            'p_value': p_value,
            'critical_values': critical_vals,
            'verdict': verdict
        }
    except Exception as e:
        raise ValueError(f"ADF test failed: {e}")

def apply_box_cox(series: pd.Series, lam: float | None) -> tuple[pd.Series, float]:
    """
    Applies Box-Cox transformation. If lam is None, automatically optimizes it.
    """
    non_nan_mask = series.notna()
    valid_values = series[non_nan_mask].values
    if (valid_values <= 0).any():
        raise ValueError("Box-Cox transformation requires strictly positive data. Found non-positive values.")
    
    if lam is None:
        transformed_valid, opt_lam = boxcox(valid_values)
        opt_lam = float(opt_lam)
    else:
        transformed_valid = boxcox(valid_values, lmbda=lam)
        opt_lam = float(lam)
        
    transformed_values = np.full(series.shape, np.nan)
    transformed_values[non_nan_mask] = transformed_valid
    transformed_series = pd.Series(transformed_values, index=series.index, name=series.name)
    return transformed_series, opt_lam

def apply_differencing(series: pd.Series, d: int, D: int, s: int) -> pd.Series:
    """
    Applies ordinary differencing of order d and seasonal differencing of order D with period s.
    """
    res = series.copy()
    for _ in range(d):
        res = res.diff()
    for _ in range(D):
        res = res.diff(periods=s)
    return res

def compute_frequency_response(d: int, D: int, s: int, n_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the squared gain function of the differencing filter.
    """
    omega = np.linspace(0, np.pi, n_points)
    term1 = (2.0 * np.abs(np.sin(omega / 2.0))) ** (2 * d)
    term2 = (2.0 * np.abs(np.sin(s * omega / 2.0))) ** (2 * D)
    gain = term1 * term2
    return omega, gain

def rolling_mean(x: pd.Series | np.ndarray, window: int) -> pd.Series | np.ndarray:
    """
    Compute rolling mean manually.
    Returns object of same type; first (window-1) values are NaN.
    """
    if window <= 0:
        raise ValueError("Window size must be greater than 0.")
    is_series = isinstance(x, pd.Series)
    values = x.values if is_series else np.asarray(x)
    n = len(values)
    result = np.full(n, np.nan)
    for t in range(window - 1, n):
        result[t] = np.mean(values[t - window + 1 : t + 1])
    if is_series:
        return pd.Series(result, index=x.index, name=x.name)
    return result

def rolling_std(x: pd.Series | np.ndarray, window: int) -> pd.Series | np.ndarray:
    """
    Compute rolling standard deviation manually (ddof=1).
    Returns object of same type; first (window-1) values are NaN.
    """
    if window <= 1:
        raise ValueError("Window size must be greater than 1 for standard deviation.")
    is_series = isinstance(x, pd.Series)
    values = x.values if is_series else np.asarray(x)
    n = len(values)
    result = np.full(n, np.nan)
    for t in range(window - 1, n):
        segment = values[t - window + 1 : t + 1]
        result[t] = np.sqrt(np.sum((segment - np.mean(segment)) ** 2) / (window - 1))
    if is_series:
        return pd.Series(result, index=x.index, name=x.name)
    return result

def sample_autocovariance(x: pd.Series | np.ndarray, h: int) -> float:
    """
    Sample autocovariance at lag h (B&D §1.4, STAT720 §2.6).
    gamma_hat(h) = (1/n) * sum_{t=1}^{n-h} (x_t - x_bar)(x_{t+h} - x_bar)
    Uses the biased estimator (divides by n, not n-h) for positive semidefiniteness.
    """
    values = x.values if isinstance(x, pd.Series) else np.asarray(x)
    n = len(values)
    if h < 0:
        raise ValueError("Lag h must be non-negative.")
    if h >= n:
        return 0.0
    x_bar = np.mean(values)
    return float(np.sum((values[: n - h] - x_bar) * (values[h:] - x_bar)) / n)

