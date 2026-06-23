import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import mackinnonp, mackinnoncrit

def load_csv(path: str) -> pd.DataFrame:
    """
    Loads a CSV or TXT file using pandas. Handles clean parsing.
    """
    try:
        df = pd.read_csv(path, sep=None, engine='python')
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV/TXT file: {e}")

def impute_series(x: np.ndarray, method: str = 'linear') -> tuple[np.ndarray, int]:
    """
    Impute missing values (NaN) in series x.
    Methods: 'linear' (interpolation), 'forward_fill', 'mean'.
    Returns (imputed_array, n_imputed).
    """
    method_norm = method.lower().replace(" ", "_")
    if method_norm == "linear_interpolation":
        method_norm = "linear"
    elif method_norm == "mean_imputation":
        method_norm = "mean"
    
    x = x.astype(float)
    mask = np.isnan(x)
    n_imputed = int(np.sum(mask))

    if n_imputed == 0:
        return x, 0

    if method_norm == "linear":
        indices = np.arange(len(x))
        valid = ~mask
        if np.sum(valid) == 0:
            x_out = np.zeros_like(x)
        else:
            x_out = np.interp(indices, indices[valid], x[valid])
    elif method_norm == "forward_fill":
        x_out = x.copy()
        if np.isnan(x_out[0]):
            first_valid = x[~np.isnan(x)]
            x_out[0] = first_valid[0] if len(first_valid) > 0 else 0.0
        for i in range(1, len(x_out)):
            if np.isnan(x_out[i]):
                x_out[i] = x_out[i - 1]
    elif method_norm == "mean":
        x_out = x.copy()
        if np.all(mask):
            mean_val = 0.0
        else:
            mean_val = np.nanmean(x)
        x_out[mask] = mean_val
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return x_out, n_imputed

def run_adf_test(series: pd.Series) -> dict:
    """
    Runs the Augmented Dickey-Fuller test using manual OLS regression from scratch.
    """
    clean_series = series.dropna()
    N = len(clean_series)
    if N < 10:
        raise ValueError("Insufficient data points (at least 10 required) to run ADF test.")
    
    try:
        x = clean_series.values.astype(float)
        if x.max() == x.min():
            raise ValueError("Invalid input, x is constant")
        
        # 1. Determine maxlag (Schwert 1989 rule)
        p_max = int(np.floor(12 * (N / 100.0) ** 0.25))
        # Limit maxlag as in statsmodels: min(nobs // 2 - ntrend - 1, maxlag)
        # For regression = 'c', ntrend = 1
        p_max = min(N // 2 - 2, p_max)
        if p_max < 0:
            raise ValueError("sample size is too short to use selected regression component")
        
        xdiff = np.diff(x)
        
        # 2. Lag selection via AIC minimization on fixed sample size (restricted by maxlag)
        nobs_select = N - 1 - p_max
        yd = xdiff[p_max:]
        
        aics = []
        for p in range(p_max + 1):
            cols = [np.ones((nobs_select, 1)), x[p_max : N - 1, None]]
            for j in range(1, p + 1):
                cols.append(xdiff[p_max - j : N - 1 - j, None])
            X_p = np.hstack(cols)
            
            # Solve OLS via least squares
            beta, _, _, _ = np.linalg.lstsq(X_p, yd, rcond=None)
            residuals = yd - X_p @ beta
            RSS = np.sum(residuals ** 2)
            
            # Calculate log-likelihood and AIC
            llf = -nobs_select / 2.0 * (np.log(2.0 * np.pi) + np.log(RSS / nobs_select) + 1.0)
            k = X_p.shape[1]
            aic = -2.0 * llf + 2.0 * k
            aics.append((aic, p))
            
        best_aic, best_p = min(aics)
        
        # 3. Rerun OLS on full available sample size for the selected best_p
        nobs_final = N - 1 - best_p
        yd_final = xdiff[best_p:]
        
        # Regressors: statsmodels has prepend=False for final regression,
        # so level variable x_{t-1} is column 0, lagged differences are cols 1..best_p,
        # and constant is appended as the last column.
        cols_final = [x[best_p : N - 1, None]]
        for j in range(1, best_p + 1):
            cols_final.append(xdiff[best_p - j : N - 1 - j, None])
        cols_final.append(np.ones((nobs_final, 1)))
        X_final = np.hstack(cols_final)
        
        beta_final, _, _, _ = np.linalg.lstsq(X_final, yd_final, rcond=None)
        residuals_final = yd_final - X_final @ beta_final
        RSS_final = np.sum(residuals_final ** 2)
        
        df_final = nobs_final - X_final.shape[1]
        s_final2 = RSS_final / df_final
        
        # Standard errors of coefficients
        X_T_X_inv = np.linalg.inv(X_final.T @ X_final)
        var_beta = s_final2 * X_T_X_inv
        se_beta = np.sqrt(np.diag(var_beta))
        
        # ADF statistic is the t-statistic of the lagged level (column 0)
        tau = beta_final[0] / se_beta[0]
        
        # Calculate critical values and approximate p-value
        c_vals = mackinnoncrit(N=1, regression='c', nobs=nobs_final)
        p_value = mackinnonp(tau, regression='c', N=1)
        
        c1, c5, c10 = float(c_vals[0]), float(c_vals[1]), float(c_vals[2])
        p_value = float(p_value)
        verdict = 'Stationary' if p_value < 0.05 else 'Non-Stationary'
        
        return {
            'adf_stat': float(tau),
            'p_value': p_value,
            'critical_values': {'1%': c1, '5%': c5, '10%': c10},
            'verdict': verdict,
            'used_lag': best_p,
            'nobs': nobs_final,
            'regression_eq': f"ΔX_t = α + β·X_{{t-1}}" + (f" + Σ γ_j·ΔX_{{t-j}} (j=1..{best_p})" if best_p > 0 else "") + " + ε_t"
        }
    except Exception as e:
        raise ValueError(f"ADF test failed: {e}")

def box_cox_transform(x: np.ndarray, lam: float) -> np.ndarray:
    """
    Box-Cox transformation (B&D §9.4, eq. 9.4.1).
    Y_t = (X_t^lambda - 1) / lambda   if lambda != 0
    Y_t = ln(X_t)                      if lambda == 0
    Requires x > 0.
    """
    if np.any(x <= 0):
        raise ValueError("Box-Cox requires strictly positive series.")
    if abs(lam) < 1e-10:
        return np.log(x)
    return (x**lam - 1.0) / lam

def box_cox_inverse(y: np.ndarray, lam: float) -> np.ndarray:
    """
    Inverse Box-Cox transformation (for back-transform in Axis 5).
    X_t = (lambda * Y_t + 1)^(1/lambda)  if lambda != 0
    X_t = exp(Y_t)                         if lambda == 0
    """
    if abs(lam) < 1e-10:
        return np.exp(y)
    return (lam * y + 1.0) ** (1.0 / lam)

def box_cox_profile_loglik(x: np.ndarray, lam: float) -> float:
    """
    Profile log-likelihood for Box-Cox lambda (B&D §9.4).
    l(lambda) = -(n/2) * ln(sigma_hat^2(lambda)) + (lambda-1) * sum(ln(x_t))
    where sigma_hat^2 = (1/n) * sum((y_t - y_bar)^2)
    """
    if np.any(x <= 0):
        raise ValueError("Box-Cox requires strictly positive series.")
    n = len(x)
    if n == 0:
        return -np.inf
    y = box_cox_transform(x, lam)
    y_bar = np.mean(y)
    sigma2_hat = np.sum((y - y_bar) ** 2) / n
    if sigma2_hat <= 0:
        return -np.inf
    log_lik = -(n / 2.0) * np.log(sigma2_hat) + (lam - 1.0) * np.sum(np.log(x))
    return float(log_lik)

def box_cox_auto(x: np.ndarray, lam_grid: np.ndarray | None = None) -> float:
    """
    Find lambda that maximizes the profile log-likelihood.
    Uses a grid search (400 points) over [-2, 2] to find optimal lambda.
    """
    if lam_grid is None:
        lam_grid = np.linspace(-2.0, 2.0, 400)
    logliks = np.array([box_cox_profile_loglik(x, lam) for lam in lam_grid])
    return float(lam_grid[np.argmax(logliks)])

def apply_box_cox(series: pd.Series, lam: float | None) -> tuple[pd.Series, float]:
    """
    Applies Box-Cox transformation. If lam is None, automatically optimizes it.
    """
    non_nan_mask = series.notna()
    valid_values = series[non_nan_mask].values
    if (valid_values <= 0).any():
        raise ValueError("Box-Cox transformation requires strictly positive data. Found non-positive values.")
    
    if lam is None:
        opt_lam = box_cox_auto(valid_values)
    else:
        opt_lam = float(lam)
        
    transformed_valid = box_cox_transform(valid_values, opt_lam)
    transformed_values = np.full(series.shape, np.nan)
    transformed_values[non_nan_mask] = transformed_valid
    transformed_series = pd.Series(transformed_values, index=series.index, name=series.name)
    return transformed_series, opt_lam

def apply_differencing(series: pd.Series, d: int, D: int, s: int) -> pd.Series:
    """
    Applies ordinary differencing of order d and seasonal differencing of order D with period s.
    """
    if d < 0 or D < 0:
        raise ValueError("Differencing orders d and D must be non-negative.")
    if D > 0 and s <= 0:
        raise ValueError("Seasonal period s must be greater than 0 when D > 0.")
    
    arr = series.values.astype(float).copy()
    for _ in range(d):
        new_arr = np.full_like(arr, np.nan)
        new_arr[1:] = arr[1:] - arr[:-1]
        arr = new_arr
    for _ in range(D):
        new_arr = np.full_like(arr, np.nan)
        if s > 0 and len(arr) >= s:
            new_arr[s:] = arr[s:] - arr[:-s]
        arr = new_arr
    return pd.Series(arr, index=series.index, name=series.name)

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
