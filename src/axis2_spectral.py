import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize

def compute_periodogram(series: pd.Series, taper: str | None, taper_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute raw periodogram ordinate:
    I(lambda_j) = 1/(2*pi*n) |sum_{t=1}^n x_t e^{-it lambda_j}|^2
    for lambda_j in [0, pi] using np.fft.fft.
    """
    # Demean the series values
    x = series.values - series.mean()
    n = len(x)
    
    # Handle empty/short series
    if n == 0:
        return np.array([]), np.array([])
        
    # Apply data taper if specified
    if taper == 'Cosine Bell':
        w = np.ones(n)
        p = int(taper_pct * n)
        if p > 0:
            for idx in range(p):
                val = 0.5 * (1.0 - np.cos(np.pi * (idx + 1) / p))
                w[idx] = val
                w[n - 1 - idx] = val
        x = x * w
    elif taper == 'Hann':
        if n > 1:
            w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
        else:
            w = np.ones(n)
        x = x * w
    elif taper == 'Hamming':
        if n > 1:
            w = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
        else:
            w = np.ones(n)
        x = x * w

    # Compute raw periodogram ordinate
    fft_vals = np.fft.fft(x)
    periodogram = (1.0 / (2.0 * np.pi * n)) * (np.abs(fft_vals) ** 2)
    
    num_freqs = n // 2 + 1
    frequencies = 2.0 * np.pi * np.arange(num_freqs) / n
    periodogram = periodogram[:num_freqs]
    
    return frequencies, periodogram

def smooth_spectrum(series: pd.Series, window: str, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the consistent lag-window spectral density estimator:
    f_hat(lambda) = 1/(2*pi) * (gamma_hat(0) + 2 * sum_{h=1}^M w(h/M) gamma_hat(h) cos(h lambda))
    evaluated at 512 frequencies in [0, pi].
    """
    x = series.values - series.mean()
    n = len(x)
    
    # Evaluation frequencies
    n_points = 512
    frequencies = np.linspace(0, np.pi, n_points)
    
    if n <= 1:
        # Trivial fallback for degenerate data
        zero_arr = np.zeros(n_points)
        return frequencies, zero_arr, zero_arr, zero_arr

    M_eff = min(M, n - 1)
    
    # Compute autocovariances gamma_hat for h = 0, ..., M_eff
    gamma = np.zeros(M_eff + 1)
    for h in range(M_eff + 1):
        gamma[h] = np.sum(x[:n-h] * x[h:]) / n
        
    # Compute lag window values w(h/M)
    w_vals = np.zeros(M_eff + 1)
    for h in range(1, M_eff + 1):
        val = h / M
        if window == 'Daniell':
            w_vals[h] = 1.0
        elif window == 'Bartlett':
            w_vals[h] = 1.0 - abs(val)
        elif window == 'Parzen':
            abs_val = abs(val)
            if abs_val <= 0.5:
                w_vals[h] = 1.0 - 6.0 * abs_val**2 + 6.0 * abs_val**3
            else:
                w_vals[h] = 2.0 * (1.0 - abs_val)**3
        elif window == 'Hann':
            w_vals[h] = 0.5 * (1.0 + np.cos(np.pi * val))
            
    # Calculate equivalent degrees of freedom nu
    sum_w2 = np.sum(w_vals[1:]**2)
    nu = (2.0 * n) / (1.0 + 2.0 * sum_w2)
    
    # Compute consistent lag-window spectral density estimator f_hat
    term = np.zeros_like(frequencies)
    for h in range(1, M_eff + 1):
        term += w_vals[h] * gamma[h] * np.cos(h * frequencies)
        
    f_hat = (1.0 / (2.0 * np.pi)) * (gamma[0] + 2.0 * term)
    f_hat = np.clip(f_hat, 1e-15, None)  # Prevent negative/zero values
    
    # Compute 95% confidence intervals
    chi2_upper_val = scipy.stats.chi2.ppf(0.975, df=nu)
    chi2_lower_val = scipy.stats.chi2.ppf(0.025, df=nu)
    
    ci_lower = nu * f_hat / chi2_upper_val
    ci_upper = nu * f_hat / chi2_lower_val
    
    return frequencies, f_hat, ci_lower, ci_upper

def compute_parametric_spectrum(ar_coeffs: list, ma_coeffs: list, sigma2: float, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute theoretical spectral density of ARMA(p, q) process:
    f_X(lambda) = sigma^2 / (2*pi) * |1 + theta_1 e^{-i lambda} + ... + theta_q e^{-iq lambda}|^2 / 
                                     |1 - phi_1 e^{-i lambda} - ... - phi_p e^{-ip lambda}|^2
    """
    frequencies = np.linspace(0, np.pi, n_points)
    
    if ar_coeffs is None:
        ar_coeffs = []
    if ma_coeffs is None:
        ma_coeffs = []
        
    # Denominator
    den = np.ones_like(frequencies, dtype=complex)
    for j, phi in enumerate(ar_coeffs):
        den -= phi * np.exp(-1j * (j + 1) * frequencies)
        
    # Numerator
    num = np.ones_like(frequencies, dtype=complex)
    for k, theta in enumerate(ma_coeffs):
        num += theta * np.exp(-1j * (k + 1) * frequencies)
        
    den_abs = np.clip(np.abs(den), 1e-15, None)
    theoretical_spectrum = (sigma2 / (2.0 * np.pi)) * (np.abs(num) ** 2) / (den_abs ** 2)
    
    return frequencies, theoretical_spectrum

def detect_cycles(frequencies: np.ndarray, f_hat: np.ndarray, ci_upper: np.ndarray, nu: float | None = None) -> list[dict]:
    """
    Identify statistically significant cycles from the smoothed spectrum.
    Accounts for the spectral slope by fitting a log-quadratic baseline.
    """
    if len(f_hat) < 3:
        return []
        
    # Identify local maximum peaks
    peaks_indices = []
    for i in range(1, len(f_hat) - 1):
        if f_hat[i] > f_hat[i-1] and f_hat[i] > f_hat[i+1]:
            peaks_indices.append(i)
            
    # Solve for equivalent degrees of freedom nu if not provided
    if nu is None:
        ratios = ci_upper / f_hat
        valid = np.isfinite(ratios) & (f_hat > 0)
        if not np.any(valid):
            ratio = 39.4978  # Default corresponding to nu = 2.0
        else:
            ratio = np.mean(ratios[valid])
        ratio = max(ratio, 1.0001)
        
        # Solve nu / chi2.ppf(0.025, df=nu) = ratio
        def obj(nu_val):
            denom = scipy.stats.chi2.ppf(0.025, df=nu_val)
            if denom <= 0:
                return -ratio
            return nu_val / denom - ratio
            
        try:
            a = 1e-5
            b = 1e7
            if obj(a) < 0:
                while obj(a) < 0 and a > 1e-15:
                    a /= 10.0
            if obj(b) > 0:
                while obj(b) > 0 and b < 1e15:
                    b *= 10.0
            nu = scipy.optimize.brentq(obj, a, b)
        except Exception:
            nu = 2.0
            
    # Account for spectral slope by fitting a baseline:
    # Fit a log-quadratic baseline (a quadratic polynomial to log(f_hat) vs frequencies)
    # This captures the smooth, frequency-dependent background (slope + curvature)
    try:
        log_f = np.log(np.clip(f_hat, 1e-15, None))
        poly = np.polyfit(frequencies, log_f, deg=2)
        baseline = np.exp(np.polyval(poly, frequencies))
    except Exception:
        # Fallback to white noise level if fitting fails
        baseline = np.full_like(frequencies, np.mean(f_hat))
        
    # Statistical significance threshold at each frequency
    thresholds = baseline * scipy.stats.chi2.ppf(0.95, df=nu) / nu
    
    results = []
    for idx in peaks_indices:
        freq = frequencies[idx]
        val = f_hat[idx]
        sig = bool(val > thresholds[idx])
        period = 2.0 * np.pi / freq if freq > 0 else float('inf')
        results.append({
            "frequency": float(freq),
            "period": float(period),
            "significant": sig,
            "height": float(val)
        })
        
    # Sort by height descending
    results.sort(key=lambda x: x["height"], reverse=True)
    for r in results:
        del r["height"]
        
    return results
