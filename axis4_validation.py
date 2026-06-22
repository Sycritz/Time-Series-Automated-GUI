import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from axis2_spectral import smooth_spectrum

def compute_cumulative_periodogram(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the normalized cumulative periodogram C(omega_k) and the 95% Kolmogorov-Smirnov bounds.
    Eq. 10-11. K_0.05 approx 1.358.
    Returns:
        frequencies (np.ndarray): cycles per unit time (k/n)
        C_omega (np.ndarray): normalized cumulative periodogram
        ks_upper (np.ndarray): upper KS bound
        ks_lower (np.ndarray): lower KS bound
    """
    # Clean residuals of any NaNs
    res = residuals[~np.isnan(residuals)]
    n = len(res)
    if n <= 1:
        empty = np.array([])
        return empty, empty, empty, empty

    q = (n - 1) // 2
    if q <= 0:
        empty = np.array([])
        return empty, empty, empty, empty

    # Compute raw periodogram ordinates using FFT at Fourier frequencies 2*pi*j/n
    fft_vals = np.fft.fft(res)
    # The ordinates are |fft_vals[j]|^2. We sum over j = 1 to q (excluding frequency 0)
    I_Z = np.abs(fft_vals[1:q+1]) ** 2

    cum_sum = np.cumsum(I_Z)
    if cum_sum[-1] == 0:
        C_omega = np.zeros(q)
    else:
        C_omega = cum_sum / cum_sum[-1]

    # Frequencies in cycles per unit time (omega_k = k/n)
    frequencies = np.arange(1, q + 1) / n

    # K_0.05 is approximately 1.358
    ks_limit = 1.358 / np.sqrt(q)

    # Diagonal reference line (k/q)
    diagonal = np.arange(1, q + 1) / q

    ks_upper = diagonal + ks_limit
    ks_lower = diagonal - ks_limit

    return frequencies, C_omega, ks_upper, ks_lower

def compute_residual_spectrum(residuals: np.ndarray, window: str, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the smoothed spectral density of standardized residuals.
    Reuses smooth_spectrum from axis2_spectral.py.
    """
    # Clean residuals
    res = residuals[~np.isnan(residuals)]
    if len(res) <= 1:
        freqs = np.linspace(0, np.pi, 512)
        zero_arr = np.zeros(512)
        return freqs, zero_arr, zero_arr, zero_arr

    # Standardize residuals
    res_std = np.std(res)
    std_res = res / res_std if res_std > 1e-10 else res
    series = pd.Series(std_res)
    
    # Reuse smooth_spectrum
    return smooth_spectrum(series, window, M)

def run_all_diagnostics(residuals: np.ndarray, p: int, q: int) -> dict:
    """
    Run all model validation diagnostics.
    """
    # Clean residuals
    res = residuals[~np.isnan(residuals)]
    n = len(res)
    
    # 1. Ljung-Box Portmanteau Test
    h_max = min(20, n - 1)
    if h_max <= 0:
        lb_stats = np.full(20, np.nan)
        lb_pvalues = np.full(20, np.nan)
        lb_p_value_20 = np.nan
        lb_pass = True
    else:
        lb_df = acorr_ljungbox(res, lags=h_max, model_df=p+q)
        lb_stats = lb_df['lb_stat'].values
        lb_pvalues = lb_df['lb_pvalue'].values
        
        # Pad with NaNs if necessary to reach length 20
        if len(lb_stats) < 20:
            pad = 20 - len(lb_stats)
            lb_stats = np.concatenate([lb_stats, np.full(pad, np.nan)])
            lb_pvalues = np.concatenate([lb_pvalues, np.full(pad, np.nan)])
            
        lb_p_value_20 = lb_pvalues[19] if len(lb_pvalues) > 19 else np.nan
        if np.isnan(lb_p_value_20):
            non_nan_pvals = lb_pvalues[~np.isnan(lb_pvalues)]
            if len(non_nan_pvals) > 0:
                lb_p_value_20 = non_nan_pvals[-1]
                lb_pass = (lb_p_value_20 >= 0.05)
            else:
                lb_pass = True
        else:
            lb_pass = (lb_p_value_20 >= 0.05)

    # 2. Jarque-Bera Test for Normality
    if n < 2:
        jb_stat = 0.0
        jb_pvalue = 1.0
        jb_pass = True
    else:
        jb_stat, jb_pvalue = jarque_bera(res)
        jb_pass = (jb_pvalue >= 0.05)

    # 3. Cumulative Periodogram Test
    freqs_cp, C_omega, ks_up, ks_lo = compute_cumulative_periodogram(res)
    if len(C_omega) > 0:
        q_val = len(C_omega)
        diagonal = np.arange(1, q_val + 1) / q_val
        ks_limit = 1.358 / np.sqrt(q_val)
        cp_pass = np.all(np.abs(C_omega - diagonal) <= ks_limit)
    else:
        cp_pass = True

    # 4. Residual Spectrum Flatness
    M = max(2, int(np.sqrt(n)))
    freqs_sp, f_hat, sp_lo, sp_up = compute_residual_spectrum(res, 'Bartlett', M)
    if len(sp_lo) > 0:
        flat_level = 1.0 / (2.0 * np.pi)
        flatness_pass = np.all((sp_lo <= flat_level) & (flat_level <= sp_up))
    else:
        flatness_pass = True

    return {
        'lb_stats': lb_stats,
        'lb_pvalues': lb_pvalues,
        'lb_p_value_20': lb_p_value_20,
        'lb_pass': lb_pass,
        'h_max': h_max,
        'jb_stat': jb_stat,
        'jb_pvalue': jb_pvalue,
        'jb_pass': jb_pass,
        'cp_frequencies': freqs_cp,
        'cp_C_omega': C_omega,
        'cp_ks_upper': ks_up,
        'cp_ks_lower': ks_lo,
        'cp_pass': cp_pass,
        'sp_frequencies': freqs_sp,
        'sp_f_hat': f_hat,
        'sp_ci_lower': sp_lo,
        'sp_ci_upper': sp_up,
        'flatness_pass': flatness_pass
    }
