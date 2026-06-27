import numpy as np
import pandas as pd
from scipy.stats import chi2

from axis2_spectral import smooth_spectrum


def ljung_box_test(
    res: np.ndarray, lags: int, p: int, q: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Ljung-Box test statistic and p-values from scratch.
    """
    n = len(res)
    if n == 0:
        return np.array([]), np.array([])
    res_c = res - np.mean(res)
    gamma0 = np.sum(res_c**2) / n
    if gamma0 < 1e-14:
        return np.zeros(lags), np.ones(lags)

    rho = np.zeros(lags + 1)
    for h in range(1, lags + 1):
        rho[h] = np.dot(res_c[: n - h], res_c[h:]) / (n * gamma0)

    stats = np.zeros(lags)
    pvalues = np.zeros(lags)
    cumsum = 0.0
    for h in range(1, lags + 1):
        cumsum += (rho[h] ** 2) / (n - h)
        q_stat = n * (n + 2) * cumsum
        stats[h - 1] = q_stat
        df = max(1, h - p - q)
        pvalues[h - 1] = 1.0 - chi2.cdf(q_stat, df=df)
    return stats, pvalues


def jarque_bera_test(res: np.ndarray) -> tuple[float, float]:
    """
    Compute Jarque-Bera statistic and p-value from scratch.
    """
    n = len(res)
    if n < 2:
        return 0.0, 1.0
    res_c = res - np.mean(res)
    m2 = np.sum(res_c**2) / n
    m3 = np.sum(res_c**3) / n
    m4 = np.sum(res_c**4) / n
    if m2 < 1e-14:
        return 0.0, 1.0
    S = m3 / (m2**1.5)
    K = m4 / (m2**2)
    jb_stat = n * (S**2 / 6.0 + (K - 3.0) ** 2 / 24.0)
    pvalue = 1.0 - chi2.cdf(jb_stat, df=2)
    return float(jb_stat), float(pvalue)


def compute_cumulative_periodogram(
    residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    I_Z = np.abs(fft_vals[1 : q + 1]) ** 2

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


def compute_residual_spectrum(
    residuals: np.ndarray, window: str, M: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        lb_stats, lb_pvalues = ljung_box_test(res, h_max, p, q)

        # Pad with NaNs if necessary to reach length 20
        if len(lb_stats) < 20:
            pad = 20 - len(lb_stats)
            lb_stats = np.concatenate([lb_stats, np.full(pad, np.nan)])
            lb_pvalues = np.concatenate([lb_pvalues, np.full(pad, np.nan)])

        lb_p_value_20 = lb_pvalues[19] if len(lb_pvalues) > 19 else np.nan
        valid_pvals = lb_pvalues[~np.isnan(lb_pvalues)]
        if len(valid_pvals) > 0:
            lb_pass = bool(np.all(valid_pvals > 0.05))
        else:
            lb_pass = True

    # 2. Jarque-Bera Test for Normality
    if n < 2:
        jb_stat = 0.0
        jb_pvalue = 1.0
        jb_pass = True
    else:
        jb_stat, jb_pvalue = jarque_bera_test(res)
        jb_pass = jb_pvalue >= 0.05

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
    freqs_sp, f_hat, sp_lo, sp_up = compute_residual_spectrum(res, "Bartlett", M)
    if len(sp_lo) > 0:
        flat_level = 1.0 / (2.0 * np.pi)
        flatness_pass = np.all((sp_lo <= flat_level) & (flat_level <= sp_up))
    else:
        flatness_pass = True

    return {
        "lb_stats": lb_stats,
        "lb_pvalues": lb_pvalues,
        "lb_p_value_20": lb_p_value_20,
        "lb_pass": lb_pass,
        "h_max": h_max,
        "jb_stat": jb_stat,
        "jb_pvalue": jb_pvalue,
        "jb_pass": jb_pass,
        "cp_frequencies": freqs_cp,
        "cp_C_omega": C_omega,
        "cp_ks_upper": ks_up,
        "cp_ks_lower": ks_lo,
        "cp_pass": cp_pass,
        "sp_frequencies": freqs_sp,
        "sp_f_hat": f_hat,
        "sp_ci_lower": sp_lo,
        "sp_ci_upper": sp_up,
        "flatness_pass": flatness_pass,
    }
