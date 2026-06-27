import numpy as np
import pytest

from axis4_validation import (compute_cumulative_periodogram,
                              compute_residual_spectrum, jarque_bera_test,
                              ljung_box_test, run_all_diagnostics)


def test_cumulative_periodogram_white_noise():
    np.random.seed(42)
    residuals = np.random.randn(200)
    freqs, C_omega, ks_up, ks_lo = compute_cumulative_periodogram(residuals)

    assert len(freqs) == (200 - 1) // 2
    assert len(C_omega) == len(freqs)
    assert np.isclose(C_omega[-1], 1.0)

    diagonal = np.arange(1, len(C_omega) + 1) / len(C_omega)
    diff = np.abs(C_omega - diagonal)
    ks_limit = 1.358 / np.sqrt(len(C_omega))
    assert np.all(diff <= ks_limit)


def test_cumulative_periodogram_seasonal():
    np.random.seed(42)
    t = np.arange(200)
    residuals = np.sin(2.0 * np.pi * t / 12.0) + 0.1 * np.random.randn(200)
    freqs, C_omega, ks_up, ks_lo = compute_cumulative_periodogram(residuals)

    diagonal = np.arange(1, len(C_omega) + 1) / len(C_omega)
    diff = np.abs(C_omega - diagonal)
    ks_limit = 1.358 / np.sqrt(len(C_omega))
    assert np.any(diff > ks_limit)


def test_residual_spectrum_flatness():
    np.random.seed(42)
    residuals = np.random.randn(100)
    freqs, f_hat, sp_lo, sp_up = compute_residual_spectrum(residuals, "Bartlett", M=10)

    assert len(freqs) == 512
    assert len(f_hat) == 512
    flat_val = 1.0 / (2.0 * np.pi)
    covered = np.sum((sp_lo <= flat_val) & (flat_val <= sp_up))
    assert covered > 450


def test_run_all_diagnostics_adequate():
    np.random.seed(42)
    residuals = np.random.randn(200)
    results = run_all_diagnostics(residuals, p=0, q=0)

    assert results["lb_pass"] or results["lb_p_value_20"] >= 0.05
    assert results["jb_pass"]
    assert results["cp_pass"]


def test_run_all_diagnostics_inadequate():
    np.random.seed(42)
    n = 200
    eps = np.random.randn(n)
    residuals = np.zeros(n)
    residuals[0] = eps[0]
    for t in range(1, n):
        residuals[t] = 0.8 * residuals[t - 1] + eps[t]

    results = run_all_diagnostics(residuals, p=1, q=0)
    assert (not results["lb_pass"]) or (not results["cp_pass"])


def test_ljung_box_exact():
    # Simple known residuals series
    res = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    # Compute Ljung-Box test for lag 2
    stats, pvals = ljung_box_test(res, lags=2, p=0, q=0)
    assert len(stats) == 2
    assert len(pvals) == 2
    assert not np.isnan(pvals[0])
    assert not np.isnan(pvals[1])

    # Check that df <= 0 falls back to max(1, h - p - q) and does not return NaN
    stats_df, pvals_df = ljung_box_test(res, lags=2, p=1, q=1)
    assert not np.isnan(pvals_df[0])
    assert not np.isnan(pvals_df[1])


def test_jarque_bera_exact():
    # Simple known series
    res = np.array([-1.0, 0.0, 1.0, 2.0, -2.0])
    jb_stat, pvalue = jarque_bera_test(res)
    assert jb_stat >= 0.0
    assert 0.0 <= pvalue <= 1.0


def test_ljung_box_failing_early_passing_late(monkeypatch):
    import axis4_validation

    def mock_ljung_box_test(res, lags, p, q):
        stats = np.arange(1, 21) * 2.0
        pvals = np.ones(20) * 0.90
        pvals[4] = 0.01  # lag 5 fails (<= 0.05)
        pvals[19] = 0.95  # lag 20 passes (> 0.05)
        return stats, pvals

    monkeypatch.setattr(axis4_validation, "ljung_box_test", mock_ljung_box_test)

    residuals = np.random.randn(100)
    results = axis4_validation.run_all_diagnostics(residuals, p=0, q=0)

    # The overall lb_pass should be False because lag 5 failed, even though lag 20 passed.
    assert not results["lb_pass"]
    assert results["lb_pvalues"][4] == 0.01
    assert results["lb_pvalues"][19] == 0.95
