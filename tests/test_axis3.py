import numpy as np
import pandas as pd
import pytest

from axis3_modeling import (compute_acf_pacf, fit_model, grid_search,
                            suggest_model_from_acf_pacf)


def test_compute_acf_pacf():
    # Simple random series
    n = 100
    series = pd.Series(np.random.randn(n))

    acf_vals, pacf_vals = compute_acf_pacf(series, nlags=20)
    # Lags are 0 to min(nlags, n//2 - 1). With nlags=20, actual_lags is 20.
    # So length of array is actual_lags + 1 = 21.
    assert len(acf_vals) == 21
    assert len(pacf_vals) == 21
    assert np.isclose(acf_vals[0], 1.0)
    assert np.isclose(pacf_vals[0], 1.0)

    # Degenerate cases
    empty_series = pd.Series([], dtype=float)
    acf_empty, pacf_empty = compute_acf_pacf(empty_series, 20)
    assert len(acf_empty) == 1
    assert acf_empty[0] == 1.0

    short_series = pd.Series([1.0, 2.0])
    acf_short, pacf_short = compute_acf_pacf(short_series, 20)
    # actual_lags capped at min(20, 2//2 - 1) = 0 -> falls to min(1, n-1) = 1
    # So length is 2.
    assert len(acf_short) == 2


def test_suggest_model_from_acf_pacf():
    # 1. White Noise case
    acf_vals = np.array([1.0, 0.05, -0.02, 0.01])
    pacf_vals = np.array([1.0, 0.04, 0.03, -0.01])
    res = suggest_model_from_acf_pacf(acf_vals, pacf_vals, n=100)
    assert res["p"] == 0
    assert res["q"] == 0
    assert "White Noise" in res["explanation"]

    # 2. MA(1) case (ACF cut off, PACF tails off)
    # n=100 -> threshold = 1.96 / 10 = 0.196
    acf_vals = np.array([1.0, 0.4, 0.05, 0.02])
    pacf_vals = np.array([1.0, -0.3, 0.25, -0.21])
    res = suggest_model_from_acf_pacf(acf_vals, pacf_vals, n=100)
    assert res["q"] == 1
    assert res["p"] == 0

    # 3. AR(2) case (PACF cut off, ACF tails off)
    acf_vals = np.array([1.0, 0.5, 0.35, 0.22])
    pacf_vals = np.array([1.0, 0.4, -0.3, 0.02])
    res = suggest_model_from_acf_pacf(acf_vals, pacf_vals, n=100)
    assert res["p"] == 2
    assert res["q"] == 0

    # 4. Mixed ARMA(1, 1) case (both tail off)
    acf_vals = np.array([1.0, 0.5, 0.4, 0.3])
    pacf_vals = np.array([1.0, 0.4, 0.3, 0.25])
    res = suggest_model_from_acf_pacf(acf_vals, pacf_vals, n=100)
    assert res["p"] == 1
    assert res["q"] == 1


def test_fit_model_and_grid_search():
    # Create simple AR(1) process
    np.random.seed(42)
    n = 100
    eps = np.random.randn(n)
    y = np.zeros(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + eps[t]
    series = pd.Series(y)

    # Test fitting non-seasonal
    res = fit_model(series, order=(1, 0, 0), seasonal_order=None)
    assert res is not None
    assert len(res.params) == 2  # ar.L1, sigma2

    # Test fitting seasonal
    res_seas = fit_model(series, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    assert res_seas is not None

    # Test grid search
    df = grid_search(series, max_p=1, max_q=1, max_P=0, max_Q=0, s=1, d=0, D=0)
    assert not df.empty
    assert "Rank" in df.columns
    assert "Model" in df.columns
    assert df.loc[0, "Rank"] == 1


def test_custom_acf_pacf_exact():
    # Simple series
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # Under biased autocovariance:
    # Mean is 3.0. Deviations: [-2.0, -1.0, 0.0, 1.0, 2.0]
    # gamma(0) = (4 + 1 + 0 + 1 + 4) / 5 = 10 / 5 = 2.0
    # gamma(1) = ((-2)*(-1) + (-1)*0 + 0*1 + 1*2) / 5 = (2 + 2) / 5 = 0.8
    # gamma(2) = ((-2)*0 + (-1)*1 + 0*2) / 5 = -1 / 5 = -0.2
    # So acf = [1.0, 0.8 / 2.0, -0.2 / 2.0] = [1.0, 0.4, -0.1]
    # Durbin-Levinson for PACF:
    # phi_11 = rho(1) = 0.4
    # v_1 = 1 - phi_11^2 = 1 - 0.16 = 0.84
    # h = 2:
    # num = rho(2) - phi_11 * rho(1) = -0.1 - 0.4 * 0.4 = -0.26
    # phi_22 = num / v_1 = -0.26 / 0.84 = -13 / 42 approx -0.3095238
    acf_vals, pacf_vals = compute_acf_pacf(series, nlags=2)
    np.testing.assert_allclose(acf_vals, [1.0, 0.4, -0.1])
    np.testing.assert_allclose(pacf_vals, [1.0, 0.4, -13.0 / 42.0])


def test_suggest_model_stray_lags_and_tails_off():
    # 1. Stray lag in ACF (cuts off at 1 with stray lag at 5, threshold = 0.196)
    acf_vals = np.array([1.0, 0.4, 0.05, 0.05, 0.05, 0.3])
    pacf_vals = np.array([1.0, 0.05, 0.05, 0.05, 0.05, 0.05])
    res = suggest_model_from_acf_pacf(acf_vals, pacf_vals, n=100)
    # ACF cutoff should be 1. PACF cutoff should be None.
    # So it should suggest MA(1).
    assert res["q"] == 1
    assert res["p"] == 0

    # 2. Stray lag in PACF (cuts off at 2 with stray lag at 6)
    acf_vals = np.array([1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    pacf_vals = np.array([1.0, 0.35, 0.4, 0.05, 0.05, 0.05, 0.3])
    res = suggest_model_from_acf_pacf(acf_vals, pacf_vals, n=100)
    # PACF cutoff should be 2. ACF cutoff should be None.
    # So it should suggest AR(2).
    assert res["p"] == 2
    assert res["q"] == 0

    # 3. Tails off (at least 3 significant lags and no cutoff at <= 3)
    acf_vals = np.array([1.0, 0.3, 0.3, 0.05, 0.3, 0.3])
    pacf_vals = np.array([1.0, 0.3, 0.3, 0.05, 0.3, 0.3])
    res = suggest_model_from_acf_pacf(acf_vals, pacf_vals, n=100)
    assert res["p"] == 1
    assert res["q"] == 1
