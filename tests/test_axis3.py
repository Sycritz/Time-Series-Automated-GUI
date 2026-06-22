import numpy as np
import pandas as pd
import pytest
from axis3_modeling import (
    compute_acf_pacf, suggest_model_from_acf_pacf,
    suggest_model_from_spectrum, fit_model, grid_search
)

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

def test_suggest_model_from_spectrum():
    # 1. No significant cycles
    cycles = [
        {"frequency": 0.1, "period": 10.0, "significant": False},
        {"frequency": 0.2, "period": 5.0, "significant": False}
    ]
    res = suggest_model_from_spectrum(cycles, seasonal_period=12)
    assert res["P"] == 0
    assert res["Q"] == 0
    
    # 2. Significant cycle matching seasonal period
    cycles = [
        {"frequency": 2.0 * np.pi / 12.0, "period": 12.0, "significant": True},
        {"frequency": 0.2, "period": 5.0, "significant": False}
    ]
    res = suggest_model_from_spectrum(cycles, seasonal_period=12)
    assert res["P"] == 1
    assert res["Q"] == 1
    assert res["s"] == 12
    assert "Significant cyclical/seasonal component detected" in res["explanation"]

def test_fit_model_and_grid_search():
    # Create simple AR(1) process
    np.random.seed(42)
    n = 100
    eps = np.random.randn(n)
    y = np.zeros(n)
    y[0] = eps[0]
    for t in range(1, n):
        y[t] = 0.5 * y[t-1] + eps[t]
    series = pd.Series(y)
    
    # Test fitting non-seasonal
    res = fit_model(series, order=(1, 0, 0), seasonal_order=None)
    assert res is not None
    assert len(res.params) == 3 # const, ar.L1, sigma2
    
    # Test fitting seasonal
    res_seas = fit_model(series, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    assert res_seas is not None
    
    # Test grid search
    df = grid_search(series, max_p=1, max_q=1, max_P=0, max_Q=0, s=1, d=0, D=0)
    assert not df.empty
    assert "Rank" in df.columns
    assert "Model" in df.columns
    assert df.loc[0, "Rank"] == 1
