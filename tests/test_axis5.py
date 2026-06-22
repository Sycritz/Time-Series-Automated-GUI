import numpy as np
import pandas as pd
import pytest
from state import AnalysisState
from axis5_forecasting import (
    compute_psi_weights,
    undo_box_cox,
    undo_diff,
    undo_seasonal_diff,
    back_transform,
    generate_forecasts
)
from statsmodels.tsa.arima.model import ARIMA

def test_compute_psi_weights():
    # AR(1) with phi = 0.5
    # psi_0 = 1.0, psi_1 = 0.5, psi_2 = 0.25, psi_3 = 0.125
    ar = [0.5]
    ma = []
    psi = compute_psi_weights(ar, ma, 4)
    np.testing.assert_allclose(psi, [1.0, 0.5, 0.25, 0.125])
    
    # MA(1) with theta = 0.4
    # psi_0 = 1.0, psi_1 = 0.4, psi_2 = 0.0, psi_3 = 0.0
    ar = []
    ma = [0.4]
    psi = compute_psi_weights(ar, ma, 4)
    np.testing.assert_allclose(psi, [1.0, 0.4, 0.0, 0.0])
    
    # ARMA(1, 1) with phi = 0.5, theta = 0.4
    # psi_0 = 1.0
    # psi_1 = 0.4 + 0.5 * 1.0 = 0.9
    # psi_2 = 0 + 0.5 * 0.9 = 0.45
    # psi_3 = 0 + 0.5 * 0.45 = 0.225
    ar = [0.5]
    ma = [0.4]
    psi = compute_psi_weights(ar, ma, 4)
    np.testing.assert_allclose(psi, [1.0, 0.9, 0.45, 0.225])

def test_undo_box_cox():
    # Lambda = 0 (exp)
    val = np.array([0.0, 1.0, 2.0])
    res = undo_box_cox(val, 0.0)
    np.testing.assert_allclose(res, [1.0, np.exp(1.0), np.exp(2.0)])
    
    # Lambda = 2.0
    # y = (x^2 - 1)/2 => 2y + 1 = x^2 => x = sqrt(2y + 1)
    res_2 = undo_box_cox(val, 2.0)
    np.testing.assert_allclose(res_2, [1.0, np.sqrt(3.0), np.sqrt(5.0)])

def test_undo_diff():
    # Ordinary differencing order 1
    # Y = [1, 2, 3], prev_series = [10, 11, 12]
    # Reconstructed X_1 = 12 + 1 = 13, X_2 = 13 + 2 = 15, X_3 = 15 + 3 = 18
    forecasts = np.array([1.0, 2.0, 3.0])
    prev = pd.Series([10.0, 11.0, 12.0])
    reconstructed = undo_diff(forecasts, prev, 1)
    np.testing.assert_allclose(reconstructed, [13.0, 15.0, 18.0])
    
    # Order 2 differencing
    # Let X = [1, 4, 9, 16] => diff1 = [3, 5, 7] => diff2 = [2, 2]
    # If forecast is [2, 2], and prev = X
    # It should reconstruct [25, 36] (next values of X)
    forecasts_2 = np.array([2.0, 2.0])
    prev_2 = pd.Series([1.0, 4.0, 9.0, 16.0])
    reconstructed_2 = undo_diff(forecasts_2, prev_2, 2)
    np.testing.assert_allclose(reconstructed_2, [25.0, 36.0])

def test_undo_seasonal_diff():
    # Seasonal differencing order 1, s = 4
    # prev_series = [1, 2, 3, 4, 5, 6, 7, 8]
    # forecasts = [1, 1, 1, 1, 2, 2]
    # step 1: X_9 = X_5 + Y_9 = 5 + 1 = 6
    # step 2: X_10 = X_6 + Y_10 = 6 + 1 = 7
    # step 3: X_11 = X_7 + Y_11 = 7 + 1 = 8
    # step 4: X_12 = X_8 + Y_12 = 8 + 1 = 9
    # step 5: X_13 = X_9 + Y_13 = 6 + 2 = 8
    # step 6: X_14 = X_10 + Y_14 = 7 + 2 = 9
    forecasts = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    prev = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    reconstructed = undo_seasonal_diff(forecasts, prev, 1, 4)
    np.testing.assert_allclose(reconstructed, [6.0, 7.0, 8.0, 9.0, 8.0, 9.0])

def test_back_transform():
    # Test a combination chain: Box-Cox (lam=0) then diff(d=1)
    # Original series: [e^1, e^2, e^3] = [2.718, 7.389, 20.086]
    # After Box-Cox (lam=0): [1.0, 2.0, 3.0]
    # After diff (d=1): [1.0, 1.0] (history length 2)
    # Forecast on diff scale: [1.0, 1.0]
    # Back-transform should:
    # 1. Undo diff: [3.0 + 1 = 4.0, 4.0 + 1 = 5.0]
    # 2. Undo Box-Cox: [e^4, e^5]
    orig = pd.Series([np.exp(1.0), np.exp(2.0), np.exp(3.0)])
    transformations = [
        {"type": "boxcox", "lambda": 0.0},
        {"type": "diff", "d": 1}
    ]
    forecasts = {"point": np.array([1.0, 1.0])}
    res = back_transform(forecasts, transformations, orig)
    np.testing.assert_allclose(res["point"], [np.exp(4.0), np.exp(5.0)])

def test_generate_forecasts_integration():
    np.random.seed(42)
    y = pd.Series(np.random.randn(100))
    model = ARIMA(y, order=(1, 0, 1))
    res = model.fit(method='innovations_mle')
    
    state = AnalysisState()
    state.original_series = y
    state.stationary_series = y
    
    forecasts = generate_forecasts(res, 5, state)
    
    assert "point" in forecasts
    assert "lower_95" in forecasts
    assert len(forecasts["point"]) == 5
    assert len(forecasts["lower_95"]) == 5
    assert (forecasts["lower_95"] < forecasts["point"]).all()
    assert (forecasts["upper_95"] > forecasts["point"]).all()
