import numpy as np
import pandas as pd
import tempfile
import os
import pytest
from axis1_preprocessing import (
    load_csv, run_adf_test,
    apply_box_cox, apply_differencing,
    box_cox_transform, box_cox_inverse, box_cox_profile_loglik,
    box_cox_auto, impute_series, rolling_mean, rolling_std,
    sample_autocovariance
)

def test_load_csv():
    print("Testing load_csv...")
    # Create temp CSV
    content = "time,value\n0,1.2\n1,3.4\n2,5.6\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        temp_path = f.name
        
    try:
        df = load_csv(temp_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ['time', 'value']
    finally:
        os.remove(temp_path)
    print("load_csv passed.")

def test_handle_missing():
    print("Testing handle_missing (via impute_series)...")
    # Series with missing values
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    
    # 1. Forward Fill
    res_ff, n_imp = impute_series(s.values, 'forward_fill')
    pct = (n_imp / len(s)) * 100.0
    assert pct == 40.0
    assert np.isnan(res_ff).sum() == 0
    assert list(res_ff) == [1.0, 1.0, 3.0, 3.0, 5.0]
    
    # 2. Linear Interpolation
    res_li, _ = impute_series(s.values, 'linear')
    assert np.isnan(res_li).sum() == 0
    assert list(res_li) == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # 3. Mean Imputation
    res_mean, _ = impute_series(s.values, 'mean')
    assert np.isnan(res_mean).sum() == 0
    assert list(res_mean) == [1.0, 3.0, 3.0, 3.0, 5.0]
    print("handle_missing passed.")

def test_run_adf_test():
    print("Testing run_adf_test...")
    from statsmodels.tsa.stattools import adfuller

    # 1. Stationary series (white noise)
    np.random.seed(42)
    stat_series = pd.Series(np.random.randn(100))
    
    # Calculate p_max manually for comparison
    N = len(stat_series)
    p_max = int(np.floor(12 * (N / 100.0) ** 0.25))
    p_max = min(N // 2 - 2, p_max)
    
    res = run_adf_test(stat_series)
    res_sm = adfuller(stat_series.dropna(), maxlag=p_max, autolag='AIC')
    
    # Verify values match statsmodels for statistics and custom values for p-values/critical values
    np.testing.assert_allclose(res['adf_stat'], res_sm[0], rtol=1e-7, atol=1e-7)
    assert res['critical_values'] == {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    
    tau = res['adf_stat']
    if tau <= -3.43:
        expected_p = 0.01
    elif tau <= -2.86:
        expected_p = 0.01 + (tau - (-3.43)) / (-2.86 - (-3.43)) * (0.05 - 0.01)
    elif tau <= -2.57:
        expected_p = 0.05 + (tau - (-2.86)) / (-2.57 - (-2.86)) * (0.10 - 0.05)
    else:
        if tau >= 0.0:
            expected_p = 0.99
        else:
            expected_p = 0.10 + (tau - (-2.57)) / (0.0 - (-2.57)) * (0.99 - 0.10)
    assert np.isclose(res['p_value'], expected_p)
    
    nobs_sm = res_sm[3]
    used_lag_sm = res_sm[2]
    assert nobs_sm == N - 1 - used_lag_sm
    assert res['verdict'] == ('Stationary' if expected_p < 0.05 else 'Non-Stationary')
    
    # 2. Non-stationary series (random walk)
    np.random.seed(42)
    non_stat_series = pd.Series(np.random.randn(100).cumsum())
    
    res_ns = run_adf_test(non_stat_series)
    res_ns_sm = adfuller(non_stat_series.dropna(), maxlag=p_max, autolag='AIC')
    
    np.testing.assert_allclose(res_ns['adf_stat'], res_ns_sm[0], rtol=1e-7, atol=1e-7)
    assert res_ns['critical_values'] == {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    
    tau_ns = res_ns['adf_stat']
    if tau_ns <= -3.43:
        expected_p_ns = 0.01
    elif tau_ns <= -2.86:
        expected_p_ns = 0.01 + (tau_ns - (-3.43)) / (-2.86 - (-3.43)) * (0.05 - 0.01)
    elif tau_ns <= -2.57:
        expected_p_ns = 0.05 + (tau_ns - (-2.86)) / (-2.57 - (-2.86)) * (0.10 - 0.05)
    else:
        if tau_ns >= 0.0:
            expected_p_ns = 0.99
        else:
            expected_p_ns = 0.10 + (tau_ns - (-2.57)) / (0.0 - (-2.57)) * (0.99 - 0.10)
    assert np.isclose(res_ns['p_value'], expected_p_ns)
    assert res_ns['verdict'] == ('Stationary' if expected_p_ns < 0.05 else 'Non-Stationary')
    print("run_adf_test passed.")

def test_apply_box_cox():
    print("Testing apply_box_cox...")
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 1. Manual lambda
    res, lam = apply_box_cox(s, 0.0)
    assert lam == 0.0
    np.testing.assert_allclose(res.values, np.log(s.values))
    
    # 2. Auto-optimize
    res_auto, lam_auto = apply_box_cox(s, None)
    assert isinstance(lam_auto, float)
    
    # 3. Invalid non-positive data
    s_invalid = pd.Series([1.0, 0.0, -1.0])
    try:
        apply_box_cox(s_invalid, 1.0)
        assert False, "Should raise ValueError for non-positive data"
    except ValueError:
        pass
    print("apply_box_cox passed.")

def test_apply_differencing():
    print("Testing apply_differencing...")
    s = pd.Series([1, 2, 4, 7, 11, 16], dtype=float)
    
    # d = 1, D = 0
    diff_1 = apply_differencing(s, 1, 0, 1)
    assert diff_1.iloc[0] is np.nan or pd.isna(diff_1.iloc[0])
    assert list(diff_1.dropna()) == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # d = 0, D = 1, s = 2
    diff_seas = apply_differencing(s, 0, 1, 2)
    # x_t - x_{t-2}
    # 4 - 1 = 3, 7 - 2 = 5, 11 - 4 = 7, 16 - 7 = 9
    assert list(diff_seas.dropna()) == [3.0, 5.0, 7.0, 9.0]
    print("apply_differencing passed.")

def test_rolling_and_autocovariance():
    print("Testing custom rolling mean, std and autocovariance...")
    x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    # Test rolling_mean
    rm = rolling_mean(x, 3)
    np.testing.assert_allclose(rm[2:], [4.0, 6.0, 8.0])
    assert np.isnan(rm[0]) and np.isnan(rm[1])
    
    # Test rolling_std
    rs = rolling_std(x, 3)
    np.testing.assert_allclose(rs[2:], [2.0, 2.0, 2.0])
    assert np.isnan(rs[0]) and np.isnan(rs[1])
    
    # Test sample_autocovariance
    # For [2.0, 4.0, 6.0, 8.0, 10.0]: mean = 6.0
    # diffs = [-4, -2, 0, 2, 4]
    # h = 0: (16 + 4 + 0 + 4 + 16)/5 = 40/5 = 8.0
    # h = 1: ((-4)*(-2) + (-2)*0 + 0*2 + 2*4)/5 = (8 + 8)/5 = 3.2
    assert np.isclose(sample_autocovariance(x, 0), 8.0)
    assert np.isclose(sample_autocovariance(x, 1), 3.2)
    
    # Test pd.Series support and metadata preservation
    series_x = pd.Series(x, index=[10, 20, 30, 40, 50], name="test_col")
    rm_s = rolling_mean(series_x, 3)
    assert isinstance(rm_s, pd.Series)
    assert rm_s.name == "test_col"
    pd.testing.assert_index_equal(rm_s.index, series_x.index)
    np.testing.assert_allclose(rm_s.values[2:], [4.0, 6.0, 8.0])
    
    rs_s = rolling_std(series_x, 3)
    assert isinstance(rs_s, pd.Series)
    assert rs_s.name == "test_col"
    pd.testing.assert_index_equal(rs_s.index, series_x.index)
    np.testing.assert_allclose(rs_s.values[2:], [2.0, 2.0, 2.0])
    
    # Test exception handling and validation boundaries
    with pytest.raises(ValueError, match="Window size must be greater than 0"):
        rolling_mean(x, 0)
    with pytest.raises(ValueError, match="Window size must be greater than 1"):
        rolling_std(x, 1)
    with pytest.raises(ValueError, match="Lag h must be non-negative"):
        sample_autocovariance(x, -1)
    
    # Test edge case window > n and h >= n
    rm_large = rolling_mean(x, 10)
    assert np.all(np.isnan(rm_large))
    rs_large = rolling_std(x, 10)
    assert np.all(np.isnan(rs_large))
    assert sample_autocovariance(x, 10) == 0.0
    
    print("Rolling and autocovariance tests passed.")

def test_box_cox_detailed():
    print("Testing Box-Cox transformations and grid search...")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 1. Transform
    y_log = box_cox_transform(x, 0.0)
    np.testing.assert_allclose(y_log, np.log(x))
    
    y_power = box_cox_transform(x, 2.0)
    np.testing.assert_allclose(y_power, (x**2 - 1.0) / 2.0)
    
    with pytest.raises(ValueError):
        box_cox_transform(np.array([1.0, 0.0, -1.0]), 1.0)
        
    # 2. Inverse
    np.testing.assert_allclose(box_cox_inverse(y_log, 0.0), x)
    np.testing.assert_allclose(box_cox_inverse(y_power, 2.0), x)
    
    # 3. Profile Log-Likelihood
    # Non-positive values raise ValueError
    with pytest.raises(ValueError):
        box_cox_profile_loglik(np.array([1.0, -1.0]), 1.0)
    # Correct calculation check
    ll_0 = box_cox_profile_loglik(x, 0.0)
    ll_2 = box_cox_profile_loglik(x, 2.0)
    assert isinstance(ll_0, float)
    assert isinstance(ll_2, float)
    
    # 4. Auto-optimize
    best_lam = box_cox_auto(x)
    assert isinstance(best_lam, float)
    assert -2.0 <= best_lam <= 2.0
    
    print("Box-Cox detailed tests passed.")

def test_impute_series_detailed():
    print("Testing impute_series...")
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    
    # 1. Linear interpolation
    x_linear, n_imp = impute_series(x, "linear")
    assert n_imp == 2
    np.testing.assert_allclose(x_linear, [1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 2. Forward fill
    x_ff, n_imp_ff = impute_series(x, "forward_fill")
    assert n_imp_ff == 2
    np.testing.assert_allclose(x_ff, [1.0, 1.0, 3.0, 3.0, 5.0])
    
    # 3. Mean Imputation
    x_mean, n_imp_mean = impute_series(x, "mean")
    assert n_imp_mean == 2
    np.testing.assert_allclose(x_mean, [1.0, 3.0, 3.0, 3.0, 5.0])
    
    # Test boundary case where first element is NaN for forward fill
    x_nan_first = np.array([np.nan, 2.0, 3.0])
    x_nan_first_ff, _ = impute_series(x_nan_first, "forward_fill")
    np.testing.assert_allclose(x_nan_first_ff, [2.0, 2.0, 3.0])
    
    # Test when all elements are NaN
    x_all_nan = np.array([np.nan, np.nan])
    x_all_nan_li, _ = impute_series(x_all_nan, "linear")
    np.testing.assert_allclose(x_all_nan_li, [0.0, 0.0])
    
    x_all_nan_ff, _ = impute_series(x_all_nan, "forward_fill")
    np.testing.assert_allclose(x_all_nan_ff, [0.0, 0.0])
    
    x_all_nan_mean, _ = impute_series(x_all_nan, "mean")
    np.testing.assert_allclose(x_all_nan_mean, [0.0, 0.0])

    print("impute_series detailed tests passed.")

if __name__ == "__main__":
    test_load_csv()
    test_handle_missing()
    test_run_adf_test()
    test_apply_box_cox()
    test_apply_differencing()
    test_rolling_and_autocovariance()
    test_box_cox_detailed()
    test_impute_series_detailed()
    print("All preprocessing tests passed successfully!")
