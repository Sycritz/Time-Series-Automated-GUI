import numpy as np
import pandas as pd
import tempfile
import os
from axis1_preprocessing import (
    load_csv, handle_missing, run_adf_test,
    apply_box_cox, apply_differencing, compute_frequency_response
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
    print("Testing handle_missing...")
    # Series with missing values
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    
    # 1. Forward Fill
    res_ff, pct = handle_missing(s, 'Forward Fill')
    assert pct == 40.0
    assert res_ff.isna().sum() == 0
    assert list(res_ff) == [1.0, 1.0, 3.0, 3.0, 5.0]
    
    # 2. Linear Interpolation
    res_li, _ = handle_missing(s, 'Linear Interpolation')
    assert res_li.isna().sum() == 0
    assert list(res_li) == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # 3. Mean Imputation
    res_mean, _ = handle_missing(s, 'Mean Imputation')
    assert res_mean.isna().sum() == 0
    assert list(res_mean) == [1.0, 3.0, 3.0, 3.0, 5.0]
    print("handle_missing passed.")

def test_run_adf_test():
    print("Testing run_adf_test...")
    # 1. Stationary series (white noise)
    np.random.seed(42)
    stat_series = pd.Series(np.random.randn(100))
    res = run_adf_test(stat_series)
    assert 'adf_stat' in res
    assert 'p_value' in res
    assert 'critical_values' in res
    assert res['verdict'] == 'Stationary'
    
    # 2. Non-stationary series (random walk)
    non_stat_series = pd.Series(np.random.randn(100).cumsum())
    res_ns = run_adf_test(non_stat_series)
    assert res_ns['verdict'] == 'Non-Stationary'
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

def test_compute_frequency_response():
    print("Testing compute_frequency_response...")
    w, mag = compute_frequency_response(1, 0, 1, n_points=100)
    assert len(w) == 100
    assert len(mag) == 100
    assert mag[0] == 0.0  # magnitude response of first diff is 0 at frequency 0
    
    w_seas, mag_seas = compute_frequency_response(0, 1, 12, n_points=100)
    assert mag_seas[0] == 0.0
    print("compute_frequency_response passed.")

if __name__ == "__main__":
    test_load_csv()
    test_handle_missing()
    test_run_adf_test()
    test_apply_box_cox()
    test_apply_differencing()
    test_compute_frequency_response()
    print("All preprocessing tests passed successfully!")
