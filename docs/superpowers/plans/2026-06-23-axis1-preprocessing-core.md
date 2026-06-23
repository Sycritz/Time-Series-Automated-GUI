# Axis 1 Preprocessing Core Functions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement manual, from-scratch computations of rolling mean, rolling standard deviation, and sample autocovariance, and integrate them into the PySide6 user interface.

**Architecture:** We will implement manual calculations using NumPy and Python loops in `src/axis1_preprocessing.py`, preserving pandas Series wrapping where needed to align with existing matplotlib plots.

**Tech Stack:** Python 3.12, NumPy, Pandas, PySide6, Matplotlib, Pytest

## Global Constraints

- Preserve all existing comments and documentation style.
- The functions must support both pandas `Series` and numpy `ndarray` inputs.
- First `window - 1` data points must return `NaN` for rolling statistics to avoid partial-window estimation bias.
- Divisor for rolling standard deviation must be `window - 1` (unbiased standard deviation).
- Divisor for sample autocovariance must be `n` (biased estimator).

---

### Task 1: Core Mathematical Functions in axis1_preprocessing.py

**Files:**
- Modify: `src/axis1_preprocessing.py`
- Test: `tests/test_axis1.py`

**Interfaces:**
- Consumes: None
- Produces:
  - `rolling_mean(x: pd.Series | np.ndarray, window: int) -> pd.Series | np.ndarray`
  - `rolling_std(x: pd.Series | np.ndarray, window: int) -> pd.Series | np.ndarray`
  - `sample_autocovariance(x: pd.Series | np.ndarray, h: int) -> float`

- [ ] **Step 1: Write the failing tests**
  Add the following test function at the end of `tests/test_axis1.py`:
  ```python
  def test_rolling_and_autocovariance():
      print("Testing custom rolling mean, std and autocovariance...")
      from axis1_preprocessing import rolling_mean, rolling_std, sample_autocovariance
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
      print("Rolling and autocovariance tests passed.")
  ```

- [ ] **Step 2: Run tests to verify they fail**
  Run: `pytest tests/test_axis1.py::test_rolling_and_autocovariance -v`
  Expected: FAIL with "ImportError: cannot import name 'rolling_mean'"

- [ ] **Step 3: Write implementation**
  Add the functions at the end of `src/axis1_preprocessing.py`:
  ```python
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
  ```

- [ ] **Step 4: Run tests to verify they pass**
  Run: `pytest tests/test_axis1.py::test_rolling_and_autocovariance -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add src/axis1_preprocessing.py tests/test_axis1.py
  git commit -m "feat(axis1): implement manual rolling mean, std and autocovariance"
  ```

---

### Task 2: Integrate into main.py (Tab 1 and Tab 2 UI)

**Files:**
- Modify: `src/main.py`
- Test: `pytest tests/test_gui_axis1.py -v`

**Interfaces:**
- Consumes:
  - `rolling_mean(x: pd.Series | np.ndarray, window: int) -> pd.Series | np.ndarray`
  - `rolling_std(x: pd.Series | np.ndarray, window: int) -> pd.Series | np.ndarray`
- Produces: Updated GUI plots showing manual rolling stats.

- [ ] **Step 1: Write inline imports and call new functions in DataLoadTab.plot_data**
  In `src/main.py` lines 471-480, replace:
  ```python
          rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
          ax1.plot(series.index, rolling_mean, color='#2563EB', label=f'Rolling Mean ({window_size})')
          ax1.set_ylabel('Values')
          ax1.set_xlabel(series.index.name if series.index.name else 'Time')
          ax1.legend(loc='best')
          ax1.set_title("Rolling Mean")
          
          # Subplot 2: Rolling Std
          rolling_std = series.rolling(window=window_size, min_periods=1).std()
  ```
  With:
  ```python
          from axis1_preprocessing import rolling_mean, rolling_std
          rolling_mean_vals = rolling_mean(series, window_size)
          ax1.plot(series.index, rolling_mean_vals, color='#2563EB', label=f'Rolling Mean ({window_size})')
          ax1.set_ylabel('Values')
          ax1.set_xlabel(series.index.name if series.index.name else 'Time')
          ax1.legend(loc='best')
          ax1.set_title("Rolling Mean")
          
          # Subplot 2: Rolling Std
          rolling_std_vals = rolling_std(series, window_size)
  ```
  And replace line 480:
  ```python
          ax2.plot(series.index, rolling_std, color='#16A34A', label=f'Rolling Std ({window_size})', linestyle='--')
  ```
  With:
  ```python
          ax2.plot(series.index, rolling_std_vals, color='#16A34A', label=f'Rolling Std ({window_size})', linestyle='--')
  ```

- [ ] **Step 2: Write inline imports and call new functions in TransformTab.plot_transformed**
  In `src/main.py` lines 691-700, replace:
  ```python
          rolling_mean = clean_series.rolling(window=window_size, min_periods=1).mean()
          ax1.plot(clean_series.index, rolling_mean, color='#2563EB', label=f'Rolling Mean ({window_size})')
          ax1.set_ylabel('Transformed Values')
          ax1.set_xlabel(clean_series.index.name if clean_series.index.name else 'Time')
          ax1.legend(loc='best')
          ax1.set_title("Rolling Mean")
          
          rolling_std = clean_series.rolling(window=window_size, min_periods=1).std()
  ```
  With:
  ```python
          from axis1_preprocessing import rolling_mean, rolling_std
          rolling_mean_vals = rolling_mean(clean_series, window_size)
          ax1.plot(clean_series.index, rolling_mean_vals, color='#2563EB', label=f'Rolling Mean ({window_size})')
          ax1.set_ylabel('Transformed Values')
          ax1.set_xlabel(clean_series.index.name if clean_series.index.name else 'Time')
          ax1.legend(loc='best')
          ax1.set_title("Rolling Mean")
          
          rolling_std_vals = rolling_std(clean_series, window_size)
  ```
  And replace line 699:
  ```python
          ax2.plot(clean_series.index, rolling_std, color='#16A34A', label=f'Rolling Std ({window_size})', linestyle='--')
  ```
  With:
  ```python
          ax2.plot(clean_series.index, rolling_std_vals, color='#16A34A', label=f'Rolling Std ({window_size})', linestyle='--')
  ```

- [ ] **Step 3: Run the GUI unit tests**
  Run: `pytest tests/test_gui_axis1.py -v`
  Expected: PASS

- [ ] **Step 4: Run the full test suite (except the known validation/gating failures)**
  Run: `pytest -v`
  Expected: All pass except `test_gui_validation_workflow_fail` and `test_gui_gating_and_indicators` (failures are known and documented in `triage_and_tackling_proposal.md`).

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add src/main.py
  git commit -m "feat(axis1): integrate manual rolling statistics into UI Tab 1 & Tab 2"
  ```
