# Product Requirements Document (PRD)

## Problem Statement

The user is experiencing discrepancies between the automated GUI application's results (in model identification, validation, and forecasting) and the theoretical expectations from Brockwell & Davis (B&D), as well as the reference implementation created by their project partner in the `Time-Series-Forcasting` repository. 

Key issues include:
1. **Unstructured & Duplicate Code**: Rolling mean and rolling standard deviation calculations are implemented inline and duplicated across multiple tabs (`main.py`), utilizing pandas built-in functions. The academic spec and the partner require these to be standalone functions implemented from scratch using NumPy and `for` loops in Axis 1.
2. **Algorithmic Inconsistencies**: The current ACF/PACF model suggestion rules, ADF stationarity checks, Durbin-Levinson PACF computations, Ljung-Box tests, and Jarque-Bera tests rely on statsmodels/scipy wrappers which can return results that disagree with textbook formulas.
3. **Forecasting Discrepancies**: Back-transformation of prediction intervals for log-transformed series can explode asymmetrically, failing to match the partner's correct multiplicative scaling.

---

## Solution

Rebuild and align the core time-series algorithms in the GUI codebase to match the partner's codebase (serving as the ground truth) and Brockwell & Davis methodologies.

1. **Extract & Rewrite Preprocessing**: Implement manual rolling mean, rolling standard deviation, sample autocovariance, differencing, Box-Cox transformations, and the ADF test from scratch using NumPy and Python loops in `axis1_preprocessing.py`.
2. **Implement Hand-Crafted Identification & Validation**: Implement Durbin-Levinson PACF recursion, Box-Jenkins pattern suggestions, Ljung-Box portmanteau statistic, and Jarque-Bera normality tests from scratch.
3. **Refine Forecasting Scale**: Integrate multiplicative forecasting bounds back-transformation for log-transformed inputs.
4. **Integrate into GUI**: Cleanly import and display these from-scratch calculations in `main.py`, replacing the duplicated pandas rolling calls while preserving the warning behavior on the forecasting tab if validation fails.

---

## User Stories

1. As a student presenting Axis 1, I want the rolling mean and standard deviation to be computed manually with loops and NumPy, so that my grade is not penalized for using pandas wrappers.
2. As a user, I want the rolling statistics to return NaN for the first `window - 1` data points, so that there is no partial-window estimation bias.
3. As a user, I want the automatic Box-Cox parameter optimization to use a manual profile log-likelihood grid search over $[-2, 2]$, so that it matches B&D textbook methods.
4. As a student presenting Axis 3, I want the PACF to be computed using the recursive Durbin-Levinson algorithm, so that I can explain the steps during the oral defense.
5. As a user, I want the automated model suggestions to handle stray significant lags (up to a tolerance of 1) and tailing-off states correctly, so that the suggestions match classical Box-Jenkins rules.
6. As a student presenting Axis 4, I want the Ljung-Box and Jarque-Bera statistics computed from scratch, so that we demonstrate full mathematical understanding to the evaluators.
7. As a user, I want the forecasting bounds of log-transformed series to be back-transformed multiplicatively using exponents, so that the prediction intervals do not explode asymmetrically.
8. As an exploratory researcher, I want the forecasting tab to be unlocked when a model is fitted even if validation fails, so that I can generate forecasts with a warning warning.

---

## Implementation Decisions

### 1. Preprocessing & Stationarity (Axis 1)
- **Rolling Mean and Standard Deviation**:
  - Implement `rolling_mean(x, window)` and `rolling_std(x, window)` inside `axis1_preprocessing.py`.
  - Use Python `for` loops.
  - Return `NaN` for the first `window - 1` positions.
  - Compute standard deviation using the unbiased divisor ($window - 1$).
- **Sample Autocovariance**:
  - Implement `sample_autocovariance(x, h)` utilizing the biased estimator formula (dividing by the series length $n$).
- **ADF Test**:
  - Rewrite ADF test from scratch via manual OLS regression, AIC lag selection (Schwert 1989 rule), and hardcoded MacKinnon critical values.
- **Box-Cox & Differencing**:
  - Implement Box-Cox transformation, inverse transformation, and log-likelihood profile grid search manually.
  - Rewrite differencing and seasonal differencing manually using NumPy operations.
  - Add `impute_series` from the partner's code to replace `handle_missing`.

### 2. Time-Domain Modeling & Selection (Axis 3)
- **ACF & PACF**:
  - Compute ACF from scratch using the new `sample_autocovariance` function.
  - Compute PACF from scratch using the Durbin-Levinson recursion.
- **Model Suggestions**:
  - Rewrite `suggest_model_from_acf_pacf` to incorporate stray-lag tolerance and tailing-off heuristics.
- **Fitting Uniformity**:
  - Standardize all fitting calls on `statsmodels.tsa.statespace.sarimax.SARIMAX` to ensure identical MLE results for seasonal and non-seasonal models.

### 3. Model Auditor & Diagnostics (Axis 4)
- **Ljung-Box Test**:
  - Implement $Q_{LB}(h) = n(n+2) \sum_{k=1}^h \hat{\rho}^2(k)/(n-k)$ from scratch.
  - Adjust degrees of freedom to $h - p - q$ for p-value computation.
- **Jarque-Bera Test**:
  - Compute skewness $S$ and kurtosis $K$ manually from central moments, and calculate the test statistic $JB = n[S^2/6 + (K-3)^2/24]$.

### 4. Forecasting Tab (Axis 5 & main.py)
- **Back-Transformation**:
  - For log-transformed series, calculate prediction intervals using multiplicative scaling: $PI = \hat{X}_{original} \cdot \exp(\pm z \cdot \sigma_{stationary}(h))$.
- **GUI Tab Gating**:
  - Keep the Forecasting tab (Tab 6) active if a model has been fitted. If validation tests failed, present a prominent warning dialog and message rather than locking the tab completely.
- **Rolling Stats Integration**:
  - Import the new `rolling_mean` and `rolling_std` functions into `main.py` and replace the pandas `.rolling()` calls in `DataLoadTab.plot_data()` and `TransformTab.plot_transformed()`.

---

## Testing Decisions

- **Test Seams**:
  - We will test the mathematical equivalence of the manual from-scratch functions against statsmodels/scipy references (where applicable) and ensure they produce expected outputs.
- **Target Modules for Tests**:
  - `tests/test_axis1.py` - Verify manual rolling mean, rolling std, sample autocovariance, and Box-Cox transforms.
  - `tests/test_axis3.py` - Verify Durbin-Levinson PACF and Box-Jenkins suggestion logic.
  - `tests/test_axis4.py` - Verify Ljung-Box and Jarque-Bera outputs.
  - `tests/test_axis5.py` - Verify multiplicative PI back-transformation.
- **Test Integrity**:
  - Ensure all assertions run successfully using the existing test runner configurations.

---

## Out of Scope

- Modifying the spectral estimation code (`axis2_spectral.py`) or cycle detection algorithm, as they are already implemented from scratch and correct.
- Rewriting the statsmodels Kalman filter/MLE numerical solver.
- Changing PySide6 visual themes, fonts, or colors.

---

## Further Notes

- The partner's code at `~/School-Projects/Time-Series-Forcasting` remains the authoritative template.
- All code must preserve existing comments and documentation style.
