# Axis 3 — The Model Architect: Time Domain Modeling & Identification
### Defense Preparation Guide · Tab 4

> **Source file:** [`src/axis3_modeling.py`](../src/axis3_modeling.py)  
> **Audience:** Student team preparing for project defense  
> **B&D Reference:** Brockwell & Davis — *Introduction to Time Series and Forecasting*, Chapters 5 (Forecasting), 8 (Estimation), and 9 (Model Building)

---

## Table of Contents

1. [Role & Purpose](#1-role--purpose)
2. [Theory: The B&D Foundation](#2-theory-the-bd-foundation)
   - 2.1 [ACF & PACF Behavior](#21-acf--pacf-behavior)
   - 2.2 [Asymptotic Bounds under White Noise](#22-asymptotic-bounds-under-white-noise)
   - 2.3 [Box-Jenkins Model Identification Rules](#23-box-jenkins-model-identification-rules)
   - 2.4 [Information Criteria: AIC, BIC, & AICc](#24-information-criteria-aic-bic--aicc)
   - 2.5 [Estimation: Maximum Likelihood & Innovations Algorithm](#25-estimation-maximum-likelihood--innovations-algorithm)
   - 2.6 [Dual Suggestion Strategy](#26-dual-suggestion-strategy)
3. [Implementation Walkthrough](#3-implementation-walkthrough)
   - 3.1 [`compute_acf_pacf`](#31-compute_acf_pacf)
   - 3.2 [`suggest_model_from_acf_pacf`](#32-suggest_model_from_acf_pacf)
   - 3.3 [`suggest_model_from_spectrum`](#33-suggest_model_from_spectrum)
   - 3.4 [`fit_model`](#34-fit_model)
   - 3.5 [`grid_search`](#35-grid_search)
   - 3.6 [`GridSearchWorker`](#36-gridsearchworker)
4. [Hand-Implemented vs Library Table](#4-hand-implemented-vs-library-table)
5. [Data Flow](#5-data-flow)
6. [Design Decisions & Gotchas](#6-design-decisions--gotchas)
7. [Likely Defense Questions & Answers](#7-likely-defense-questions--answers)
8. [Quick Reference — Key Formulas](#8-quick-reference--key-formulas)

---

## 1. Role & Purpose

Axis 3 is the **System Architect** of the time-domain model. Once Axis 1 has prepared a stationary time series, Axis 3's task is to identify, estimate, and select a candidate set of ARMA/ARIMA or SARIMA models. It bridges the gap between raw data properties and concrete parametric specifications by:
- Presenting the Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots.
- Suggesting model orders using time-domain heuristics (Box-Jenkins rules) and frequency-domain suggestions (spectral peak matching).
- Performing a multi-threaded, non-blocking Grid Search over hundreds of model order configurations, ranking them by corrected Information Criteria (AICc).
- Fitting the final model parameters via Maximum Likelihood Estimation (MLE) using the Kalman filter or the Innovations Algorithm.

---

## 2. Theory: The B&D Foundation

### 2.1 ACF & PACF Behavior

The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) are the primary time-domain fingerprints of stationary stochastic processes:

- **Autocorrelation Function (ACF):** Measures the linear correlation between $X_t$ and $X_{t-k}$:
  $$\rho(k) = \frac{\gamma(k)}{\gamma(0)}$$
- **Partial Autocorrelation Function (PACF):** Measures the correlation between $X_t$ and $X_{t-k}$ after removing the linear influence of the intermediate variables $X_{t-1}, \dots, X_{t-k+1}$. Under the hood, the PACF lag $k$ coefficient $\phi_{kk}$ is the last coefficient of the best linear predictor of $X_{t+k}$ based on $\{X_{t+k-1}, \dots, X_t\}$:
  $$X_{t+k} = \phi_{k1} X_{t+k-1} + \dots + \phi_{kk} X_t + W_t$$
  It is recursively computed using the **Durbin-Levinson Algorithm**.

---

### 2.2 Asymptotic Bounds under White Noise

To determine whether an ACF or PACF coefficient is statistically different from zero, we use the classical asymptotic result (B&D Theorem 7.2.2). Under the null hypothesis that $\{X_t\}$ is i.i.d. white noise with finite variance:

$$\hat{\rho}(k) \;\overset{\text{approx}}{\sim}\; \mathcal{N}\left(0, \frac{1}{n}\right) \quad \text{for } k \ge 1$$

Therefore, the $95\%$ critical bounds under the null hypothesis of white noise are:

$$\text{Bounds} = \pm \frac{1.96}{\sqrt{n}}$$

If a sample autocorrelation or partial autocorrelation falls outside these bounds, we reject the null hypothesis at the $5\%$ significance level and treat the correlation at that lag as statistically significant.

---

### 2.3 Box-Jenkins Model Identification Rules

The Box-Jenkins methodology relies on the matching of sample ACF/PACF signatures to the theoretical behaviors of pure AR, pure MA, and mixed ARMA processes:

| Model | Theoretical ACF | Theoretical PACF |
|---|---|---|
| **$\text{AR}(p)$** | Tails off (decays exponentially or as a damped sine wave) | Cuts off after lag $p$ ($\alpha(k) = 0$ for $k > p$) |
| **$\text{MA}(q)$** | Cuts off after lag $q$ ($\rho(k) = 0$ for $k > q$) | Tails off (decays geometrically or as a damped sine wave) |
| **$\text{ARMA}(p, q)$** | Tails off starting at lag $q$ | Tails off starting at lag $p$ |

---

### 2.4 Information Criteria: AIC, BIC, & AICc

While the Box-Jenkins rules provide a starting point, formal model selection is guided by Information Criteria that balance **goodness-of-fit** against **model parsimony** (penalizing too many parameters to avoid overfitting).

1. **Akaike Information Criterion (AIC):**
   $$\text{AIC} = -2\ln(L) + 2k$$
   where $L$ is the maximized likelihood and $k$ is the number of estimated parameters (including the white noise variance $\sigma^2$).
2. **Bayesian Information Criterion (BIC):**
   $$\text{BIC} = -2\ln(L) + k\ln(n)$$
   BIC penalizes parameters more heavily than AIC for $n \ge 8$.
3. **Corrected AIC (AICc):**
   For small samples, AIC tends to select overfitted models. The corrected AIC adds a second-order correction term (B&D §9.3):
   $$\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1} = -2\ln(L) + 2k + \frac{2k(k+1)}{n - k - 1}$$
   As $n \to \infty$, the correction term goes to zero, converging to standard AIC. The app strictly uses AICc to rank models in the Grid Search.

---

### 2.5 Estimation: Maximum Likelihood & Innovations Algorithm

For an ARMA model, the parameter vector is estimated by maximizing the Gaussian likelihood. Because the covariance matrix of an ARMA process is complex, computing its inverse directly is computationally expensive.

The **Innovations Algorithm** (B&D §5.2) solves this by recursively computing the one-step-ahead predictors $\hat{X}_{t+1}$ and their mean squared errors $v_t = \text{E}(X_{t+1} - \hat{X}_{t+1})^2$. This allows us to write the Gaussian log-likelihood in terms of the independent one-step-ahead prediction errors (innovations) $e_t = X_t - \hat{X}_t$:

$$\ln L(\beta, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2}\sum_{t=1}^{n}\ln(v_{t-1}) - \frac{1}{2\sigma^2}\sum_{t=1}^{n}\frac{(X_t - \hat{X}_t)^2}{v_{t-1}}$$

For non-seasonal models, the application sets `method='innovations_mle'` in `statsmodels`, which invokes this exact recursion. For seasonal models, it uses `SARIMAX` which optimizes the likelihood using the Kalman filter state-space representation.

---

### 2.6 Dual Suggestion Strategy

Rather than relying purely on time-domain ACF/PACF heuristics, the app implements a **dual-suggestion strategy**:

1. **Time-Domain Suggester:** Examines the max significant lags in ACF and PACF and suggests order bounds (e.g. if ACF is zero after lag 2 and PACF decays, suggest $\text{MA}(2)$).
2. **Frequency-Domain Suggester:** Looks at the significant cycles detected by Axis 2.
   - If a significant spectral peak matches the seasonal period $s$ (within $15\%$ tolerance) or its main harmonics ($s/2, s/3, s/4$), it recommends a seasonal SARIMA model with seasonal orders $P=1, Q=1, s$.
   - Otherwise, it recommends a non-seasonal model.

---

## 3. Implementation Walkthrough

### 3.1 `compute_acf_pacf`

Computes ACF and PACF using statsmodels. It caps the number of lags at $n/2 - 1$ to avoid statsmodels exceptions, and uses the Yule-Walker method (`ywm`) for PACF to ensure stability:

```python
clean_series = series.dropna()
n = len(clean_series)
actual_lags = min(nlags, n // 2 - 1)
# ...
acf_vals = acf(clean_series, nlags=actual_lags)
pacf_vals = pacf(clean_series, nlags=actual_lags, method='ywm')
```

### 3.2 `suggest_model_from_acf_pacf`

Applies the Box-Jenkins heuristic. It counts the number of lags exceeding $\pm 1.96/\sqrt{n}$:

```python
threshold = 1.96 / np.sqrt(n) if n > 0 else 0.2
sig_acf = [lag for lag in range(1, len(acf_vals)) if abs(acf_vals[lag]) > threshold]
sig_pacf = [lag for lag in range(1, len(pacf_vals)) if abs(pacf_vals[lag]) > threshold]
# ...
if acf_max <= 3 and (pacf_max > acf_max + 1 or len(sig_pacf) > len(sig_acf)):
    # ACF cuts off, PACF tails off
    return {"p": 0, "q": acf_max, "explanation": ...}
elif pacf_max <= 3 and (acf_max > pacf_max + 1 or len(sig_acf) > len(sig_pacf)):
    # PACF cuts off, ACF tails off
    return {"p": pacf_max, "q": 0, "explanation": ...}
else:
    # Both tail off
    return {"p": 1, "q": 1, "explanation": ...}
```

### 3.3 `suggest_model_from_spectrum`

Matches the detected cycles from Axis 2 against the seasonal period $s$:

```python
matched = False
for c in significant_cycles:
    p = c["period"]
    tolerance = 0.15 * seasonal_period
    if abs(p - seasonal_period) <= tolerance:
        matched = True
        break
    else:
        for h in [2, 3, 4]:
            harmonic_p = seasonal_period / h
            if abs(p - harmonic_p) <= 0.15 * harmonic_p:
                matched = True
                break
# Suggest SARIMA(p,d,q)x(1,D,1)_s if matched
```

### 3.4 `fit_model`

Fits the specified model. It switches between `ARIMA` with `method='innovations_mle'` for non-seasonal models (faster and theoretically elegant) and `SARIMAX` for seasonal models:

```python
if is_seas:
    model = SARIMAX(clean_series, order=order, seasonal_order=seasonal_order)
    res = model.fit(disp=False)
else:
    model = ARIMA(clean_series, order=order)
    res = model.fit(method='innovations_mle')
```

### 3.5 `grid_search`

Iterates through all permutations of $(p, d, q) \times (P, D, Q)_s$. It catches and ignores convergence and user warnings to prevent the loop from aborting, and manually computes the AICc:

```python
# Loop combos...
res = fit_model(series, order, seasonal_order)
k = len(res.params)
n_eff = res.nobs
aic = res.aic

if n_eff - k - 1 > 0:
    aicc = aic + (2.0 * k * (k + 1)) / (n_eff - k - 1)
else:
    aicc = aic
```

### 3.6 `GridSearchWorker`

This is a PySide6 `QThread` subclass. It wraps `grid_search` to run in the background. It sends percentage progress signals to update the UI progress bar and emits the final pandas DataFrame when done, ensuring the application remains responsive:

```python
class GridSearchWorker(QThread):
    progress = Signal(int)
    finished = Signal(pd.DataFrame)
    
    def run(self):
        # Calls grid_search in separate thread...
        self.finished.emit(df_results)
```

---

## 4. Hand-Implemented vs Library Table

| Component | Hand-Implemented | Library | Why |
|---|---|---|---|
| **ACF/PACF Bounds** | Threshold equation $\pm 1.96/\sqrt{n}$. | — | Direct implementation of the white noise asymptotic distribution. |
| **ACF/PACF Values** | — | `statsmodels.tsa.stattools.acf/pacf` | Reuses optimized autocovariance Fast Fourier Transform logic. |
| **BJ Heuristics** | Rule matching logic & natural text explanations. | — | Translates Box-Jenkins cut-off and tail-off theory into code. |
| **Spectral Suggestion** | 15% tolerance harmonic search. | — | Combines frequency domain inputs to suggest seasonal orders. |
| **AICc Calculation** | The corrected formula: $\text{AIC} + \frac{2k(k+1)}{n-k-1}$. | — | `statsmodels` does not natively compute AICc for all model classes. |
| **Parameter Estimation** | — | `statsmodels` (ARIMA/SARIMAX) | Solving the non-linear optimization of MLE requires robust algorithms. |
| **Grid Search Threading** | Non-blocking QThread worker. | — | Standard PySide6 threads isolate heavy computations from the UI loop. |

---

## 5. Data Flow

```
state.stationary_series & state.detected_cycles (from Axis 1 & 2)
  │
  ▼
[Tab 4: Model Identification]
  ├── compute_acf_pacf() -> Plots ACF & PACF with hover tooltips
  ├── suggest_model_from_acf_pacf() ──┐
  │                                    ├──► populates Order Suggestions
  ├── suggest_model_from_spectrum() ───┘
  │
  ├── GridSearchWorker (runs in background)
  │      └─► grid_search() -> outputs sorted ranking table
  │
  ▼ (User selects model & clicks Fit)
fit_model() -> statsmodels Result Object
  │
  ├─► writes to state.fitted_model
  ├─► writes to state.model_order & state.model_params
  ├─► writes to state.residuals
  └─► resets state.validation_passed = False
```

---

## 6. Design Decisions & Gotchas

1. **Why `innovations_mle`?** For non-seasonal ARMA, statsmodels can compute exact MLE using the Innovations Algorithm. This is faster and has better convergence properties than the full state-space representation, and aligns perfectly with Chapter 5/8 of the B&D textbook.
2. **Why Cap $nlags$?** If we ask statsmodels to compute ACF for lags larger than $n/2$, it raises an error. The code dynamically caps `nlags` at $n/2 - 1$ to prevent the GUI from crashing on short datasets.
3. **Downstream Resets:** Fitting a new model automatically invalidates the validation state: `state.validation_passed = False`. This forces the user to run the diagnostics on Tab 5 before they can forecast with the new model.
4. **Ignoring Convergence Failures:** During grid search, some models (especially overparameterized ones) fail to converge, throwing `ConvergenceWarning` or `MaxiterExceeded` exceptions. The search catches these exceptions and skips the invalid models silently to avoid interrupting the search process.

---

## 7. Likely Defense Questions & Answers

### Q1: What is the PACF and how does it differ from the ACF?
**Answer:** The ACF measures the correlation between $X_t$ and $X_{t-k}$ including the effects of all intermediate lags. The PACF measures the correlation between $X_t$ and $X_{t-k}$ *after* removing the linear dependence of the intervening variables $X_{t-1}, \dots, X_{t-k+1}$. For example, in an AR(1) process, $X_t$ and $X_{t-2}$ are correlated because $X_t$ depends on $X_{t-1}$ which depends on $X_{t-2}$. The ACF at lag 2 is non-zero, but the PACF at lag 2 is zero because the influence of $X_{t-1}$ is removed.

### Q2: Why does the PACF cut off after lag $p$ for an $\text{AR}(p)$ process?
**Answer:** An $\text{AR}(p)$ process is defined by $X_t = \phi_1 X_{t-1} + \dots + \phi_p X_{t-p} + Z_t$, where $Z_t$ is uncorrelated with past values. For any lag $k > p$, the best linear predictor of $X_t$ based on the preceding $k$ values only assigns non-zero coefficients to the first $p$ variables. The coefficient for the lag $t-k$ is exactly zero, which is the definition of the PACF at lag $k$.

### Q3: Why does the ACF cut off after lag $q$ for an $\text{MA}(q)$ process?
**Answer:** An $\text{MA}(q)$ process is defined as $X_t = Z_t + \theta_1 Z_{t-1} + \dots + \theta_q Z_{t-q}$. The covariance between $X_t$ and $X_{t-k}$ is $\text{E}[(Z_t + \dots + \theta_q Z_{t-q})(Z_{t-k} + \dots + \theta_q Z_{t-k-q})]$. Since the white noise terms $\{Z_t\}$ are independent, any product $Z_{t-i}Z_{t-k-j}$ has an expectation of zero unless their indices match. For $k > q$, there is no overlap in the white noise components of $X_t$ and $X_{t-k}$, so the covariance is exactly zero.

### Q4: What is the difference between AIC, BIC, and AICc, and why is AICc preferred here?
**Answer:** 
- **AIC** penalizes parameters by $2k$.
- **BIC** penalizes parameters by $k\ln(n)$, which is more severe for $n \ge 8$, preferring simpler models.
- **AICc** includes an extra correction term: $\frac{2k(k+1)}{n-k-1}$.
AICc is preferred because AIC has a tendency to choose models with too many parameters (overfitting) when the sample size $n$ is small. AICc corrects for this bias. As $n$ grows, the correction term decays to zero, making AICc asymptotically equivalent to AIC.

### Q5: What estimation algorithm is statsmodels using under the hood when you specify `method='innovations_mle'`?
**Answer:** It uses the **Innovations Algorithm** to recursively project the series onto its past values, yielding the one-step-ahead prediction errors (innovations) and their variances. These are plugged directly into the Gaussian log-likelihood function, which is then maximized using non-linear optimization.

### Q6: Why do we run the grid search in a separate thread (QThread)?
**Answer:** The grid search fits dozens or hundreds of models, which can take several seconds or minutes. If we ran this directly in the main thread, it would block the PySide6 event loop, causing the GUI to freeze, show "(Not Responding)", and prevent the user from clicking other buttons. A `QThread` runs the search in the background, communicating progress updates via Qt signals while keeping the UI responsive.

### Q7: Why do we cap the number of lags in the ACF/PACF plots at $n/2 - 1$?
**Answer:** Autocorrelations at high lags are computed from very few data pairs (e.g. $\hat{\rho}(n-1)$ is computed from just one pair), leading to high variance and numerical instability. Statsmodels enforces a restriction that $nlags < n/2$ to ensure numerical reliability. The cap prevents runtime crashes on short datasets.

### Q8: What does the "dual-suggestion" algorithm do when a spectral peak matches a harmonic of the seasonal period?
**Answer:** If the seasonal period is $s$ (e.g., 12 for monthly data), the seasonal component will manifest peaks not only at the fundamental frequency $2\pi/12$, but also at its integer harmonics (e.g., $2\pi/6, 2\pi/4, 2\pi/3$). The algorithm checks if any significant spectral peak is within $15\%$ of these harmonic periods. If found, it correctly suggests a seasonal SARIMA model rather than a non-seasonal model.

---

## 8. Quick Reference — Key Formulas

| Concept | Formula |
|---|---|
| **ACF** | $\rho(k) = \frac{\gamma(k)}{\gamma(0)}$ |
| **PACF (Yule-Walker)** | $\phi_{kk} = \frac{\rho(k) - \sum_{j=1}^{k-1} \phi_{k-1, j} \rho(k-j)}{1 - \sum_{j=1}^{k-1} \phi_{k-1, j} \rho(j)}$ |
| **White Noise Bounds** | $\pm \frac{1.96}{\sqrt{n}}$ |
| **Corrected AIC (AICc)** | $\text{AICc} = -2\ln(L) + 2k + \frac{2k(k+1)}{n - k - 1}$ |
| **BIC** | $\text{BIC} = -2\ln(L) + k\ln(n)$ |
| **Gaussian Log-Likelihood** | $\ln L = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2}\sum_{t=1}^{n}\ln(v_{t-1}) - \frac{1}{2\sigma^2}\sum_{t=1}^{n}\frac{(X_t - \hat{X}_t)^2}{v_{t-1}}$ |
| **Spectral Peak Tolerance** | $|P_{\text{peak}} - s| \le 0.15 \times s$ |
