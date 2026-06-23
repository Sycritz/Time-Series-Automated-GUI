# Axis 5 — The Probabilist: Forecasting & Uncertainty Quantification
### Defense Preparation Guide · Tab 6

> **Source file:** [`src/axis5_forecasting.py`](../src/axis5_forecasting.py)  
> **Audience:** Student team preparing for project defense  
> **B&D Reference:** Brockwell & Davis — *Introduction to Time Series and Forecasting*, Chapters 5 (Forecasting) and 9.5 (Forecasting ARIMA Models)

---

## Table of Contents

1. [Role & Purpose](#1-role--purpose)
2. [Theory: The B&D Foundation](#2-theory-the-bd-foundation)
   - 2.1 [Best Linear Predictor & Orthogonal Projection](#21-best-linear-predictor--orthogonal-projection)
   - 2.2 [The $\text{MA}(\infty)$ Representation & $\psi$-Weights](#22-the-mainfty-representation--psi-weights)
   - 2.3 [Prediction Error Variance](#23-prediction-error-variance)
   - 2.4 [Recursive $\psi$-Weight Computation](#24-recursive-psi-weight-computation)
   - 2.5 [The Inverse Transformation Chain](#25-the-inverse-transformation-chain)
   - 2.6 [Spectral Interpretation of Forecast Behavior](#26-spectral-interpretation-of-forecast-behavior)
3. [Implementation Walkthrough](#3-implementation-walkthrough)
   - 3.1 [`compute_psi_weights`](#31-compute_psi_weights)
   - 3.2 [`get_model_ar_ma_coeffs`](#32-get_model_ar_ma_coeffs)
   - 3.3 [`generate_forecasts`](#33-generate_forecasts)
   - 3.4 [Back-Transformation Functions](#34-back-transformation-functions)
   - 3.5 [`generate_spectral_insight`](#35-generate_spectral_insight)
4. [Hand-Implemented vs Library Table](#4-hand-implemented-vs-library-table)
5. [Data Flow](#5-data-flow)
6. [Design Decisions & Gotchas](#6-design-decisions--gotchas)
7. [Likely Defense Questions & Answers](#7-likely-defense-questions--answers)
8. [Quick Reference — Key Formulas](#8-quick-reference--key-formulas)

---

## 1. Role & Purpose

Axis 5 is the **Forecasting Engine** of the application. Once a model has been estimated (Axis 3) and passed all residual validation checks (Axis 4), Axis 5 takes over to project the time series into the future. 

Its core role is to:
- Compute point forecasts on the stationary scale using the fitted model coefficients.
- Recursively calculate the theoretical $\psi$-weights of the model to determine the forecast error variances.
- Construct nominal $50\%$, $80\%$, and $95\%$ prediction intervals (PI) to display as an expanding "fan chart."
- Systematically undo the differencing and Box-Cox transformations in reverse order to return forecasts and intervals to the original scale.
- Provide a "Spectral Insight" panel connecting the qualitative shape of the forecasts (mean reversion, trends, cycles) to the model's frequency-domain properties.

---

## 2. Theory: The B&D Foundation

### 2.1 Best Linear Predictor & Orthogonal Projection

Let $\{X_t\}$ be a stationary time series. We want to predict $X_{n+h}$ (where $h \ge 1$ is the forecast horizon) using the observed data $\{X_1, \dots, X_n\}$. The best linear predictor is denoted:

$$\hat{X}_{n+h|n} = P_{\text{sp}\{X_1, \dots, X_n\}} X_{n+h}$$

where $P$ is the orthogonal projection operator onto the closed linear span of the observations. This projection minimizes the mean squared error (MSE):

$$\text{E}\left(X_{n+h} - \hat{X}_{n+h|n}\right)^2$$

For an invertible ARMA process, the predictor satisfies the model difference equations, replacing future values $X_{t+k}$ with their forecasts $\hat{X}_{t+k|t}$ and future noise terms $Z_{t+k}$ with their expectation ($0$).

---

### 2.2 The $\text{MA}(\infty)$ Representation & $\psi$-Weights

Any causal ARMA$(p, q)$ process can be expressed as an infinite moving average process:

$$X_t = \sum_{j=0}^{\infty} \psi_j Z_{t-j} \quad \text{where } \psi_0 = 1, \; \sum_{j=0}^{\infty} |\psi_j| < \infty$$

The coefficients $\{\psi_j\}$ are the **$\psi$-weights**. They represent the impulse response function of the process — how a one-unit shock in the white noise $Z_{t-j}$ propagates through time to affect the value of $X_t$.

---

### 2.3 Prediction Error Variance

Using the $\text{MA}(\infty)$ representation, we write the future value $X_{n+h}$ as:

$$X_{n+h} = \sum_{j=0}^{\infty} \psi_j Z_{n+h-j} = \sum_{j=0}^{h-1} \psi_j Z_{n+h-j} + \sum_{j=h}^{\infty} \psi_j Z_{n+h-j}$$

The best linear predictor $\hat{X}_{n+h|n}$ is the projection of this series. Since the shocks occurring after time $n$ ($Z_{n+1}, \dots, Z_{n+h}$) are uncorrelated with $\{X_1, \dots, X_n\}$, their projection is zero. The shocks up to time $n$ are known, so they project onto themselves. Thus:

$$\hat{X}_{n+h|n} = \sum_{j=h}^{\infty} \psi_j Z_{n+h-j}$$

Subtracting these two equations yields the **$h$-step prediction error**:

$$X_{n+h} - \hat{X}_{n+h|n} = \sum_{j=0}^{h-1} \psi_j Z_{n+h-j}$$

Since the white noise terms $\{Z_t\}$ are independent with variance $\sigma^2$, the covariance terms vanish, and the **prediction error variance** $\sigma^2(h)$ is:

$$\sigma^2(h) = \text{Var}\left(X_{n+h} - \hat{X}_{n+h|n}\right) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$$

---

### 2.4 Recursive $\psi$-Weight Computation

To calculate $\sigma^2(h)$ in code, we must compute the $\psi$-weights. For an $\text{ARMA}(p, q)$ process, we solve the polynomial identity:

$$\phi(z)\psi(z) = \theta(z) \implies \left(1 - \sum_{k=1}^{p} \phi_k z^k\right) \left(\sum_{j=0}^{\infty} \psi_j z^j\right) = 1 + \sum_{j=1}^{q} \theta_j z^j$$

Equating coefficients of $z^j$ yields the **$\psi$-weight recursion** (B&D Equation 5.1.4):

$$\psi_0 = 1$$

$$\psi_j = \theta_j + \sum_{k=1}^{\min(p, j)} \phi_k \psi_{j-k}, \quad j \ge 1$$

with the convention that $\theta_j = 0$ for $j > q$.

---

### 2.5 The Inverse Transformation Chain

The model is fitted to transformed stationary data $Y_t$. The forecasts $\hat{Y}_{n+h|n}$ and their prediction intervals must be mapped back to the original scale of $X_t$. The transformations applied in Tab 2 are recorded in `state.transformations`. To back-transform, we replay this chain in **reverse order**:

```
Transformed scale (Y_t) ──► Undo Seasonal Diff ──► Undo Diff ──► Undo Box-Cox ──► Original scale (X_t)
```

#### Undoing Ordinary Differencing ($d$)
Differencing is undone by cumulative summation. For $d=1$:
$$\hat{X}_{n+h} = X_n + \sum_{j=1}^{h} \hat{Y}_{n+j}$$
For $d > 1$, we compute this recursively using the last available values of the intermediate differenced series as anchors.

#### Undoing Seasonal Differencing ($D$) with period $s$
For seasonal differencing with period $s$, the value at step $h$ depends on the value $s$ steps prior:
$$\hat{X}_{n+h} = \hat{X}_{n+h-s} + \hat{Y}_{n+h}$$
- For $h \le s$, we anchor to actual historical data points: $X_{n - s + h}$.
- For $h > s$, we anchor to previously reconstructed forecasts: $\hat{X}_{n + h - s}$.

#### Undoing Box-Cox ($\lambda$)
We invert the power transformation equation:
$$\hat{X}_t = \begin{cases}
\left(\lambda \hat{Y}_t + 1\right)^{1/\lambda} & \lambda \neq 0 \\
\exp(\hat{Y}_t) & \lambda = 0
\end{cases}$$
This non-linear transformation is applied directly to the point forecasts and all prediction interval bounds.

---

### 2.6 Spectral Interpretation of Forecast Behavior

The qualitative shape of the forecast function is mathematically tied to the theoretical spectral density $f_X(\lambda)$ of the model:

1. **Stationary ARMA ($d=0, D=0$):**
   - The spectrum $f_X(\lambda)$ is continuous and bounded.
   - The forecast function decays geometrically towards the process mean.
   - The prediction error variance $\sigma^2(h)$ converges to the process variance $\gamma(0)$ as $h \to \infty$, and the prediction intervals stabilize.
2. **Integrated ARIMA ($d \ge 1$):**
   - The spectrum has an infinite singularity at frequency $\lambda = 0$ ($f(0) = \infty$).
   - The forecast function follows a polynomial trend of degree $d-1$ (e.g. flat line for $d=1$, linear trend for $d=2$).
   - The prediction intervals widen indefinitely, flaring out at a rate of $\sqrt{h}$ for $d=1$.
3. **Seasonal SARIMA ($D \ge 1$ or seasonal AR coefficients):**
   - The spectrum shows sharp peaks at the seasonal frequency $\lambda_s = 2\pi/s$ and its harmonics.
   - The forecast function displays persistent, quasi-periodic oscillations with period $s$.

---

## 3. Implementation Walkthrough

### 3.1 `compute_psi_weights`

Recursively computes $\psi$-weights using the AR and MA coefficients:

```python
def compute_psi_weights(ar_coeffs: list, ma_coeffs: list, h: int) -> np.ndarray:
    if h <= 0:
        return np.array([], dtype=float)
        
    psi = np.zeros(h, dtype=float)
    psi[0] = 1.0
    
    p = len(ar_coeffs)
    q = len(ma_coeffs)
    
    for j in range(1, h):
        theta_j = ma_coeffs[j - 1] if j - 1 < q else 0.0
        val = theta_j
        for k in range(1, min(p + 1, j + 1)):
            val += ar_coeffs[k - 1] * psi[j - k]
        psi[j] = val
        
    return psi
```

### 3.2 `get_model_ar_ma_coeffs`

Extracts combined polynomials from statsmodels, adjusting for sign conventions (in statsmodels, the AR polynomial is $1 - \phi_1 B - \dots$, so $\phi_k$ is the negative of the coefficient):

```python
poly_ar = getattr(res, 'polynomial_reduced_ar', None) # holds expanded AR * seasonal AR
poly_ma = getattr(res, 'polynomial_reduced_ma', None) # holds expanded MA * seasonal MA

if poly_ar is not None:
    ar_coeffs = [-c for c in poly_ar[1:]]
else:
    ar_coeffs = []
    
if poly_ma is not None:
    ma_coeffs = list(poly_ma[1:])
else:
    ma_coeffs = []
```

### 3.3 `generate_forecasts`

Computes the prediction intervals on the stationary scale using standard normal quantiles:

```python
point_forecasts = model_result.forecast(steps=h)
ar_coeffs, ma_coeffs = get_model_ar_ma_coeffs(model_result)
psi = compute_psi_weights(ar_coeffs, ma_coeffs, h)
sigma2 = getattr(model_result, 'sigma2', 1.0)

# Prediction standard error: standard deviation accumulates
psi_sq_cumsum = np.cumsum(psi**2)
prediction_std = np.sqrt(sigma2 * psi_sq_cumsum)

# Quantiles
z_50 = norm.ppf(0.75)    # ~0.6745
z_80 = norm.ppf(0.90)    # ~1.2816
z_95 = norm.ppf(0.975)   # ~1.9600

# CI = Point +/- z * std
# ...
```

### 3.4 Back-Transformation Functions

To undo differencing, the code replays the forward transformations on the historical series first to obtain the correct intermediate histories for anchoring. Then it walks the transformation log in reverse:

```python
# Undo Box-Cox
def undo_box_cox(x: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam) < 1e-7:
        return np.exp(x)
    else:
        val = lam * x + 1.0
        val = np.maximum(val, 1e-9)  # Avoid negative bases
        return val ** (1.0 / lam)

# Undo ordinary diff
def undo_diff(forecast_vals: np.ndarray, prev_series: pd.Series, d: int) -> np.ndarray:
    # Reconstructs intermediate histories, then performs cumulative summation:
    reconstructed = np.zeros(len(current_forecast))
    for h in range(len(current_forecast)):
        prev = last_val if h == 0 else reconstructed[h - 1]
        reconstructed[h] = prev + current_forecast[h]
    # Loops for each d...
```

### 3.5 `generate_spectral_insight`

Analyzes the fitted model structure to output a detailed explanation for the user, connecting the forecast shape to the spectral density:

```python
# Check integration
if total_d > 0:
    lines.append(f"The model requires differencing of total order {total_d}.")
    lines.append(f"Theoretical spectrum has an infinite singularity at frequency lambda = 0.")
# Check seasonality
if is_seasonal and s > 1:
    lines.append(f"Spectral peaks at seasonal frequency 2*pi/{s}.")
# Check complex roots for quasi-periodic cycles
if p >= 2:
    roots = np.roots(poly)
    # phase = angle, period = 2*pi/phase
```

---

## 4. Hand-Implemented vs Library Table

| Component | Hand-Implemented | Library | Why |
|---|---|---|---|
| **$\psi$-weights** | Recursive computation loop (Equation 17). | — | Standard libraries do not expose raw $\psi$-weights directly. |
| **Prediction Intervals** | Multi-level interval calculations ($50\%, 80\%, 95\%$) using normal quantiles. | — | Enforces correct math and custom nominal coverage levels. |
| **Inverse Box-Cox** | Piecewise formula with safety clamping. | — | Translates Box-Cox inversion theory into code. |
| **Inverse Differencing** | Multi-step cumulative summation and historical anchoring. | — | Libraries like statsmodels only back-transform their internal forecasts, not custom intervals. |
| **History Rebuilder** | Re-applying the forward transformations step-by-step. | — | Essential for identifying correct anchor values for differencing. |
| **Spectral Insight** | Mathematical analysis of AR roots, integration orders, and peak periods. | `numpy.roots` for AR roots | Explains time-domain forecast features through frequency-domain properties. |
| **Point Forecast** | — | `model_result.forecast` | Leverage the fitted statsmodels state-space engine for prediction. |

---

## 5. Data Flow

```
state.fitted_model & state.transformations (from Tab 4 & 2)
  │
  ▼
[Tab 6: Forecasting]
  ├── generate_forecasts() ──► stationary forecasts & PIs
  │                              │
  ├── back_transform()  ◄────────┴─── (reads state.original_series)
  │      │
  │      ├─► undo_seasonal_diff()
  │      ├─► undo_diff()
  │      └─► undo_box_cox()
  │
  ├── generate_spectral_insight() ──► Generates explanatory Markdown
  │
  ▼
Interactive Fan Chart (original or transformed scale toggle)
Exportable CSV Forecast Table
```

---

## 6. Design Decisions & Gotchas

1. **The Reversal Ordering:** Differencing is applied *after* Box-Cox during preprocessing. To back-transform, we must invert them in **reverse order**: first cumulative summation to undo differencing, and *then* the power transform inversion. Doing this out of order breaks mathematical correctness.
2. **Box-Cox Base Guard:** The inverse Box-Cox formula contains a term $(\lambda Y_t + 1)^{1/\lambda}$. If $Y_t$ is highly negative, the base $\lambda Y_t + 1$ can become negative. Raising a negative number to a fractional power yields a complex number (crashing the float array). We guard against this by clipping the base to a small positive value: `np.maximum(val, 1e-9)`.
3. **Seasonal Anchoring ($h \ge s$):** When undoing seasonal differencing of period $s$ for horizon $h$, if $h < s$, we can anchor to the actual historical data point $X_{n - s + h}$. However, if we project past one season ($h \ge s$), we must anchor to the *previously reconstructed forecast* $\hat{X}_{n + h - s}$.
4. **Sign Convention in statsmodels:** In standard equations, the AR polynomial is written $1 - \phi_1 B - \phi_2 B^2$. Statsmodels fits the parameters directly, meaning `polynomial_reduced_ar` is represented as $[1.0, -\phi_1, -\phi_2]$. We must negate the coefficients to obtain the standard positive values of $\phi_j$ used in the $\psi$-weight recursion.

---

## 7. Likely Defense Questions & Answers

### Q1: Derive the formula for the $h$-step prediction error variance.
**Answer:** The future value is $X_{n+h} = \sum_{j=0}^{h-1} \psi_j Z_{n+h-j} + \sum_{j=h}^{\infty} \psi_j Z_{n+h-j}$. The best linear predictor based on information up to time $n$ is the projection $\hat{X}_{n+h|n} = \sum_{j=h}^{\infty} \psi_j Z_{n+h-j}$ since future shocks project to zero. The prediction error is the difference: $X_{n+h} - \hat{X}_{n+h|n} = \sum_{j=0}^{h-1} \psi_j Z_{n+h-j}$. Since $\{Z_t\}$ are independent with variance $\sigma^2$, the variance of this sum is the sum of the variances: $\sigma^2(h) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$.

### Q2: Write down the recursion relation for the $\psi$-weights of an $\text{ARMA}(p,q)$ process.
**Answer:** The relation is:
- $\psi_0 = 1$
- $\psi_j = \theta_j + \sum_{k=1}^{\min(p, j)} \phi_k \psi_{j-k}$ for $j \ge 1$, where $\theta_j = 0$ for $j > q$.
This is obtained by equating coefficients in the polynomial equation $\phi(z)\psi(z) = \theta(z)$.

### Q3: Why do prediction intervals for stationary models stabilize as $h \to \infty$, while those for integrated models grow indefinitely?
**Answer:** 
- For a **stationary model**, the $\psi$-weights decay geometrically to zero because the roots of the AR polynomial lie outside the unit circle. The sum $\sum_{j=0}^{h-1} \psi_j^2$ converges to a finite constant, meaning the prediction error variance converges to the process variance $\gamma(0)$.
- For an **integrated model** (like a random walk where $\psi_j = 1$ for all $j$), the $\psi$-weights do not decay. The sum $\sum_{j=0}^{h-1} \psi_j^2$ grows to infinity as $h \to \infty$, meaning our uncertainty about the future grows without bound.

### Q4: Why must we back-transform both the point forecast and the prediction interval bounds, rather than just back-transforming the point forecast and calculating the interval around it?
**Answer:** Because the Box-Cox transformation is **non-linear**. If we back-transformed the prediction standard deviation and added it to the back-transformed point forecast, we would get symmetrical intervals. However, a non-linear power transform scales values differently at high vs. low levels, meaning the intervals on the original scale should be **asymmetric** (reflecting log-normality or power-skewness). Reverting the upper and lower bounds individually preserves this skewness.

### Q5: How do you undo a seasonal difference of period $s$ for a forecast horizon $h$ that is larger than $s$?
**Answer:** For $h < s$, we add the forecast to the historical value observed $s$ steps before the end of the series: $\hat{X}_{n+h} = X_{n-s+h} + \hat{Y}_{n+h}$. For $h \ge s$, the value $s$ steps prior is itself a forecast. We anchor to that reconstructed forecast: $\hat{X}_{n+h} = \hat{X}_{n+h-s} + \hat{Y}_{n+h}$.

### Q6: What does the "Spectral Insight" panel explain when a model has complex roots in its AR polynomial?
**Answer:** Complex conjugate roots in the AR polynomial indicate a cyclical tendency in the time domain. This manifests as a peak in the theoretical spectral density at the phase angle frequency $\lambda = \text{angle}(\text{root})$. In the forecast function, this complex conjugate pair causes the forecasts to exhibit decaying sinusoidal oscillations with a period of $T = 2\pi / \lambda$.

### Q7: Why do we clip the base of the inverse Box-Cox transform to $10^{-9}$?
**Answer:** The inverse Box-Cox formula is $(\lambda Y_t + 1)^{1/\lambda}$. If $Y_t$ is negative and $\lambda$ is positive (or vice-versa), the term $\lambda Y_t + 1$ can fall below zero. Raising a negative number to a fractional power (such as $1/\lambda$) is mathematically undefined in real numbers (yielding complex numbers), which would crash the float arrays in the plot and tables. We clamp it to a tiny positive number ($10^{-9}$) to ensure mathematical stability.

### Q8: How are the standard normal quantiles $z_{\alpha/2}$ derived for the $50\%$, $80\%$, and $95\%$ prediction intervals?
**Answer:** Under the assumption of Gaussian innovations, the prediction errors are normally distributed: $e(h) \sim \mathcal{N}(0, \sigma^2(h))$. The symmetric intervals are:
- **$50\%$ interval:** Covers the middle $50\%$, leaving $25\%$ in each tail. The quantile is $z_{0.75} \approx 0.6745$.
- **$80\%$ interval:** Leaves $10\%$ in each tail. The quantile is $z_{0.90} \approx 1.2816$.
- **$95\%$ interval:** Leaves $2.5\%$ in each tail. The quantile is $z_{0.975} \approx 1.9600$.

---

## 8. Quick Reference — Key Formulas

| Concept | Formula |
|---|---|
| **Best Linear Predictor** | $\hat{X}_{n+h|n} = P_{\text{sp}\{X_1, \dots, X_n\}} X_{n+h}$ |
| **$\psi$-Weight Recursion** | $\psi_j = \theta_j + \sum_{k=1}^{\min(p, j)} \phi_k \psi_{j-k}$ |
| **Prediction Error Variance** | $\sigma^2(h) = \sigma^2 \sum_{j=0}^{h-1} \psi_j^2$ |
| **Normal Prediction Interval** | $\hat{X}_{n+h|n} \pm z_{\alpha/2} \sigma(h)$ |
| **Inverse Box-Cox** | $X = (\lambda Y + 1)^{1/\lambda}$ |
| **Inverse Ordinary Diff ($d=1$)** | $\hat{X}_{n+h} = X_n + \sum_{j=1}^{h} \hat{Y}_{n+j}$ |
| **Inverse Seasonal Diff ($D=1$)** | $\hat{X}_{n+h} = \hat{X}_{n+h-s} + \hat{Y}_{n+h}$ |
