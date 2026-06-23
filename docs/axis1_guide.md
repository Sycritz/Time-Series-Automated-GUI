# Axis 1 — The Data Prospector: Pre-processing & Stationarity
**Defense Prep Guide** · Time-Series Automated GUI · University Group Project

> [!NOTE]
> This guide is written for the whole team, not just the Axis 1 owner.
> Every section maps directly to something a panel examiner is likely to probe.

---

## 1. Role & Purpose

Axis 1 is the **entry point** of the entire pipeline. It is responsible for loading raw data, diagnosing and repairing missing values, testing whether the series is stationary, and applying the transformations (Box-Cox, differencing) needed to make it stationary. Without a clean, stationary series, every downstream model (ARIMA, spectral analysis, forecasting) is built on a lie — that is why this axis resets all downstream shared state whenever its core output changes.

---

## 2. Theory: The Box & Jenkins Foundation

### 2.1 Augmented Dickey-Fuller (ADF) Test

The ADF test answers: *"Is there a unit root?"*

**Regression estimated:**

$$
\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{j=1}^{p} \delta_j \Delta X_{t-j} + \varepsilon_t
$$

- $\alpha$ — constant (drift)
- $\beta t$ — deterministic time trend (optional; statsmodels defaults include it)
- $\gamma X_{t-1}$ — this is the key coefficient; a unit root means $\gamma = 0$
- The lagged differences $\Delta X_{t-j}$ augment the basic DF test to absorb serial correlation in $\varepsilon_t$

**Hypotheses:**

| | Statement |
|---|---|
| $H_0$ | $\gamma = 0$ — unit root present, series is **non-stationary** |
| $H_1$ | $\gamma < 0$ — no unit root, series is **stationary** |

**Decision rule:** reject $H_0$ when $p < 0.05$. Because the ADF statistic has a non-standard distribution (MacKinnon, 1994), we compare against tabulated critical values at 1%, 5%, and 10% levels rather than the normal distribution. The implementation returns all three so the user can see how close the borderline cases are.

---

### 2.2 Box-Cox Transformation

Box-Cox is a family of power transformations that **stabilises variance** across time.

$$
Y_t =
\begin{cases}
\dfrac{X_t^{\lambda} - 1}{\lambda} & \lambda \neq 0 \\[6pt]
\ln(X_t) & \lambda = 0
\end{cases}
$$

**Auto-optimise lambda** uses the **profile log-likelihood**. For a given $\lambda$, transform the series, then evaluate:

$$
\ell(\lambda) = -\frac{n}{2} \ln\!\left(\hat{\sigma}^2_\lambda\right) + (\lambda - 1)\sum_{t=1}^{n} \ln X_t
$$

where $\hat{\sigma}^2_\lambda$ is the variance of the transformed series. `scipy.stats.boxcox` maximises this over $\lambda$ using Brent's method — when `lam=None` is passed, that is exactly what happens.

**Why strictly positive?** The power $X_t^\lambda$ is undefined (or complex-valued) for $X_t \leq 0$ when $\lambda$ is non-integer. The log branch ($\lambda=0$) also requires $X_t > 0$. There is no mathematically valid generalisation of Box-Cox to non-positive data.

---

### 2.3 Differencing

**Regular differencing** removes deterministic and stochastic trends by computing first differences:

$$
\nabla X_t = X_t - X_{t-1} = (1 - B) X_t
$$

A linear trend $X_t = a + bt + \varepsilon_t$ becomes $\nabla X_t = b + \varepsilon_t$ — the trend term vanishes, leaving a constant plus noise. Differencing order $d$ means applying the operator $d$ times: $(1 - B)^d X_t$.

**Seasonal differencing** removes periodic structure with period $s$:

$$
\nabla_s X_t = X_t - X_{t-s} = (1 - B^s) X_t
$$

Seasonal differencing order $D$ means applying $(1 - B^s)^D X_t$. In the SARIMA notation $(p,d,q)(P,D,Q)_s$, Axis 1 determines $d$ and $D$.

---

### 2.4 Frequency Response of the Differencing Filter

A linear filter's effect can be characterised in the frequency domain by its **squared gain function** $G(\omega)$ — how much power at each frequency $\omega \in [0, \pi]$ passes through vs. is attenuated.

For the regular differencing operator $(1 - e^{-i\omega})$:

$$
|1 - e^{-i\omega}|^2 = \left(2\sin\frac{\omega}{2}\right)^2
$$

For $d$ regular diffs and $D$ seasonal diffs with period $s$, the combined squared gain is:

$$
G(\omega) = \left(2\left|\sin\frac{\omega}{2}\right|\right)^{2d} \cdot \left(2\left|\sin\frac{s\omega}{2}\right|\right)^{2D}
$$

At $\omega = 0$ (DC / long-run), $G(0) = 0$ — meaning differencing **completely suppresses low frequencies** (trends). High frequencies ($\omega \approx \pi$) are passed with gain 2 or higher. This is the definition of a **high-pass filter**. The plot is a bonus feature that gives the Axis 2 spectral team an intuition for what has been done to the series before they see it.

---

## 3. Implementation Walkthrough

### 3.1 `load_csv(path)` → `pd.DataFrame`

```python
df = pd.read_csv(path, sep=None, engine='python')
```

- `sep=None` tells pandas to **auto-detect the delimiter** using Python's `csv.Sniffer`. This handles comma-separated, semicolon-separated, and tab-delimited files transparently.
- `engine='python'` is required when `sep=None`; the C engine does not support delimiter sniffing.
- The try/except wraps the entire call so the GUI receives a clean `ValueError` with a human-readable message rather than a raw pandas traceback.

**What to say:** *"We chose `sep=None` so users don't have to pre-process their files or select a delimiter manually — it's one fewer friction point at data ingestion."*

---

### 3.2 `handle_missing(series, method)` → `(series, pct_missing)`

```python
n_missing = series.isna().sum()
pct_missing = (n_missing / len(series)) * 100.0
```

Three methods are supported:

| Method | Key line | Notes |
|---|---|---|
| Forward Fill | `imputed.ffill().bfill()` | Propagates the last observed value forward; `bfill()` catches leading NaNs |
| Linear Interpolation | `imputed.interpolate(method='linear').ffill().bfill()` | Linearly interpolates between neighbours; fallback handles edges |
| Mean Imputation | `imputed.fillna(mean_val)` | Replaces every NaN with the global mean; defaults to 0 if all-NaN |

**The `bfill()` fallback:** `ffill()` alone cannot fill NaNs at the very start of the series (there is no preceding value). The chained `.bfill()` fills those from the right, ensuring no NaN leaks through. This is an explicit design decision, not a pandas default.

**The 5% warning:** The function returns `pct_missing`; the GUI layer compares it to 5 and shows a warning banner. Imputing more than 5% of a time series risks distorting autocorrelation structure, which directly corrupts ADF results and ACF/PACF plots downstream.

---

### 3.3 `run_adf_test(series)` → `dict`

```python
res = adfuller(clean_series)
verdict = 'Stationary' if p_value < 0.05 else 'Non-Stationary'
```

- `series.dropna()` is called first to prevent statsmodels raising on NaN — important if the user runs the test before imputation.
- The 10-point minimum guard (`len(clean_series) < 10`) prevents degenerate results from micro-datasets.
- The returned dict unpacks `res[0]` (ADF statistic), `res[1]` (p-value), and `res[4]` (critical values dict) explicitly, keeping the GUI layer decoupled from statsmodels' tuple format.
- `critical_vals` is a dict keyed `"1%"`, `"5%"`, `"10%"` — all cast to `float` for JSON-safe serialisation.

**What to say:** *"We delegate the computation to statsmodels because adfuller implements lag selection (AIC by default) and MacKinnon critical values that would be nontrivial to replicate correctly. Our job is structuring the output for the GUI."*

---

### 3.4 `apply_box_cox(series, lam)` → `(series, lam)`

```python
if (valid_values <= 0).any():
    raise ValueError("Box-Cox transformation requires strictly positive data...")

if lam is None:
    transformed_valid, opt_lam = boxcox(valid_values)
else:
    transformed_valid = boxcox(valid_values, lmbda=lam)
```

- Only non-NaN values (`non_nan_mask`) are passed to `scipy.stats.boxcox`. NaN positions are preserved in the output using `np.full(..., np.nan)` with index-aligned placement — so the series length is unchanged.
- When `lam=None`, scipy returns the optimised lambda from profile log-likelihood maximisation.
- When `lam` is user-specified, the fixed-lambda branch is used. Both paths cast `opt_lam` to `float` for consistent downstream storage.
- The returned `(series, lambda)` tuple lets the state layer record the exact lambda used, which is needed for the inverse transform in forecasting.

---

### 3.5 `apply_differencing(series, d, D, s)` → `pd.Series`

```python
for _ in range(d):
    res = res.diff()
for _ in range(D):
    res = res.diff(periods=s)
```

- Regular diffs use `diff()` (period=1 default); each call is $\nabla$.
- Seasonal diffs use `diff(periods=s)`; each call is $\nabla_s$.
- The loops compose them: regular differencing is applied first, then seasonal. This matches the SARIMA convention where the full filter is $(1-B)^d (1-B^s)^D$.
- Each `diff()` introduces one leading NaN; after $d$ regular and $D$ seasonal diffs, the first $d + D \cdot s$ values will be NaN. Downstream axes must handle this truncation.

---

### 3.6 `compute_frequency_response(d, D, s, n_points=512)` → `(freqs, gain)`

```python
omega = np.linspace(0, np.pi, n_points)
term1 = (2.0 * np.abs(np.sin(omega / 2.0))) ** (2 * d)
term2 = (2.0 * np.abs(np.sin(s * omega / 2.0))) ** (2 * D)
gain = term1 * term2
```

This is **entirely hand-implemented** — no signal-processing library is used. The formula is derived analytically from the z-transform of the differencing operator and evaluated numerically on a grid of 512 frequency points between 0 and $\pi$ rad/sample. The `np.abs` inside the `sin` is technically redundant for $\omega \in [0, \pi]$ but is defensive against edge cases.

---

## 4. Hand-Implemented vs. Library Table

| Component | Hand-Implemented | Library | Why |
|---|---|---|---|
| ADF test statistics & critical values | ✗ | `statsmodels.tsa.stattools.adfuller` | Non-standard distribution; MacKinnon tables are complex to replicate accurately |
| Box-Cox transform + profile log-likelihood | ✗ | `scipy.stats.boxcox` | Numerically stable Brent optimisation; log-likelihood derivation is non-trivial |
| Differencing operator loop | ✓ | — | Trivially `pd.Series.diff()`; explicit loop makes order and structure transparent |
| Frequency response squared gain | ✓ | — | Closed-form formula from z-transform; numpy vectorisation is sufficient; adds no new dependency |
| Missing value imputation | ✓ (logic) | `pandas` (mechanics) | Imputation strategies are custom-selected; pandas provides the fast fill primitives |
| CSV delimiter sniffing | ✗ | `pandas` + Python `csv.Sniffer` | Well-tested across edge cases; reimplementing would be fragile |

---

## 5. Data Flow

```
[User uploads file]
        │
        ▼
  load_csv(path)
        │ pd.DataFrame
        ▼
  [User selects column]
        │ pd.Series (raw)
        ▼
  handle_missing(series, method)
        │ pd.Series (clean), pct_missing
        ├──────────────────────────────► state.original_series   ← cascades: resets ALL downstream state
        ▼
  run_adf_test(series)   ← informational; does NOT mutate state
        │ {adf_stat, p_value, critical_values, verdict}
        ▼
  [User applies transforms]
        │
        ├── apply_box_cox(series, lam)  ──► state.transformations.append({"type":"boxcox","lambda":λ})
        │
        └── apply_differencing(s,d,D,s) ─► state.transformations.append({"type":"diff","d":d})
                                           state.transformations.append({"type":"seasonal_diff","D":D,"s":s})
        │ pd.Series (stationary)
        ▼
  state.stationary_series               ← read by Axis 2, 3, 4
```

**State fields written by Axis 1:**

| Field | Type | Set when |
|---|---|---|
| `state.original_series` | `pd.Series` | Column selected after load/impute |
| `state.stationary_series` | `pd.Series` | After final transform step |
| `state.transformations` | `list[dict]` | Appended per transform; ordered |

> [!IMPORTANT]
> Setting `state.original_series` triggers a cascade that clears `state.stationary_series`, `state.transformations`, and all downstream state. This is intentional: a new series means none of the old transforms, model fits, or forecasts are valid.

---

## 6. Design Decisions & Gotchas

### 6.1 Why strictly positive is required for Box-Cox

$X_t^\lambda$ is complex-valued for non-integer $\lambda$ when $X_t < 0$, and $\ln(X_t)$ is undefined for $X_t \leq 0$. The transformation is not mathematically defined on non-positive data. Common workarounds (shift the series by $\min + \varepsilon$) exist but are not implemented here because they change the interpretation of the resulting $\lambda$ and the inverse transform.

### 6.2 Why `bfill()` after `ffill()` (and after `interpolate()`)

`ffill()` propagates the last **known** value forward. If the series starts with NaNs, there is no previous value to propagate — those leading NaNs survive. The chained `.bfill()` fills them from the first observed value. Without it, ADF would silently receive NaNs and either crash (`ValueError: x is constant`) or operate on a shorter effective series than the user expects.

### 6.3 The 5% imputation warning threshold

5% is a soft heuristic grounded in the time-series literature (e.g., Schafer & Graham, 2002): imputing more than 5% of observations with simple methods (FF, linear, mean) can artificially smooth the series, suppress variance, and bias autocorrelation estimates. The check is informational — it does not block the user. The threshold is defensible but somewhat arbitrary; the key is that *we made a conscious decision rather than ignoring the issue*.

### 6.4 Frequency response as an Axis 1 → Axis 2 bridge

`compute_frequency_response` is a **bonus feature** that creates a conceptual link between the pre-processing axis and the spectral analysis axis. By showing that differencing is a high-pass filter, it motivates why the differenced series will have a different spectral density from the original — something Axis 2 will then measure formally. It cost one vectorised numpy call to implement; the pedagogical payoff is high.

### 6.5 NaN propagation from differencing

Each `diff()` call introduces one NaN at the head. After $d$ regular and $D$ seasonal diffs, the first $d + D \cdot s$ rows are NaN. The function preserves the original index — downstream axes that align `original_series` with `stationary_series` for plotting must account for this offset.

---

## 7. Likely Defense Questions & Answers

**Q1: What regression equation does the ADF test estimate, and what is the null hypothesis?**

> The ADF estimates $\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum \delta_j \Delta X_{t-j} + \varepsilon_t$. The null hypothesis is $H_0: \gamma = 0$ — a unit root exists and the series is non-stationary. We reject when $p < 0.05$, which means we have enough evidence that $\gamma < 0$ and the series is stationary.

---

**Q2: Why does Box-Cox require strictly positive data?**

> The transformation is $Y_t = (X_t^\lambda - 1)/\lambda$ for $\lambda \neq 0$ and $\ln X_t$ for $\lambda = 0$. Both forms require $X_t > 0$: the logarithm is undefined at zero or below, and the power $X_t^\lambda$ is complex-valued for non-integer $\lambda$ when $X_t < 0$. Our code raises `ValueError` early and clearly rather than propagating a cryptic numpy warning.

---

**Q3: What is the profile log-likelihood and how does "Auto-Optimize Lambda" use it?**

> For each candidate $\lambda$, we transform the data, then evaluate $\ell(\lambda) = -(n/2)\ln\hat\sigma^2_\lambda + (\lambda-1)\sum\ln X_t$. The first term rewards a tight (low-variance) transformed series; the second term is the Jacobian of the transformation ensuring we compare densities on the same scale. `scipy.stats.boxcox` maximises this over $\lambda$ using Brent's bracketed scalar optimisation. When we pass `lam=None`, that is exactly what happens under the hood.

---

**Q4: Explain how differencing acts as a high-pass filter.**

> The squared gain of the first-difference filter at frequency $\omega$ is $(2\sin(\omega/2))^2$. At $\omega=0$ (zero-frequency / long-run trend), $\sin(0) = 0$, so $G(0) = 0$ — the trend is completely removed. At $\omega = \pi$ (Nyquist / highest frequency), $\sin(\pi/2) = 1$, so $G(\pi) = 4$ — high-frequency components are amplified. This is the hallmark of a high-pass filter. The frequency response plot in Tab 2 shows this shape directly.

---

**Q5: Why do you warn users when more than 5% of data is imputed?**

> Simple imputation methods — forward fill, linear interpolation, mean substitution — artificially smooth the series. Smoothing suppresses variance and inflates autocorrelation. If 5% or more of the data is fabricated this way, the ADF test, ACF, and PACF plots that follow are all biased. We do not block the user because their data may be fine despite the percentage, but the warning prompts them to think carefully before proceeding.

---

**Q6: What does the frequency response plot show, and why is it hand-implemented?**

> It plots the squared gain $G(\omega) = (2|\sin(\omega/2)|)^{2d} \cdot (2|\sin(s\omega/2)|)^{2D}$ over $[0, \pi]$ rad/sample, showing which frequencies the differencing filter attenuates (low, near zero) and which it passes (high). We implemented it by hand because the formula is a direct analytic result from the z-transform of the differencing operator and requires only numpy vectorisation — pulling in `scipy.signal` for two lines of numpy would be unnecessary complexity.

---

**Q7 (bonus): Why does `handle_missing` chain `.bfill()` after `.ffill()`?**

> `ffill()` propagates the last observed value forward, so any NaN at the very beginning of the series — before the first valid observation — cannot be filled. The chained `.bfill()` then fills those from the next observed value. Without it, leading NaNs would silently survive into the ADF test, which would either crash or operate on a shorter effective series than the user expects.

---

## 8. Quick Reference — Key Formulas

| Name | Formula | Where used |
|---|---|---|
| ADF regression | $\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_j \delta_j \Delta X_{t-j} + \varepsilon_t$ | `run_adf_test` → statsmodels |
| ADF decision | reject $H_0$ if $p < 0.05$ | `verdict` field |
| Box-Cox ($\lambda \neq 0$) | $Y_t = (X_t^\lambda - 1)/\lambda$ | `apply_box_cox` → scipy |
| Box-Cox ($\lambda = 0$) | $Y_t = \ln X_t$ | `apply_box_cox` → scipy |
| Profile log-likelihood | $\ell(\lambda) = -\tfrac{n}{2}\ln\hat\sigma^2_\lambda + (\lambda-1)\sum\ln X_t$ | Auto-optimize branch |
| Regular difference | $\nabla X_t = (1-B)X_t = X_t - X_{t-1}$ | `apply_differencing` |
| Seasonal difference | $\nabla_s X_t = (1-B^s)X_t = X_t - X_{t-s}$ | `apply_differencing` |
| Squared gain (combined) | $G(\omega) = (2\lvert\sin\tfrac{\omega}{2}\rvert)^{2d}\cdot(2\lvert\sin\tfrac{s\omega}{2}\rvert)^{2D}$ | `compute_frequency_response` |
| High-pass property | $G(0) = 0$, $G(\pi) = 4^d$ | Interpretation |

---

*Last updated: June 2026 · Source: [`src/axis1_preprocessing.py`](../src/axis1_preprocessing.py)*
