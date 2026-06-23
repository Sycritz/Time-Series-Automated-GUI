# Axis 4 — The Model Auditor: Residual Diagnostics & Validation
### Defense Preparation Guide · Tab 5

> **Source file:** [`src/axis4_validation.py`](../src/axis4_validation.py)  
> **Audience:** Student team preparing for project defense  
> **B&D Reference:** Box, Jenkins & Reinsel — *Time Series Analysis*, Chapter 8 (Diagnostic Checking)

---

## Table of Contents

1. [Role & Purpose](#1-role--purpose)
2. [Theory: The B&D Foundation](#2-theory-the-bd-foundation)
   - 2.1 [What Adequate Residuals Look Like](#21-what-adequate-residuals-look-like)
   - 2.2 [Ljung-Box Portmanteau Test](#22-ljung-box-portmanteau-test)
   - 2.3 [Jarque-Bera Normality Test](#23-jarque-bera-normality-test)
   - 2.4 [Cumulative Periodogram Test](#24-cumulative-periodogram-test)
   - 2.5 [Residual Spectral Density](#25-residual-spectral-density)
   - 2.6 [The 2×3 Diagnostic Plot Grid](#26-the-23-diagnostic-plot-grid)
3. [Implementation Walkthrough](#3-implementation-walkthrough)
   - 3.1 [`compute_cumulative_periodogram`](#31-compute_cumulative_periodogram)
   - 3.2 [`compute_residual_spectrum`](#32-compute_residual_spectrum)
   - 3.3 [`run_all_diagnostics`](#33-run_all_diagnostics)
4. [Hand-Implemented vs Library Table](#4-hand-implemented-vs-library-table)
5. [Data Flow](#5-data-flow)
6. [Design Decisions & Gotchas](#6-design-decisions--gotchas)
7. [Likely Defense Questions & Answers](#7-likely-defense-questions--answers)
8. [Quick Reference — Key Formulas](#8-quick-reference--key-formulas)

---

## 1. Role & Purpose

Axis 4 is the **quality-control gate** of the entire pipeline. Its single job is to answer one question: *"Are the residuals of the fitted model indistinguishable from white noise?"*

The Box & Jenkins (B&D) philosophy is unambiguous on this point:

> **A model is NEVER used for forecasting until its residuals are demonstrably white noise in BOTH the time domain AND the frequency domain.**

Axis 4 enforces this rule programmatically. It:

- Runs four independent diagnostic tests (Ljung-Box, Jarque-Bera, Cumulative Periodogram, Spectral Flatness).
- Renders a 2×3 diagnostic plot grid giving a visual summary.
- Issues a binary verdict: **ADEQUATE** or **INADEQUATE**.
- **Hard-gates Tab 6** (Forecasting) so that a failed model literally cannot produce forecasts.

If any single test fails, the team must return to Tab 4 (Model Fitting) and revise the order $(p, q)$ before forecasting is permitted.

---

## 2. Theory: The B&D Foundation

### 2.1 What Adequate Residuals Look Like

For an ARIMA$(p, d, q)$ model fitted to a time series $\{z_t\}$, the one-step-ahead residuals are:

$$\hat{a}_t = z_t - \hat{z}_t(1)$$

If the model is **correctly specified**, these residuals should be a realization of a sequence of i.i.d. Gaussian random variables:

$$\hat{a}_t \sim \mathcal{N}(0,\, \sigma_a^2) \quad \text{(independent)}$$

Operationally, this means:

| Property | Implication |
|---|---|
| **Zero mean** | No systematic bias |
| **Constant variance** | No heteroskedasticity |
| **No autocorrelation** | Model has captured all linear structure |
| **Gaussian distribution** | Prediction intervals are valid |
| **Flat spectrum** | No uncaptured periodicity at any frequency |

Failing any of these means the model has *left structure in the residuals* — structure that could be modelled and would improve forecasts.

### 2.2 Ljung-Box Portmanteau Test

The **Ljung-Box** test is a *joint* test of whether the first $h$ sample autocorrelations of the residuals are all zero.

**Intuition:** Instead of testing each lag individually (which inflates Type I error), it pools all lags into one statistic.

**Statistic:**

$$Q_{LB}(h) = n(n+2)\sum_{k=1}^{h} \frac{\hat{\rho}^2(\hat{a}_k)}{n - k}$$

where:
- $n$ = number of residuals
- $\hat{\rho}(\hat{a}_k)$ = sample autocorrelation of residuals at lag $k$
- $h$ = number of lags tested (here: $\min(20,\, n-1)$)

**Null hypothesis:** $H_0: \rho(\hat{a}_1) = \rho(\hat{a}_2) = \cdots = \rho(\hat{a}_h) = 0$ (residuals are uncorrelated)

**Distribution under $H_0$:**

$$Q_{LB}(h) \;\overset{\text{approx}}{\sim}\; \chi^2_{h - p - q}$$

The degrees of freedom are **reduced by $p+q$** to correct for the parameters that were estimated in fitting the ARIMA model. See Section 6 for why.

**Pass criterion:** $p\text{-value at lag }h \geq 0.05$

In code, the library call is:
```python
acorr_ljungbox(res, lags=h_max, model_df=p+q)
```

The `model_df=p+q` argument tells `statsmodels` to subtract $p+q$ from the nominal degrees of freedom.

### 2.3 Jarque-Bera Normality Test

The ARIMA model assumes Gaussian innovations $a_t \sim \mathcal{N}(0, \sigma_a^2)$. The **Jarque-Bera** test checks whether the residual distribution matches a Gaussian by examining its *shape* — specifically its skewness and excess kurtosis.

**Statistic:**

$$JB = \frac{n}{6}\left[S^2 + \frac{(K - 3)^2}{4}\right] \;\overset{\text{approx}}{\sim}\; \chi^2_2$$

where:
- $S = \hat{\mu}_3 / \hat{\sigma}^3$ is the **skewness** (third standardized moment)
- $K = \hat{\mu}_4 / \hat{\sigma}^4$ is the **kurtosis** (fourth standardized moment)
- A perfect Gaussian has $S = 0$ and $K = 3$, so $(K-3)$ measures *excess* kurtosis

**Null hypothesis:** $H_0:$ residuals are drawn from a Gaussian distribution

**Pass criterion:** $p\text{-value} \geq 0.05$

**Why it matters for forecasting:** ARIMA prediction intervals are computed as:

$$\hat{z}_t(l) \pm z_{\alpha/2} \cdot \sigma_a \sqrt{\psi_0^2 + \psi_1^2 + \cdots + \psi_{l-1}^2}$$

where the $z_{\alpha/2}$ quantile comes from the **standard normal distribution**. If the residuals are heavily skewed or fat-tailed, those intervals are wrong — potentially dangerously so. Jarque-Bera is the gate that protects interval validity.

### 2.4 Cumulative Periodogram Test

The cumulative periodogram is the **frequency-domain analogue** of the Kolmogorov-Smirnov test. It is often *more sensitive* than Ljung-Box for detecting low-frequency or seasonal patterns that are spread across many lags.

**Periodogram ordinates:** Let $\text{FFT}[j]$ be the $j$-th coefficient of the DFT of the residuals. The raw periodogram at Fourier frequency $\omega_j = 2\pi j/n$ is:

$$I(\omega_j) = |\text{FFT}[j]|^2, \quad j = 1, 2, \ldots, q, \quad q = \left\lfloor \frac{n-1}{2} \right\rfloor$$

Frequency index $j=0$ (the DC component / sample mean) is excluded.

**Normalized cumulative periodogram:**

$$C(\omega_k) = \frac{\sum_{j=1}^{k} I(\omega_j)}{\sum_{j=1}^{q} I(\omega_j)}, \quad k = 1, 2, \ldots, q$$

**Interpretation for white noise:** If the residuals are truly white noise, their power is *uniformly distributed* across all frequencies. Therefore, $C(\omega_k)$ should follow the 45° diagonal line $k/q$ — a straight line from $(0, 0)$ to $(1, 1)$ (equivalently, from frequency $0$ to $0.5$ cycles/unit in normalized terms).

**Kolmogorov-Smirnov bounds:** The test uses the asymptotic KS distribution. At the 5% level, the 95% confidence band around the diagonal is:

$$\text{diagonal}_k \pm \frac{1.358}{\sqrt{q}} = \frac{k}{q} \pm \frac{1.358}{\sqrt{q}}$$

The constant $1.358$ is the $\alpha = 0.05$ critical value of the KS distribution ($K_{0.05} \approx 1.358$).

**Pass criterion:** $C(\omega_k)$ stays within the KS bounds for all $k = 1, \ldots, q$.

**Why cumulative periodogram often beats Ljung-Box:**

- Ljung-Box tests autocorrelations at individual integer lags. A slowly-decaying or seasonally-structured spectrum might produce autocorrelations too small to trigger Ljung-Box at any single lag, yet still be detectable as a systematic *deviation from the diagonal* in the cumulative periodogram.
- Spectral energy from an uncaptured seasonal component (e.g., an annual cycle) concentrates at a specific frequency and causes a visible *jump* in $C(\omega_k)$ at that frequency band — unmistakable visually and statistically.
- The cumulative periodogram is therefore the *complementary* check: Ljung-Box is powerful for low-lag autocorrelations; the cumulative periodogram is powerful for periodic or band-limited residual structure.

### 2.5 Residual Spectral Density

White noise has a **constant (flat) spectral density**. For a unit-variance white noise process, the theoretical spectral density is:

$$f(\omega) = \frac{1}{2\pi} \approx 0.1592 \quad \text{for all } \omega \in [0, \pi]$$

Axis 4 checks this by:
1. **Standardizing** the residuals to unit variance (so the flat level is exactly $1/(2\pi)$, regardless of $\sigma_a^2$).
2. Calling **`smooth_spectrum`** from Axis 2 (reusing the Bartlett-windowed spectral estimator already implemented there).
3. Verifying that the $1/(2\pi)$ level falls **inside the confidence interval** $[\hat{f}_{lo}(\omega), \hat{f}_{hi}(\omega)]$ at every frequency.

**Pass criterion:** $\hat{f}_{lo}(\omega) \leq \frac{1}{2\pi} \leq \hat{f}_{hi}(\omega)$ for all $\omega$.

The bandwidth parameter is $M = \max(2, \lfloor\sqrt{n}\rfloor)$. See Section 6 for why.

### 2.6 The 2×3 Diagnostic Plot Grid

The six-panel plot provides a holistic visual check. Each panel targets a specific property:

| Panel | Plot | What It Checks | Failure Signature |
|:---:|---|---|---|
| **1** | Standardized residuals vs. time | No trend, no mean-shift, constant variance | Drift, level shifts, funnel/fan shape (heteroskedasticity) |
| **2** | ACF of standardized residuals | No autocorrelation | Spikes at early lags → AR/MA structure missed; spikes at seasonal lags → seasonal component missed |
| **3** | PACF of standardized residuals | No partial autocorrelation | Significant PACF at lag $p+1$ → underfitted AR order |
| **4** | Normal Q-Q plot | Gaussian distribution | S-curve → symmetric heavy tails; curved ends → skewness |
| **5** | Histogram + KDE + $\mathcal{N}(0,1)$ overlay | Shape of distribution | Bimodality, skewness, fat tails |
| **6** | Cumulative periodogram + KS bounds | Flat frequency spectrum | Deviation from diagonal at specific frequencies → uncaptured periodicities |

> **Defense tip:** Be ready to *describe* what each panel should look like for a **passing** model. Panels 2 and 3 are the most commonly probed.

---

## 3. Implementation Walkthrough

### 3.1 `compute_cumulative_periodogram`

**Signature:**
```python
def compute_cumulative_periodogram(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

**Returns:** `(frequencies, C_omega, ks_upper, ks_lower)`

**Step-by-step:**

```python
res = residuals[~np.isnan(residuals)]      # (1) strip NaNs — guard against gaps
n = len(res)
q = (n - 1) // 2                           # (2) number of usable Fourier frequencies
                                            #     for a real series of length n

fft_vals = np.fft.fft(res)                 # (3) full complex FFT
I_Z = np.abs(fft_vals[1:q+1]) ** 2        # (4) raw periodogram: |FFT[j]|^2, j=1..q
                                            #     j=0 excluded (DC / mean component)

cum_sum = np.cumsum(I_Z)                    # (5) running total of spectral power
C_omega = cum_sum / cum_sum[-1]             # (6) normalize by total power -> values in [0,1]

frequencies = np.arange(1, q + 1) / n      # (7) Fourier frequencies in cycles/unit
diagonal    = np.arange(1, q + 1) / q      # (8) expected diagonal under white noise

ks_limit = 1.358 / np.sqrt(q)              # (9) half-width of 95% KS band
ks_upper = diagonal + ks_limit             # (10) upper confidence bound
ks_lower = diagonal - ks_limit             # (11) lower confidence bound
```

**Key implementation notes:**

- `q = (n-1)//2` gives $\lfloor (n-1)/2 \rfloor$ — this is the number of *independent* Fourier frequencies for a real-valued series. Frequencies beyond $q$ are conjugates and carry no new information.
- The guard `if cum_sum[-1] == 0` handles the (pathological) case where all residuals are identical — it returns a flat zero curve rather than dividing by zero.
- `frequencies` are in **cycles per unit time** ($\omega_k = k/n$, ranging from $1/n$ to $q/n \approx 0.5$), not in angular frequency ($\omega = 2\pi k/n$). This matches the conventional periodogram axis.

### 3.2 `compute_residual_spectrum`

**Signature:**
```python
def compute_residual_spectrum(residuals: np.ndarray, window: str, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

**Returns:** `(freqs, f_hat, ci_lower, ci_upper)`

```python
res_std = np.std(res)
std_res = res / res_std if res_std > 1e-10 else res   # standardize to unit variance
series = pd.Series(std_res)

return smooth_spectrum(series, window, M)              # delegate to Axis 2
```

The standardization step is critical: by dividing by $\hat{\sigma}_a$, the white-noise level becomes exactly $1/(2\pi)$ regardless of the fitted model's innovation variance. This makes the pass threshold **universal**.

### 3.3 `run_all_diagnostics`

**Signature:**
```python
def run_all_diagnostics(residuals: np.ndarray, p: int, q: int) -> dict
```

The function orchestrates all four tests and packages results into a single dictionary consumed by the Tab 5 UI.

**Test 1 — Ljung-Box:**
```python
h_max = min(20, n - 1)
lb_df = acorr_ljungbox(res, lags=h_max, model_df=p+q)
lb_p_value_20 = lb_pvalues[19]   # pass/fail keyed to lag 20
lb_pass = (lb_p_value_20 >= 0.05)
```

If `h_max < 20` (short series), results are NaN-padded to length 20 for consistent array shapes. When the lag-20 p-value is NaN (series too short), the code falls back to the last available non-NaN p-value.

**Test 2 — Jarque-Bera:**
```python
jb_stat, jb_pvalue = jarque_bera(res)
jb_pass = (jb_pvalue >= 0.05)
```

**Test 3 — Cumulative Periodogram:**
```python
freqs_cp, C_omega, ks_up, ks_lo = compute_cumulative_periodogram(res)
diagonal = np.arange(1, q_val + 1) / q_val
ks_limit = 1.358 / np.sqrt(q_val)
cp_pass = np.all(np.abs(C_omega - diagonal) <= ks_limit)
```

**Test 4 — Residual Spectrum Flatness:**
```python
M = max(2, int(np.sqrt(n)))
freqs_sp, f_hat, sp_lo, sp_up = compute_residual_spectrum(res, 'Bartlett', M)
flat_level = 1.0 / (2.0 * np.pi)
flatness_pass = np.all((sp_lo <= flat_level) & (flat_level <= sp_up))
```

**Verdict logic (in the UI layer, reading the returned dict):**
```
ALL of {lb_pass, jb_pass, cp_pass, flatness_pass} -> state.validation_passed = True
ANY False                                           -> state.validation_passed = False
```

---

## 4. Hand-Implemented vs Library Table

| Component | Implementation | Library / Function | Why |
|---|---|---|---|
| Cumulative periodogram $C(\omega_k)$ | **Hand-implemented** | `numpy.fft.fft` (primitive only) | B&D Eq. 10-11; pedagogical requirement; full control over $q$, normalization, and KS bound formula |
| KS confidence bounds | **Hand-implemented** | — | Simple closed-form formula; no library needed |
| Ljung-Box $Q_{LB}$ | **Library** | `statsmodels.stats.diagnostic.acorr_ljungbox` | Standard validated implementation; `model_df` parameter handles d.f. correction automatically |
| Jarque-Bera | **Library** | `scipy.stats.jarque_bera` | Standard; numerically stable skewness/kurtosis computation |
| Smoothed spectrum + CI | **Cross-axis reuse** | `axis2_spectral.smooth_spectrum` | DRY principle; Axis 2 already implements Bartlett-windowed estimator with confidence intervals |
| NaN stripping | **Hand-implemented** | `numpy` boolean indexing | Guard for missing values / gaps in the series |

---

## 5. Data Flow

```
state.residuals ──────────────────────────────────────────────────────────────┐
state.model_order (p, q) ─────────────────────────────────────────────────┐   │
                                                                           │   │
                                                                           v   v
                                                           run_all_diagnostics(res, p, q)
                                                                           │
                    ┌──────────────────────────────────────────────────────┘
                    │
        ┌───────────┼──────────────────────────────────────────┐
        v           v                                          v
acorr_ljungbox  jarque_bera       compute_cumulative_periodogram(res)
(statsmodels)  (scipy.stats)                  │
        │           │                         │   compute_residual_spectrum(res, 'Bartlett', M)
        │           │                         │                 │
        │           │                         │     axis2_spectral.smooth_spectrum(...)
        │           │                         │                 │
        └───────────┴─────────────────────────┴─────────────────┘
                                        │
                               returns dict {
                                   lb_pass, lb_stats, lb_pvalues,
                                   jb_pass, jb_stat, jb_pvalue,
                                   cp_pass, cp_C_omega, cp_frequencies, ...
                                   flatness_pass, sp_f_hat, sp_frequencies, ...
                               }
                                        │
                               Tab 5 UI renders:
                                   - 2×3 diagnostic grid
                                   - Verdict banner
                                        │
                               state.validation_passed --> Tab 6 enabled / disabled
```

---

## 6. Design Decisions & Gotchas

### Why `model_df = p + q` in Ljung-Box (not `p + q + P + Q`)

The `model_df` argument subtracts the number of **estimated parameters** from the nominal degrees of freedom $h$:

$$df = h - (\text{model\_df})$$

For a pure ARIMA$(p,d,q)$ model, the number of parameters that "use up" autocorrelation structure is exactly $p + q$ (the AR and MA polynomial orders). Seasonal orders $(P, Q)$ are **not currently implemented** in this project, so `model_df = p + q` is correct.

If seasonal ARIMA$(p,d,q)(P,D,Q)_s$ were added, this would need to be `model_df = p + q + P + Q`.

Omitting the adjustment (`model_df=0`) would make the test **too conservative** — it would reject models that are actually adequate, because the test would mistakenly attribute fitted-model autocorrelation as evidence of residual autocorrelation.

### Why `M = max(2, √n)` for the Residual Spectrum

The Bartlett window bandwidth parameter $M$ controls the **bias-variance tradeoff** of the spectral estimator:

- **Large $M$** → narrow smoothing window → high variance (noisy estimate), low bias
- **Small $M$** → wide smoothing window → low variance (smooth estimate), high bias

The rule $M = \lfloor\sqrt{n}\rfloor$ is a classical data-adaptive choice (Priestley, 1981) that achieves asymptotically optimal mean-squared error for a general spectrum. The `max(2, ...)` floor prevents degenerate cases when $n$ is very small.

### Why the Residual Spectrum Must Be Flat at `1/(2π)`

White noise with variance $\sigma^2$ has spectral density $f(\omega) = \sigma^2 / (2\pi)$ for all $\omega \in [0, \pi]$ (one-sided convention). After standardizing residuals to $\sigma = 1$, the level is exactly $1/(2\pi) \approx 0.1592$. A residual spectral estimate whose CI does not cover this level at some frequency indicates that the model has not absorbed all energy at that frequency — i.e., there is still structure to be modelled.

### Why Tab 6 Is Hard-Gated

The hard gate enforces the B&D workflow at the software level:

1. **Prevents spurious forecasts.** An inadequate model can produce confidently-wrong predictions.
2. **Forces model revision.** "INADEQUATE — Return to Tab 4" is not a suggestion; it is the only exit.
3. **Audit trail.** `state.validation_passed` is a persistent boolean that can be inspected at any point in the session.

### The Cross-Axis Reuse of `axis2_spectral.smooth_spectrum`

`compute_residual_spectrum` imports and calls `smooth_spectrum` from Axis 2 directly. This is an intentional **DRY** (Don't Repeat Yourself) design:

- Axis 2 already implements the Bartlett-windowed periodogram smoother with correct confidence interval calculation.
- Reimplementing it in Axis 4 would introduce two independent code paths that could diverge over time.
- The only pre-processing Axis 4 adds is **standardization** (dividing by $\hat{\sigma}_a$) so that the universal threshold $1/(2\pi)$ applies regardless of the model's noise level.

Be ready to explain this dependency at the defense — it demonstrates awareness of software engineering principles applied to scientific code.

---

## 7. Likely Defense Questions & Answers

---

**Q1. "Why is the Ljung-Box statistic chi-squared with $h - p - q$ degrees of freedom, not $h$?"**

**A.** When we fit an ARIMA$(p, d, q)$ model by MLE or CSS, the optimizer *minimizes* the sum of squared one-step-ahead residuals. In doing so, it implicitly forces the first $p + q$ sample autocorrelations of the residuals toward zero — they are constrained by the fitting process. If we then tested those same $p + q$ autocorrelations against a $\chi^2_h$ distribution as if they were free, we would be overconfident: we would be counting autocorrelations as "independent evidence" when they were actually forced to be small by the fitting criterion. Box and Pierce (1970) and Ljung and Box (1978) derived the correction: subtract the number of fitted parameters from the degrees of freedom to obtain the *effective* number of free autocorrelations being tested.

---

**Q2. "Why does the Jarque-Bera test matter for forecasting?"**

**A.** ARIMA $l$-step-ahead prediction intervals take the form $\hat{z}_t(l) \pm z_{\alpha/2}\,\sigma_e(l)$, where $z_{\alpha/2}$ is the standard normal quantile (e.g., 1.96 for 95%). This formula is only valid when innovations are Gaussian. If residuals are fat-tailed (high kurtosis), the actual coverage of those intervals will be *narrower* than claimed — the tails contain more probability mass than the Gaussian assumes, so the user is systematically underestimating forecast uncertainty. Jarque-Bera detects non-Gaussianity through deviations in skewness ($S \neq 0$) and excess kurtosis ($K - 3 \neq 0$), and failing it signals that prediction intervals cannot be trusted at face value.

---

**Q3. "Explain what the cumulative periodogram tests and how the KS bounds work."**

**A.** The cumulative periodogram $C(\omega_k)$ is the *running fraction* of total spectral power up to Fourier frequency $k$. For white noise, all frequencies carry equal power, so $C(\omega_k)$ should rise as a straight diagonal line ($k/q$). Any deviation from that diagonal indicates unequal power distribution — some frequencies carry more energy than others, which means the spectrum is not flat, which means the process is not white noise.

The Kolmogorov-Smirnov bounds come from the asymptotic distribution of the maximum deviation between an empirical CDF and its theoretical counterpart. At the 5% significance level, the maximum absolute deviation from the diagonal should not exceed $1.358/\sqrt{q}$ if the null (white noise) holds. So the test band is $k/q \pm 1.358/\sqrt{q}$, and the model passes if $C(\omega_k)$ stays within this band for all $k$.

---

**Q4. "Why is the cumulative periodogram often more sensitive than Ljung-Box for detecting uncaptured periodicities?"**

**A.** Ljung-Box aggregates the *squared autocorrelations* at lags $1, 2, \ldots, h$. A seasonal component at period $s$ shows up as autocorrelation at lags $s, 2s, 3s, \ldots$ — but with only $\lfloor h/s \rfloor$ contributing lags in the sum, each one possibly below the individual significance threshold, the power of the test decreases.

The cumulative periodogram works in the *frequency domain*: an uncaptured seasonal component at frequency $1/s$ concentrates spectral energy at that exact Fourier band, causing a sharp *upward jump* in $C(\omega)$ at frequency $1/s$. This localized spike is much easier to detect as a KS violation than the diffuse signal in the Ljung-Box sum. In short: Ljung-Box tests many lags jointly with diluted power; the cumulative periodogram focuses all power at the right frequency.

---

**Q5. "What does a significant autocorrelation at a seasonal lag in the residual ACF mean?"**

**A.** A spike at lag $s$ (e.g., lag 12 for monthly data) in the residual ACF means the model has *not* captured the seasonal structure of the series. The ARIMA$(p,d,q)$ model fitted in Tab 4 is treating the series as non-seasonal, but the residuals are still seasonally correlated. The correct remedies are: (1) extend to a seasonal ARIMA$(p,d,q)(P,D,Q)_s$ model, or (2) apply seasonal differencing ($D = 1$) before refitting. Operationally, this sends the team back to Tab 4 to revise model order.

---

**Q6. "What should the residual spectral density look like for a well-specified model?"**

**A.** It should be approximately **flat** (constant) across all frequencies $\omega \in [0, 0.5]$ (cycles/unit), at a level of $1/(2\pi) \approx 0.1592$ after standardization. The smoothed spectral estimate $\hat{f}(\omega)$ will have sampling variability, so the confidence interval will be a band — but that $1/(2\pi)$ level should remain inside the CI at every frequency. Departures look like:

- **Peaks** at specific frequencies → uncaptured periodicity (seasonal component missed).
- **Elevated low-frequency power** → long-memory or trend-like behavior (under-differencing).
- **Elevated high-frequency power** → over-differencing or model misspecification.

---

**Q7. "If Ljung-Box passes but the cumulative periodogram fails, what does that mean?"**

**A.** It means the residuals contain a *periodic or band-limited* signal that does not produce a large enough autocorrelation at any individual lag to trigger Ljung-Box, but does produce detectable non-uniformity in the frequency domain. This is the classic case where the two tests are **complementary rather than redundant**. The most common cause is a near-seasonal frequency that is slightly irregular (autocorrelation distributed over a range of lags rather than concentrated at one), or a very high-frequency component that contributes little to the autocorrelation at integer lags but is clearly visible in the spectrum. The practical conclusion is the same: the model is inadequate and must be revised.

---

**Q8. "Why is Tab 6 disabled until validation passes?"**

**A.** This is a hard enforcement of the B&D modeling discipline. An inadequate model — one whose residuals are not white noise — has, by definition, left predictable structure in the error terms. Any forecasts from such a model would be systematically biased or have incorrect uncertainty bounds. By disabling Tab 6 programmatically (checking `state.validation_passed` before enabling the tab), the application makes it *impossible* to produce forecasts from an unvalidated model. The gate is set in `state.validation_passed`, which is `False` by default and only flipped to `True` when all four diagnostic tests pass simultaneously — enforcing the B&D workflow at the software level.

---

## 8. Quick Reference — Key Formulas

### Ljung-Box Statistic

$$Q_{LB}(h) = n(n+2)\sum_{k=1}^{h} \frac{\hat{\rho}^2(\hat{a}_k)}{n - k} \;\sim\; \chi^2_{h-p-q}$$

- Pass: $p\text{-value}(Q_{LB}(20)) \geq 0.05$

### Jarque-Bera Statistic

$$JB = \frac{n}{6}\left[S^2 + \frac{(K-3)^2}{4}\right] \;\sim\; \chi^2_2$$

- $S$ = sample skewness, $K$ = sample kurtosis
- Pass: $p\text{-value}(JB) \geq 0.05$

### Normalized Cumulative Periodogram

$$C(\omega_k) = \frac{\displaystyle\sum_{j=1}^{k} I(\omega_j)}{\displaystyle\sum_{j=1}^{q} I(\omega_j)}, \qquad q = \left\lfloor\frac{n-1}{2}\right\rfloor, \qquad I(\omega_j) = |\text{FFT}[j]|^2$$

### KS Confidence Bounds

$$\text{Upper}(k) = \frac{k}{q} + \frac{1.358}{\sqrt{q}}, \qquad \text{Lower}(k) = \frac{k}{q} - \frac{1.358}{\sqrt{q}}$$

- Pass: $|C(\omega_k) - k/q| \leq 1.358/\sqrt{q}$ for all $k$

### White Noise Spectral Level

$$f_{\text{WN}}(\omega) = \frac{1}{2\pi} \approx 0.1592 \quad \text{(after standardizing residuals to unit variance)}$$

- Pass: $\hat{f}_{lo}(\omega) \leq \frac{1}{2\pi} \leq \hat{f}_{hi}(\omega)$ for all $\omega$

### Smoothing Bandwidth

$$M = \max\!\left(2,\; \left\lfloor\sqrt{n}\right\rfloor\right)$$

### Prediction Interval (context for JB importance)

$$\hat{z}_t(l) \pm z_{\alpha/2} \cdot \sigma_a \sqrt{\sum_{j=0}^{l-1} \psi_j^2}$$

- Valid only when innovations are Gaussian — hence why JB matters.

---

> **Verdict logic (implemented in the UI layer):**
>
> - `lb_pass AND jb_pass AND cp_pass AND flatness_pass` → `state.validation_passed = True` → Tab 6 **enabled**
>   → *"MODEL IS ADEQUATE — Proceed to Forecasting"*
> - **Any** test fails → `state.validation_passed = False` → Tab 6 **disabled**
>   → *"MODEL IS INADEQUATE — Return to Tab 4"*

---

*Guide written for: Group Project Defense Preparation · Tab 5 — Model Validation & Residual Diagnostics*
