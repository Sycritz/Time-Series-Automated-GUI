# Axis 2 — The Spectral Analyst: Frequency Domain Exploration
### Defense Preparation Guide · Tab 3

> **Source file:** [`src/axis2_spectral.py`](../src/axis2_spectral.py)  
> **Audience:** Student team preparing for project defense  
> **B&D Reference:** Brockwell & Davis — *Introduction to Time Series and Forecasting*, Chapters 4 (Spectral Representation) and 10 (Spectral Analysis)

---

## Table of Contents

1. [Role & Purpose](#1-role--purpose)
2. [Theory: The B&D Foundation](#2-theory-the-bd-foundation)
   - 2.1 [The Raw Periodogram & Inconsistency](#21-the-raw-periodogram--inconsistency)
   - 2.2 [Data Tapering & Spectral Leakage](#22-data-tapering--spectral-leakage)
   - 2.3 [Lag-Window Smoothed Spectral Estimator](#23-lag-window-smoothed-spectral-estimator)
   - 2.4 [Degrees of Freedom & Confidence Bands](#24-degrees-of-freedom--confidence-bands)
   - 2.5 [Parametric ARMA Spectral Density](#25-parametric-arma-spectral-density)
   - 2.6 [Cycle Detection & Statistical Significance](#26-cycle-detection--statistical-significance)
   - 2.7 [The Duality Principle](#27-the-duality-principle)
3. [Implementation Walkthrough](#3-implementation-walkthrough)
   - 3.1 [`compute_periodogram`](#31-compute_periodogram)
   - 3.2 [`smooth_spectrum`](#32-smooth_spectrum)
   - 3.3 [`compute_parametric_spectrum`](#33-compute_parametric_spectrum)
   - 3.4 [`detect_cycles`](#34-detect_cycles)
4. [Hand-Implemented vs Library Table](#4-hand-implemented-vs-library-table)
5. [Data Flow](#5-data-flow)
6. [Design Decisions & Gotchas](#6-design-decisions--gotchas)
7. [Likely Defense Questions & Answers](#7-likely-defense-questions--answers)
8. [Quick Reference — Key Formulas](#8-quick-reference--key-formulas)

---

## 1. Role & Purpose

Axis 2 is the **Frequency Domain Detective** of the application. While the time domain examines how a series correlates with its own lagged versions, the frequency domain decomposes the series into its constituent sine and cosine waves of varying frequencies. 

Its core role is to:
- Dissect the stationary time series into periodic cycles.
- Identify the dominant periodicities (e.g., annual cycles, business cycles, weekly seasonality).
- Check if the time-domain model captures these cycles (via a side-by-side parametric spectral overlay).
- Provide automated recommendations to the model builder (Axis 3) regarding seasonal periods and cyclical AR(2) components.

---

## 2. Theory: The B&D Foundation

### 2.1 The Raw Periodogram & Inconsistency

For a real-valued time series $\{X_1, X_2, \dots, X_n\}$, the **raw periodogram** $I(\lambda)$ measures the intensity of periodicities at Fourier frequencies $\lambda_j = 2\pi j / n$:

$$I(\lambda_j) = \frac{1}{2\pi n} \left| \sum_{t=1}^{n} (X_t - \bar{X}) e^{-i t \lambda_j} \right|^2$$

where $j = 1, 2, \dots, \lfloor (n-1)/2 \rfloor$.

#### The Inconsistency Theorem
A major gotcha of the raw periodogram is that **it is not a consistent estimator of the true spectral density $f(\lambda)$**. 
As the sample size $n \to \infty$:
- The Fourier frequencies get closer together, providing higher frequency resolution.
- However, the variance of $I(\lambda_j)$ does not decrease to zero. Under mild conditions, the ordinates at any fixed frequency $\lambda$ behave asymptotically like:

$$I(\lambda) \;\dot{\sim}\; \frac{1}{2} f(\lambda) \chi^2_2$$

Since a $\chi^2_2$ distribution has a standard deviation equal to its mean, the standard deviation of $I(\lambda)$ converges to $f(\lambda)$, which is constant and does not go to zero as $n \to \infty$. The plot remains extremely jagged and "noisy" regardless of how much data we collect. Thus, smoothing is mathematically mandatory for consistent estimation.

---

### 2.2 Data Tapering & Spectral Leakage

When we compute the FFT of a finite series of length $n$, we are implicitly multiplying the infinite series by a **rectangular window** $w_t = 1$ for $1 \le t \le n$ (and $0$ otherwise). In the frequency domain, multiplication by this boxcar window corresponds to convolving the true spectrum with the Dirichlet kernel:

$$D_n(\lambda) = \frac{\sin(n\lambda/2)}{\sin(\lambda/2)}$$

This kernel has high sidelobes (leakage), meaning power from a dominant peak leaks into neighboring frequencies. 
**Tapering** resolves this by smoothly scaling the data near the boundaries to zero. The app implements three tapers:

1. **Cosine Bell (Split Cosine):** Applies a cosine taper to a proportion $p$ (e.g., $10\%$) of the data on both ends:
   $$h_t = \begin{cases} 
   0.5 \left(1 - \cos\frac{\pi t}{p n}\right) & 1 \le t \le pn \\ 
   1 & pn < t \le n(1-p) \\ 
   0.5 \left(1 - \cos\frac{\pi (n - t + 1)}{p n}\right) & n(1-p) < t \le n
   \end{cases}$$
2. **Hann:** Tapers the entire series: $w_t = 0.5 \left(1 - \cos\frac{2\pi t}{n-1}\right)$
3. **Hamming:** Similar to Hann but does not touch zero at the boundaries: $w_t = 0.54 - 0.46 \cos\frac{2\pi t}{n-1}$

---

### 2.3 Lag-Window Smoothed Spectral Estimator

To obtain a consistent spectral estimator $\hat{f}(\lambda)$, we damp high-lag sample autocovariances $\hat{\gamma}(h)$ because they are computed from fewer data pairs and have high variance. We use the **lag-window estimator**:

$$\hat{f}(\lambda) = \frac{1}{2\pi} \left[ \hat{\gamma}(0) + 2 \sum_{h=1}^{M} w\left(\frac{h}{M}\right) \hat{\gamma}(h) \cos(h\lambda) \right]$$

where:
- $\hat{\gamma}(h) = \frac{1}{n} \sum_{t=1}^{n-h} (X_t - \bar{X})(X_{t+h} - \bar{X})$ is the sample autocovariance.
- $w(x)$ is a symmetric lag window satisfying $w(0) = 1$ and $w(x) = 0$ for $|x| > 1$.
- $M$ is the **bandwidth/truncation parameter**.

The application implements four lag windows:
- **Daniell (Rectangular):** $w(x) = 1$ for $|x| \le 1$.
- **Bartlett (Triangular):** $w(x) = 1 - |x|$ for $|x| \le 1$.
- **Parzen (Piecewise Cubic):**
  $$w(x) = \begin{cases} 
  1 - 6x^2 + 6|x|^3 & |x| \le 0.5 \\ 
  2(1 - |x|)^3 & 0.5 < |x| \le 1 
  \end{cases}$$
- **Hann (Raised Cosine):** $w(x) = 0.5(1 + \cos(\pi x))$ for $|x| \le 1$.

---

### 2.4 Degrees of Freedom & Confidence Bands

The smoothed spectral density estimator $\hat{f}(\lambda)$ behaves approximately as a scaled chi-squared variable:

$$\frac{\nu \hat{f}(\lambda)}{f(\lambda)} \;\dot{\sim}\; \chi^2_\nu$$

where $\nu$ represents the **equivalent degrees of freedom (EDF)**. The EDF depends on the sample size $n$, the truncation point $M$, and the choice of lag window:

$$\nu = \frac{2n}{\sum_{h=-M}^{M} w^2(h/M)} = \frac{2n}{1 + 2\sum_{h=1}^{M} w^2(h/M)}$$

Using this distribution, the $95\%$ confidence interval for the true spectral density $f(\lambda)$ at frequency $\lambda$ is:

$$\left[ \frac{\nu \hat{f}(\lambda)}{\chi^2_{\nu, 0.975}}, \; \frac{\nu \hat{f}(\lambda)}{\chi^2_{\nu, 0.025}} \right]$$

where $\chi^2_{\nu, \alpha}$ is the $\alpha$-quantile of a chi-squared distribution with $\nu$ degrees of freedom. On a logarithmic plot, this interval has a constant width across all frequencies, which is why spectral densities are frequently plotted on a log scale.

---

### 2.5 Parametric ARMA Spectral Density

If a process follows a stationary and invertible ARMA$(p, q)$ model:

$$\phi(B)X_t = \theta(B)Z_t, \quad \{Z_t\} \sim \text{WN}(0, \sigma^2)$$

its **theoretical spectral density** $f_X(\lambda)$ is completely determined by the model parameters:

$$f_X(\lambda) = \frac{\sigma^2}{2\pi} \frac{|\theta(e^{-i\lambda})|^2}{|\phi(e^{-i\lambda})|^2} = \frac{\sigma^2}{2\pi} \frac{\left|1 + \theta_1 e^{-i\lambda} + \cdots + \theta_q e^{-iq\lambda}\right|^2}{\left|1 - \phi_1 e^{-i\lambda} - \cdots - \phi_p e^{-ip\lambda}\right|^2}$$

By plotting this smooth curve on top of the nonparametric lag-window spectrum, we gain immediate visual confirmation of model specification:
- If the curves match closely, the ARMA model has successfully captured the covariance structure of the data.
- If there is a peak in the nonparametric spectrum that the parametric curve misses, the model has failed to capture that periodicity (e.g., missing seasonal parameters).

---

### 2.6 Cycle Detection & Statistical Significance

Simply picking the highest peaks of the periodogram is dangerous because the spectrum is rarely flat (except for white noise). A series with high positive autoregressive correlation has a spectrum that slopes steeply upwards towards $\lambda = 0$ (a "red noise" spectrum).

To detect cycles, the application:
1. Fits a **log-quadratic baseline** to the log of the smoothed spectrum $\ln \hat{f}(\lambda)$ vs. $\lambda$ using a degree-2 polynomial. This models the overall low-frequency slope and curvature.
2. Formulates a threshold at each frequency based on the local baseline $b(\lambda)$ and the $95\%$ quantile of the chi-squared distribution with $\nu$ degrees of freedom:
   $$\text{Threshold}(\lambda) = b(\lambda) \frac{\chi^2_{\nu, 0.95}}{\nu}$$
3. Runs a peak-finding algorithm (3-point local maxima) and labels peaks exceeding this threshold as statistically significant.
4. Converts the peak frequencies to periods ($T = 2\pi / \lambda$) and logs them in `state.detected_cycles`.

---

### 2.7 The Duality Principle

The Core Duality Principle states that time-domain features and frequency-domain features are two descriptions of the exact same underlying stochastic process:

| Time Domain | Frequency Domain |
|---|---|
| A slowly decaying, quasi-periodic sinusoidal ACF | A sharp peak in the spectral density $f(\lambda)$ at frequency $\lambda_0$ |
| A seasonal spike in the ACF at lag $s$ | Spectral peaks at the seasonal frequency $2\pi/s$ and its integer harmonics |
| A fitted AR(2) model with complex roots $e^{\pm i \lambda_0}$ | A theoretical spectrum peak centered at frequency $\lambda_0$ |

---

## 3. Implementation Walkthrough

### 3.1 `compute_periodogram`

This function extracts the raw values of the pandas Series, subtracts the mean (to remove the spike at frequency 0), applies one of three tapers if requested, computes the FFT, and returns the frequencies and periodogram values up to the Nyquist frequency ($f = \pi$):

```python
x = series.values - series.mean()
n = len(x)
# [Apply Taper, e.g., Cosine Bell]
p = int(taper_pct * n)
for idx in range(p):
    val = 0.5 * (1.0 - np.cos(np.pi * (idx + 1) / p))
    w[idx] = val
    w[n - 1 - idx] = val
x = x * w

# Compute FFT
fft_vals = np.fft.fft(x)
periodogram = (1.0 / (2.0 * np.pi * n)) * (np.abs(fft_vals) ** 2)
num_freqs = n // 2 + 1
frequencies = 2.0 * np.pi * np.arange(num_freqs) / n
periodogram = periodogram[:num_freqs]
```

### 3.2 `smooth_spectrum`

This computes the consistent lag-window estimator. Note that autocovariances $\hat{\gamma}(h)$ are hand-calculated up to the truncation point $M$, lag window weights are computed, and degrees of freedom are evaluated using the chi-squared distribution:

```python
# Compute autocovariances gamma_hat for h = 0, ..., M
gamma = np.zeros(M_eff + 1)
for h in range(M_eff + 1):
    gamma[h] = np.sum(x[:n-h] * x[h:]) / n

# Compute lag window w(h/M) [e.g. Parzen]
# ... (code computes Parzen/Bartlett piecewise values)

# Equivalent degrees of freedom
sum_w2 = np.sum(w_vals[1:]**2)
nu = (2.0 * n) / (1.0 + 2.0 * sum_w2)

# Compute estimator
term = np.zeros_like(frequencies)
for h in range(1, M_eff + 1):
    term += w_vals[h] * gamma[h] * np.cos(h * frequencies)
f_hat = (1.0 / (2.0 * np.pi)) * (gamma[0] + 2.0 * term)
f_hat = np.clip(f_hat, 1e-15, None) # Guard against values <= 0
```

### 3.3 `compute_parametric_spectrum`

This constructs the complex polynomials for the numerator (MA) and denominator (AR) to compute the theoretical spectrum at 512 points over $[0, \pi]$:

```python
den = np.ones_like(frequencies, dtype=complex)
for j, phi in enumerate(ar_coeffs):
    den -= phi * np.exp(-1j * (j + 1) * frequencies)
    
num = np.ones_like(frequencies, dtype=complex)
for k, theta in enumerate(ma_coeffs):
    num += theta * np.exp(-1j * (k + 1) * frequencies)
    
den_abs = np.clip(np.abs(den), 1e-15, None)
theoretical_spectrum = (sigma2 / (2.0 * np.pi)) * (np.abs(num) ** 2) / (den_abs ** 2)
```

### 3.4 `detect_cycles`

Finds peaks and evaluates them against the log-quadratic baseline. If the degrees of freedom $\nu$ is not supplied (e.g. if called on a series without knowing the window parameters), it recovers $\nu$ from the ratio of `ci_upper` to `f_hat` by solving $\nu / \chi^2_{0.025}(\nu) = \text{ratio}$ using Brent's method:

```python
# Recover nu from CI upper limit ratio
def obj(nu_val):
    denom = scipy.stats.chi2.ppf(0.025, df=nu_val)
    return nu_val / denom - ratio if denom > 0 else -ratio
nu = scipy.optimize.brentq(obj, 1e-5, 1e7)

# Log-quadratic baseline fit
log_f = np.log(np.clip(f_hat, 1e-15, None))
poly = np.polyfit(frequencies, log_f, deg=2)
baseline = np.exp(np.polyval(poly, frequencies))

# Check significance
thresholds = baseline * scipy.stats.chi2.ppf(0.95, df=nu) / nu
sig = bool(val > thresholds[idx])
```

---

## 4. Hand-Implemented vs Library Table

| Component | Hand-Implemented | Library | Why |
|---|---|---|---|
| **Data Tapering** | Taper sequences (Cosine Bell, Hann, Hamming) are generated mathematically. | — | Deep pedagogical validation of window scaling equations. |
| **FFT** | — | `numpy.fft.fft` | Reusing standard optimized FFT implementation. |
| **Autocovariance** | Calculation of $\hat{\gamma}(h)$ via shifted slice dot products. | — | Avoids depending on time-domain library settings. |
| **Lag Windows** | Daniell, Bartlett, Parzen, Hann piecewise functions. | — | Enforces exact window geometries from B&D Chapter 10. |
| **Degrees of Freedom** | In-code calculation of $\nu$ from window weights. | — | Direct implementation of Brockwell & Davis equation 10.4.9. |
| **Parametric Spectrum** | Complex polynomial evaluation of $e^{-i j \lambda}$. | — | Direct mapping of the ARMA rational spectral density formula. |
| **Cycle Detection** | 3-point local max, log-quadratic baseline, significance threshold. | `scipy.optimize.brentq` for $\nu$ recovery, `np.polyfit` for baseline | Custom baseline detrending is more robust than a flat white noise threshold. |

---

## 5. Data Flow

```
state.stationary_series (Pandas Series from Tab 2)
  │
  ▼
[Tab 3: Spectral Exploration]
  ├── compute_periodogram() -> (freqs, raw_periodogram)
  ├── smooth_spectrum() -> (freqs, smoothed_spec, CI_lower, CI_upper)
  ├── detect_cycles() -> list of dicts (frequency, period, significance)
  │      │
  │      └─► writes to state.detected_cycles
  │
  ▼ (after Tab 4 model fitting)
compute_parametric_spectrum() -> (freqs, theoretical_spec) 
  (Displays overlay comparison in the UI)
```

---

## 6. Design Decisions & Gotchas

1. **Clamping to $10^{-15}$:** In both `smooth_spectrum` and `compute_parametric_spectrum`, the estimated densities or denominators are clipped. If they hit zero, the log-spectral plots and division would fail with zero-division errors.
2. **Why Log-Quadratic Baseline?** If we just test peaks against a flat line (like the white noise variance), any trend-like series with high low-frequency power will flag *every* low-frequency peak as significant, even if it's just a smooth autoregressive decay. The quadratic baseline removes this slope before evaluating peaks.
3. **Brent's Method for $\nu$ Recovery:** If we read a spectrum back in another axis (like Axis 4 residual checking) where we only have the plotted arrays (`f_hat` and `ci_upper`) but not the original window choice or $M$, we can reverse-engineer $\nu$ since the width of the confidence band is a monotonic function of $\nu$. Brent's root-finder solves this efficiently in fractions of a millisecond.

---

## 7. Likely Defense Questions & Answers

### Q1: Why is the raw periodogram not a consistent estimator of the spectral density?
**Answer:** Because its variance does not go to zero as $n \to \infty$. Instead, each ordinate $I(\lambda_j)$ asymptotically behaves like $\frac{1}{2} f(\lambda_j) \chi^2_2$. The standard deviation of the estimate is equal to the value we are trying to estimate, resulting in a permanent "hairy" and erratic plot regardless of sample size.

### Q2: What is the tradeoff between selecting a small vs. large bandwidth parameter $M$ in the lag window?
**Answer:** This is the classical **bias-variance tradeoff**:
- **Small $M$ (narrow window):** Wipes out autocovariances at low lags, yielding a highly smoothed spectrum. This has **low variance** (smooth curve) but **high bias** (can smooth out and erase real narrow spectral peaks).
- **Large $M$ (wide window):** Retains autocovariances up to high lags. This has **low bias** (can resolve close peaks) but **high variance** (the spectrum is jagged and unstable).

### Q3: What is "spectral leakage" and how does data tapering mitigate it?
**Answer:** Spectral leakage occurs because we observe the series over a finite interval $[1, n]$, which is mathematically equivalent to multiplying the infinite process by a rectangular window. In the frequency domain, this convolves the true spectrum with the Dirichlet kernel, whose large sidelobes leak energy from major peaks into distant frequencies. Tapering smoothly scales the data down to zero at the boundaries, dampening these sidelobes at the expense of slightly widening the main peak.

### Q4: Explain the difference between the Daniell, Bartlett, and Parzen windows.
**Answer:** The windows apply different weighting profiles to the autocovariances $\hat{\gamma}(h)$ as lag increases:
- **Daniell (Rectangular):** Weights all lags up to $M$ equally ($w=1$). It has the smallest peak width but poor sidelobe suppression.
- **Bartlett (Triangular):** Linear decay ($1 - |x|$). Reduces sidelobes but introduces more bias.
- **Parzen (Cubic Spline):** Smooth piecewise cubic decay. It has excellent sidelobe suppression (no negative values) and is highly recommended in B&D.

### Q5: How do we determine the statistical significance of a cycle in the spectrum?
**Answer:** We fit a quadratic baseline to the log-smoothed spectrum to capture the overall shape (the red-noise background). Then, at each frequency, we set a critical threshold using the baseline value multiplied by the $95\%$ quantile of the scaled chi-squared distribution: $\text{Threshold}(\lambda) = b(\lambda) \frac{\chi^2_{\nu, 0.95}}{\nu}$. Any peak exceeding this threshold is statistically significant at the $5\%$ significance level.

### Q6: How does the parametric spectrum overlay serve as a model diagnostic?
**Answer:** The parametric spectrum is the theoretical spectrum derived directly from the fitted ARMA parameters. The nonparametric spectrum is computed directly from the data. If the two curves match closely, the model captures the correlation structure. If the nonparametric spectrum shows a significant peak that the parametric spectrum lacks, it indicates the model is misspecified (e.g., missed seasonal lag).

### Q7: Why is it helpful to plot the spectrum on a logarithmic scale?
**Answer:** The confidence interval for the spectral density is multiplicative: $\hat{f}(\lambda)$ multiplied or divided by a factor. Taking the logarithm converts this multiplication into addition: $\ln \hat{f}(\lambda) \pm \text{constant}$. This means the confidence interval has a **constant height** across the entire plot, allowing us to visually compare peak significance at high and low power regions easily.

### Q8: What are "equivalent degrees of freedom" ($\nu$)?
**Answer:** The equivalent degrees of freedom represent the size of a white noise series that would yield an estimate with the same variance as our smoothed estimate. It is computed as $\nu = 2n / (1 + 2\sum_{h=1}^M w^2(h/M))$, showing that smoothing over more lags (smaller $M$) increases $\nu$ and reduces variance.

---

## 8. Quick Reference — Key Formulas

| Concept | Formula |
|---|---|
| **Raw Periodogram** | $I(\lambda_j) = \frac{1}{2\pi n} \left| \sum_{t=1}^{n} (X_t - \bar{X}) e^{-i t \lambda_j} \right|^2$ |
| **Lag-Window Smoother** | $\hat{f}(\lambda) = \frac{1}{2\pi} \left[ \hat{\gamma}(0) + 2 \sum_{h=1}^{M} w\left(\frac{h}{M}\right) \hat{\gamma}(h) \cos(h\lambda) \right]$ |
| **Equivalent df ($\nu$)** | $\nu = \frac{2n}{1 + 2\sum_{h=1}^{M} w^2(h/M)}$ |
| **95% Confidence Band** | $\left[ \frac{\nu \hat{f}(\lambda)}{\chi^2_{\nu, 0.975}}, \; \frac{\nu \hat{f}(\lambda)}{\chi^2_{\nu, 0.025}} \right]$ |
| **ARMA Spectral Density** | $f_X(\lambda) = \frac{\sigma^2}{2\pi} \frac{\left|1 + \theta_1 e^{-i\lambda} + \cdots + \theta_q e^{-iq\lambda}\right|^2}{\left|1 - \phi_1 e^{-i\lambda} - \cdots - \phi_p e^{-ip\lambda}\right|^2}$ |
| **Log-Quadratic Baseline** | $\ln b(\lambda) = \beta_0 + \beta_1 \lambda + \beta_2 \lambda^2$ |
| **Significance Threshold** | $\text{Threshold}(\lambda) = b(\lambda) \frac{\chi^2_{\nu, 0.95}}{\nu}$ |
