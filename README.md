<div align="center">
  <h1>📈 Time Series Automated GUI</h1>
  <p><strong>Box-Jenkins Pipeline App</strong></p>

  <p>
    <img alt="Python Version" src="https://img.shields.io/badge/Python-3.12%2B-blue.svg">
    <img alt="PySide6" src="https://img.shields.io/badge/PySide6-GUI-green.svg">
    <img alt="Statsmodels" src="https://img.shields.io/badge/Statsmodels-Backend-red.svg">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-purple.svg">
  </p>
</div>

---

A professional desktop application built with **PySide6** that guides users through the complete **Box-Jenkins methodology** for time series analysis and forecasting. The application enforces a strict wizard-style workflow, ensuring data preprocessing, transformation to stationarity, cycle detection, model identification, and residual validation are completed before generating forecasts.

## 🧠 Theoretical Foundation

> This project is grounded in the **Box-Jenkins methodology** (Brockwell & Davis, *Introduction to Time Series and Forecasting*): a principled three-stage cycle of *Identification → Estimation → Diagnostic Checking*, applied iteratively until a statistically adequate model is found.

The intellectual core is the **Duality Principle** — the mathematical equivalence between the time domain and the frequency domain. Any periodic structure in a time series simultaneously manifests as a slowly decaying, quasi-periodic pattern in the Autocorrelation Function (ACF) and as a peak in the Spectral Density. The application forces the analyst to reconcile both views: a validated model must be adequate in both domains.

---

## ✨ Features

- 📊 **Axis 1 — Data Ingestion & Preprocessing**: Load CSV files, impute missing values (Forward Fill, Linear Interpolation, Mean Imputation), analyze rolling statistics, and perform the Augmented Dickey-Fuller (ADF) test for stationarity.
- 🌊 **Axis 2 — Spectral Analysis & Cycle Detection**: Hand-implemented nonparametric smoothed spectral estimators (Daniell, Bartlett, Parzen, and Hann lag windows) with chi-squared 95% confidence bands, data tapers, and automated cyclical period detection.
- ⚙️ **Axis 3 — Model Identification & Selection**: Plot ACF/PACF with automatic lag bounding and hover tooltips. Automated order suggestion using Box-Jenkins heuristics and spectral cycles. Fast, multi-threaded grid search optimizing AIC, BIC, and AICc, plus parameter estimation using Innovations MLE.
- ✅ **Axis 4 — Model Validation & Residual Diagnostics**: Interactive 2×3 diagnostic grid featuring standardized residuals, ACF/PACF of residuals, Normal Q-Q plot, histogram with KDE, and the Cumulative Periodogram with Kolmogorov-Smirnov (KS) bounds. Formal verdict summary combining Ljung-Box, Jarque-Bera, and Cumulative Periodogram tests.
- 🔮 **Axis 5 — Forecasting & Uncertainty Quantification**: Prediction engine using recursive psi-weights to compute forecast variances. Renders an interactive fan chart with 50%, 80%, and 95% prediction intervals. Includes a "Spectral Insight" panel connecting forecast behaviour (mean reversion, trends, oscillations) to the theoretical spectral density.

---

## 🏗️ Project Architecture

The codebase is organized around the five analytical axes, each implemented as a self-contained source module:

| Module | Responsibility |
|---|---|
| `axis1_preprocessing.py` | Data ingestion, cleaning, stationarity testing & transformation |
| `axis2_spectral.py` | Nonparametric & parametric spectral analysis, cycle detection |
| `axis3_model_id.py` | ACF/PACF diagnostics, grid search, model fitting |
| `axis4_validation.py` | Residual diagnostics, white-noise tests, adequacy verdict |
| `axis5_forecasting.py` | Forecast engine, fan chart, back-transformation |

A **shared state bus** (a central data object passed between tabs) carries the output contract of each axis — the stationary series, fitted model parameters, and transformation metadata — to the next. The application enforces a **wizard-style workflow**: downstream tabs are locked until upstream steps are validated, preventing forecasting with an unvalidated model.

---

## 📚 Documentation

Detailed guides for each axis and the overall architecture are in the `docs/` directory:

- 📖 [Axis 1 Guide](docs/axis1_guide.md) — Data ingestion, stationarity, and transformations
- 📖 [Axis 2 Guide](docs/axis2_guide.md) — Spectral analysis and cycle detection
- 📖 [Axis 3 Guide](docs/axis3_guide.md) — Model identification and grid search
- 📖 [Axis 4 Guide](docs/axis4_guide.md) — Residual diagnostics and validation
- 📖 [Axis 5 Guide](docs/axis5_guide.md) — Forecasting and uncertainty quantification
- 🏗️ [Architecture Guide](docs/architecture_guide.md) — Shared state bus, interface contracts, and module boundaries

---

## 💻 Technology Stack

The application is written in Python with a minimal, highly optimized dependency set:

| Package | Version | Role |
|---|---|---|
| **Python** | `3.12+` | Core language |
| **PySide6** | `>= 6.5` | Qt UI framework |
| **numpy** | `>= 1.24` | Numerical computing |
| **scipy** | `>= 1.10` | Scientific computing, statistical distributions, tapers |
| **pandas** | `>= 2.0` | Data ingestion and cleaning |
| **matplotlib** | `>= 3.7` | Interactive plot embedding |
| **statsmodels** | `>= 0.14` | ARIMA/SARIMAX fitting engine and diagnostic helpers |

---

## 🚀 Getting Started

### Prerequisites

Ensure [Conda](https://docs.conda.io/) is installed on your machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sycritz/Time-Series-Automated-GUI.git
   cd Time-Series-Automated-GUI
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda create -n ML python=3.12 -y
   conda activate ML
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 End-to-End Walkthrough

To run the application, ensure the `ML` Conda environment is active:
```bash
python main.py
```

*The following walkthrough uses the included `datasets/international-airline-passengers.csv` to demonstrate the full pipeline.*

1. **Tab 1 — Data Load & Explore**
   - Click **Browse CSV File** and select the airline passengers dataset.
   - Set the time and value columns, then review the Data Preview and rolling statistics plots.
   - Click **Run ADF Test** and observe the stationarity verdict for the raw series.

2. **Tab 2 — Transform to Stationarity**
   - Click **Auto-Optimize Lambda** to find the optimal Box-Cox transformation parameter.
   - Set regular and seasonal differencing orders to remove trend and seasonal non-stationarity.
   - Click **Apply Transformations** and confirm that rolling statistics stabilize and the ADF test reports stationarity.

3. **Tab 3 — Spectral Exploration**
   - Select a lag window and bandwidth, then click **Estimate Spectrum** to view the smoothed spectral estimate with confidence bands.
   - Click **Detect Cycles** to identify dominant periodic components in the data.

4. **Tab 4 — Model Identification**
   - Click **Plot ACF/PACF** and inspect seasonal and non-seasonal lag patterns.
   - Use **Suggest (ACF/PACF)** or **Suggest (Spectral)** for heuristic model guidance.
   - Configure Grid Search limits and click **Run Grid Search** (a real-time progress bar will update). Select a model from the ranked Top-10 table.

5. **Tab 5 — Model Validation**
   - Click **Run Diagnostics** to generate the 2×3 diagnostic plot grid.
   - Review the formal verdict combining Ljung-Box, Jarque-Bera, and Cumulative Periodogram tests. The forecasting tab only unlocks on a **PASS** verdict.

6. **Tab 6 — Generate Forecasts**
   - Set the forecast horizon and click **Generate Forecast** to render the interactive fan chart with nested prediction intervals.
   - Toggle between **Original Scale** and **Transformed Scale** to verify back-transformation.
   - Export the forecast table or plot using the provided export buttons.

---

## 🧪 Testing

To run the test suite, use pytest:
```bash
pytest
```
