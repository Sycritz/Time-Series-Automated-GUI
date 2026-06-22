# Time Series Automated GUI (Box-Jenkins Pipeline App)

A professional desktop application built with PySide6 that guides users through the complete Box-Jenkins methodology for time series analysis and forecasting. The application enforces a strict wizard-style workflow to ensure mathematical rigor, requiring data preprocessing, transformation to stationarity, cycle detection, model identification/selection, and residual validation before generating forecasts.

## Features

- **Axis 1: Data Ingestion & Preprocessing**: Load CSV files, impute missing values (Forward Fill, Linear Interpolation, Mean Imputation), analyze rolling statistics, and perform the Augmented Dickey-Fuller (ADF) test for stationarity.
- **Axis 2: Spectral Analysis & Cycle Detection**: Hand-implemented nonparametric smoothed spectral estimators (Daniell, Bartlett, Parzen, and Hann lag windows) with chi-squared 95% confidence bands, data tapers, and automated cyclical period detection.
- **Axis 3: Model Identification & Selection**: Plot ACF/PACF with automatic lag bounding and hover tooltips. Automated order suggestion using Box-Jenkins heuristics and spectral cycles. Fast, multi-threaded grid search optimizing AIC, BIC, and AICc, plus parameter estimation using Innovations MLE.
- **Axis 4: Model Validation & Residual Diagnostics**: Interactive 2x3 diagnostic grid featuring standardized residuals, ACF/PACF of residuals, Normal Q-Q plot, histogram with KDE, and the Cumulative Periodogram with Kolmogorov-Smirnov (KS) bounds. Formal verdict summary combining Ljung-Box, Jarque-Bera, and Cumulative Periodogram tests.
- **Axis 5: Forecasting & Uncertainty Quantification**: Prediction engine utilizing recursive $\psi$-weights to compute forecast variances. Renders an interactive fan chart with 50%, 80%, and 95% prediction intervals. Offers a "Spectral Insight" panel connecting forecast behaviors (reversion, trends, oscillations) to the theoretical spectral density.

## Technology Stack

The application is written in Python and uses a minimal, highly optimized dependency set:
- **Python**: 3.12+
- **PySide6**: >= 6.5 (Qt UI Framework)
- **numpy**: >= 1.24 (Numerical Computing)
- **scipy**: >= 1.10 (Scientific computing, statistical distributions, and tapers)
- **pandas**: >= 2.0 (Data ingestion and cleaning)
- **matplotlib**: >= 3.7 (Interactive plot embedding)
- **statsmodels**: >= 0.14 (ARIMA/SARIMAX fitting engine and diagnostic helpers)

## Installation Instructions

Follow these step-by-step instructions to set up the project locally.

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/) installed on your machine.

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sycritz/Time-Series-Automated-GUI.git
   cd Time-Series-Automated-GUI
   ```

2. **Create the Conda Environment**:
   Create and activate a new virtual environment named `ML` with Python 3.12:
   ```bash
   conda create -n ML python=3.12 -y
   conda activate ML
   ```

3. **Install Dependencies**:
   Install the required packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Ensure the `ML` conda environment is active, then run the entry script:

```bash
python main.py
```

### End-to-End Walkthrough (using the Airline Passengers Dataset)

1. **Tab 1: Data Load & Explore**
   - Click **Browse CSV File** and select `datasets/international-airline-passengers.csv`.
   - Set the Time Column to **Month** and the Value Column to **Passengers**.
   - Review the Data Preview table and rolling plots.
   - Click **Run ADF Test** to observe that the raw series is non-stationary.

2. **Tab 2: Transform to Stationarity**
   - Click **Auto-Optimize Lambda** (returns $\lambda \approx 0$ for a Log transform).
   - Set differencing orders: $d=1$ (trend) and $D=1$, $s=12$ (seasonal trend).
   - Click **Apply Transformations**. Verify the rolling mean/variance stabilizes and the ADF test confirms stationarity.

3. **Tab 3: Spectral Exploration**
   - Select a window (e.g. Daniell) and Bandwidth $M=10$.
   - Click **Estimate Spectrum** to view the smoothed spectrum and confidence band.
   - Click **Detect Cycles** to identify the annual cycle (Period $\approx 12.00$ months).

4. **Tab 4: Model Identification**
   - Click **Plot ACF/PACF**. Notice the seasonal spikes at lag 12.
   - Click **Suggest (ACF/PACF)** or **Suggest (Spectral)** for heuristic guidance.
   - Set Grid Search Limits (e.g., Max $p=2$, Max $q=2$, Max $P=1$, Max $Q=1$, Period $s=12$).
   - Click **Run Grid Search** (progress bar will update in real-time).
   - Select a model (e.g. `SARIMA(1,1,0)x(0,1,0)_12` or `SARIMA(0,1,1)x(0,1,1)_12`) by clicking its **Select** button in the Top-10 list.

5. **Tab 5: Model Validation**
   - Click **Run Diagnostics** to generate the 2x3 diagnostic plots and tests.
   - Verify all tests (Ljung-Box, Jarque-Bera, Cumulative Periodogram) pass, displaying an **ADEQUATE [PASS]** verdict.

6. **Tab 6: Generate Forecasts**
   - Set the Horizon (e.g., $h=24$ or $h=36$).
   - Click **Generate Forecast** to view the interactive fan chart (with 50%, 80%, and 95% bands) and the forecast table.
   - Switch between **Original Scale** and **Transformed Scale** to check back-transform correctness.
   - Click **Export Table to CSV** or click **Export Plot** on the chart.

## Demo Video

[Watch the 2-3 minute demonstration video here](https://example.com/demo-video-placeholder) (Link Placeholder)

## Running Tests

To run the automated test suite, execute:
```bash
pytest
```
