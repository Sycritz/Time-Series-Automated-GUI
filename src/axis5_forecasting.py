import numpy as np
import pandas as pd
from scipy.stats import norm

from state import AnalysisState


def compute_psi_weights(ar_coeffs: list, ma_coeffs: list, h: int) -> np.ndarray:
    """
    Custom implementation of recursive psi-weights computation (Equation 17).
    psi_0 = 1, and for j >= 1:
        psi_j = theta_j + sum_{k=1}^{min(p, j)} phi_k * psi_{j-k}
    with convention theta_j = 0 for j > q.
    """
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


def get_model_ar_ma_coeffs(res) -> tuple[list, list]:
    """
    Helper to extract AR and MA coefficients from the statsmodels results object.
    Supports both ARIMA and SARIMAX by pulling the combined polynomials.
    """
    poly_ar = getattr(res, "polynomial_reduced_ar", None)
    poly_ma = getattr(res, "polynomial_reduced_ma", None)

    if poly_ar is None:
        poly_ar = getattr(res, "polynomial_ar", None)
    if poly_ma is None:
        poly_ma = getattr(res, "polynomial_ma", None)

    if poly_ar is not None:
        # Sign convention in statsmodels: polynomial_ar is [1.0, -phi_1, -phi_2, ...]
        # So phi_k = -poly_ar[k]
        ar_coeffs = [-c for c in poly_ar[1:]]
    else:
        ar_coeffs = []

    if poly_ma is not None:
        # Sign convention in statsmodels: polynomial_ma is [1.0, theta_1, theta_2, ...]
        # So theta_j = poly_ma[j]
        ma_coeffs = list(poly_ma[1:])
    else:
        ma_coeffs = []

    return ar_coeffs, ma_coeffs


def extend_dates(original_index: pd.Index, h: int) -> pd.Index:
    """
    Extends the original time index by h steps forward.
    Handles DatetimeIndex and generic range/numeric indices.
    """
    if original_index is None or len(original_index) == 0:
        return pd.Index(range(1, h + 1))

    if isinstance(original_index, pd.DatetimeIndex):
        freq = original_index.freq
        if freq is None:
            freq = pd.infer_freq(original_index)
        if freq is not None:
            return pd.date_range(start=original_index[-1], periods=h + 1, freq=freq)[1:]
        else:
            diffs = pd.Series(original_index).diff().dropna()
            if not diffs.empty:
                delta = diffs.median()
                return pd.DatetimeIndex(
                    [original_index[-1] + (i + 1) * delta for i in range(h)]
                )
            else:
                return pd.DatetimeIndex(
                    [original_index[-1] + pd.Timedelta(days=i + 1) for i in range(h)]
                )
    else:
        last_val = original_index[-1]
        try:
            step = (
                original_index[1] - original_index[0] if len(original_index) > 1 else 1
            )
        except Exception:
            step = 1
        return pd.Index([last_val + (i + 1) * step for i in range(h)])


def generate_forecasts(model_result, h: int, state: AnalysisState) -> dict:
    """
    Generates point forecasts and mathematical prediction intervals on the stationary scale.
    Returns a dictionary of results.
    """
    if model_result is None:
        raise ValueError("No model result object provided.")

    # Generate point forecast using statsmodels forecast method (stationary scale)
    point_forecasts = model_result.forecast(steps=h)
    if isinstance(point_forecasts, pd.Series):
        point_vals = point_forecasts.values
    else:
        point_vals = np.array(point_forecasts)

    # Extract coefficients and compute psi-weights
    ar_coeffs, ma_coeffs = get_model_ar_ma_coeffs(model_result)
    psi = compute_psi_weights(ar_coeffs, ma_coeffs, h)

    # Extract white noise variance sigma2
    sigma2 = getattr(model_result, "sigma2", 1.0)
    if sigma2 is None:
        params_dict = (
            model_result.params.to_dict()
            if hasattr(model_result.params, "to_dict")
            else {}
        )
        sigma2 = params_dict.get("sigma2", 1.0)

    # Compute prediction error standard error for h-steps
    psi_sq_cumsum = np.cumsum(psi**2)
    prediction_std = np.sqrt(sigma2 * psi_sq_cumsum)

    # Calculate quantiles
    z_50 = norm.ppf(0.75)  # ~0.6745
    z_80 = norm.ppf(0.90)  # ~1.2816
    z_95 = norm.ppf(0.975)  # ~1.9600

    # Extend index
    orig_idx = (
        state.original_series.index if state.original_series is not None else None
    )
    extended_idx = extend_dates(orig_idx, h)

    forecasts = {
        "steps": np.arange(1, h + 1),
        "dates": extended_idx,
        "point": point_vals,
        "lower_50": point_vals - z_50 * prediction_std,
        "upper_50": point_vals + z_50 * prediction_std,
        "lower_80": point_vals - z_80 * prediction_std,
        "upper_80": point_vals + z_80 * prediction_std,
        "lower_95": point_vals - z_95 * prediction_std,
        "upper_95": point_vals + z_95 * prediction_std,
    }

    return forecasts


def undo_box_cox(x: np.ndarray, lam: float) -> np.ndarray:
    """
    Inverts the Box-Cox transformation:
    (lam * x + 1)**(1 / lam) for lam != 0, exp(x) for lam == 0.
    """
    if abs(lam) < 1e-7:
        return np.exp(x)
    else:
        val = lam * x + 1.0
        # Clip to a small positive value to avoid negative bases before fractional power
        val = np.maximum(val, 1e-9)
        return val ** (1.0 / lam)


def undo_diff(forecast_vals: np.ndarray, prev_series: pd.Series, d: int) -> np.ndarray:
    """
    Undoes ordinary differencing of order d via cumulative summation,
    anchored to the last values of the series before differencing.
    """
    if d <= 0:
        return forecast_vals

    history = prev_series.dropna()
    series_list = [history]

    for _ in range(d - 1):
        series_list.append(series_list[-1].diff().dropna())

    current_forecast = np.array(forecast_vals, dtype=float)

    for i in reversed(range(d)):
        hist_series = series_list[i]
        if len(hist_series) == 0:
            last_val = 0.0
        else:
            last_val = hist_series.iloc[-1]

        reconstructed = np.zeros(len(current_forecast))
        for h in range(len(current_forecast)):
            prev = last_val if h == 0 else reconstructed[h - 1]
            reconstructed[h] = prev + current_forecast[h]
        current_forecast = reconstructed

    return current_forecast


def undo_seasonal_diff(
    forecast_vals: np.ndarray, prev_series: pd.Series, D: int, s: int
) -> np.ndarray:
    """
    Undoes seasonal differencing of order D with period s via cumulative summation,
    anchored to the last s-spaced values of the series before differencing.
    """
    if D <= 0 or s <= 1:
        return forecast_vals

    history = prev_series.dropna()
    series_list = [history]

    for _ in range(D - 1):
        series_list.append(series_list[-1].diff(periods=s).dropna())

    current_forecast = np.array(forecast_vals, dtype=float)

    for i in reversed(range(D)):
        hist_series = series_list[i]
        hist_len = len(hist_series)

        reconstructed = np.zeros(len(current_forecast))
        for h in range(len(current_forecast)):
            if h < s:
                idx = hist_len - s + h
                if idx >= 0:
                    prev = hist_series.iloc[idx]
                else:
                    prev = hist_series.iloc[h % hist_len] if hist_len > 0 else 0.0
            else:
                prev = reconstructed[h - s]
            reconstructed[h] = prev + current_forecast[h]
        current_forecast = reconstructed

    return current_forecast


def get_transformation_histories(
    original_series: pd.Series, transformations: list[dict]
) -> list[pd.Series]:
    """
    Reconstructs the series histories at each step of the transformation chain.
    """
    histories = [original_series]
    current = original_series.copy()

    for trans in transformations:
        if trans["type"] == "boxcox":
            from axis1_preprocessing import apply_box_cox

            current, _ = apply_box_cox(current, trans["lambda"])
        elif trans["type"] == "diff":
            from axis1_preprocessing import apply_differencing

            current = apply_differencing(current, trans["d"], 0, 1)
        elif trans["type"] == "seasonal_diff":
            from axis1_preprocessing import apply_differencing

            current = apply_differencing(current, 0, trans["D"], trans["s"])
        histories.append(current)

    return histories


def back_transform(
    forecasts: dict, transformations: list[dict], original_series: pd.Series
) -> dict:
    """
    Reads the transformations list in reverse and reverts them sequentially on all forecast arrays.
    For log-transformed series, prediction intervals are back-transformed multiplicatively:
    PI = X_original * exp(+- z * sigma_stationary)
    """
    if not transformations or original_series is None:
        return forecasts.copy()

    histories = get_transformation_histories(original_series, transformations)
    transformed_forecasts = forecasts.copy()

    # Check if any Box-Cox transformation was applied and if differencing was also applied
    has_bc = any(
        t["type"] == "boxcox" and t.get("lambda") is not None for t in transformations
    )
    has_diff = any(
        (t["type"] == "diff" and t.get("d", 0) > 0)
        or (t["type"] == "seasonal_diff" and t.get("D", 0) > 0)
        for t in transformations
    )

    # We must have upper_95 and lower_95 to calculate the multiplicative intervals
    use_multiplicative = (
        has_bc and has_diff and "upper_95" in forecasts and "lower_95" in forecasts
    )

    if use_multiplicative:
        from scipy.stats import norm

        # 1. Back-transform point forecasts normally
        point_original = np.array(forecasts["point"], dtype=float)
        for k in reversed(range(len(transformations))):
            trans = transformations[k]
            prev_hist = histories[k]
            if trans["type"] == "boxcox":
                point_original = undo_box_cox(point_original, trans["lambda"])
            elif trans["type"] == "diff":
                point_original = undo_diff(point_original, prev_hist, trans["d"])
            elif trans["type"] == "seasonal_diff":
                point_original = undo_seasonal_diff(
                    point_original, prev_hist, trans["D"], trans["s"]
                )

        transformed_forecasts["point"] = point_original

        # 2. Extract sigma_stationary from 95% interval on stationary scale
        z_95 = norm.ppf(0.975)
        sigma_stationary = (forecasts["upper_95"] - forecasts["lower_95"]) / (
            2.0 * z_95
        )

        # 3. Calculate multiplicative prediction intervals on original scale
        for level_str, z in [
            ("50", norm.ppf(0.75)),
            ("80", norm.ppf(0.90)),
            ("95", norm.ppf(0.975)),
        ]:
            lower_key = f"lower_{level_str}"
            upper_key = f"upper_{level_str}"
            if lower_key in forecasts and upper_key in forecasts:
                lo_bt = point_original * np.exp(-z * sigma_stationary)
                hi_bt = point_original * np.exp(z * sigma_stationary)
                # Clip lower bounds to be non-negative
                transformed_forecasts[lower_key] = np.maximum(lo_bt, 0.0)
                transformed_forecasts[upper_key] = hi_bt

        # Also back-transform any other keys in forecasts that are not handled above
        handled_keys = {
            "point",
            "lower_50",
            "upper_50",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
        }
        for key, val in forecasts.items():
            if key not in handled_keys and key not in ["steps", "dates"]:
                vals = np.array(val, dtype=float)
                for k in reversed(range(len(transformations))):
                    trans = transformations[k]
                    prev_hist = histories[k]
                    if trans["type"] == "boxcox":
                        vals = undo_box_cox(vals, trans["lambda"])
                    elif trans["type"] == "diff":
                        vals = undo_diff(vals, prev_hist, trans["d"])
                    elif trans["type"] == "seasonal_diff":
                        vals = undo_seasonal_diff(
                            vals, prev_hist, trans["D"], trans["s"]
                        )
                transformed_forecasts[key] = vals
    else:
        # Standard additive back-transformations
        keys_to_transform = [
            "point",
            "lower_50",
            "upper_50",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
        ]
        for key in keys_to_transform:
            if key not in forecasts:
                continue

            vals = np.array(forecasts[key], dtype=float)

            # Apply inverse transformations in reverse order
            for k in reversed(range(len(transformations))):
                trans = transformations[k]
                prev_hist = histories[k]

                if trans["type"] == "boxcox":
                    vals = undo_box_cox(vals, trans["lambda"])
                elif trans["type"] == "diff":
                    vals = undo_diff(vals, prev_hist, trans["d"])
                elif trans["type"] == "seasonal_diff":
                    vals = undo_seasonal_diff(vals, prev_hist, trans["D"], trans["s"])

            transformed_forecasts[key] = vals

    return transformed_forecasts


def generate_spectral_insight(state: AnalysisState) -> str:
    """
    Generates a markdown text explaining the qualitative behavior of the forecasts
    by linking them to the model's spectral properties.
    """
    if state.model_order is None:
        return "No model fitted yet. Fit and validate a model first."

    order = state.model_order
    p = order[0]
    d = state.diff_order_d
    q = order[2]

    is_seasonal = state.is_seasonal or (len(order) == 7 and order[6] > 1)
    s = state.seasonal_period_s if is_seasonal else 1
    D = state.diff_order_D
    total_d = d + D

    lines = []
    lines.append("### Why do these forecasts look the way they do?")
    lines.append("#### A Spectral Interpretation")
    lines.append("")

    # 1. Trend / Integration Analysis (singularities at freq = 0)
    if total_d > 0:
        lines.append(f"**Integration & Trend (d={d}, D={D}, s={s}):**")
        lines.append(
            f"- The model requires differencing of total order {total_d}. This implies that the theoretical spectrum $f_X(\\lambda)$ has an infinite singularity at frequency $\\lambda = 0$ (unbounded spectrum)."
        )
        if total_d == 1:
            lines.append(
                "- For a random-walk-like process ($d+D=1$), the forecast function converges to a flat line equal to the last observation (once back-transformed)."
            )
            lines.append(
                "- The prediction intervals expand proportional to $\\sqrt{h}$, reflecting a constant rate of uncertainty accumulation."
            )
        else:
            lines.append(
                f"- For higher integration orders ($d+D={total_d}$), the forecast function asymptotically follows a polynomial trend of degree {total_d - 1} as $h \\to \\infty$."
            )
            lines.append(
                "- The prediction intervals flare out very rapidly due to the accumulation of integrated noise."
            )
    else:
        lines.append("**Stationary ARMA Process (d=0, D=0):**")
        lines.append(
            "- The theoretical spectrum $f_X(\\lambda)$ is continuous and bounded, indicating no unit roots or integration."
        )
        lines.append(
            "- As a result, the forecast function decays geometrically and reverts to the process mean as $h \\to \\infty$."
        )
        if p > 0:
            lines.append(
                f"- The speed of reversion is governed by the modulus of the largest AR roots. Moduli closer to 1.0 result in slower decay."
            )

    # 2. Seasonality / Quasi-periodicity (peaks at seasonal frequencies)
    if is_seasonal and s > 1:
        lambda_s = 2.0 * np.pi / s
        lines.append("")
        lines.append(f"**Seasonal Component (s={s}):**")
        lines.append(
            f"- The model captures seasonal dynamics, corresponding to distinct spectral peaks at the seasonal frequency $\\lambda_s = 2\\pi/{s} \\approx {lambda_s:.4f}$ radians and its harmonics."
        )
        lines.append(
            f"- In the time domain, these peaks translate directly to a persistent, quasi-periodic seasonal oscillation in the forecast function with a period of {s} steps."
        )
    elif p >= 2 and state.model_params:
        ar_coeffs = state.model_params.get("ar", [])
        if len(ar_coeffs) >= 2:
            # check complex roots of the AR polynomial
            poly = [-c for c in reversed(ar_coeffs)] + [1.0]
            try:
                roots = np.roots(poly)
                complex_roots = [r for r in roots if np.abs(np.imag(r)) > 1e-5]
                if complex_roots:
                    r = complex_roots[0]
                    modulus = 1.0 / np.abs(r)
                    phase = np.abs(np.angle(r))
                    period = 2.0 * np.pi / phase if phase > 0 else np.nan
                    if modulus > 0.7:
                        lines.append("")
                        lines.append(f"**Cyclical/Complex Root dynamics:**")
                        lines.append(
                            f"- The autoregressive polynomial has complex conjugate roots with modulus {modulus:.3f} and frequency {phase:.4f} radians."
                        )
                        lines.append(
                            f"- This creates a peak in the theoretical spectrum $f_X(\\lambda)$ at $\\lambda \\approx {phase:.4f}$, causing the forecast function to exhibit a decaying quasi-periodic cycle of period $\\approx {period:.2f}$ steps."
                        )
            except Exception:
                pass

    return "\n".join(lines)
