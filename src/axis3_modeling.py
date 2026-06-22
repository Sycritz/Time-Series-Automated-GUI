import numpy as np
import pandas as pd
import warnings
from PySide6.QtCore import QThread, Signal
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def compute_acf_pacf(series: pd.Series, nlags: int) -> tuple[np.ndarray, np.ndarray]:
    clean_series = series.dropna()
    n = len(clean_series)
    if n == 0:
        return np.array([1.0]), np.array([1.0])
    
    # Cap nlags based on statsmodels constraints (nlags < n // 2)
    actual_lags = min(nlags, n // 2 - 1)
    if actual_lags < 1:
        actual_lags = min(1, n - 1)
    if actual_lags < 1:
        return np.array([1.0]), np.array([1.0])
        
    try:
        acf_vals = acf(clean_series, nlags=actual_lags)
        pacf_vals = pacf(clean_series, nlags=actual_lags, method='ywm')
    except Exception:
        try:
            acf_vals = acf(clean_series, nlags=actual_lags)
            pacf_vals = pacf(clean_series, nlags=actual_lags)
        except Exception:
            acf_vals = np.array([1.0] + [0.0]*actual_lags)
            pacf_vals = np.array([1.0] + [0.0]*actual_lags)
            
    return acf_vals, pacf_vals

def suggest_model_from_acf_pacf(acf_vals: np.ndarray, pacf_vals: np.ndarray, n: int) -> dict:
    threshold = 1.96 / np.sqrt(n) if n > 0 else 0.2
    
    # Lags are 0-indexed, so lag 1 is index 1
    sig_acf = [lag for lag in range(1, len(acf_vals)) if abs(acf_vals[lag]) > threshold]
    sig_pacf = [lag for lag in range(1, len(pacf_vals)) if abs(pacf_vals[lag]) > threshold]
    
    explanation = []
    explanation.append(f"ACF/PACF Analysis (N={n}, 95% Confidence Threshold={threshold:.4f}):")
    explanation.append(f"- Significant ACF lags: {sig_acf}")
    explanation.append(f"- Significant PACF lags: {sig_pacf}")
    
    if not sig_acf and not sig_pacf:
        explanation.append("\nConclusion: Neither ACF nor PACF have significant lags. The series behaves like White Noise.")
        return {"p": 0, "q": 0, "explanation": "\n".join(explanation)}
        
    p_suggest = 0
    q_suggest = 0
    
    acf_max = max(sig_acf) if sig_acf else 0
    pacf_max = max(sig_pacf) if sig_pacf else 0
    
    if acf_max > 0 and pacf_max > 0:
        # Both have significant lags
        if acf_max <= 3 and (pacf_max > acf_max + 1 or len(sig_pacf) > len(sig_acf)):
            q_suggest = acf_max
            p_suggest = 0
            explanation.append(f"\nConclusion: ACF cuts off at lag {acf_max} while PACF tails off. This suggests a Moving Average MA({acf_max}) model.")
        elif pacf_max <= 3 and (acf_max > pacf_max + 1 or len(sig_acf) > len(sig_pacf)):
            p_suggest = pacf_max
            q_suggest = 0
            explanation.append(f"\nConclusion: PACF cuts off at lag {pacf_max} while ACF tails off. This suggests an Autoregressive AR({pacf_max}) model.")
        else:
            p_suggest = 1
            q_suggest = 1
            explanation.append("\nConclusion: Both ACF and PACF show significant values at multiple lags, indicating tailing off behavior. A mixed ARMA(1, 1) model is suggested.")
    elif acf_max > 0:
        q_suggest = min(acf_max, 3)
        explanation.append(f"\nConclusion: Only ACF has significant lags, cutting off at lag {q_suggest}. Suggesting MA({q_suggest}).")
    else:
        p_suggest = min(pacf_max, 3)
        explanation.append(f"\nConclusion: Only PACF has significant lags, cutting off at lag {p_suggest}. Suggesting AR({p_suggest}).")
        
    return {"p": p_suggest, "q": q_suggest, "explanation": "\n".join(explanation)}

def suggest_model_from_spectrum(detected_cycles: list[dict], seasonal_period: int) -> dict:
    explanation = []
    explanation.append(f"Spectral Suggestion Analysis (Seasonal Period s={seasonal_period}):")
    
    significant_cycles = [c for c in detected_cycles if c.get("significant", False)]
    explanation.append(f"- Total detected significant cycles: {len(significant_cycles)}")
    for i, c in enumerate(significant_cycles[:5]):
        explanation.append(f"  Cycle {i+1}: Period = {c['period']:.2f} samples (frequency = {c['frequency']:.4f})")
        
    if not significant_cycles:
        explanation.append("\nConclusion: No significant cycles detected in the spectrum. Non-seasonal model suggested.")
        return {"P": 0, "Q": 0, "s": seasonal_period, "explanation": "\n".join(explanation)}
        
    matched = False
    for c in significant_cycles:
        p = c["period"]
        tolerance = 0.15 * seasonal_period
        if abs(p - seasonal_period) <= tolerance:
            matched = True
            explanation.append(f"\nFound significant cycle with period {p:.2f} matching seasonal period {seasonal_period} (within 15% tolerance).")
            break
        elif seasonal_period > 1:
            for h in [2, 3, 4]:
                harmonic_p = seasonal_period / h
                if abs(p - harmonic_p) <= 0.15 * harmonic_p:
                    matched = True
                    explanation.append(f"\nFound significant cycle with period {p:.2f} matching seasonal harmonic {harmonic_p:.2f} (within 15% tolerance).")
                    break
            if matched:
                break
                
    if matched:
        explanation.append(f"\nConclusion: Significant cyclical/seasonal component detected. A seasonal SARIMA model with P=1, Q=1, s={seasonal_period} is suggested.")
        return {"P": 1, "Q": 1, "s": seasonal_period, "explanation": "\n".join(explanation)}
    else:
        explanation.append("\nConclusion: Significant cycles detected, but none match the seasonal period or its main harmonics. A non-seasonal model is suggested, though cyclical features could be modeled with AR components.")
        return {"P": 0, "Q": 0, "s": seasonal_period, "explanation": "\n".join(explanation)}

def fit_model(series: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int] | None):
    clean_series = series.dropna()
    is_seas = False
    if seasonal_order is not None:
        P, D, Q, s = seasonal_order
        if s > 1 and (P > 0 or D > 0 or Q > 0):
            is_seas = True
            
    if is_seas:
        model = SARIMAX(clean_series, order=order, seasonal_order=seasonal_order)
        res = model.fit(disp=False)
    else:
        model = ARIMA(clean_series, order=order)
        res = model.fit(method='innovations_mle')
        
    return res

def grid_search(series: pd.Series, max_p: int, max_q: int, max_P: int, max_Q: int, s: int, d: int, D: int, progress_callback=None) -> pd.DataFrame:
    combinations = []
    is_seasonal = (s > 1) and (max_P > 0 or max_Q > 0 or D > 0)
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if is_seasonal:
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        combinations.append((p, d, q, P, D, Q, s))
            else:
                combinations.append((p, d, q, 0, 0, 0, 0))
                
    total_models = len(combinations)
    results = []
    
    for idx, combo in enumerate(combinations):
        p, _, q, P, _, Q, _ = combo
        order = (p, d, q)
        if is_seasonal:
            seasonal_order = (P, D, Q, s)
        else:
            seasonal_order = None
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                res = fit_model(series, order, seasonal_order)
                
            k = len(res.params)
            n_eff = res.nobs
            aic = res.aic
            bic = res.bic
            log_lik = res.llf
            
            if n_eff - k - 1 > 0:
                aicc = aic + (2.0 * k * (k + 1)) / (n_eff - k - 1)
            else:
                aicc = aic
                
            if is_seasonal:
                model_name = f"SARIMA({p},{d},{q})x({P},{D},{Q})_{s}"
            else:
                model_name = f"ARIMA({p},{d},{q})"
                
            results.append({
                "Model": model_name,
                "AICc": aicc,
                "BIC": bic,
                "LogLik": log_lik,
                "k": k,
                "p": p,
                "d": d,
                "q": q,
                "P": P if is_seasonal else 0,
                "D": D if is_seasonal else 0,
                "Q": Q if is_seasonal else 0,
                "s": s if is_seasonal else 0
            })
        except Exception:
            pass
            
        if progress_callback:
            progress_callback(int((idx + 1) / total_models * 100))
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="AICc").reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)
    else:
        df = pd.DataFrame(columns=["Rank", "Model", "AICc", "BIC", "LogLik", "k", "p", "d", "q", "P", "D", "Q", "s"])
        
    return df

class GridSearchWorker(QThread):
    progress = Signal(int)
    finished = Signal(pd.DataFrame)
    
    def __init__(self, series: pd.Series, max_p: int, max_q: int, max_P: int, max_Q: int, s: int, d: int, D: int, parent=None):
        super().__init__(parent)
        self.series = series
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.s = s
        self.d = d
        self.D = D
        self.error = None
        
    def run(self):
        try:
            def cb(val):
                self.progress.emit(val)
            df_results = grid_search(
                self.series, self.max_p, self.max_q, self.max_P, self.max_Q,
                self.s, self.d, self.D, progress_callback=cb
            )
            self.finished.emit(df_results)
        except Exception as e:
            self.error = e
            self.finished.emit(pd.DataFrame())
