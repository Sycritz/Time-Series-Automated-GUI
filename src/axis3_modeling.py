import numpy as np
import pandas as pd
import warnings
from PySide6.QtCore import QThread, Signal
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def compute_acf_pacf(series: pd.Series, nlags: int) -> tuple[np.ndarray, np.ndarray]:
    from axis1_preprocessing import sample_autocovariance
    
    clean_series = series.dropna()
    n = len(clean_series)
    if n == 0:
        return np.array([1.0]), np.array([1.0])
    
    # Cap nlags based on statsmodels constraints (nlags <= n // 2)
    actual_lags = min(nlags, n // 2)
    if actual_lags < 1:
        actual_lags = min(1, n - 1)
    if actual_lags < 1:
        return np.array([1.0]), np.array([1.0])
        
    gamma0 = sample_autocovariance(clean_series, 0)
    if gamma0 == 0:
        acf_vals = np.zeros(actual_lags + 1)
        acf_vals[0] = 1.0
    else:
        acf_vals = np.array([sample_autocovariance(clean_series, h) / gamma0 for h in range(actual_lags + 1)])
        
    # Durbin-Levinson algorithm for PACF
    pacf_vals = np.zeros(actual_lags + 1)
    pacf_vals[0] = 1.0
    
    if actual_lags > 0:
        phi = np.zeros(actual_lags + 1)
        phi[1] = acf_vals[1]
        pacf_vals[1] = acf_vals[1]
        v = 1.0 - acf_vals[1] ** 2
        
        for h in range(2, actual_lags + 1):
            num = acf_vals[h] - np.sum(phi[1:h] * acf_vals[h - 1 : 0 : -1])
            phi_hh = num / v if v > 1e-14 else 0.0
            pacf_vals[h] = phi_hh
            
            phi_old = phi.copy()
            for j in range(1, h):
                phi[j] = phi_old[j] - phi_hh * phi_old[h - j]
            phi[h] = phi_hh
            
            v = v * (1.0 - phi_hh ** 2)
            if v <= 0:
                v = 1e-14
                
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
        
    # Cutoff detection helper with 1 stray lag tolerance
    def get_cutoff_and_tails_off(sig_lags):
        cutoff = None
        valid_candidates = []
        for k in sig_lags:
            if k <= 3:
                after_lags = [l for l in sig_lags if l > k]
                if len(after_lags) <= 1:
                    valid_candidates.append(k)
        if valid_candidates:
            cutoff = max(valid_candidates)
        tails_off = (cutoff is None) and (len(sig_lags) >= 3)
        return cutoff, tails_off
        
    acf_cutoff, acf_tails_off = get_cutoff_and_tails_off(sig_acf)
    pacf_cutoff, pacf_tails_off = get_cutoff_and_tails_off(sig_pacf)
    
    explanation.append(f"- ACF Cutoff: {acf_cutoff} (Tails-off: {acf_tails_off})")
    explanation.append(f"- PACF Cutoff: {pacf_cutoff} (Tails-off: {pacf_tails_off})")
    
    p_suggest = 0
    q_suggest = 0
    
    if acf_cutoff is not None and pacf_cutoff is None:
        q_suggest = acf_cutoff
        p_suggest = 0
        explanation.append(f"\nConclusion: ACF cuts off at lag {acf_cutoff} while PACF tails off (or does not cut off). Suggesting MA({acf_cutoff}).")
    elif pacf_cutoff is not None and acf_cutoff is None:
        p_suggest = pacf_cutoff
        q_suggest = 0
        explanation.append(f"\nConclusion: PACF cuts off at lag {pacf_cutoff} while ACF tails off (or does not cut off). Suggesting AR({pacf_cutoff}).")
    elif acf_cutoff is not None and pacf_cutoff is not None:
        if acf_cutoff < pacf_cutoff:
            q_suggest = acf_cutoff
            p_suggest = 0
            explanation.append(f"\nConclusion: ACF cuts off earlier (at lag {acf_cutoff}) than PACF (at lag {pacf_cutoff}). Suggesting MA({acf_cutoff}).")
        elif pacf_cutoff < acf_cutoff:
            p_suggest = pacf_cutoff
            q_suggest = 0
            explanation.append(f"\nConclusion: PACF cuts off earlier (at lag {pacf_cutoff}) than ACF (at lag {acf_cutoff}). Suggesting AR({pacf_cutoff}).")
        else:
            p_suggest = 1
            q_suggest = 1
            explanation.append(f"\nConclusion: Both ACF and PACF cut off at lag {acf_cutoff}. Suggesting mixed ARMA(1, 1).")
    elif acf_tails_off and pacf_tails_off:
        p_suggest = 1
        q_suggest = 1
        explanation.append("\nConclusion: Both ACF and PACF tail off. Suggesting mixed ARMA(1, 1).")
    else:
        # Fallback if neither clear pattern is found
        acf_max = max(sig_acf) if sig_acf else 0
        pacf_max = max(sig_pacf) if sig_pacf else 0
        if acf_max > 0 and pacf_max > 0:
            p_suggest = 1
            q_suggest = 1
            explanation.append("\nConclusion: Mixed significant lags detected. Suggesting ARMA(1, 1).")
        elif acf_max > 0:
            q_suggest = min(acf_max, 3)
            explanation.append(f"\nConclusion: Significant lags present in ACF. Suggesting MA({q_suggest}).")
        else:
            p_suggest = min(pacf_max, 3)
            explanation.append(f"\nConclusion: Significant lags present in PACF. Suggesting AR({p_suggest}).")
            
    return {"p": p_suggest, "q": q_suggest, "explanation": "\n".join(explanation)}

def fit_model(series: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int] | None):
    clean_series = series.dropna()
    p, d, q = order
    is_seas = False
    D = 0
    if seasonal_order is not None:
        P_s, D_s, Q_s, s_s = seasonal_order
        if s_s > 1 and (P_s > 0 or D_s > 0 or Q_s > 0):
            is_seas = True
            D = D_s
            
    actual_seasonal_order = seasonal_order if is_seas else None
    
    model = SARIMAX(
        clean_series,
        order=order,
        seasonal_order=actual_seasonal_order,
        trend='n',
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    res = model.fit(disp=False, method='lbfgs', maxiter=200)
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
