from dataclasses import dataclass, field
import pandas as pd
import numpy as np

@dataclass
class AnalysisState:
    # Tab 1 outputs
    original_series: pd.Series | None = None
    time_index: pd.Index | None = None

    # Tab 2 outputs
    stationary_series: pd.Series | None = None
    box_cox_lambda: float | None = None
    diff_order_d: int = 0
    diff_order_D: int = 0
    seasonal_period_s: int = 1
    transformations: list[dict] = field(default_factory=list)
    # ponytail: list[dict] — simplest ordered serializable log
    # Each dict: {"type": "boxcox"|"diff"|"seasonal_diff", **params}

    # Tab 3 outputs
    detected_cycles: list[dict] = field(default_factory=list)

    # Tab 4 outputs
    fitted_model: object | None = None
    model_order: tuple | None = None    # (p, d, q) or (p, d, q, P, D, Q, s)
    model_params: dict | None = None    # {"ar": [...], "ma": [...], "sigma2": float}
    residuals: np.ndarray | None = None
    is_seasonal: bool = False

    # Tab 5 outputs
    validation_passed: bool = False

    @property
    def data_loaded(self) -> bool:
        return self.original_series is not None

    @property
    def stationarity_done(self) -> bool:
        return self.stationary_series is not None

    @property
    def model_fitted(self) -> bool:
        return self.fitted_model is not None
