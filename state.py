from dataclasses import dataclass, field
import pandas as pd
import numpy as np

@dataclass
class AnalysisState:
    # Tab 1 outputs
    _original_series: pd.Series | None = None
    time_index: pd.Index | None = None

    # Tab 2 outputs
    _stationary_series: pd.Series | None = None
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
    _fitted_model: object | None = None
    model_order: tuple | None = None    # (p, d, q) or (p, d, q, P, D, Q, s)
    model_params: dict | None = None    # {"ar": [...], "ma": [...], "sigma2": float}
    residuals: np.ndarray | None = None
    is_seasonal: bool = False
    preview_model_params: dict | None = None

    # Tab 5 outputs
    _validation_passed: bool = False
    validation_run: bool = False

    @property
    def original_series(self) -> pd.Series | None:
        return self._original_series

    @original_series.setter
    def original_series(self, val: pd.Series | None):
        self._original_series = val
        self._stationary_series = None
        self.box_cox_lambda = None
        self.diff_order_d = 0
        self.diff_order_D = 0
        self.seasonal_period_s = 1
        self.transformations = []
        self.detected_cycles = []
        self._fitted_model = None
        self.model_order = None
        self.model_params = None
        self.preview_model_params = None
        self.residuals = None
        self.is_seasonal = False
        self._validation_passed = False
        self.validation_run = False

    @property
    def stationary_series(self) -> pd.Series | None:
        return self._stationary_series

    @stationary_series.setter
    def stationary_series(self, val: pd.Series | None):
        self._stationary_series = val
        self.detected_cycles = []
        self._fitted_model = None
        self.model_order = None
        self.model_params = None
        self.preview_model_params = None
        self.residuals = None
        self.is_seasonal = False
        self._validation_passed = False
        self.validation_run = False

    @property
    def fitted_model(self) -> object | None:
        return self._fitted_model

    @fitted_model.setter
    def fitted_model(self, val: object | None):
        self._fitted_model = val
        self._validation_passed = False
        self.validation_run = False

    @property
    def validation_passed(self) -> bool:
        return self._validation_passed

    @validation_passed.setter
    def validation_passed(self, val: bool):
        self._validation_passed = val
        self.validation_run = True

    @property
    def data_loaded(self) -> bool:
        return self.original_series is not None

    @property
    def stationarity_done(self) -> bool:
        return self.stationary_series is not None

    @property
    def model_fitted(self) -> bool:
        return self.fitted_model is not None
