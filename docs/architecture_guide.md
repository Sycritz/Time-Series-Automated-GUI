# Architecture Guide — Time Series Analysis GUI
**Defense prep · Coding & architecture side · Read before the viva**

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Shared State Bus (`state.py`)](#2-the-shared-state-bus-statepy)
3. [Data Flow Between Axes](#3-data-flow-between-axes)
4. [The GUI Layer (`main.py` overview)](#4-the-gui-layer-mainpy-overview)
5. [Reusable Widgets (`widgets.py`)](#5-reusable-widgets-widgetspy)
6. [Design Decisions](#6-design-decisions)
7. [Likely Defense Questions — Architecture](#7-likely-defense-questions--architecture)

---

## 1. Architecture Overview

The app is a **wizard-style desktop GUI** built with PySide6 (Qt) and matplotlib. Three patterns work together:

| Pattern | Role |
|---|---|
| **Wizard-style tab flow** | Six tabs represent analysis stages. Each tab is only enabled when its prerequisites are satisfied. The user cannot jump ahead. |
| **Shared state bus** | A single `AnalysisState` dataclass instance lives on `MainWindow`. Every tab reads and writes through it. No data is passed via function arguments between tabs. |
| **Module-per-axis** | Pure analytical logic lives in `axis1_preprocessing.py`, `axis2_spectral.py`, `axis3_modeling.py`, `axis4_validation.py`, `axis5_forecasting.py`. The GUI imports from these on demand; the axis modules have no knowledge of Qt. |

This separation keeps analytical code testable (you can run it headlessly in pytest) while keeping the GUI layer thin — tabs are largely wiring between controls and module calls.

---

## 2. The Shared State Bus (`state.py`)

### What it is

`AnalysisState` is a Python `@dataclass`. One instance is created in `MainWindow.__init__` and stored as `self.state`. Every tab receives a reference to `self` (the `MainWindow`), and accesses state as `self.main_window.state`.

### Fields grouped by pipeline stage

```python
@dataclass
class AnalysisState:
    # ── Tab 1: Data Load ──────────────────────────────────────────
    _original_series: pd.Series | None = None   # managed by property setter
    time_index: pd.Index | None = None

    # ── Tab 2: Transform to Stationarity ─────────────────────────
    _stationary_series: pd.Series | None = None # managed by property setter
    box_cox_lambda: float | None = None
    diff_order_d: int = 0
    diff_order_D: int = 0
    seasonal_period_s: int = 1
    transformations: list[dict] = field(default_factory=list)

    # ── Tab 3: Spectral / Cycle Detection ─────────────────────────
    detected_cycles: list[dict] = field(default_factory=list)

    # ── Tab 4: Model Identification ───────────────────────────────
    _fitted_model: object | None = None         # managed by property setter
    model_order: tuple | None = None            # (p,d,q) or (p,d,q,P,D,Q,s)
    model_params: dict | None = None            # {"ar":[...], "ma":[...], "sigma2": float}
    residuals: np.ndarray | None = None
    is_seasonal: bool = False
    preview_model_params: dict | None = None    # ephemeral — overlay before committing

    # ── Tab 5: Validation ─────────────────────────────────────────
    _validation_passed: bool = False            # managed by property setter
    validation_run: bool = False
```

### Convenience properties

Three read-only boolean properties provide clean gate checks without littering the GUI code with `is not None`:

```python
@property
def data_loaded(self) -> bool:
    return self.original_series is not None

@property
def stationarity_done(self) -> bool:
    return self.stationary_series is not None

@property
def model_fitted(self) -> bool:
    return self.fitted_model is not None
```

These are the exact expressions used in `update_ui_from_state()` to enable/disable tabs.

### Cascading invalidation via property setters

The four "managed" fields use property setter/getter pairs to enforce a **downstream reset rule**: when a user changes an upstream result, all results that depended on it are automatically wiped.

**The chain, from widest to narrowest:**

```
original_series.setter
  -> clears: stationary_series, box_cox_lambda, diff_order_d/D, seasonal_period_s,
             transformations, detected_cycles, fitted_model, model_order,
             model_params, preview_model_params, residuals, is_seasonal,
             validation_passed, validation_run

stationary_series.setter
  -> clears: detected_cycles, fitted_model, model_order, model_params,
             preview_model_params, residuals, is_seasonal,
             validation_passed, validation_run

fitted_model.setter
  -> clears: validation_passed, validation_run

validation_passed.setter
  -> sets: validation_run = True   (opposite direction — records that a run happened)
```

**Example — what happens when the user loads a new CSV:**

```python
# DataLoadTab.impute_data(), line ~429
self.main_window.state.original_series = imputed_series
# One assignment triggers the setter, which resets 14+ downstream fields atomically.
# No manual "reset everything" logic is needed anywhere in the GUI code.
```

The setter for `original_series` (state.py L40–57):

```python
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
    self._fitted_model = None        # note: bypass setter intentionally
    self.model_order = None
    self.model_params = None
    self.preview_model_params = None
    self.residuals = None
    self.is_seasonal = False
    self._validation_passed = False  # note: bypass setter intentionally
    self.validation_run = False
```

> **Why bypass inner setters?** When `original_series.setter` runs, it writes `_fitted_model` and `_validation_passed` directly (private backing fields), not via their own setters. This avoids triggering a chain-within-a-chain — the outer setter is already clearing everything, so triggering `fitted_model.setter` would be a redundant no-op.

### The `transformations` log

`state.transformations` is an ordered `list[dict]`. It grows in Tab 2 as the user applies transformations:

```python
# Each entry appended in TransformTab.apply_transformations()
state.transformations.append({"type": "boxcox", "lambda": applied_lam})
state.transformations.append({"type": "diff", "d": d})
state.transformations.append({"type": "seasonal_diff", "D": D, "s": s})
```

Possible dict shapes:

| `"type"` | Additional keys |
|---|---|
| `"boxcox"` | `"lambda": float` |
| `"diff"` | `"d": int` |
| `"seasonal_diff"` | `"D": int`, `"s": int` |

Axis 5 (`back_transform`) walks this list **in reverse** to undo each transformation in the correct order. An ordered list of dicts is the right structure because: it preserves insertion order (critical — differencing must be undone before Box-Cox), each step carries its own parameters, and it is trivially JSON-serialisable.

---

## 3. Data Flow Between Axes

### Interface contract

| Module | Reads from state | Writes to state |
|---|---|---|
| `axis1_preprocessing` | *(called by Tab 1 & 2 — no state reads in module itself)* | `original_series`, `time_index`, `stationary_series`, `transformations`, `box_cox_lambda`, `diff_order_d/D`, `seasonal_period_s` |
| `axis2_spectral` | `stationary_series` | `detected_cycles` |
| `axis3_modeling` | `stationary_series`, `detected_cycles` | `fitted_model`, `model_order`, `model_params`, `residuals`, `is_seasonal` |
| `axis4_validation` | `residuals`, `model_order` | `validation_passed`, `validation_run` |
| `axis5_forecasting` | `fitted_model`, `original_series`, `transformations` | *(returns forecasts; nothing written back to state)* |

### ASCII data flow diagram

```
+-------------+
|   Tab 1     |  loads CSV -> handles missing -> sets:
| DataLoadTab |  state.original_series
+------+------+
       | original_series
       v
+--------------+
|    Tab 2     |  Box-Cox + differencing -> sets:
| TransformTab |  state.stationary_series
|              |  state.transformations [ordered log]
+------+-------+
       | stationary_series
       v
+--------------+
|    Tab 3     |  periodogram + smoothing -> detect_cycles -> sets:
| SpectralTab  |  state.detected_cycles
+------+-------+
       | stationary_series + detected_cycles
       v
+--------------+
|    Tab 4     |  ACF/PACF, grid search, model fit -> sets:
| ModelingTab  |  state.fitted_model, model_order, model_params, residuals
+------+-------+
       | residuals + model_order
       v
+----------------+
|     Tab 5      |  Ljung-Box, Jarque-Bera, cum. periodogram -> sets:
| ValidationTab  |  state.validation_passed
+------+---------+
       | fitted_model + original_series + transformations
       v
+------------------+
|     Tab 6        |  generates forecasts on transformed scale ->
| ForecastingTab   |  back_transform() reverses transformations log
+------------------+
```

### Cross-axis imports

**Tab 4 (`ModelingTab`) imports `axis1_preprocessing.apply_box_cox`**
`get_series_for_fitting()` re-applies the Box-Cox transform to the original series before handing it to the ARIMA fitter, because `stationary_series` contains NaN leading rows from differencing that would confuse the MLE optimizer.

**Tab 5 (`ValidationTab`) imports `axis4_validation.run_all_diagnostics`**
The residual spectrum flatness check inside validation reuses the smoothed-spectrum machinery.

**Tab 6 (`ForecastingTab`) imports `axis5_forecasting.back_transform`**
`back_transform` internally uses `axis1_preprocessing.apply_box_cox` and `apply_differencing` to replay the transformations log in reverse.

**Why these cross-axis imports are fine:** Every function imported across axis boundaries is a *pure function* — it takes inputs, returns outputs, touches no global state, and has no Qt dependencies. The dependency graph is acyclic. There is no circular import risk. Treating them as a shared math utility layer is the correct abstraction; duplicating `apply_box_cox` inside axis5 would violate DRY and create a maintenance burden.

---

## 4. The GUI Layer (`main.py` overview)

### Tab class names and layout

All tab classes extend `BaseTab(QWidget)`, which establishes a 30/70 split layout:
- **Left 30%** — `QFrame#controlPanel` with controls, labels, spinboxes, buttons
- **Right 70%** — one or more `PlotWidget` instances plus any result tables

| Index | Class | Tab label |
|---|---|---|
| 0 | `DataLoadTab` | "1. Data Load & Explore" |
| 1 | `TransformTab` | "2. Transform to Stationarity" |
| 2 | `SpectralTab` | "3. Spectral Exploration & Cycle Detection" |
| 3 | `ModelingTab` | "4. Model Identification & Selection" |
| 4 | `ValidationTab` | "5. Model Validation & Residual Diagnostics" |
| 5 | `ForecastingTab` | "6. Generate & View Forecasts" |

All six tabs are stored in `self.tabs = [...]` on `MainWindow` and registered with `self.tab_widget` (a `QTabWidget`).

### How state is passed to each tab

Each tab receives `main_window` (the `MainWindow` instance) in its constructor:

```python
# ModelingTab.__init__
def __init__(self, main_window, parent=None):
    self.main_window = main_window
    ...
    state = self.main_window.state  # accessed whenever needed
```

There are no Qt signals carrying data between tabs. All inter-tab data sharing goes through `self.main_window.state`. This is intentional — see [Section 6](#6-design-decisions).

### Workflow enforcement: `update_ui_from_state()`

`MainWindow.update_ui_from_state()` is called at the end of every significant action (file load, transform applied, model selected, diagnostics run). It re-evaluates the entire state and enforces tab gating:

```python
def update_ui_from_state(self):
    tab2_enabled = self.state.data_loaded       # original_series is not None
    tab3_enabled = self.state.stationarity_done # stationary_series is not None
    tab4_enabled = self.state.stationarity_done
    tab5_enabled = self.state.model_fitted      # fitted_model is not None
    tab6_enabled = self.state.model_fitted      # enabled for any fitted model; warning shown if not validated

    self.tab_widget.setTabEnabled(1, tab2_enabled)
    self.tab_widget.setTabEnabled(2, tab3_enabled)
    self.tab_widget.setTabEnabled(3, tab4_enabled)
    self.tab_widget.setTabEnabled(4, tab5_enabled)
    self.tab_widget.setTabEnabled(5, tab6_enabled)
    ...
```

> **Note on Tab 6:** Forecasting is enabled as soon as a model is fitted, not only after validation passes. If the user generates a forecast from a model that failed diagnostics, a `QMessageBox.warning` is shown first. The user can still proceed — this allows inspecting forecasts from an inadequate model (useful for learning) while making the failure unmissable.

### Progress indicators

Each tab has a `StatusIndicator` installed in its tab bar button slot (`tabBar().setTabButton(i, LeftSide, indicator)`). `update_ui_from_state()` drives all six:

```python
# Tab 1 — green once data is loaded
self.indicators[0].set_status('pass' if self.state.data_loaded else 'neutral')

# Tab 5 — three-way: green (pass), red (fail), gray (not yet run)
if self.state.model_fitted:
    if self.state.validation_run:
        status = 'pass' if self.state.validation_passed else 'fail'
    else:
        status = 'neutral'
else:
    status = 'neutral'
self.indicators[4].set_status(status)
```

### Thread safety: `GridSearchWorker` QThread pattern

ARIMA grid search can take tens of seconds for large (p, q, P, Q) ranges. Running it on the main thread would freeze Qt's event loop. `GridSearchWorker` (defined in `axis3_modeling.py`) is a `QThread` subclass:

```python
class GridSearchWorker(QThread):
    progress = Signal(int)           # 0-100, drives QProgressBar
    finished = Signal(pd.DataFrame)  # emitted on completion (or failure -> empty DataFrame)

    def run(self):
        # Runs in the worker thread — no Qt widgets touched here
        df_results = grid_search(
            self.series, self.max_p, self.max_q, self.max_P, self.max_Q,
            self.s, self.d, self.D, progress_callback=lambda v: self.progress.emit(v)
        )
        self.finished.emit(df_results)
```

Usage in `ModelingTab.run_grid_search()`:

```python
self.worker = GridSearchWorker(series=series, max_p=..., ...)
self.worker.progress.connect(self.progress_bar.setValue)  # safe cross-thread signal
self.worker.finished.connect(self.on_grid_search_finished)
self.worker.start()
# Controls disabled while running; re-enabled in on_grid_search_finished
```

Qt's signal/slot mechanism uses a queued connection for cross-thread signals, so `progress.emit(int)` from the worker thread safely updates the progress bar on the main thread. No explicit mutex needed.

On application close, `MainWindow.closeEvent()` calls `worker.terminate(); worker.wait()` to avoid a dangling thread.

---

## 5. Reusable Widgets (`widgets.py`)

### `QT_API` environment variable

```python
# widgets.py, line 2 — must be set before ANY matplotlib/Qt import
os.environ["QT_API"] = "pyside6"
```

Matplotlib's Qt backend (`backend_qtagg`) uses a compatibility shim that auto-detects which Qt binding is present. Without `QT_API = "pyside6"`, it might pick PyQt5 or PyQt6 if they are installed, causing import conflicts. Setting it at the top of `widgets.py` (which is imported before any matplotlib backend loads) locks in PySide6. `main.py` also sets it at line 3 as a belt-and-braces measure.

### `StatusIndicator(QWidget)`

```python
class StatusIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(16, 16))    # exact pixel budget
        self.color = QColor("#9CA3AF")      # gray = neutral

    def set_status(self, status: str):
        if status == 'pass':   self.color = QColor("#16A34A")   # green
        elif status == 'fail': self.color = QColor("#DC2626")   # red
        else:                  self.color = QColor("#9CA3AF")   # gray
        self.update()   # schedules a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(3, 3, 10, 10)   # centered 10x10 circle in 16x16 box
```

**Why `QPainter` instead of a colored `QLabel`?** A `QLabel` with a background-color stylesheet would inherit the application's global stylesheet rules and look wrong (border-radius, padding, wrong size). `QPainter` gives pixel-perfect control independent of any stylesheet cascade, and produces an anti-aliased professional circle.

### `PlotCanvas(FigureCanvasQTAgg)`

A thin subclass of matplotlib's `FigureCanvasQTAgg` that pre-configures the figure background (`#FFFFFF`) and tick/label colors (`#1F2937`) to match the application's stylesheet. Every `PlotWidget` owns one `PlotCanvas`.

### `PlotWidget(QWidget)` — composite design

```
PlotWidget
├── top_layout (QHBoxLayout)
│   ├── NavigationToolbar2QT  (zoom/pan/home/back/forward)
│   ├── <stretch>
│   └── QPushButton("Export Plot")  -> QFileDialog -> figure.savefig()
└── PlotCanvas
```

**Why composite?** Users need three things for every chart: the chart itself, interactive navigation (zoom/pan), and a reliable export path (the matplotlib default dialog doesn't enforce file extensions or offer PDF). Wrapping all three into one reusable widget means every tab gets these capabilities for free — no boilerplate repeated across six classes.

The export logic forces the correct extension:

```python
def export_plot(self):
    file_path, selected_filter = QFileDialog.getSaveFileName(
        self, "Export Plot", "", "PNG Image (*.png);;PDF Document (*.pdf)"
    )
    if file_path:
        ext = ".pdf" if "pdf" in selected_filter.lower() else ".png"
        if not file_path.lower().endswith(ext):
            file_path += ext
        self.canvas.figure.savefig(file_path, facecolor=..., bbox_inches='tight')
```

---

## 6. Design Decisions

### Why a single `AnalysisState` dataclass instead of signals between tabs?

Signals between tabs would require every tab to know about every other tab's signal interface — tight coupling. With a shared state object, tabs only need to know about `AnalysisState`. Adding a new field (e.g., a user-configured confidence level) means editing one dataclass, not updating signal signatures across multiple files. The state is also intrinsically inspectable: `print(state)` in a test shows everything without running the GUI.

### Why cascading invalidation instead of manual reset buttons?

A "Reset downstream" button is something users forget to click. Forget it once and you have a validated model sitting on top of changed data. The property setter makes stale state structurally impossible: the moment `original_series` is reassigned, every downstream field is cleared. The guard lives in the data model, not in user discipline.

### Why `QThread` for grid search instead of `threading.Thread` or `asyncio`?

- **`threading.Thread`:** Qt widgets are not thread-safe. Calling any Qt method from a raw Python thread risks crashes or deadlocks. `QThread` is Qt's own threading primitive; cross-thread signal delivery via Qt's queued connection mechanism is safe by design.
- **`asyncio`:** Qt's event loop and asyncio's event loop do not mix natively. Additional libraries (`qasync`, etc.) would be required, adding dependencies. More fundamentally, fitting ARIMA models via statsmodels is CPU-bound and blocking — asyncio's cooperative multitasking offers no benefit against a blocking C extension call.

`QThread` is the correct tool: system thread for CPU work, safe cross-thread signals for progress reporting, and proper cleanup via `terminate()/wait()` in `closeEvent`.

### Why module-per-axis structure?

Each analytical stage maps cleanly to one file. This means:
- Tests for `axis2_spectral` require no running Qt application — plain `pytest` works.
- The defense question "show me the spectral estimation code" has one clear answer.
- Two team members can work on `axis3_modeling` and `axis4_validation` simultaneously with no risk of GUI merge conflicts.

The axis modules contain no Qt imports and no global state — they are plain Python backed by pandas/numpy/statsmodels.

### Why is `transformations` a list of dicts, not separate boolean fields?

Separate flags like `did_boxcox: bool`, `did_diff: bool` would force axis5 to reconstruct application order heuristically. The ordered list is self-describing. Axis5's `back_transform` iterates it in reverse:

```python
for step in reversed(state.transformations):
    if step["type"] == "seasonal_diff": ...
    elif step["type"] == "diff": ...
    elif step["type"] == "boxcox": ...
```

This is also serialisation-ready: `json.dumps(state.transformations)` works out of the box, which opens the door to saving/loading analysis sessions.

---

## 7. Likely Defense Questions — Architecture

---

**"Why does changing the original data reset all downstream results?"**

> Because `state.original_series` has a property setter that atomically clears every downstream field. When the user loads a new CSV or re-applies imputation, `DataLoadTab` assigns to `state.original_series`. The setter runs immediately, resetting `stationary_series`, `fitted_model`, `validation_passed`, and everything in between. Then `update_ui_from_state()` is called, re-evaluating tab-enabled flags and status indicators — tabs 2–6 all go back to disabled or neutral. This is not manual bookkeeping; it is enforced by the data model itself.

---

**"How does the application prevent the user from forecasting with an unvalidated model?"**

> Two layers. First, Tab 6 (`ForecastingTab`) is only enabled after `state.model_fitted` is `True`, which requires Tab 4's grid search and model selection to complete — you cannot reach the forecasting tab without a fitted model. Second, if the user clicks "Generate Forecast" and `state.validation_run` is `True` but `state.validation_passed` is `False`, a `QMessageBox.warning` dialog is shown before any forecasting runs: *"This model failed one or more adequacy tests. Forecasts may be unreliable."* The user can still proceed, but the failure is unmissable.

---

**"Why did you use `QThread` instead of Python's `threading.Thread` for the grid search?"**

> ARIMA grid search can block the CPU for 10–30 seconds. Running it on the main thread would freeze Qt's event loop — the window goes unresponsive. We cannot use `threading.Thread` because Qt widgets are not thread-safe: calling `self.progress_bar.setValue()` from a raw Python thread is undefined behaviour in Qt and can crash the application. `QThread` is Qt's own threading primitive. Qt's signal/slot mechanism delivers cross-thread signals via a queued connection — `worker.progress.emit(50)` from the worker thread is posted as an event into the main thread's event queue and handled safely there. No explicit locks or mutexes are needed.

---

**"How does axis5 know how to reverse all the transformations applied in axis1?"**

> `state.transformations` is an ordered list of dicts built incrementally in Tab 2 as each transformation is applied. Each dict records the type and its parameters: `{"type": "boxcox", "lambda": 0.32}`, `{"type": "diff", "d": 1}`, `{"type": "seasonal_diff", "D": 1, "s": 12}`. Axis5's `back_transform` function receives this list and iterates it in *reverse* — undoing seasonal differencing, then ordinary differencing, then Box-Cox, in exactly the right order. The pure functions used (`apply_box_cox`, `apply_differencing`) are imported directly from `axis1_preprocessing`, so no re-implementation is needed.

---

**"Why are cross-axis imports (axis5 importing axis1, axis4 reusing axis2 functions) acceptable here?"**

> Because every function imported across axis boundaries is a pure function — it takes inputs, returns outputs, touches no global state, and has no Qt dependencies. The dependency graph is acyclic: axis5 imports from axis1, not the reverse. There is no circular import risk. Treating them as a shared math utility layer is the correct abstraction; the alternative — duplicating `apply_box_cox` inside axis5 — would violate DRY and create a divergence maintenance burden.

---

**"Explain the `StatusIndicator` widget and how tabs show their completion state."**

> `StatusIndicator` is a 16×16 `QWidget` that uses `QPainter` to draw an anti-aliased filled circle. Calling `set_status('pass')` sets the fill to green (`#16A34A`), `'fail'` to red (`#DC2626`), and `'neutral'` to gray (`#9CA3AF`), then calls `self.update()` to schedule a repaint. Six indicators are created in `MainWindow.__init__`, one per tab, and installed into the tab bar via `tabBar().setTabButton(i, LeftSide, indicator)`.
>
> Every time anything significant happens, `update_ui_from_state()` is called and it drives all six indicators from the current state. Tab 5's indicator is the most nuanced: it shows green only if `validation_run == True` AND `validation_passed == True`; red if diagnostics ran but failed; gray otherwise. This gives the team an at-a-glance pipeline health readout without any extra UI clutter.

---

*Last updated: 2026-06-22. Written against the committed source in `src/`.*
