import sys
import os
os.environ["QT_API"] = "pyside6"
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QTextBrowser,
    QProgressBar, QTabBar
)
from PySide6.QtCore import Qt
from state import AnalysisState
from widgets import StatusIndicator, PlotWidget

STYLESHEET = """
QMainWindow {
    background-color: #FFFFFF;
}

QWidget {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #1F2937;
}

/* Tab Widget styling */
QTabWidget::pane {
    border: 1px solid #D1D5DB;
    background-color: #FFFFFF;
    top: -1px;
}

QTabBar::tab {
    background-color: #F5F6F8;
    border: 1px solid #D1D5DB;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 16px;
    margin-right: 2px;
    color: #6B7280;
}

QTabBar::tab:selected {
    background-color: #FFFFFF;
    border-color: #D1D5DB;
    border-bottom: 2px solid #2563EB;
    color: #2563EB;
    font-weight: bold;
}

QTabBar::tab:disabled {
    color: #9CA3AF;
    background-color: #F5F6F8;
    border-color: #E5E7EB;
}

/* Control Panel Frame */
QFrame#controlPanel {
    background-color: #F5F6F8;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
}

/* Headings */
QLabel#heading {
    font-size: 15px;
    font-weight: bold;
    color: #1F2937;
    margin-bottom: 8px;
}

QLabel#sectionHeading {
    font-size: 13px;
    font-weight: bold;
    color: #1F2937;
    margin-top: 10px;
    margin-bottom: 4px;
}

QLabel#caption {
    font-size: 11px;
    color: #6B7280;
}

/* Buttons */
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 500;
    color: #1F2937;
}

QPushButton:hover {
    background-color: #F5F6F8;
    border-color: #2563EB;
}

QPushButton:pressed {
    background-color: #E5E7EB;
}

QPushButton:disabled {
    color: #9CA3AF;
    background-color: #F5F6F8;
    border-color: #E5E7EB;
}

/* Highlighted/Primary Buttons */
QPushButton#primaryButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #2563EB;
}

QPushButton#primaryButton:hover {
    background-color: #1D4ED8;
}

QPushButton#primaryButton:pressed {
    background-color: #1E40AF;
}

QPushButton#primaryButton:disabled {
    background-color: #9CA3AF;
    border-color: #9CA3AF;
}

/* Inputs and Views */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    padding: 4px 8px;
    color: #1F2937;
}

QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border-color: #2563EB;
}

QTableView, QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    gridline-color: #E5E7EB;
}

QHeaderView::section {
    background-color: #F5F6F8;
    color: #1F2937;
    padding: 6px;
    border: 1px solid #D1D5DB;
    font-weight: bold;
}

QTextBrowser {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    padding: 8px;
    color: #1F2937;
}
"""

class BaseTab(QWidget):
    """
    Standard layout structure for each tab: 30/70 split.
    - Left 30%: Control panel inside a QFrame (controlPanel).
    - Right 70%: Plot panel with a PlotWidget.
    """
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(12)
        
        # Left Panel (Controls)
        self.control_panel = QFrame(self)
        self.control_panel.setObjectName("controlPanel")
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_layout.setContentsMargins(12, 12, 12, 12)
        self.control_layout.setSpacing(8)
        
        # Standard Title
        self.heading = QLabel(title, self.control_panel)
        self.heading.setObjectName("heading")
        self.control_layout.addWidget(self.heading)
        
        # Add Left Panel to Main Layout with stretch factor 3 (30%)
        self.layout.addWidget(self.control_panel, 3)
        
        # Right Panel (Container)
        self.right_container = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.right_container, 7)


class DataLoadTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Data Load & Explore", parent)
        self.main_window = main_window
        
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        # Add placeholders for Data Load Controls
        self.control_layout.addWidget(QLabel("Select Time Series CSV:", self.control_panel))
        self.load_btn = QPushButton("Browse CSV File", self.control_panel)
        self.control_layout.addWidget(self.load_btn)
        
        self.control_layout.addWidget(QLabel("Time Column:", self.control_panel))
        self.time_col_combo = QComboBox(self.control_panel)
        self.time_col_combo.addItem("Select Column...")
        self.control_layout.addWidget(self.time_col_combo)
        
        self.control_layout.addWidget(QLabel("Value Column:", self.control_panel))
        self.val_col_combo = QComboBox(self.control_panel)
        self.val_col_combo.addItem("Select Column...")
        self.control_layout.addWidget(self.val_col_combo)
        
        self.control_layout.addWidget(QLabel("Missing Values Imputation:", self.control_panel))
        self.impute_combo = QComboBox(self.control_panel)
        self.impute_combo.addItems(["Forward Fill", "Linear Interpolation", "Mean Imputation"])
        self.control_layout.addWidget(self.impute_combo)
        
        self.control_layout.addWidget(QLabel("Rolling Window size:", self.control_panel))
        self.roll_spin = QSpinBox(self.control_panel)
        self.roll_spin.setRange(2, 365)
        self.roll_spin.setValue(12)
        self.control_layout.addWidget(self.roll_spin)
        
        self.adf_btn = QPushButton("Run ADF Test", self.control_panel)
        self.control_layout.addWidget(self.adf_btn)
        
        self.control_layout.addStretch()
        
        # Temporary simulation button for Task 0
        self.sim_btn = QPushButton("[Simulate Data Load]", self.control_panel)
        self.sim_btn.setObjectName("primaryButton")
        self.sim_btn.clicked.connect(self.simulate_load)
        self.control_layout.addWidget(self.sim_btn)
        
    def simulate_load(self):
        # Simulate loading data
        self.main_window.state.original_series = pd.Series(np.random.randn(100))
        self.main_window.state.time_index = pd.Index(range(100))
        self.main_window.update_ui_from_state()


class TransformTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Transform to Stationarity", parent)
        self.main_window = main_window
        
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        # Add placeholders for Transformation Controls
        self.control_layout.addWidget(QLabel("Box-Cox Lambda:", self.control_panel))
        self.lambda_spin = QDoubleSpinBox(self.control_panel)
        self.lambda_spin.setRange(-2.0, 2.0)
        self.lambda_spin.setSingleStep(0.1)
        self.lambda_spin.setValue(1.0)
        self.control_layout.addWidget(self.lambda_spin)
        
        self.auto_lambda_btn = QPushButton("Auto-Optimize Lambda", self.control_panel)
        self.control_layout.addWidget(self.auto_lambda_btn)
        
        self.control_layout.addWidget(QLabel("Differencing Orders:", self.control_panel))
        
        # Diff parameters layout
        diff_grid = QHBoxLayout()
        self.d_spin = QSpinBox(self.control_panel)
        self.d_spin.setRange(0, 3)
        self.d_spin.setValue(0)
        diff_grid.addWidget(QLabel("d:", self.control_panel))
        diff_grid.addWidget(self.d_spin)
        
        self.D_spin = QSpinBox(self.control_panel)
        self.D_spin.setRange(0, 2)
        self.D_spin.setValue(0)
        diff_grid.addWidget(QLabel("D:", self.control_panel))
        diff_grid.addWidget(self.D_spin)
        
        self.s_spin = QSpinBox(self.control_panel)
        self.s_spin.setRange(1, 52)
        self.s_spin.setValue(1)
        diff_grid.addWidget(QLabel("s:", self.control_panel))
        diff_grid.addWidget(self.s_spin)
        self.control_layout.addLayout(diff_grid)
        
        self.apply_btn = QPushButton("Apply Transformations", self.control_panel)
        self.control_layout.addWidget(self.apply_btn)
        
        self.control_layout.addStretch()
        
        # Temporary simulation button for Task 0
        self.sim_btn = QPushButton("[Simulate Stationarity]", self.control_panel)
        self.sim_btn.setObjectName("primaryButton")
        self.sim_btn.clicked.connect(self.simulate_stationarity)
        self.control_layout.addWidget(self.sim_btn)
        
    def simulate_stationarity(self):
        # Simulate stationarity processing
        self.main_window.state.stationary_series = pd.Series(np.random.randn(100))
        self.main_window.update_ui_from_state()


class SpectralTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Spectral Exploration & Cycle Detection", parent)
        self.main_window = main_window
        
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        # Add placeholders for Spectral Controls
        self.control_layout.addWidget(QLabel("Data Taper:", self.control_panel))
        self.taper_combo = QComboBox(self.control_panel)
        self.taper_combo.addItems(["None", "Cosine Bell", "Hann", "Hamming"])
        self.control_layout.addWidget(self.taper_combo)
        
        self.taper_pct_spin = QDoubleSpinBox(self.control_panel)
        self.taper_pct_spin.setRange(0.0, 0.5)
        self.taper_pct_spin.setSingleStep(0.05)
        self.taper_pct_spin.setValue(0.1)
        self.control_layout.addWidget(self.taper_pct_spin)
        
        self.control_layout.addWidget(QLabel("Smoothed Estimator Window:", self.control_panel))
        self.window_combo = QComboBox(self.control_panel)
        self.window_combo.addItems(["Daniell", "Bartlett", "Parzen", "Hann"])
        self.control_layout.addWidget(self.window_combo)
        
        self.control_layout.addWidget(QLabel("Bandwidth M:", self.control_panel))
        self.bandwidth_spin = QSpinBox(self.control_panel)
        self.bandwidth_spin.setRange(1, 100)
        self.bandwidth_spin.setValue(10)
        self.control_layout.addWidget(self.bandwidth_spin)
        
        self.estimate_btn = QPushButton("Estimate Spectrum", self.control_panel)
        self.control_layout.addWidget(self.estimate_btn)
        
        self.detect_btn = QPushButton("Detect Cycles", self.control_panel)
        self.control_layout.addWidget(self.detect_btn)
        
        self.control_layout.addStretch()
        
        # Temporary simulation button for Task 0
        self.sim_btn = QPushButton("[Simulate Cycle Detection]", self.control_panel)
        self.sim_btn.setObjectName("primaryButton")
        self.sim_btn.clicked.connect(self.simulate_cycles)
        self.control_layout.addWidget(self.sim_btn)
        
    def simulate_cycles(self):
        # Simulate cycle detection results
        self.main_window.state.detected_cycles = [{"frequency": 0.083, "period": 12.0}]
        self.main_window.update_ui_from_state()


class ModelingTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Model Identification & Selection", parent)
        self.main_window = main_window
        
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        # Add placeholders for Modeling Controls
        self.control_layout.addWidget(QLabel("ACF/PACF Max Lag:", self.control_panel))
        self.lag_spin = QSpinBox(self.control_panel)
        self.lag_spin.setRange(1, 100)
        self.lag_spin.setValue(20)
        self.control_layout.addWidget(self.lag_spin)
        
        self.plot_acf_btn = QPushButton("Plot ACF/PACF", self.control_panel)
        self.control_layout.addWidget(self.plot_acf_btn)
        
        self.suggest_acf_btn = QPushButton("Suggest Model (ACF/PACF)", self.control_panel)
        self.control_layout.addWidget(self.suggest_acf_btn)
        
        self.suggest_spec_btn = QPushButton("Suggest Model (Spectral)", self.control_panel)
        self.control_layout.addWidget(self.suggest_spec_btn)
        
        self.control_layout.addWidget(QLabel("Grid Search Limits:", self.control_panel))
        self.grid_p_spin = QSpinBox(self.control_panel)
        self.grid_p_spin.setRange(0, 5)
        self.grid_p_spin.setValue(2)
        self.control_layout.addWidget(QLabel("Max p:", self.control_panel))
        self.control_layout.addWidget(self.grid_p_spin)
        
        self.run_grid_btn = QPushButton("Run Grid Search", self.control_panel)
        self.control_layout.addWidget(self.run_grid_btn)
        
        self.progress_bar = QProgressBar(self.control_panel)
        self.progress_bar.setValue(0)
        self.control_layout.addWidget(self.progress_bar)
        
        self.control_layout.addStretch()
        
        # Temporary simulation button for Task 0
        self.sim_btn = QPushButton("[Simulate Model Fit]", self.control_panel)
        self.sim_btn.setObjectName("primaryButton")
        self.sim_btn.clicked.connect(self.simulate_fit)
        self.control_layout.addWidget(self.sim_btn)
        
    def simulate_fit(self):
        # Simulate model fitting
        self.main_window.state.fitted_model = object()  # Dummy object
        self.main_window.state.model_order = (1, 0, 1)
        self.main_window.state.model_params = {"ar": [0.5], "ma": [-0.3], "sigma2": 1.0}
        self.main_window.state.residuals = np.random.randn(100)
        self.main_window.update_ui_from_state()


class ValidationTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Model Validation & Residual Diagnostics", parent)
        self.main_window = main_window
        
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        # Add placeholders for Validation Controls
        self.control_layout.addWidget(QLabel("Ljung-Box Lags:", self.control_panel))
        self.lb_lag_spin = QSpinBox(self.control_panel)
        self.lb_lag_spin.setRange(1, 40)
        self.lb_lag_spin.setValue(20)
        self.control_layout.addWidget(self.lb_lag_spin)
        
        self.run_tests_btn = QPushButton("Run Diagnostics", self.control_panel)
        self.control_layout.addWidget(self.run_tests_btn)
        
        self.verdict_label = QLabel("Verdict: NOT COMPLETED", self.control_panel)
        self.verdict_label.setObjectName("sectionHeading")
        self.control_layout.addWidget(self.verdict_label)
        
        self.control_layout.addStretch()
        
        # Temporary simulation buttons for Task 0 (Pass/Fail)
        self.sim_pass_btn = QPushButton("[Simulate Pass Verdict]", self.control_panel)
        self.sim_pass_btn.setObjectName("primaryButton")
        self.sim_pass_btn.clicked.connect(self.simulate_pass)
        self.control_layout.addWidget(self.sim_pass_btn)
        
        self.sim_fail_btn = QPushButton("[Simulate Fail Verdict]", self.control_panel)
        self.sim_fail_btn.clicked.connect(self.simulate_fail)
        self.control_layout.addWidget(self.sim_fail_btn)
        
    def simulate_pass(self):
        # Simulate validation passing
        self.main_window.state.validation_passed = True
        self.main_window.update_ui_from_state()
        self.verdict_label.setText("Verdict: PASS")
        
    def simulate_fail(self):
        # Simulate validation failing
        self.main_window.state.validation_passed = False
        self.main_window.update_ui_from_state()
        self.verdict_label.setText("Verdict: FAIL")


class ForecastingTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Generate & View Forecasts", parent)
        self.main_window = main_window
        
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        # Add placeholders for Forecasting Controls
        self.control_layout.addWidget(QLabel("Forecast Horizon h:", self.control_panel))
        self.horizon_spin = QSpinBox(self.control_panel)
        self.horizon_spin.setRange(1, 100)
        self.horizon_spin.setValue(24)
        self.control_layout.addWidget(self.horizon_spin)
        
        self.scale_combo = QComboBox(self.control_panel)
        self.scale_combo.addItems(["Original Scale", "Transformed Scale"])
        self.control_layout.addWidget(self.scale_combo)
        
        self.generate_btn = QPushButton("Generate Forecast", self.control_panel)
        self.generate_btn.setObjectName("primaryButton")
        self.control_layout.addWidget(self.generate_btn)
        
        self.control_layout.addWidget(QLabel("Spectral Insight:", self.control_panel))
        self.insight_browser = QTextBrowser(self.control_panel)
        self.control_layout.addWidget(self.insight_browser)
        
        self.control_layout.addStretch()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Series Analysis GUI")
        self.resize(1200, 800)
        self.setStyleSheet(STYLESHEET)
        
        # Global analysis state
        self.state = AnalysisState()
        
        # Setup main tab widget
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        
        # Add tabs
        self.tabs = [
            DataLoadTab(self, self.tab_widget),
            TransformTab(self, self.tab_widget),
            SpectralTab(self, self.tab_widget),
            ModelingTab(self, self.tab_widget),
            ValidationTab(self, self.tab_widget),
            ForecastingTab(self, self.tab_widget)
        ]
        
        tab_names = [
            "1. Data Load & Explore",
            "2. Transform to Stationarity",
            "3. Spectral Exploration & Cycle Detection",
            "4. Model Identification & Selection",
            "5. Model Validation & Residual Diagnostics",
            "6. Generate & View Forecasts"
        ]
        
        # Create indicators and add tabs
        self.indicators = []
        for i, (tab, name) in enumerate(zip(self.tabs, tab_names)):
            self.tab_widget.addTab(tab, name)
            
            # Setup custom status indicator
            indicator = StatusIndicator(self.tab_widget)
            self.indicators.append(indicator)
            self.tab_widget.tabBar().setTabButton(
                i, 
                QTabBar.ButtonPosition.LeftSide, 
                indicator
            )
            
        # Initial UI Update
        self.update_ui_from_state()

    def update_ui_from_state(self):
        # Determine tab enabled states based on gating rules
        tab2_enabled = self.state.data_loaded
        tab3_enabled = self.state.stationarity_done
        tab4_enabled = self.state.stationarity_done
        tab5_enabled = self.state.model_fitted
        tab6_enabled = self.state.validation_passed
        
        # Set tab enabled states (Tab 1 is index 0, always enabled)
        self.tab_widget.setTabEnabled(1, tab2_enabled)
        self.tab_widget.setTabEnabled(2, tab3_enabled)
        self.tab_widget.setTabEnabled(3, tab4_enabled)
        self.tab_widget.setTabEnabled(4, tab5_enabled)
        self.tab_widget.setTabEnabled(5, tab6_enabled)
        
        # Update Status Indicators
        # Tab 1: Data Loaded
        if self.state.data_loaded:
            self.indicators[0].set_status('pass')
        else:
            self.indicators[0].set_status('neutral')
            
        # Tab 2: Stationarity done
        if self.state.stationarity_done:
            self.indicators[1].set_status('pass')
        else:
            self.indicators[1].set_status('neutral')
            
        # Tab 3: Cycles detected (if any)
        if self.state.detected_cycles:
            self.indicators[2].set_status('pass')
        else:
            self.indicators[2].set_status('neutral')
            
        # Tab 4: Model fitted
        if self.state.model_fitted:
            self.indicators[3].set_status('pass')
        else:
            self.indicators[3].set_status('neutral')
            
        # Tab 5: Validation passed / failed
        if self.state.model_fitted:
            if self.state.validation_run:
                if self.state.validation_passed:
                    self.indicators[4].set_status('pass')
                else:
                    self.indicators[4].set_status('fail')
            else:
                self.indicators[4].set_status('neutral')
        else:
            self.indicators[4].set_status('neutral')
            
        # Tab 6: Forecasts enabled/completed
        if self.state.validation_passed:
            self.indicators[5].set_status('pass')
        else:
            self.indicators[5].set_status('neutral')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
