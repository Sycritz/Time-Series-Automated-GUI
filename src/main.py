import sys
import os
os.environ["QT_API"] = "pyside6"
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QTextBrowser,
    QProgressBar, QTabBar, QTableView, QMessageBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QSplitter, QGridLayout
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
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


class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=QModelIndex()):
        return min(len(self._df), 10)

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iloc[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(self._df.index[section])
        return None


class DataLoadTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Data Load & Explore", parent)
        self.main_window = main_window
        self.df = None
        self.raw_series = None
        
        # Repopulate right layout: table preview at top, plot below
        table_container = QWidget(self)
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(4)
        
        table_header_layout = QHBoxLayout()
        table_label = QLabel("Data Preview (First 10 Rows)", self)
        table_label.setStyleSheet("font-weight: bold; color: #1F2937;")
        self.export_table_btn = QPushButton("Export Table CSV", self)
        self.export_table_btn.setEnabled(False)
        self.export_table_btn.clicked.connect(self.export_table_csv)
        
        table_header_layout.addWidget(table_label)
        table_header_layout.addStretch()
        table_header_layout.addWidget(self.export_table_btn)
        
        table_layout.addLayout(table_header_layout)
        
        self.preview_table = QTableView(self)
        self.preview_table.setFixedHeight(180)
        table_layout.addWidget(self.preview_table)
        
        self.right_layout.addWidget(table_container)
        
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        # Control panel controls
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
        
        self.apply_impute_btn = QPushButton("Apply Imputation", self.control_panel)
        self.control_layout.addWidget(self.apply_impute_btn)
        
        # Warning label for high imputation rates
        self.warning_label = QLabel("", self.control_panel)
        self.warning_label.setStyleSheet("color: #DC2626; font-weight: bold;")
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()
        self.control_layout.addWidget(self.warning_label)
        
        self.control_layout.addWidget(QLabel("Rolling Window size:", self.control_panel))
        self.roll_spin = QSpinBox(self.control_panel)
        self.roll_spin.setRange(2, 365)
        self.roll_spin.setValue(12)
        self.control_layout.addWidget(self.roll_spin)
        
        self.adf_btn = QPushButton("Run ADF Test", self.control_panel)
        self.control_layout.addWidget(self.adf_btn)
        
        self.control_layout.addWidget(QLabel("ADF Test Results:", self.control_panel))
        self.adf_results_browser = QTextBrowser(self.control_panel)
        self.adf_results_browser.setFontFamily("Courier New")
        self.adf_results_browser.setMinimumHeight(150)
        self.control_layout.addWidget(self.adf_results_browser)
        
        self.control_layout.addStretch()
        
        # Connect signals
        self.load_btn.clicked.connect(self.browse_file)
        self.time_col_combo.currentIndexChanged.connect(self.on_columns_selected)
        self.val_col_combo.currentIndexChanged.connect(self.on_columns_selected)
        self.apply_impute_btn.clicked.connect(self.impute_data)
        self.roll_spin.valueChanged.connect(self.plot_data)
        self.adf_btn.clicked.connect(self.run_adf)
        
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Time Series File", "", "CSV Files (*.csv);;TXT Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                from axis1_preprocessing import load_csv
                self.df = load_csv(file_path)
                
                # Clear previous state
                self.raw_series = None
                self.main_window.state.original_series = None
                
                # Clear Tab 1 plot and ADF
                self.plot_widget.canvas.figure.clear()
                self.plot_widget.canvas.draw()
                self.adf_results_browser.clear()
                
                # Clear Tab 2 plots and ADF
                tab2 = self.main_window.tabs[1]
                tab2.plot_widget.canvas.figure.clear()
                tab2.plot_widget.canvas.draw()
                tab2.freq_plot_widget.canvas.figure.clear()
                tab2.freq_plot_widget.canvas.draw()
                tab2.adf_results_browser.clear()
                
                # Update dropdown columns
                self.time_col_combo.blockSignals(True)
                self.val_col_combo.blockSignals(True)
                
                self.time_col_combo.clear()
                self.time_col_combo.addItem("Select Column...")
                self.time_col_combo.addItems(list(self.df.columns))
                
                self.val_col_combo.clear()
                self.val_col_combo.addItem("Select Column...")
                self.val_col_combo.addItems(list(self.df.columns))
                
                self.time_col_combo.blockSignals(False)
                self.val_col_combo.blockSignals(False)
                
                # Set table model preview
                model = PandasModel(self.df)
                self.preview_table.setModel(model)
                self.export_table_btn.setEnabled(True)
                
                # Update main UI to enforce tab gating
                self.main_window.update_ui_from_state()
                
            except Exception as e:
                QMessageBox.critical(self, "Error Loading File", str(e))

    def export_table_csv(self):
        if self.df is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Table to CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            try:
                self.df.to_csv(file_path, index=False)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export table: {str(e)}")
                
    def on_columns_selected(self):
        time_col = self.time_col_combo.currentText()
        val_col = self.val_col_combo.currentText()
        
        if time_col == "Select Column..." or val_col == "Select Column...":
            return
            
        if time_col == val_col:
            QMessageBox.critical(self, "Invalid Selection", "Time Column and Value Column must be different.")
            return
            
        try:
            # Parse time index
            try:
                time_idx = pd.to_datetime(self.df[time_col])
            except Exception:
                time_idx = pd.Index(range(len(self.df)))
                
            # Values Series
            raw_series = pd.to_numeric(self.df[val_col], errors='coerce')
            raw_series.index = time_idx
            raw_series.name = val_col
            self.raw_series = raw_series
            
            # Impute and plot automatically
            self.impute_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Error Extracting Series", str(e))
            
    def impute_data(self):
        if self.raw_series is None:
            QMessageBox.warning(self, "No Data Selected", "Please select Time and Value columns first.")
            return
            
        try:
            from axis1_preprocessing import handle_missing
            method = self.impute_combo.currentText()
            imputed_series, pct_missing = handle_missing(self.raw_series, method)
            
            # Update state
            self.main_window.state.original_series = imputed_series
            self.main_window.state.time_index = imputed_series.index
            
            # Show red warning label if >5% missing/imputed
            if pct_missing > 5.0:
                self.warning_label.setText(f"Warning: {pct_missing:.1f}% missing values imputed via {method}.")
                self.warning_label.show()
            else:
                self.warning_label.hide()
                
            self.plot_data()
            self.main_window.update_ui_from_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Imputation Error", str(e))
            
    def plot_data(self):
        series = self.main_window.state.original_series
        if series is None:
            return
            
        window_size = self.roll_spin.value()
        if window_size >= len(series):
            window_size = max(2, len(series) - 1)
            
        # Draw plot
        fig = self.plot_widget.canvas.figure
        fig.clear()
        
        # Two subplots side-by-side
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Configure style
        for ax in [ax1, ax2]:
            ax.tick_params(colors='#1F2937')
            ax.xaxis.label.set_color('#1F2937')
            ax.yaxis.label.set_color('#1F2937')
            ax.title.set_color('#1F2937')
            
        # Subplot 1: Series Values & Rolling Mean
        ax1.plot(series.index, series.values, color='#1F2937', label='Series Values', alpha=0.8)
        from axis1_preprocessing import rolling_mean, rolling_std
        rolling_mean_vals = rolling_mean(series, window_size)
        ax1.plot(series.index, rolling_mean_vals, color='#2563EB', label=f'Rolling Mean ({window_size})')
        ax1.set_ylabel('Values')
        ax1.set_xlabel(series.index.name if series.index.name else 'Time')
        ax1.legend(loc='best')
        ax1.set_title("Rolling Mean")
        
        # Subplot 2: Rolling Std
        rolling_std_vals = rolling_std(series, window_size)
        ax2.plot(series.index, rolling_std_vals, color='#16A34A', label=f'Rolling Std ({window_size})', linestyle='--')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_xlabel(series.index.name if series.index.name else 'Time')
        ax2.legend(loc='best')
        ax2.set_title("Rolling Standard Deviation")
        
        fig.tight_layout()
        self.plot_widget.canvas.draw()
        
    def run_adf(self):
        series = self.main_window.state.original_series
        if series is None:
            QMessageBox.warning(self, "No Data Selected", "Please select Time and Value columns first.")
            return
            
        try:
            from axis1_preprocessing import run_adf_test
            res = run_adf_test(series)
            
            crit_str = "\n".join([f"  {k}: {v:.4f}" for k, v in res['critical_values'].items()])
            english_verdict = f"Data is {'stationary' if res['p_value'] < 0.05 else 'non-stationary'} at the 5% significance level"
            out = (
                f"Augmented Dickey-Fuller (ADF) Test:\n"
                f"-----------------------------------\n"
                f"ADF Statistic: {res['adf_stat']:.4f}\n"
                f"p-value:       {res['p_value']:.6f}\n"
                f"Verdict:       {res['verdict']} ({english_verdict})\n\n"
                f"Critical Values:\n"
                f"{crit_str}\n"
            )
            self.adf_results_browser.setText(out)
        except Exception as e:
            QMessageBox.critical(self, "ADF Test Error", str(e))


class TransformTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Transform to Stationarity", parent)
        self.main_window = main_window
        
        # Re-populate right layout: transformed series plot and frequency response plot
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        self.freq_plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.freq_plot_widget)
        
        # Control panel controls
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
        
        self.control_layout.addWidget(QLabel("ADF Test (Transformed):", self.control_panel))
        self.adf_results_browser = QTextBrowser(self.control_panel)
        self.adf_results_browser.setFontFamily("Courier New")
        self.adf_results_browser.setMinimumHeight(150)
        self.control_layout.addWidget(self.adf_results_browser)
        
        self.control_layout.addStretch()
        
        # Connect signals
        self.auto_lambda_btn.clicked.connect(self.auto_optimize_lambda)
        self.apply_btn.clicked.connect(self.apply_transformations)
        self.d_spin.valueChanged.connect(self.plot_frequency_response)
        self.D_spin.valueChanged.connect(self.plot_frequency_response)
        self.s_spin.valueChanged.connect(self.plot_frequency_response)
        
    def auto_optimize_lambda(self):
        original_series = self.main_window.state.original_series
        if original_series is None:
            QMessageBox.warning(self, "No Data Loaded", "Please load a dataset on Tab 1 first.")
            return
            
        try:
            from axis1_preprocessing import apply_box_cox
            _, opt_lam = apply_box_cox(original_series, None)
            self.lambda_spin.setValue(opt_lam)
        except Exception as e:
            QMessageBox.critical(self, "Optimization Error", str(e))
            
    def apply_transformations(self):
        state = self.main_window.state
        original_series = state.original_series
        if original_series is None:
            QMessageBox.warning(self, "No Data Loaded", "Please load a dataset on Tab 1 first.")
            return
            
        try:
            from axis1_preprocessing import apply_box_cox, apply_differencing, run_adf_test
            
            # Reset transformations list
            state.transformations = []
            
            # 1. Box-Cox
            lam = self.lambda_spin.value()
            if abs(lam - 1.0) > 1e-7:
                transformed_series, applied_lam = apply_box_cox(original_series, lam)
                state.box_cox_lambda = applied_lam
                state.transformations.append({"type": "boxcox", "lambda": applied_lam})
            else:
                transformed_series = original_series.copy()
                state.box_cox_lambda = None
                
            # 2. Differencing
            d = self.d_spin.value()
            D = self.D_spin.value()
            s = self.s_spin.value()
            
            if d > 0 or D > 0:
                transformed_series = apply_differencing(transformed_series, d, D, s)
                if d > 0:
                    state.transformations.append({"type": "diff", "d": d})
                if D > 0:
                    state.transformations.append({"type": "seasonal_diff", "D": D, "s": s})
                    
            state.diff_order_d = d
            state.diff_order_D = D
            state.seasonal_period_s = s
            
            # Update state with stationary series
            state.stationary_series = transformed_series
            
            # Plot
            self.plot_transformed()
            self.plot_frequency_response()
            
            # Run ADF on stationary series
            try:
                res = run_adf_test(transformed_series)
                crit_str = "\n".join([f"  {k}: {v:.4f}" for k, v in res['critical_values'].items()])
                english_verdict = f"Data is {'stationary' if res['p_value'] < 0.05 else 'non-stationary'} at the 5% significance level"
                out = (
                    f"ADF Test (Transformed):\n"
                    f"-----------------------\n"
                    f"ADF Statistic: {res['adf_stat']:.4f}\n"
                    f"p-value:       {res['p_value']:.6f}\n"
                    f"Verdict:       {res['verdict']} ({english_verdict})\n\n"
                    f"Critical Values:\n"
                    f"{crit_str}\n"
                )
                self.adf_results_browser.setText(out)
            except Exception as adf_err:
                self.adf_results_browser.setText(f"ADF Test could not be run: {adf_err}")
                
            # Enable downstream tabs
            self.main_window.update_ui_from_state()
            
        except Exception as e:
            QMessageBox.critical(self, "Transformation Error", str(e))
            
    def plot_transformed(self):
        series = self.main_window.state.stationary_series
        if series is None:
            return
            
        try:
            window_size = self.main_window.tabs[0].roll_spin.value()
        except Exception:
            window_size = 12
            
        fig = self.plot_widget.canvas.figure
        fig.clear()
        
        # Two subplots side-by-side
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        for ax in [ax1, ax2]:
            ax.tick_params(colors='#1F2937')
            ax.xaxis.label.set_color('#1F2937')
            ax.yaxis.label.set_color('#1F2937')
            ax.title.set_color('#1F2937')
            
        # Drop NaNs for plotting rolling statistics
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return
            
        if window_size >= len(clean_series):
            window_size = max(2, len(clean_series) - 1)
            
        ax1.plot(clean_series.index, clean_series.values, color='#1F2937', label='Transformed Series', alpha=0.8)
        from axis1_preprocessing import rolling_mean, rolling_std
        rolling_mean_vals = rolling_mean(clean_series, window_size)
        ax1.plot(clean_series.index, rolling_mean_vals, color='#2563EB', label=f'Rolling Mean ({window_size})')
        ax1.set_ylabel('Transformed Values')
        ax1.set_xlabel(clean_series.index.name if clean_series.index.name else 'Time')
        ax1.legend(loc='best')
        ax1.set_title("Rolling Mean")
        
        rolling_std_vals = rolling_std(clean_series, window_size)
        ax2.plot(clean_series.index, rolling_std_vals, color='#16A34A', label=f'Rolling Std ({window_size})', linestyle='--')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_xlabel(clean_series.index.name if clean_series.index.name else 'Time')
        ax2.legend(loc='best')
        ax2.set_title("Rolling Standard Deviation")
        
        fig.tight_layout()
        self.plot_widget.canvas.draw()
        
    def plot_frequency_response(self):
        d = self.d_spin.value()
        D = self.D_spin.value()
        s = self.s_spin.value()
        
        from axis1_preprocessing import compute_frequency_response
        omega, magnitude = compute_frequency_response(d, D, s)
        
        fig = self.freq_plot_widget.canvas.figure
        fig.clear()
        
        ax = fig.add_subplot(111)
        ax.tick_params(colors='#1F2937')
        ax.xaxis.label.set_color('#1F2937')
        ax.yaxis.label.set_color('#1F2937')
        ax.title.set_color('#1F2937')
        
        ax.plot(omega, magnitude, color='#2563EB', label='Differencing Filter')
        ax.set_title('Frequency Response of Differencing Filter')
        ax.set_xlabel('Frequency (radians/sample)')
        ax.set_ylabel('Magnitude Response |H(e^{-iω})|')
        ax.set_xlim(0, np.pi)
        
        if D > 0 and s > 1:
            w_seas = 2 * np.pi / s
            h = 1
            while h * w_seas <= np.pi:
                ax.axvline(h * w_seas, color='#DC2626', linestyle=':', alpha=0.6, label='Seasonal Harmonics' if h == 1 else "")
                h += 1
                
        ax.legend(loc='best')
        fig.tight_layout()
        self.freq_plot_widget.canvas.draw()


class SpectralTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Spectral Exploration & Cycle Detection", parent)
        self.main_window = main_window
        
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget)
        
        # Layout for cycle results and its export button (added to right layout below the plot)
        results_container = QWidget(self)
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(4)
        
        results_header = QHBoxLayout()
        results_label = QLabel("Cycle Detection Results", self)
        results_label.setStyleSheet("font-weight: bold; color: #1F2937;")
        self.export_cycles_btn = QPushButton("Export Cycles CSV", self)
        self.export_cycles_btn.setEnabled(False)
        self.export_cycles_btn.clicked.connect(self.export_cycles_csv)
        
        results_header.addWidget(results_label)
        results_header.addStretch()
        results_header.addWidget(self.export_cycles_btn)
        results_layout.addLayout(results_header)
        
        self.results_browser = QTextBrowser(self)
        self.results_browser.setMaximumHeight(150)
        results_layout.addWidget(self.results_browser)
        
        self.right_layout.addWidget(results_container)
        
        # Left Panel (Controls)
        self.control_layout.addWidget(QLabel("Data Taper:", self.control_panel))
        self.taper_combo = QComboBox(self.control_panel)
        self.taper_combo.addItems(["None", "Cosine Bell", "Hann", "Hamming"])
        self.control_layout.addWidget(self.taper_combo)
        
        self.taper_pct_spin = QDoubleSpinBox(self.control_panel)
        self.taper_pct_spin.setRange(0.0, 0.5)
        self.taper_pct_spin.setSingleStep(0.05)
        self.taper_pct_spin.setValue(0.1)
        self.control_layout.addWidget(self.taper_pct_spin)
        
        self.control_layout.addWidget(QLabel("Scale:", self.control_panel))
        self.freq_period_toggle = QComboBox(self.control_panel)
        self.freq_period_toggle.addItems(["Frequency Scale", "Period Scale"])
        self.control_layout.addWidget(self.freq_period_toggle)
        
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
        
        # Statistical Warning Label
        self.warning_label = QLabel(
            "The raw periodogram is asymptotically unbiased for the spectral density but is not a consistent estimator. "
            "Its variance does not decrease as sample size n increases. Each ordinate I(λj) behaves approximately like "
            "1/2 f(λj) chi^2_2. Use the smoothed estimator below for reliable inference.",
            self.control_panel
        )
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #DC2626; font-size: 11px;")
        self.control_layout.addWidget(self.warning_label)
        
        self.control_layout.addStretch()
        
        # Instance variables for data storage to enable fast scaling toggling
        self.raw_freqs = None
        self.raw_pgram = None
        self.smooth_freqs = None
        self.smooth_val = None
        self.ci_lower = None
        self.ci_upper = None
        
        # Connect signals
        self.estimate_btn.clicked.connect(self.estimate_spectrum)
        self.detect_btn.clicked.connect(self.detect_cycles_click)
        self.freq_period_toggle.currentIndexChanged.connect(self.plot_spectrum)
        
    def estimate_spectrum(self):
        series = self.main_window.state.stationary_series
        if series is None:
            QMessageBox.warning(self, "No Stationary Data", "Please apply transformations on Tab 2 first.")
            return
            
        clean_series = series.dropna()
        if len(clean_series) < 3:
            QMessageBox.warning(self, "Insufficient Data", "The stationary series must contain at least 3 points.")
            return
            
        taper = self.taper_combo.currentText()
        taper_arg = None if taper == "None" else taper
        taper_pct = self.taper_pct_spin.value()
        
        window = self.window_combo.currentText()
        M = self.bandwidth_spin.value()
        
        try:
            from axis2_spectral import compute_periodogram, smooth_spectrum
            
            raw_freqs, raw_pgram = compute_periodogram(clean_series, taper_arg, taper_pct)
            smooth_freqs, smooth_val, ci_lower, ci_upper = smooth_spectrum(clean_series, window, M)
            
            self.raw_freqs = raw_freqs
            self.raw_pgram = raw_pgram
            self.smooth_freqs = smooth_freqs
            self.smooth_val = smooth_val
            self.ci_lower = ci_lower
            self.ci_upper = ci_upper
            
            # Compute equivalent degrees of freedom nu for detect_cycles
            n = len(clean_series)
            M_eff = min(M, n - 1)
            w_vals = np.zeros(M_eff + 1)
            for h in range(1, M_eff + 1):
                val = h / M
                if window == 'Daniell':
                    w_vals[h] = 1.0
                elif window == 'Bartlett':
                    w_vals[h] = 1.0 - abs(val)
                elif window == 'Parzen':
                    abs_val = abs(val)
                    if abs_val <= 0.5:
                        w_vals[h] = 1.0 - 6.0 * abs_val**2 + 6.0 * abs_val**3
                    else:
                        w_vals[h] = 2.0 * (1.0 - abs_val)**3
                elif window == 'Hann':
                    w_vals[h] = 0.5 * (1.0 + np.cos(np.pi * val))
            sum_w2 = np.sum(w_vals[1:]**2)
            self.nu = (2.0 * n) / (1.0 + 2.0 * sum_w2)
            
            self.plot_spectrum()
        except Exception as e:
            QMessageBox.critical(self, "Estimation Error", str(e))
            
    def detect_cycles_click(self):
        if self.smooth_freqs is None or self.smooth_val is None or self.ci_upper is None:
            QMessageBox.warning(self, "No Spectrum Estimated", "Please click 'Estimate Spectrum' first.")
            return
            
        try:
            from axis2_spectral import detect_cycles
            nu_val = getattr(self, "nu", None)
            cycles = detect_cycles(self.smooth_freqs, self.smooth_val, self.ci_upper, nu_val)
            self.main_window.state.detected_cycles = cycles
            self.export_cycles_btn.setEnabled(bool(cycles))
            
            if not cycles:
                html = "<p>No significant cycles detected.</p>"
            else:
                html = "<h3>Detected Cycles</h3><table border='1' cellpadding='5' style='border-collapse: collapse; width: 100%; border: 1px solid #D1D5DB;'>"
                html += "<tr style='background-color: #F3F4F6;'><th>Frequency (cycles/sample)</th><th>Period (samples)</th><th>Significant</th></tr>"
                for c in cycles:
                    sig_str = "<font color='#16A34A'><b>Yes</b></font>" if c["significant"] else "No"
                    freq_cycles = c["frequency"] / (2.0 * np.pi)
                    html += f"<tr><td>{freq_cycles:.4f}</td><td>{c['period']:.2f}</td><td>{sig_str}</td></tr>"
                html += "</table>"
                
            self.results_browser.setHtml(html)
            self.main_window.update_ui_from_state()
        except Exception as e:
            QMessageBox.critical(self, "Cycle Detection Error", str(e))

    def export_cycles_csv(self):
        cycles = self.main_window.state.detected_cycles
        if not cycles:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Detected Cycles to CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            try:
                df_cycles = pd.DataFrame(cycles)
                # Include frequency in both units for completeness
                df_cycles["frequency_cycles_per_sample"] = df_cycles["frequency"] / (2.0 * np.pi)
                df_cycles.to_csv(file_path, index=False)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export cycles: {str(e)}")
            
    def plot_spectrum(self):
        if self.raw_freqs is None:
            if self.main_window.state.stationary_series is not None:
                self.estimate_spectrum()
                return
            return
            
        fig = self.plot_widget.canvas.figure
        fig.clear()
        
        ax = fig.add_subplot(111)
        self.plot_widget.canvas.axes = ax
        ax.tick_params(colors='#1F2937')
        ax.xaxis.label.set_color('#1F2937')
        ax.yaxis.label.set_color('#1F2937')
        ax.title.set_color('#1F2937')
        
        scale = self.freq_period_toggle.currentText()
        is_period = (scale == "Period Scale")
        
        if is_period:
            valid_raw = self.raw_freqs > 0
            x_raw = 2.0 * np.pi / self.raw_freqs[valid_raw]
            y_raw = self.raw_pgram[valid_raw]
            
            valid_smooth = self.smooth_freqs > 0
            x_smooth = 2.0 * np.pi / self.smooth_freqs[valid_smooth]
            y_smooth = self.smooth_val[valid_smooth]
            y_ci_lower = self.ci_lower[valid_smooth]
            y_ci_upper = self.ci_upper[valid_smooth]
            
            xlabel = "Period (samples)"
            title = "Spectral Density Estimators (Period Scale)"
            
            # Limit the x-axis for period scale to avoid stretching to infinity
            series = self.main_window.state.stationary_series
            max_period = 100.0
            if series is not None:
                max_period = min(100.0, len(series))
            ax.set_xlim(2.0, max_period)
        else:
            x_raw = self.raw_freqs / (2.0 * np.pi)
            y_raw = self.raw_pgram
            
            x_smooth = self.smooth_freqs / (2.0 * np.pi)
            y_smooth = self.smooth_val
            y_ci_lower = self.ci_lower
            y_ci_upper = self.ci_upper
            
            xlabel = "Frequency (cycles/sample)"
            title = "Spectral Density Estimators (Frequency Scale)"
            
        # Plot raw periodogram
        ax.plot(x_raw, y_raw, color='#9CA3AF', alpha=0.6, label='Raw Periodogram', linewidth=1)
        
        # Plot smoothed spectrum
        ax.plot(x_smooth, y_smooth, color='#2563EB', alpha=0.9, label='Smoothed Spectrum', linewidth=2)
        
        # Plot 95% Confidence Interval band
        ax.fill_between(x_smooth, y_ci_lower, y_ci_upper, color='#2563EB', alpha=0.15, label='95% Confidence Band')
        
        # Overlay theoretical spectrum if model is fitted
        self.update_parametric_spectrum_overlay(ax, is_period)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Spectral Density')
        ax.set_title(title)
        ax.legend(loc='best')
        
        fig.tight_layout()
        self.plot_widget.canvas.draw()
        
    def update_parametric_spectrum_overlay(self, ax=None, is_period=False):
        if ax is None:
            # If called as a slot from outside, replot to ensure clean rendering
            self.plot_spectrum()
            return
            
        state = self.main_window.state
        
        # Check preview first, then fall back to fitted model
        params = None
        if getattr(state, 'preview_model_params', None) is not None:
            params = state.preview_model_params
        elif state.fitted_model is not None and state.model_params is not None:
            params = state.model_params
            
        if params is None:
            return
            
        ar_coeffs = params.get("ar", [])
        ma_coeffs = params.get("ma", [])
        sigma2 = params.get("sigma2", 1.0)
        
        from axis2_spectral import compute_parametric_spectrum
        param_freqs, param_spec = compute_parametric_spectrum(ar_coeffs, ma_coeffs, sigma2, n_points=512)
        
        if is_period:
            valid = param_freqs > 0
            x_vals = 2.0 * np.pi / param_freqs[valid]
            y_vals = param_spec[valid]
        else:
            x_vals = param_freqs / (2.0 * np.pi)
            y_vals = param_spec
            
        ax.plot(x_vals, y_vals, color='#D97706', linewidth=2.5, linestyle='--', label='Parametric ARMA Spectrum')


class ModelingTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Model Identification & Selection", parent)
        self.main_window = main_window
        
        self.acf_vals = None
        self.pacf_vals = None
        
        # Upper Right: PlotWidget for ACF/PACF
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget, 5)
        
        # Connect hover event on canvas
        self.plot_widget.canvas.mpl_connect('motion_notify_event', self.on_hover)
        
        # Lower Right: Splitter containing Top-10 Model table and Selected Model parameters
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        
        # Left side of splitter: Top-10 Model Table
        top10_container = QWidget(self)
        top10_layout = QVBoxLayout(top10_container)
        top10_layout.setContentsMargins(0, 0, 0, 0)
        
        top10_header_layout = QHBoxLayout()
        top10_label = QLabel("Top 10 Candidate Models", self)
        top10_label.setStyleSheet("font-weight: bold; color: #1F2937;")
        top10_header_layout.addWidget(top10_label)
        top10_header_layout.addStretch()
        self.export_models_btn = QPushButton("Export CSV", self)
        self.export_models_btn.setEnabled(False)
        top10_header_layout.addWidget(self.export_models_btn)
        top10_layout.addLayout(top10_header_layout)
        
        self.model_table = QTableWidget(self)
        self.model_table.setColumnCount(8)
        self.model_table.setHorizontalHeaderLabels([
            "Rank", "Model", "AICc", "BIC", "LogLik", "k", "Show Spectrum", "Select Model"
        ])
        top10_layout.addWidget(self.model_table)
        splitter.addWidget(top10_container)
        
        # Right side of splitter: Parameter Summary
        param_container = QWidget(self)
        param_layout = QVBoxLayout(param_container)
        param_layout.setContentsMargins(0, 0, 0, 0)
        
        param_header_layout = QHBoxLayout()
        param_label = QLabel("Fitted Model Parameters", self)
        param_label.setStyleSheet("font-weight: bold; color: #1F2937;")
        param_header_layout.addWidget(param_label)
        param_header_layout.addStretch()
        self.export_params_btn = QPushButton("Export CSV", self)
        self.export_params_btn.setEnabled(False)
        param_header_layout.addWidget(self.export_params_btn)
        param_layout.addLayout(param_header_layout)
        
        self.param_table = QTableWidget(self)
        self.param_table.setColumnCount(5)
        self.param_table.setHorizontalHeaderLabels([
            "Parameter", "Estimate", "Std. Error", "z-value", "p-value"
        ])
        param_layout.addWidget(self.param_table)
        
        info_layout = QHBoxLayout()
        self.sigma2_label = QLabel("Estimated σ̂²: N/A", self)
        self.sigma2_label.setStyleSheet("font-weight: 500; color: #1F2937;")
        self.loglik_label = QLabel("Log-Likelihood: N/A", self)
        self.loglik_label.setStyleSheet("font-weight: 500; color: #1F2937;")
        info_layout.addWidget(self.sigma2_label)
        info_layout.addStretch()
        info_layout.addWidget(self.loglik_label)
        param_layout.addLayout(info_layout)
        
        splitter.addWidget(param_container)
        self.right_layout.addWidget(splitter, 5)
        
        # Left Panel (Controls)
        self.control_layout.addWidget(QLabel("ACF/PACF Max Lag:", self.control_panel))
        self.lag_spin = QSpinBox(self.control_panel)
        self.lag_spin.setRange(1, 100)
        self.lag_spin.setValue(20)
        self.control_layout.addWidget(self.lag_spin)
        
        self.plot_acf_btn = QPushButton("Plot ACF/PACF", self.control_panel)
        self.control_layout.addWidget(self.plot_acf_btn)
        
        self.control_layout.addWidget(QLabel("Model Heuristic Suggestions:", self.control_panel))
        heuristic_btn_layout = QHBoxLayout()
        self.suggest_acf_btn = QPushButton("Suggest (ACF/PACF)", self.control_panel)
        self.suggest_spec_btn = QPushButton("Suggest (Spectral)", self.control_panel)
        heuristic_btn_layout.addWidget(self.suggest_acf_btn)
        heuristic_btn_layout.addWidget(self.suggest_spec_btn)
        self.control_layout.addLayout(heuristic_btn_layout)
        
        self.suggestion_browser = QTextBrowser(self.control_panel)
        self.suggestion_browser.setFontFamily("Courier New")
        self.suggestion_browser.setMinimumHeight(150)
        self.control_layout.addWidget(self.suggestion_browser)
        
        self.control_layout.addWidget(QLabel("Grid Search Limits:", self.control_panel))
        grid_layout = QGridLayout()
        
        self.grid_p_spin = QSpinBox(self.control_panel)
        self.grid_p_spin.setRange(0, 5)
        self.grid_p_spin.setValue(2)
        grid_layout.addWidget(QLabel("Max p:", self.control_panel), 0, 0)
        grid_layout.addWidget(self.grid_p_spin, 0, 1)
        
        self.grid_q_spin = QSpinBox(self.control_panel)
        self.grid_q_spin.setRange(0, 5)
        self.grid_q_spin.setValue(2)
        grid_layout.addWidget(QLabel("Max q:", self.control_panel), 0, 2)
        grid_layout.addWidget(self.grid_q_spin, 0, 3)
        
        self.grid_P_spin = QSpinBox(self.control_panel)
        self.grid_P_spin.setRange(0, 2)
        self.grid_P_spin.setValue(1)
        grid_layout.addWidget(QLabel("Max P:", self.control_panel), 1, 0)
        grid_layout.addWidget(self.grid_P_spin, 1, 1)
        
        self.grid_Q_spin = QSpinBox(self.control_panel)
        self.grid_Q_spin.setRange(0, 2)
        self.grid_Q_spin.setValue(1)
        grid_layout.addWidget(QLabel("Max Q:", self.control_panel), 1, 2)
        grid_layout.addWidget(self.grid_Q_spin, 1, 3)
        
        self.grid_s_spin = QSpinBox(self.control_panel)
        self.grid_s_spin.setRange(1, 365)
        self.grid_s_spin.setValue(12)
        grid_layout.addWidget(QLabel("Period s:", self.control_panel), 2, 0)
        grid_layout.addWidget(self.grid_s_spin, 2, 1, 1, 3)
        
        self.control_layout.addLayout(grid_layout)
        
        self.run_grid_btn = QPushButton("Run Grid Search", self.control_panel)
        self.run_grid_btn.setObjectName("primaryButton")
        self.control_layout.addWidget(self.run_grid_btn)
        
        self.progress_bar = QProgressBar(self.control_panel)
        self.progress_bar.setValue(0)
        self.control_layout.addWidget(self.progress_bar)
        
        self.control_layout.addStretch()
        
        # Connect signals
        self.plot_acf_btn.clicked.connect(self.plot_acf_pacf)
        self.suggest_acf_btn.clicked.connect(self.suggest_model_acf_pacf)
        self.suggest_spec_btn.clicked.connect(self.suggest_model_spectral)
        self.run_grid_btn.clicked.connect(self.run_grid_search)
        self.export_models_btn.clicked.connect(self.export_models_csv)
        self.export_params_btn.clicked.connect(self.export_params_csv)
        
    def plot_acf_pacf(self):
        state = self.main_window.state
        series = state.stationary_series
        if series is None:
            QMessageBox.warning(self, "No Data", "Please load and transform the data first.")
            return
            
        nlags = self.lag_spin.value()
        
        from axis3_modeling import compute_acf_pacf
        acf_vals, pacf_vals = compute_acf_pacf(series, nlags)
        
        self.acf_vals = acf_vals
        self.pacf_vals = pacf_vals
        
        fig = self.plot_widget.canvas.figure
        fig.clear()
        
        self.acf_ax = fig.add_subplot(121)
        self.pacf_ax = fig.add_subplot(122)
        
        n = len(series.dropna())
        bound = 1.96 / np.sqrt(n) if n > 0 else 0.2
        
        for ax, title, vals in zip([self.acf_ax, self.pacf_ax], ["ACF", "PACF"], [acf_vals, pacf_vals]):
            ax.tick_params(colors='#1F2937')
            ax.xaxis.label.set_color('#1F2937')
            ax.yaxis.label.set_color('#1F2937')
            ax.title.set_color('#1F2937')
            
            lags = range(len(vals))
            ax.vlines(lags, 0, vals, colors='#2563EB', linewidth=2)
            ax.plot(lags, vals, 'o', color='#2563EB', markersize=4)
            
            ax.axhline(0, color='#1F2937', linewidth=1)
            ax.axhline(bound, color='#2563EB', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(-bound, color='#2563EB', linestyle='--', linewidth=1, alpha=0.7)
            ax.fill_between(lags, -bound, bound, color='#2563EB', alpha=0.1)
            
            ax.set_title(title)
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(-0.5, len(vals) - 0.5)
            
        fig.tight_layout()
        self.plot_widget.canvas.draw()
        
    def on_hover(self, event):
        if getattr(self, 'acf_vals', None) is None or getattr(self, 'pacf_vals', None) is None:
            return
            
        if event.inaxes is None:
            from PySide6.QtWidgets import QToolTip
            QToolTip.hideText()
            return
            
        ax = event.inaxes
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
            
        lag = int(round(x))
        
        if ax == getattr(self, 'acf_ax', None) and 0 <= lag < len(self.acf_vals):
            val = self.acf_vals[lag]
            text = f"ACF\nLag: {lag}\nValue: {val:.4f}"
        elif ax == getattr(self, 'pacf_ax', None) and 0 <= lag < len(self.pacf_vals):
            val = self.pacf_vals[lag]
            text = f"PACF\nLag: {lag}\nValue: {val:.4f}"
        else:
            from PySide6.QtWidgets import QToolTip
            QToolTip.hideText()
            return
            
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QToolTip
        QToolTip.showText(QCursor.pos(), text, self.plot_widget)
        
    def suggest_model_acf_pacf(self):
        state = self.main_window.state
        series = state.stationary_series
        if series is None:
            QMessageBox.warning(self, "No Data", "Please load and transform the data first.")
            return
            
        if getattr(self, 'acf_vals', None) is None:
            self.plot_acf_pacf()
            
        if getattr(self, 'acf_vals', None) is None:
            return
            
        from axis3_modeling import suggest_model_from_acf_pacf
        n = len(series.dropna())
        res = suggest_model_from_acf_pacf(self.acf_vals, self.pacf_vals, n)
        self.suggestion_browser.setText(res["explanation"])
        
    def suggest_model_spectral(self):
        state = self.main_window.state
        if not state.stationarity_done:
            QMessageBox.warning(self, "No Data", "Please load and transform the data first.")
            return
            
        from axis3_modeling import suggest_model_from_spectrum
        res = suggest_model_from_spectrum(state.detected_cycles, state.seasonal_period_s)
        self.suggestion_browser.setText(res["explanation"])
        
    def get_series_for_fitting(self):
        state = self.main_window.state
        if state.original_series is None:
            return None
        if state.box_cox_lambda is not None:
            from axis1_preprocessing import apply_box_cox
            transformed, _ = apply_box_cox(state.original_series, state.box_cox_lambda)
            return transformed
        return state.original_series
        
    def run_grid_search(self):
        state = self.main_window.state
        if state.stationary_series is None:
            QMessageBox.warning(self, "No Data", "Please load and transform the data first.")
            return
            
        series = self.get_series_for_fitting()
        if series is None:
            QMessageBox.warning(self, "No Data", "No data available for fitting.")
            return
            
        self.run_grid_btn.setEnabled(False)
        self.grid_p_spin.setEnabled(False)
        self.grid_q_spin.setEnabled(False)
        self.grid_P_spin.setEnabled(False)
        self.grid_Q_spin.setEnabled(False)
        self.grid_s_spin.setEnabled(False)
        self.progress_bar.setValue(0)
        
        from axis3_modeling import GridSearchWorker
        self.worker = GridSearchWorker(
            series=series,
            max_p=self.grid_p_spin.value(),
            max_q=self.grid_q_spin.value(),
            max_P=self.grid_P_spin.value(),
            max_Q=self.grid_Q_spin.value(),
            s=self.grid_s_spin.value(),
            d=state.diff_order_d,
            D=state.diff_order_D
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_grid_search_finished)
        self.worker.start()
        
    def on_grid_search_finished(self, df):
        self.run_grid_btn.setEnabled(True)
        self.grid_p_spin.setEnabled(True)
        self.grid_q_spin.setEnabled(True)
        self.grid_P_spin.setEnabled(True)
        self.grid_Q_spin.setEnabled(True)
        self.grid_s_spin.setEnabled(True)
        
        if df.empty:
            QMessageBox.critical(self, "Grid Search Failed", "Grid search did not find any valid models.")
            self.export_models_btn.setEnabled(False)
            return
            
        self.grid_results = df
        self.export_models_btn.setEnabled(True)
        self.populate_top_10_table(df)
        
    def populate_top_10_table(self, df):
        n_rows = min(len(df), 10)
        self.model_table.setRowCount(n_rows)
        self.model_table.setColumnCount(8)
        self.model_table.setHorizontalHeaderLabels([
            "Rank", "Model", "AICc", "BIC", "LogLik", "k", "Show Spectrum", "Select Model"
        ])
        
        for i in range(n_rows):
            row_data = df.iloc[i]
            
            self.model_table.setItem(i, 0, QTableWidgetItem(str(int(row_data["Rank"]))))
            self.model_table.setItem(i, 1, QTableWidgetItem(str(row_data["Model"])))
            self.model_table.setItem(i, 2, QTableWidgetItem(f"{row_data['AICc']:.4f}"))
            self.model_table.setItem(i, 3, QTableWidgetItem(f"{row_data['BIC']:.4f}"))
            self.model_table.setItem(i, 4, QTableWidgetItem(f"{row_data['LogLik']:.4f}"))
            self.model_table.setItem(i, 5, QTableWidgetItem(str(int(row_data["k"]))))
            
            show_spec_btn = QPushButton("Show Spectrum", self)
            show_spec_btn.clicked.connect(lambda checked=False, r=row_data: self.show_spectrum_for_model(r))
            self.model_table.setCellWidget(i, 6, show_spec_btn)
            
            select_btn = QPushButton("Select", self)
            select_btn.clicked.connect(lambda checked=False, r=row_data: self.select_model(r))
            self.model_table.setCellWidget(i, 7, select_btn)
            
        self.model_table.resizeColumnsToContents()
        
    def select_model(self, row_data):
        series = self.get_series_for_fitting()
        if series is None:
            return
            
        p, d, q = int(row_data["p"]), int(row_data["d"]), int(row_data["q"])
        P, D, Q, s = int(row_data["P"]), int(row_data["D"]), int(row_data["Q"]), int(row_data["s"])
        
        is_seasonal = s > 1 and (P > 0 or D > 0 or Q > 0)
        order = (p, d, q)
        seasonal_order = (P, D, Q, s) if is_seasonal else None
        
        try:
            from axis3_modeling import fit_model
            res = fit_model(series, order, seasonal_order)
            
            state = self.main_window.state
            state.preview_model_params = None  # Clear preview when selecting a model
            state.fitted_model = res
            state.model_order = (p, d, q, P, D, Q, s) if is_seasonal else (p, d, q)
            state.model_params = self.get_model_params_dict(res)
            state.residuals = res.resid.values
            state.is_seasonal = is_seasonal
            
            self.display_parameters(res)
            self.main_window.update_ui_from_state()
            QMessageBox.information(self, "Model Selected", f"Successfully fitted and selected {row_data['Model']}.")
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Failed to fit model: {str(e)}")
            
    def show_spectrum_for_model(self, row_data):
        series = self.get_series_for_fitting()
        if series is None:
            return
            
        p, d, q = int(row_data["p"]), int(row_data["d"]), int(row_data["q"])
        P, D, Q, s = int(row_data["P"]), int(row_data["D"]), int(row_data["Q"]), int(row_data["s"])
        
        is_seasonal = s > 1 and (P > 0 or D > 0 or Q > 0)
        order = (p, d, q)
        seasonal_order = (P, D, Q, s) if is_seasonal else None
        
        try:
            from axis3_modeling import fit_model
            res = fit_model(series, order, seasonal_order)
            
            state = self.main_window.state
            state.preview_model_params = self.get_model_params_dict(res)
            
            self.main_window.tab_widget.setCurrentIndex(2)
            self.main_window.tabs[2].update_parametric_spectrum_overlay()
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Failed to fit model: {str(e)}")
            
    def get_model_params_dict(self, res):
        params_dict = res.params.to_dict()
        ar_coeffs = []
        ma_coeffs = []
        for i in range(1, 20):
            key = f"ar.L{i}"
            if key in params_dict:
                ar_coeffs.append(params_dict[key])
            else:
                break
        for i in range(1, 20):
            key = f"ma.L{i}"
            if key in params_dict:
                ma_coeffs.append(params_dict[key])
            else:
                break
        sigma2 = params_dict.get('sigma2', 1.0)
        return {
            "ar": ar_coeffs,
            "ma": ma_coeffs,
            "sigma2": sigma2
        }
        
    def display_parameters(self, res):
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        tvalues = res.tvalues
        
        self.param_table.setRowCount(len(params))
        self.param_table.setColumnCount(5)
        self.param_table.setHorizontalHeaderLabels([
            "Parameter", "Estimate", "Std. Error", "z-value", "p-value"
        ])
        
        rows = []
        for idx, name in enumerate(params.index):
            val = params[name]
            se = bse[name] if name in bse else float('nan')
            z = tvalues[name] if name in tvalues else float('nan')
            p = pvalues[name] if name in pvalues else float('nan')
            
            self.param_table.setItem(idx, 0, QTableWidgetItem(name))
            self.param_table.setItem(idx, 1, QTableWidgetItem(f"{val:.4f}"))
            self.param_table.setItem(idx, 2, QTableWidgetItem(f"{se:.4f}" if not np.isnan(se) else "N/A"))
            self.param_table.setItem(idx, 3, QTableWidgetItem(f"{z:.4f}" if not np.isnan(z) else "N/A"))
            self.param_table.setItem(idx, 4, QTableWidgetItem(f"{p:.4f}" if not np.isnan(p) else "N/A"))
            
            rows.append({
                "Parameter": name,
                "Estimate": val,
                "Std. Error": se,
                "z-value": z,
                "p-value": p
            })
            
        self.fitted_params_df = pd.DataFrame(rows)
        self.export_params_btn.setEnabled(True)
        self.param_table.resizeColumnsToContents()
        
        sigma2 = res.params.get("sigma2", float('nan'))
        loglik = res.llf
        self.sigma2_label.setText(f"Estimated σ̂²: {sigma2:.4f}" if not np.isnan(sigma2) else "Estimated σ̂²: N/A")
        self.loglik_label.setText(f"Log-Likelihood: {loglik:.4f}")

    def export_models_csv(self):
        if getattr(self, 'grid_results', None) is None or self.grid_results.empty:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Grid Search Results to CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            try:
                self.grid_results.to_csv(file_path, index=False)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export models: {str(e)}")

    def export_params_csv(self):
        if getattr(self, 'fitted_params_df', None) is None or self.fitted_params_df.empty:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Model Parameters to CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            try:
                self.fitted_params_df.to_csv(file_path, index=False)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export parameters: {str(e)}")


class ValidationTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Model Validation & Residual Diagnostics", parent)
        self.main_window = main_window
        
        # Right side: two plot widgets (Top: 2x3 grid, Bottom: 1x2 grid for Ljung-Box and Residual Spectrum)
        self.plot_widget = PlotWidget(self)
        self.bottom_plot_widget = PlotWidget(self)
        
        self.right_layout.addWidget(self.plot_widget, 6)
        self.right_layout.addWidget(self.bottom_plot_widget, 4)
        
        # Left Panel (Controls)
        self.control_layout.addWidget(QLabel("Ljung-Box Lags:", self.control_panel))
        self.lb_lag_spin = QSpinBox(self.control_panel)
        self.lb_lag_spin.setRange(1, 40)
        self.lb_lag_spin.setValue(20)
        self.control_layout.addWidget(self.lb_lag_spin)
        
        self.run_tests_btn = QPushButton("Run Diagnostics", self.control_panel)
        self.run_tests_btn.setObjectName("primaryButton")
        self.run_tests_btn.clicked.connect(self.run_diagnostics)
        self.control_layout.addWidget(self.run_tests_btn)
        
        self.control_layout.addWidget(QLabel("", self.control_panel)) # spacing
        
        self.verdict_label = QLabel("Verdict Summary:", self.control_panel)
        self.verdict_label.setObjectName("sectionHeading")
        self.control_layout.addWidget(self.verdict_label)
        
        # Monospaced text browser for final verdict
        self.verdict_browser = QTextBrowser(self.control_panel)
        from PySide6.QtGui import QFontDatabase
        mono_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono_font.setPointSize(9)
        self.verdict_browser.setFont(mono_font)
        self.verdict_browser.setReadOnly(True)
        self.control_layout.addWidget(self.verdict_browser)
        
        self.control_layout.addStretch()
        
    def run_diagnostics(self):
        state = self.main_window.state
        if not state.model_fitted or state.residuals is None:
            QMessageBox.warning(self, "No Fitted Model", "Please fit and select a model first on Tab 4.")
            return
            
        residuals = state.residuals
        order = state.model_order
        
        # Standard Box-Jenkins: number of estimated parameters (excluding constant/variance)
        # order is (p, d, q) or (p, d, q, P, D, Q, s)
        p_val = order[0]
        q_val = order[2]
        P_val = order[3] if len(order) == 7 else 0
        Q_val = order[5] if len(order) == 7 else 0
        
        # Standardize residuals
        res_std = np.std(residuals)
        std_residuals = residuals / res_std if res_std > 1e-10 else residuals
        
        # Calculate diagnostics
        try:
            from axis4_validation import run_all_diagnostics
            results = run_all_diagnostics(residuals, p_val + P_val, q_val + Q_val)
        except Exception as e:
            QMessageBox.critical(self, "Diagnostic Error", f"Failed to run diagnostics: {str(e)}")
            return
            
        # Draw Top 2x3 Plot Grid
        self.draw_top_plots(residuals, std_residuals, results)
        
        # Draw Bottom 1x2 Plot Grid
        self.draw_bottom_plots(std_residuals, results)
        
        # Decide Verdict
        lb_lag = self.lb_lag_spin.value()
        h_max = results.get('h_max', 20)
        lb_lag_actual = min(lb_lag, h_max) if h_max > 0 else 0
        
        if lb_lag_actual > 0:
            lb_pvalue = results['lb_pvalues'][lb_lag_actual - 1]
        else:
            lb_pvalue = np.nan
        
        if np.isnan(lb_pvalue):
            lb_pass = True
            lb_status_str = "PASS"
        else:
            lb_pass = bool(lb_pvalue >= 0.05)
            lb_status_str = "PASS" if lb_pass else "FAIL"
            
        jb_pvalue = results['jb_pvalue']
        jb_pass = bool(results['jb_pass'])
        jb_status_str = "PASS" if jb_pass else "FAIL"
        
        cp_pass = bool(results['cp_pass'])
        cp_status_str = "PASS" if cp_pass else "FAIL"
        
        # Overall verdict
        all_passed = bool(lb_pass and jb_pass and cp_pass)
        state.validation_passed = all_passed
        
        # Update UI state and enabled tabs
        self.main_window.update_ui_from_state()
        
        # Format verdict text
        verdict_status = "ADEQUATE" if all_passed else "INADEQUATE"
        
        if all_passed:
            verdict_details = "Residuals are consistent with Gaussian White Noise.\nProceed to Forecasting (Tab 6)."
        else:
            verdict_details = "Return to Tab 4 (Model Identification) and\nconsider alternative specifications."
            
        summary_text = (
            "===== MODEL VALIDATION SUMMARY =====\n"
            f"Ljung-Box Test (h={lb_lag_actual}):\n"
            f"p-value = {f'{lb_pvalue:.4f}' if not np.isnan(lb_pvalue) else 'NaN'}\n"
            f"[{lb_status_str}]\n\n"
            "Jarque-Bera Test:\n"
            f"p-value = {jb_pvalue:.4f}\n"
            f"[{jb_status_str}]\n\n"
            "Cum. Periodogram Test:\n"
            f"[{cp_status_str}]\n"
            "---------------------\n"
            f"VERDICT: MODEL IS {verdict_status}.\n"
            f"{verdict_details}"
        )
        
        self.verdict_browser.setText(summary_text)
        
    def draw_top_plots(self, residuals, std_residuals, results):
        fig = self.plot_widget.canvas.figure
        fig.clear()
        
        # 2x3 Subplots
        axs = fig.subplots(2, 3)
        
        # Theme colors
        text_color = '#1F2937'
        accent_color = '#2563EB'
        fail_color = '#DC2626'
        
        # Plot 1: Standardized residuals vs time
        ax = axs[0, 0]
        time_idx = self.main_window.state.time_index
        if time_idx is not None and len(time_idx) == len(std_residuals):
            ax.plot(time_idx, std_residuals, color=text_color, linewidth=1)
        else:
            ax.plot(np.arange(len(std_residuals)), std_residuals, color=text_color, linewidth=1)
        ax.axhline(0, color='gray', linestyle='-')
        ax.axhline(1.96, color=fail_color, linestyle='--')
        ax.axhline(-1.96, color=fail_color, linestyle='--')
        ax.set_title("Standardized Residuals")
        ax.tick_params(colors=text_color)
        
        # Plot 2: ACF of residuals
        ax = axs[0, 1]
        from statsmodels.tsa.stattools import acf
        n_res = len(std_residuals)
        nlags = min(20, n_res // 2 - 1)
        if nlags < 1:
            nlags = 1
        acf_vals = acf(std_residuals, nlags=nlags)
        lags = np.arange(len(acf_vals))
        ax.vlines(lags, 0, acf_vals, colors=accent_color, linewidth=2)
        ax.plot(lags, acf_vals, 'o', color=accent_color, markersize=4)
        ax.axhline(0, color=text_color, linewidth=1)
        bound = 1.96 / np.sqrt(n_res)
        ax.axhline(bound, color=fail_color, linestyle='--', linewidth=1)
        ax.axhline(-bound, color=fail_color, linestyle='--', linewidth=1)
        ax.fill_between(lags, -bound, bound, color=accent_color, alpha=0.1)
        ax.set_title("ACF of Residuals")
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(colors=text_color)
        
        # Plot 3: PACF of residuals
        ax = axs[0, 2]
        from statsmodels.tsa.stattools import pacf
        try:
            pacf_vals = pacf(std_residuals, nlags=nlags, method='ywm')
        except Exception:
            pacf_vals = pacf(std_residuals, nlags=nlags)
        lags = np.arange(len(pacf_vals))
        ax.vlines(lags, 0, pacf_vals, colors=accent_color, linewidth=2)
        ax.plot(lags, pacf_vals, 'o', color=accent_color, markersize=4)
        ax.axhline(0, color=text_color, linewidth=1)
        ax.axhline(bound, color=fail_color, linestyle='--', linewidth=1)
        ax.axhline(-bound, color=fail_color, linestyle='--', linewidth=1)
        ax.fill_between(lags, -bound, bound, color=accent_color, alpha=0.1)
        ax.set_title("PACF of Residuals")
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(colors=text_color)
        
        # Plot 4: QQ-plot
        ax = axs[1, 0]
        import scipy.stats as stats
        stats.probplot(std_residuals, plot=ax)
        ax.get_lines()[0].set_markerfacecolor(accent_color)
        ax.get_lines()[0].set_markeredgecolor(accent_color)
        ax.get_lines()[1].set_color(fail_color)
        ax.set_title("Normal Q-Q Plot")
        ax.tick_params(colors=text_color)
        
        # Plot 5: Histogram + KDE
        ax = axs[1, 1]
        ax.hist(std_residuals, bins='auto', density=True, alpha=0.5, facecolor='#F5F6F8', edgecolor='#D1D5DB')
        x_grid = np.linspace(-4, 4, 200)
        ax.plot(x_grid, stats.norm.pdf(x_grid, 0, 1), linestyle='--', color=fail_color, label='N(0,1)')
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(std_residuals)
            ax.plot(x_grid, kde(x_grid), color=accent_color, label='KDE')
        except Exception:
            pass
        ax.set_title("Residual Histogram")
        ax.legend(prop={'size': 7})
        ax.tick_params(colors=text_color)
        
        # Plot 6: Cumulative Periodogram + KS bounds
        ax = axs[1, 2]
        cp_freqs = results['cp_frequencies']
        C_omega = results['cp_C_omega']
        ks_up = results['cp_ks_upper']
        ks_lo = results['cp_ks_lower']
        if len(C_omega) > 0:
            ax.plot(cp_freqs, C_omega, color=accent_color, label='C(ω)')
            ax.plot([0, cp_freqs[-1]], [0, 1.0], color='gray', linestyle='-')
            ax.plot(cp_freqs, ks_up, color=fail_color, linestyle='--', label='95% KS Bound')
            ax.plot(cp_freqs, ks_lo, color=fail_color, linestyle='--')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(prop={'size': 7})
        ax.set_title("Cumulative Periodogram")
        ax.tick_params(colors=text_color)
        
        fig.tight_layout()
        self.plot_widget.canvas.draw()
        
    def draw_bottom_plots(self, std_residuals, results):
        fig = self.bottom_plot_widget.canvas.figure
        fig.clear()
        
        axs = fig.subplots(1, 2)
        text_color = '#1F2937'
        accent_color = '#2563EB'
        fail_color = '#DC2626'
        
        # Plot 1: Ljung-Box p-values
        ax = axs[0]
        lb_pvals = results['lb_pvalues']
        lags = np.arange(1, len(lb_pvals) + 1)
        valid_indices = ~np.isnan(lb_pvals)
        if np.any(valid_indices):
            colors = [accent_color if p >= 0.05 else fail_color for p in lb_pvals[valid_indices]]
            ax.bar(lags[valid_indices], lb_pvals[valid_indices], color=colors, alpha=0.8, edgecolor='#D1D5DB')
        ax.axhline(0.05, color=fail_color, linestyle='--', label='5% Level')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Lag h")
        ax.set_ylabel("p-value")
        ax.set_title("Ljung-Box p-values")
        ax.legend(prop={'size': 8})
        ax.tick_params(colors=text_color)
        
        # Plot 2: Residual Spectral Density Flatness Check
        ax = axs[1]
        sp_freqs = results['sp_frequencies']
        f_hat = results['sp_f_hat']
        sp_lo = results['sp_ci_lower']
        sp_up = results['sp_ci_upper']
        
        if len(f_hat) > 0:
            ax.plot(sp_freqs, f_hat, color=accent_color, label='f̂_Z(λ)')
            ax.fill_between(sp_freqs, sp_lo, sp_up, color=accent_color, alpha=0.15, label='95% CI')
            flat_level = 1.0 / (2.0 * np.pi)
            ax.axhline(flat_level, color=fail_color, linestyle='--', label='1/(2π)')
            ax.legend(prop={'size': 8})
        ax.set_title("Residual Spectrum Flatness")
        ax.set_xlabel("Frequency (radians)")
        ax.set_ylabel("Spectral Density")
        ax.tick_params(colors=text_color)
        
        fig.tight_layout()
        self.bottom_plot_widget.canvas.draw()



class ForecastingTab(BaseTab):
    def __init__(self, main_window, parent=None):
        super().__init__("Generate & View Forecasts", parent)
        self.main_window = main_window
        
        self.stationary_forecasts = None
        self.back_transformed_forecasts = None
        self.current_forecasts = None
        
        # Upper Right: Plot
        self.plot_widget = PlotWidget(self)
        self.right_layout.addWidget(self.plot_widget, 6)
        
        # Lower Right: Table and Export
        from PySide6.QtWidgets import QHeaderView
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(9)
        self.table_widget.setHorizontalHeaderLabels([
            "Step", "Date", "Point Forecast",
            "Lower 50% PI", "Upper 50% PI",
            "Lower 80% PI", "Upper 80% PI",
            "Lower 95% PI", "Upper 95% PI"
        ])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.right_layout.addWidget(self.table_widget, 4)
        
        self.export_csv_btn = QPushButton("Export Table to CSV", self)
        self.export_csv_btn.setObjectName("secondaryButton")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.right_layout.addWidget(self.export_csv_btn)
        
        # Add Forecasting Controls to Left Panel
        self.control_layout.addWidget(QLabel("Forecast Horizon h:", self.control_panel))
        self.horizon_spin = QSpinBox(self.control_panel)
        self.horizon_spin.setRange(1, 100)
        self.horizon_spin.setValue(24)
        self.control_layout.addWidget(self.horizon_spin)
        
        self.scale_combo = QComboBox(self.control_panel)
        self.scale_combo.addItems(["Original Scale", "Transformed Scale"])
        self.scale_combo.currentIndexChanged.connect(self.update_view)
        self.control_layout.addWidget(self.scale_combo)
        
        self.generate_btn = QPushButton("Generate Forecast", self.control_panel)
        self.generate_btn.setObjectName("primaryButton")
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        self.control_layout.addWidget(self.generate_btn)
        
        self.control_layout.addWidget(QLabel("Spectral Insight:", self.control_panel))
        self.insight_browser = QTextBrowser(self.control_panel)
        self.insight_browser.setMinimumHeight(200)
        self.control_layout.addWidget(self.insight_browser)
        
        self.control_layout.addStretch()

    def on_generate_clicked(self):
        state = self.main_window.state
        if not state.model_fitted or state.fitted_model is None:
            QMessageBox.warning(self, "No Model", "No fitted model found. Please fit a model first on Tab 4.")
            return

        # Warn if validation was run and failed, but still allow forecasting
        if state.validation_run and not state.validation_passed:
            QMessageBox.warning(
                self, "Model Not Validated",
                "⚠️ This model failed one or more adequacy tests.\n"
                "Forecasts may be unreliable — interpret prediction intervals with caution."
            )

        h = self.horizon_spin.value()
        try:
            from axis5_forecasting import generate_forecasts, back_transform, generate_spectral_insight
            
            # Generate forecasts on transformed scale
            self.stationary_forecasts = generate_forecasts(state.fitted_model, h, state)
            
            # Back-transform to original scale
            self.back_transformed_forecasts = back_transform(
                self.stationary_forecasts, state.transformations, state.original_series
            )
            
            # Update the spectral insight panel
            insight_text = generate_spectral_insight(state)
            self.insight_browser.setMarkdown(insight_text)
            
            # Update plots and table
            self.update_view()
            
        except Exception as e:
            QMessageBox.critical(self, "Forecasting Error", f"Failed to generate forecasts: {str(e)}")

    def update_view(self):
        if not hasattr(self, 'stationary_forecasts') or self.stationary_forecasts is None:
            return
            
        state = self.main_window.state
        scale = self.scale_combo.currentText()
        
        if scale == "Original Scale":
            forecasts = self.back_transformed_forecasts
            history = state.original_series
            scale_label = "Original Scale"
        else:
            forecasts = self.stationary_forecasts
            history = state.stationary_series
            scale_label = "Transformed Scale"
            
        self.current_forecasts = forecasts
        
        # 1. Update plot
        ax = self.plot_widget.canvas.axes
        ax.clear()
        
        # Plot last 30-50 historical values for context
        history_len = min(50, len(history))
        history_series = history.iloc[-history_len:]
        
        # Connect history and forecast by prepending last history value
        plot_dates = [history_series.index[-1]] + list(forecasts["dates"])
        
        plot_point = np.concatenate([[history_series.iloc[-1]], forecasts["point"]])
        plot_lower_50 = np.concatenate([[history_series.iloc[-1]], forecasts["lower_50"]])
        plot_upper_50 = np.concatenate([[history_series.iloc[-1]], forecasts["upper_50"]])
        plot_lower_80 = np.concatenate([[history_series.iloc[-1]], forecasts["lower_80"]])
        plot_upper_80 = np.concatenate([[history_series.iloc[-1]], forecasts["upper_80"]])
        plot_lower_95 = np.concatenate([[history_series.iloc[-1]], forecasts["lower_95"]])
        plot_upper_95 = np.concatenate([[history_series.iloc[-1]], forecasts["upper_95"]])
        
        # Plot history
        ax.plot(history_series.index, history_series.values, color='#4B5563', label='History', linewidth=1.5)
        
        # Plot point forecast
        ax.plot(plot_dates, plot_point, color='#2563EB', label='Forecast', linewidth=2.0)
        
        # Plot prediction intervals (fan chart)
        ax.fill_between(plot_dates, plot_lower_50, plot_upper_50, color='#2563EB', alpha=0.4, label='50% PI')
        ax.fill_between(plot_dates, plot_lower_80, plot_upper_80, color='#2563EB', alpha=0.25, label='80% PI')
        ax.fill_between(plot_dates, plot_lower_95, plot_upper_95, color='#2563EB', alpha=0.1, label='95% PI')
        
        ax.set_title(f"Forecast Horizon h={len(forecasts['point'])} ({scale_label})", color='#1F2937', fontsize=12, fontweight='bold')
        ax.set_xlabel("Time", color='#1F2937')
        ax.set_ylabel("Value", color='#1F2937')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best')
        
        # Rotate dates for better readability if DatetimeIndex
        if isinstance(history.index, pd.DatetimeIndex):
            self.plot_widget.canvas.figure.autofmt_xdate()
            
        self.plot_widget.canvas.draw()
        
        # 2. Update table
        h = len(forecasts["point"])
        self.table_widget.setRowCount(h)
        for idx in range(h):
            # Step
            step_item = QTableWidgetItem(str(forecasts["steps"][idx]))
            step_item.setFlags(step_item.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 0, step_item)
            
            # Date
            date_val = forecasts["dates"][idx]
            if isinstance(date_val, pd.Timestamp):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)
            date_item = QTableWidgetItem(date_str)
            date_item.setFlags(date_item.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 1, date_item)
            
            # Point
            point_item = QTableWidgetItem(f"{forecasts['point'][idx]:.4f}")
            point_item.setFlags(point_item.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 2, point_item)
            
            # 50% PI
            l50 = QTableWidgetItem(f"{forecasts['lower_50'][idx]:.4f}")
            l50.setFlags(l50.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 3, l50)
            
            u50 = QTableWidgetItem(f"{forecasts['upper_50'][idx]:.4f}")
            u50.setFlags(u50.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 4, u50)
            
            # 80% PI
            l80 = QTableWidgetItem(f"{forecasts['lower_80'][idx]:.4f}")
            l80.setFlags(l80.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 5, l80)
            
            u80 = QTableWidgetItem(f"{forecasts['upper_80'][idx]:.4f}")
            u80.setFlags(u80.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 6, u80)
            
            # 95% PI
            l95 = QTableWidgetItem(f"{forecasts['lower_95'][idx]:.4f}")
            l95.setFlags(l95.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 7, l95)
            
            u95 = QTableWidgetItem(f"{forecasts['upper_95'][idx]:.4f}")
            u95.setFlags(u95.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(idx, 8, u95)

    def export_csv(self):
        if not hasattr(self, 'current_forecasts') or self.current_forecasts is None:
            QMessageBox.warning(self, "No Forecasts", "Please generate forecasts first.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Forecasts to CSV", "", "CSV Files (*.csv)"
        )
        if file_path:
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
                
            forecasts = self.current_forecasts
            date_strings = []
            for d in forecasts["dates"]:
                if isinstance(d, pd.Timestamp):
                    date_strings.append(d.strftime('%Y-%m-%d'))
                else:
                    date_strings.append(str(d))
                    
            df = pd.DataFrame({
                "Step": forecasts["steps"],
                "Date": date_strings,
                "Point_Forecast": forecasts["point"],
                "Lower_50_PI": forecasts["lower_50"],
                "Upper_50_PI": forecasts["upper_50"],
                "Lower_80_PI": forecasts["lower_80"],
                "Upper_80_PI": forecasts["upper_80"],
                "Lower_95_PI": forecasts["lower_95"],
                "Upper_95_PI": forecasts["upper_95"]
            })
            
            try:
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Successful", f"Forecasts successfully exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {str(e)}")


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
        tab6_enabled = self.state.model_fitted  # ponytail: allow forecast for any fitted model; warning shown if not validated
        
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
            # Synchronize grid search limits s spin box in Tab 4
            if hasattr(self, 'tabs') and len(self.tabs) > 3:
                self.tabs[3].grid_s_spin.blockSignals(True)
                self.tabs[3].grid_s_spin.setValue(self.state.seasonal_period_s)
                self.tabs[3].grid_s_spin.blockSignals(False)
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
        elif self.state.model_fitted and self.state.validation_run:
            self.indicators[5].set_status('fail')  # model fitted but failed validation
        else:
            self.indicators[5].set_status('neutral')
            
        # Update tab 3 parametric overlay if model is fitted
        if hasattr(self, 'tabs') and len(self.tabs) > 2:
            self.tabs[2].update_parametric_spectrum_overlay()

    def closeEvent(self, event):
        try:
            # Safely terminate GridSearchWorker if running on exit
            modeling_tab = self.tabs[3]
            if hasattr(modeling_tab, 'worker') and modeling_tab.worker and modeling_tab.worker.isRunning():
                modeling_tab.worker.terminate()
                modeling_tab.worker.wait()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
