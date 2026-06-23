import sys
import os
import pandas as pd
import numpy as np
import pytest

# Force headless Qt
os.environ["QT_API"] = "pyside6"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from main import MainWindow
import axis4_validation

def test_gui_validation_workflow_fail():
    print("Testing GUI Tab 5 Validation Integration Workflow (FAIL)...")
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Mock QMessageBox to prevent blocking on headless execution
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
    QMessageBox.warning = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
    QMessageBox.critical = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
    
    window = MainWindow()
    
    # 1. Load dataset on Tab 1
    csv_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "international-airline-passengers.csv")
    tab1 = window.tabs[0]
    from axis1_preprocessing import load_csv
    tab1.df = load_csv(csv_path)
    
    tab1.time_col_combo.clear()
    tab1.time_col_combo.addItems(["Select Column...", "Month", "Passengers"])
    tab1.val_col_combo.clear()
    tab1.val_col_combo.addItems(["Select Column...", "Month", "Passengers"])
    tab1.time_col_combo.setCurrentIndex(1) # Month
    tab1.val_col_combo.setCurrentIndex(2) # Passengers
    tab1.on_columns_selected()
    
    # 2. Transform to stationarity on Tab 2
    tab2 = window.tabs[1]
    tab2.lambda_spin.setValue(0.0) # Log transform
    tab2.d_spin.setValue(1)
    tab2.D_spin.setValue(1)
    tab2.s_spin.setValue(12)
    tab2.apply_transformations()
    
    # 3. Fit a bad model (non-seasonal, ARIMA(1, 1, 0))
    tab4 = window.tabs[3]
    row_data_bad = {
        "p": 1, "d": 1, "q": 0,
        "P": 0, "D": 0, "Q": 0, "s": 12,
        "Model": "ARIMA(1, 1, 0)"
    }
    tab4.select_model(row_data_bad)
    
    # Verify Tab 5 is enabled
    assert window.tab_widget.isTabEnabled(4) is True
    
    # 4. Switch to Tab 5 (ValidationTab) and Run Diagnostics
    tab5 = window.tabs[4]
    tab5.lb_lag_spin.setValue(20)
    tab5.run_tests_btn.click()
    
    # Check that plots are drawn
    assert len(tab5.plot_widget.canvas.figure.axes) == 6
    assert len(tab5.bottom_plot_widget.canvas.figure.axes) == 2
    
    # Verify verdict text says INADEQUATE because of seasonal cycles
    verdict_text_bad = tab5.verdict_browser.toPlainText()
    assert "MODEL IS INADEQUATE" in verdict_text_bad
    
    # Verify Tab 6 is enabled (exploratory forecasting is allowed with warning)
    assert window.tab_widget.isTabEnabled(5) is True
    print("GUI Tab 5 FAIL Workflow tests passed.")

def test_gui_validation_workflow_pass(monkeypatch):
    print("Testing GUI Tab 5 Validation Integration Workflow (PASS via Mock)...")
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Mock QMessageBox
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
    QMessageBox.warning = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
    QMessageBox.critical = lambda *args, **kwargs: QMessageBox.StandardButton.Ok
    
    window = MainWindow()
    
    # 1. Load dataset on Tab 1
    csv_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "international-airline-passengers.csv")
    tab1 = window.tabs[0]
    from axis1_preprocessing import load_csv
    tab1.df = load_csv(csv_path)
    
    tab1.time_col_combo.clear()
    tab1.time_col_combo.addItems(["Select Column...", "Month", "Passengers"])
    tab1.val_col_combo.clear()
    tab1.val_col_combo.addItems(["Select Column...", "Month", "Passengers"])
    tab1.time_col_combo.setCurrentIndex(1) # Month
    tab1.val_col_combo.setCurrentIndex(2) # Passengers
    tab1.on_columns_selected()
    
    # 2. Transform to stationarity on Tab 2
    tab2 = window.tabs[1]
    tab2.lambda_spin.setValue(0.0) # Log transform
    tab2.d_spin.setValue(1)
    tab2.D_spin.setValue(1)
    tab2.s_spin.setValue(12)
    tab2.apply_transformations()
    
    # 3. Fit a model on Tab 4
    tab4 = window.tabs[3]
    row_data = {
        "p": 1, "d": 1, "q": 0,
        "P": 0, "D": 0, "Q": 0, "s": 12,
        "Model": "ARIMA(1, 1, 0)"
    }
    tab4.select_model(row_data)
    
    # Mock run_all_diagnostics to return passing stats
    def mock_run_all_diagnostics(residuals, p, q):
        return {
            'lb_stats': np.ones(20),
            'lb_pvalues': np.ones(20) * 0.99,
            'lb_p_value_20': 0.99,
            'lb_pass': True,
            'jb_stat': 0.1,
            'jb_pvalue': 0.95,
            'jb_pass': True,
            'cp_frequencies': np.array([0.1, 0.2, 0.3]),
            'cp_C_omega': np.array([0.33, 0.66, 1.0]),
            'cp_ks_upper': np.array([0.8, 0.9, 1.0]),
            'cp_ks_lower': np.array([0.0, 0.1, 0.2]),
            'cp_pass': True,
            'sp_frequencies': np.linspace(0, np.pi, 512),
            'sp_f_hat': np.ones(512) * (1.0 / (2.0 * np.pi)),
            'sp_ci_lower': np.ones(512) * 0.1,
            'sp_ci_upper': np.ones(512) * 0.3,
            'flatness_pass': True
        }
        
    monkeypatch.setattr(axis4_validation, "run_all_diagnostics", mock_run_all_diagnostics)
    
    # 4. Switch to Tab 5 and Run Diagnostics
    tab5 = window.tabs[4]
    tab5.run_tests_btn.click()
    
    # Verify verdict text says ADEQUATE
    verdict_text_good = tab5.verdict_browser.toPlainText()
    assert "MODEL IS ADEQUATE" in verdict_text_good
    assert "PASS" in verdict_text_good
    
    # Verify Tab 6 is enabled
    assert window.tab_widget.isTabEnabled(5) is True
    print("GUI Tab 5 PASS Workflow tests passed.")
