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

def test_gui_forecasting_workflow(monkeypatch, tmp_path):
    print("Testing GUI Tab 6 Forecasting Workflow...")
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Mock QMessageBox to prevent blocking on headless execution
    from PySide6.QtWidgets import QMessageBox, QFileDialog
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    
    window = MainWindow()
    
    # 1. Load dataset on Tab 1
    csv_path = os.path.join(os.path.dirname(__file__), "datasets", "international-airline-passengers.csv")
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
    
    # Mock validation run to pass
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
    
    # Run validation
    tab5 = window.tabs[4]
    tab5.run_tests_btn.click()
    
    # Verify Tab 6 is now enabled
    assert window.tab_widget.isTabEnabled(5) is True
    
    # Switch to Tab 6
    window.tab_widget.setCurrentIndex(5)
    tab6 = window.tabs[5]
    
    # 4. Generate forecasts
    tab6.horizon_spin.setValue(36)
    tab6.generate_btn.click()
    
    # Check that the plot is updated
    assert len(tab6.plot_widget.canvas.figure.axes) == 1
    
    # Check that the table has 36 rows and 9 columns
    assert tab6.table_widget.rowCount() == 36
    assert tab6.table_widget.columnCount() == 9
    
    # Verify first point forecast value is non-empty
    val = tab6.table_widget.item(0, 2).text()
    assert val != ""
    assert float(val) > 0.0 # because back-transformed airline data is positive
    
    # Check that the spectral insight browser is populated
    insight = tab6.insight_browser.toPlainText()
    assert "Spectral Interpretation" in insight or "Why do these forecasts look" in insight
    
    # 5. Toggle Scale and verify update
    tab6.scale_combo.setCurrentIndex(1) # Transformed scale
    val_trans = tab6.table_widget.item(0, 2).text()
    assert val_trans != val # should be different (transformed vs original)
    
    # 6. Test CSV Export
    csv_file = os.path.join(tmp_path, "forecasts.csv")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (csv_file, "CSV Files (*.csv)"))
    tab6.export_csv_btn.click()
    
    assert os.path.exists(csv_file)
    df_exported = pd.read_csv(csv_file)
    assert len(df_exported) == 36
    assert "Point_Forecast" in df_exported.columns
    assert "Lower_95_PI" in df_exported.columns
    
    print("GUI Tab 6 tests passed successfully!")
