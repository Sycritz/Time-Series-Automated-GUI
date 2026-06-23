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

def test_gui_modeling_workflow():
    print("Testing GUI Tab 4 Modeling Integration Workflow...")
    
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
    
    # Verify Tab 4 is enabled
    assert window.tab_widget.isTabEnabled(3) is True
    
    # 3. Interact with Tab 4 (ModelingTab)
    tab4 = window.tabs[3]
    
    # Run ACF/PACF plot
    tab4.plot_acf_btn.click()
    assert tab4.acf_vals is not None
    assert tab4.pacf_vals is not None
    assert len(tab4.acf_vals) > 0
    
    # Test Suggestions
    tab4.suggest_acf_btn.click()
    assert "ACF/PACF Analysis" in tab4.suggestion_browser.toPlainText()
    
    # Run Spectral Suggestion (without any cycles in state first)
    tab4.suggest_spec_btn.click()
    assert "Spectral Suggestion Analysis" in tab4.suggestion_browser.toPlainText()
    
    # 4. Asynchronous Grid Search
    # Setup grid search limits (small search space to run fast in tests: p:0-1, q:0-1, non-seasonal)
    tab4.grid_p_spin.setValue(1)
    tab4.grid_q_spin.setValue(1)
    tab4.grid_P_spin.setValue(0)
    tab4.grid_Q_spin.setValue(0)
    tab4.grid_s_spin.setValue(1)
    
    print("Clicking run_grid_btn...", flush=True)
    tab4.run_grid_btn.click()
    
    # Wait for the worker thread to finish
    assert tab4.worker is not None
    print("Waiting for grid search worker to finish...", flush=True)
    import time
    start_time = time.time()
    while tab4.worker.isRunning():
        QApplication.processEvents()
        time.sleep(0.1)
        if time.time() - start_time > 15:
            print("TIMEOUT! Grid search worker did not finish in 15 seconds.", flush=True)
            break
    print("Finished waiting for worker. Thread running status:", tab4.worker.isRunning(), flush=True)
    
    # Process events to trigger finished slot
    QApplication.processEvents()
    
    print("Model table row count:", tab4.model_table.rowCount(), flush=True)
    # Check that model table is populated
    assert tab4.model_table.rowCount() > 0
    
    # 5. Test Select Model
    # Simulate clicking "Select" for the top model
    print("Selecting top model...", flush=True)
    row_data = tab4.grid_results.iloc[0]
    tab4.select_model(row_data)
    
    # Verify state updates
    assert window.state.fitted_model is not None
    assert window.state.model_order is not None
    assert window.state.model_params is not None
    assert window.state.residuals is not None
    
    # Verify UI tables/labels populated
    assert tab4.param_table.rowCount() > 0
    assert "Estimated σ̂²" in tab4.sigma2_label.text()
    assert "Log-Likelihood" in tab4.loglik_label.text()
    
    # Verify Tab 5 is enabled
    assert window.tab_widget.isTabEnabled(4) is True
    
    # 6. Test Show Spectrum
    tab4.show_spectrum_for_model(row_data)
    
    # Check active tab is Tab 3 (index 2)
    assert window.tab_widget.currentIndex() == 2
    
    print("GUI Tab 4 Modeling Integration Workflow tests passed!")

if __name__ == "__main__":
    test_gui_modeling_workflow()
