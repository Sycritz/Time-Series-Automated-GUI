import sys
import os
import pandas as pd
import numpy as np

# Force headless Qt
os.environ["QT_API"] = "pyside6"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from main import MainWindow

def test_gui_workflow():
    print("Testing GUI Tab 1 and Tab 2 Integration Workflow...")
    
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    
    # 1. Verify initial state
    assert window.state.data_loaded is False
    assert window.state.stationarity_done is False
    
    # 2. Simulate loading the airline passengers CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "datasets", "international-airline-passengers.csv")
    print(f"Loading test file: {csv_path}")
    
    tab1 = window.tabs[0]
    # Manually trigger load
    from axis1_preprocessing import load_csv
    tab1.df = load_csv(csv_path)
    
    # Check that columns were loaded
    assert "Month" in tab1.df.columns
    assert "Passengers" in tab1.df.columns
    
    # Set combobox items (simulating populate in browse_file)
    tab1.time_col_combo.clear()
    tab1.time_col_combo.addItems(["Select Column...", "Month", "Passengers"])
    tab1.val_col_combo.clear()
    tab1.val_col_combo.addItems(["Select Column...", "Month", "Passengers"])
    
    # Set index to columns (simulating selection of Month and Passengers)
    tab1.time_col_combo.setCurrentIndex(1) # Month
    tab1.val_col_combo.setCurrentIndex(2) # Passengers
    
    # Trigger selection handler
    tab1.on_columns_selected()
    
    # Verify Tab 1 completed output in AnalysisState
    assert window.state.data_loaded is True
    assert window.state.original_series is not None
    assert len(window.state.original_series) == 144
    assert window.state.time_index is not None
    
    # Verify Tab 2 is now enabled, other downstream tabs are disabled
    assert window.tab_widget.isTabEnabled(1) is True # Tab 2 enabled
    assert window.tab_widget.isTabEnabled(2) is False # Tab 3 disabled
    assert window.tab_widget.isTabEnabled(3) is False # Tab 4 disabled
    
    # Verify ADF test on Tab 1 runs
    tab1.run_adf()
    adf_text = tab1.adf_results_browser.toPlainText()
    print("Tab 1 ADF Test Output:\n", adf_text)
    assert "ADF Statistic" in adf_text
    assert "Verdict:       Non-Stationary" in adf_text # airline passenger series is non-stationary
    
    # 3. Simulate Tab 2 (Stationarity) workflow
    tab2 = window.tabs[1]
    
    # Auto-optimize lambda
    tab2.auto_optimize_lambda()
    opt_lambda = tab2.lambda_spin.value()
    print(f"Auto-optimized lambda: {opt_lambda}")
    # Optimal Box-Cox lambda for airline passenger series is around -0.3 to 0.0 (log transform is common, log is 0.0)
    assert -1.0 <= opt_lambda <= 1.0
    
    # Set differencing: d=1, D=1, s=12
    tab2.d_spin.setValue(1)
    tab2.D_spin.setValue(1)
    tab2.s_spin.setValue(12)
    
    # Apply transformations
    tab2.apply_transformations()
    
    # Verify state updates
    assert window.state.stationarity_done is True
    assert window.state.stationary_series is not None
    assert len(window.state.stationary_series) == 144
    # The first 13 values should be NaN because of d=1 and D=1, s=12 (1 + 12 = 13 points lost)
    assert window.state.stationary_series.isna().sum() == 13
    
    # Check transformations logged
    assert len(window.state.transformations) == 3
    assert window.state.transformations[0]["type"] == "boxcox"
    assert window.state.transformations[1]["type"] == "diff"
    assert window.state.transformations[2]["type"] == "seasonal_diff"
    
    # Verify ADF test on Tab 2 runs and verdict is updated
    adf2_text = tab2.adf_results_browser.toPlainText()
    print("Tab 2 ADF Test Output:\n", adf2_text)
    assert "ADF Statistic" in adf2_text
    # Diff + seasonal diff makes the airline passenger series stationary!
    assert "Verdict:       Stationary" in adf2_text
    
    # Verify Tab 3 & 4 are now enabled
    assert window.tab_widget.isTabEnabled(2) is True
    assert window.tab_widget.isTabEnabled(3) is True
    
    print("GUI Tab 1 and Tab 2 Integration Workflow tests passed!")

if __name__ == "__main__":
    test_workflow_err = None
    try:
        test_gui_workflow()
    except Exception as e:
        import traceback
        traceback.print_exc()
        test_workflow_err = e
        
    if test_workflow_err is not None:
        sys.exit(1)
    else:
        print("All GUI workflow tests passed successfully!")
