import os
import sys
import pandas as pd
import numpy as np

# Force PySide6 for matplotlib and Qt
os.environ["QT_API"] = "pyside6"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from state import AnalysisState
from main import MainWindow

def test_analysis_state_invalidation():
    print("Testing AnalysisState Invalidation...")
    state = AnalysisState()
    
    # 1. Initial defaults
    assert state.original_series is None
    assert state.stationary_series is None
    assert state.fitted_model is None
    assert state.validation_passed is False
    assert state.validation_run is False
    assert state.box_cox_lambda is None
    assert state.diff_order_d == 0
    assert state.diff_order_D == 0
    assert state.seasonal_period_s == 1
    assert state.transformations == []
    assert state.detected_cycles == []
    assert state.model_order is None
    assert state.model_params is None
    assert state.residuals is None
    assert state.is_seasonal is False
    
    # 2. Test setting validation_passed sets validation_run to True
    state.validation_passed = True
    assert state.validation_run is True
    assert state.validation_passed is True
    
    # 3. Test setting fitted_model resets validation outputs
    state.fitted_model = object()
    assert state.fitted_model is not None
    # since we just set fitted_model, downstream validation_passed and validation_run should be reset:
    assert state.validation_passed is False
    assert state.validation_run is False
    
    # Let's set validation back to True
    state.validation_passed = True
    assert state.validation_run is True
    
    # 4. Test setting stationary_series resets Tab 3-6 outputs
    state.detected_cycles = [{"freq": 0.1}]
    state.fitted_model = object()
    state.model_order = (1, 0, 1)
    state.residuals = np.array([1, 2])
    state.is_seasonal = True
    state.validation_passed = True
    
    # Now set stationary_series
    state.stationary_series = pd.Series([1, 2, 3])
    assert state.stationary_series is not None
    assert state.detected_cycles == []
    assert state.fitted_model is None
    assert state.model_order is None
    assert state.residuals is None
    assert state.is_seasonal is False
    assert state.validation_passed is False
    assert state.validation_run is False
    
    # 5. Test setting original_series resets ALL downstream outputs
    state.stationary_series = pd.Series([1, 2])
    state.box_cox_lambda = 0.5
    state.diff_order_d = 1
    state.diff_order_D = 1
    state.seasonal_period_s = 12
    state.transformations = [{"type": "diff"}]
    state.detected_cycles = [{"freq": 0.1}]
    state.fitted_model = object()
    state.model_order = (1, 1, 1)
    state.residuals = np.array([1, 2])
    state.is_seasonal = True
    state.validation_passed = True
    
    state.original_series = pd.Series([1, 2, 3, 4])
    assert state.original_series is not None
    assert state.stationary_series is None
    assert state.box_cox_lambda is None
    assert state.diff_order_d == 0
    assert state.diff_order_D == 0
    assert state.seasonal_period_s == 1
    assert state.transformations == []
    assert state.detected_cycles == []
    assert state.fitted_model is None
    assert state.model_order is None
    assert state.residuals is None
    assert state.is_seasonal is False
    assert state.validation_passed is False
    assert state.validation_run is False
    
    print("AnalysisState Invalidation tests passed!")

def test_gui_gating_and_indicators():
    print("Testing GUI Gating and Indicators...")
    
    # Create QApplication instance if not already running
    app = QApplication.instance() or QApplication(sys.argv)
    
    window = MainWindow()
    
    # Initial state (nothing loaded)
    # Tab 0 is always enabled.
    assert window.tab_widget.isTabEnabled(0) is True
    # Tabs 1 to 5 should be disabled
    for i in range(1, 6):
        assert window.tab_widget.isTabEnabled(i) is False, f"Tab {i} should be disabled initially"
        
    # All indicators should be neutral
    for i in range(6):
        assert window.indicators[i].color.name() == "#9ca3af", f"Indicator {i} should be neutral (#9ca3af) initially, got {window.indicators[i].color.name()}"
        
    # 1. Load data
    window.state.original_series = pd.Series([1, 2, 3])
    window.update_ui_from_state()
    
    assert window.tab_widget.isTabEnabled(1) is True
    for i in range(2, 6):
        assert window.tab_widget.isTabEnabled(i) is False
        
    assert window.indicators[0].color.name() == "#16a34a"  # Green / pass
    for i in range(1, 6):
        assert window.indicators[i].color.name() == "#9ca3af"  # Neutral
        
    # 2. Make stationary
    window.state.stationary_series = pd.Series([1, 2, 3])
    window.update_ui_from_state()
    
    assert window.tab_widget.isTabEnabled(1) is True
    assert window.tab_widget.isTabEnabled(2) is True
    assert window.tab_widget.isTabEnabled(3) is True
    assert window.tab_widget.isTabEnabled(4) is False
    assert window.tab_widget.isTabEnabled(5) is False
    
    assert window.indicators[0].color.name() == "#16a34a"
    assert window.indicators[1].color.name() == "#16a34a"
    assert window.indicators[2].color.name() == "#9ca3af"  # No cycles detected yet
    
    # Detect cycles
    window.state.detected_cycles = [{"freq": 0.083}]
    window.update_ui_from_state()
    assert window.indicators[2].color.name() == "#16a34a"
    
    # 3. Model fitted (but validation not run yet)
    window.state.fitted_model = object()
    window.update_ui_from_state()
    
    assert window.tab_widget.isTabEnabled(4) is True
    assert window.tab_widget.isTabEnabled(5) is False
    
    assert window.indicators[3].color.name() == "#16a34a"
    # Indicator 4 (Validation) must be neutral because validation_run is False
    assert window.indicators[4].color.name() == "#9ca3af", f"Got validation indicator color: {window.indicators[4].color.name()}"
    
    # 4. Validation passed
    window.state.validation_passed = True
    window.update_ui_from_state()
    
    assert window.tab_widget.isTabEnabled(5) is True
    assert window.indicators[4].color.name() == "#16a34a"  # Pass
    assert window.indicators[5].color.name() == "#16a34a"  # Pass
    
    # 5. Validation failed
    window.state.validation_passed = False
    window.update_ui_from_state()
    
    assert window.tab_widget.isTabEnabled(5) is False
    assert window.indicators[4].color.name() == "#dc2626"  # Red / fail
    assert window.indicators[5].color.name() == "#9ca3af"  # Neutral
    
    # 6. Change original series -> resets everything downstream
    window.state.original_series = pd.Series([4, 5, 6])
    window.update_ui_from_state()
    
    assert window.tab_widget.isTabEnabled(1) is True
    for i in range(2, 6):
        assert window.tab_widget.isTabEnabled(i) is False
        
    assert window.indicators[0].color.name() == "#16a34a"
    for i in range(1, 6):
        assert window.indicators[i].color.name() == "#9ca3af"
        
    print("GUI Gating and Indicator tests passed!")

if __name__ == "__main__":
    test_analysis_state_invalidation()
    test_gui_gating_and_indicators()
    print("All tests completed successfully!")
