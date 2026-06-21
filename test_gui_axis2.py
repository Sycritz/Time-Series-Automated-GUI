import sys
import os
import pandas as pd
import numpy as np

# Force headless Qt
os.environ["QT_API"] = "pyside6"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from main import MainWindow

def test_gui_spectral_workflow():
    print("Testing GUI Tab 3 Spectral Integration Workflow...")
    
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    
    # 1. Load the airline passengers CSV file (from Tab 1 helper steps)
    csv_path = os.path.join(os.path.dirname(__file__), "datasets", "international-airline-passengers.csv")
    tab1 = window.tabs[0]
    from axis1_preprocessing import load_csv
    tab1.df = load_csv(csv_path)
    
    # Set selections and select columns
    tab1.time_col_combo.clear()
    tab1.time_col_combo.addItems(["Select Column...", "Month", "Passengers"])
    tab1.val_col_combo.clear()
    tab1.val_col_combo.addItems(["Select Column...", "Month", "Passengers"])
    
    tab1.time_col_combo.setCurrentIndex(1) # Month
    tab1.val_col_combo.setCurrentIndex(2) # Passengers
    tab1.on_columns_selected()
    
    # 2. Transform to stationarity (from Tab 2 helper steps) to unlock Tab 3
    tab2 = window.tabs[1]
    tab2.lambda_spin.setValue(0.0) # Log transform
    tab2.d_spin.setValue(1)
    tab2.D_spin.setValue(1)
    tab2.s_spin.setValue(12)
    tab2.apply_transformations()
    
    # Verify Tab 3 is enabled
    assert window.tab_widget.isTabEnabled(2) is True
    
    # 3. Interact with Tab 3 (SpectralTab)
    tab3 = window.tabs[2]
    
    # Verify default widget states
    assert tab3.taper_combo.currentText() == "None"
    assert tab3.window_combo.currentText() == "Daniell"
    assert tab3.freq_period_toggle.currentText() == "Frequency Scale"
    
    # Test setting parameters
    tab3.taper_combo.setCurrentText("Cosine Bell")
    tab3.taper_pct_spin.setValue(0.15)
    tab3.window_combo.setCurrentText("Bartlett")
    tab3.bandwidth_spin.setValue(12)
    
    # Estimate Spectrum
    tab3.estimate_btn.click() # triggers self.estimate_spectrum()
    
    # Assert data was calculated and cached
    assert tab3.raw_freqs is not None
    assert tab3.raw_pgram is not None
    assert tab3.smooth_freqs is not None
    assert tab3.smooth_val is not None
    assert tab3.ci_lower is not None
    assert tab3.ci_upper is not None
    
    # Frequencies must go from 0 to pi
    assert np.isclose(tab3.raw_freqs[-1], np.pi, atol=0.1)
    assert len(tab3.smooth_freqs) == 512
    assert np.isclose(tab3.smooth_freqs[-1], np.pi)
    
    # Detect Cycles
    tab3.detect_btn.click() # triggers self.detect_cycles_click()
    
    # Verify detected cycles stored in state
    assert len(window.state.detected_cycles) > 0
    # There should be a significant cycle representing seasonality of 12 or harmonic
    # Let's inspect the results browser content
    results_text = tab3.results_browser.toHtml()
    assert "Detected Cycles" in results_text
    
    # Verify indicator 2 (Spectral/Cycle detection tab status) is pass (green)
    assert window.indicators[2].color.name() == "#16a34a"
    
    # Test changing the frequency/period scale toggle
    tab3.freq_period_toggle.setCurrentText("Period Scale") # calls plot_spectrum()
    
    # Check that scale updated on the plot axes
    ax = tab3.plot_widget.canvas.axes
    assert ax.get_xlabel() == "Period (samples)"
    
    # Switch back to frequency scale
    tab3.freq_period_toggle.setCurrentText("Frequency Scale")
    ax = tab3.plot_widget.canvas.axes
    assert ax.get_xlabel() == "Frequency (radians/sample)"
    
    # 4. Test parametric spectrum overlay when a model is fitted
    window.state.fitted_model = object()  # Dummy fitted model
    # Simulate ARMA(1, 0) params
    window.state.model_params = {
        "ar": [0.7],
        "ma": [],
        "sigma2": 0.5
    }
    
    # Trigger MainWindow state update
    window.update_ui_from_state() # calls tab3.update_parametric_spectrum_overlay()
    
    # Verify that the parametric spectrum curve is drawn on the plot
    # The axes lines should contain the parametric line now (3 lines: raw periodogram, smoothed spectrum, parametric ARMA spectrum)
    ax = tab3.plot_widget.canvas.axes
    lines = ax.get_lines()
    labels = [line.get_label() for line in lines]
    assert "Parametric ARMA Spectrum" in labels
    
    print("GUI Tab 3 Spectral Integration Workflow tests passed!")

if __name__ == "__main__":
    test_workflow_err = None
    try:
        test_gui_spectral_workflow()
    except Exception as e:
        import traceback
        traceback.print_exc()
        test_workflow_err = e
        
    if test_workflow_err is not None:
        sys.exit(1)
    else:
        print("All GUI spectral workflow tests passed successfully!")
