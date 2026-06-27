import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests
from PySide6.QtWidgets import QApplication, QLineEdit

from chat_assistant import (APIKeyDialog, ChatPanel, GeminiClient,
                            StreamWorker, build_context_snapshot,
                            render_latex_to_image)
from state import AnalysisState


def test_build_context_snapshot_empty():
    state = AnalysisState()
    snapshot = build_context_snapshot(state, current_tab_index=0)

    assert "Active Tab: 1. Data Load & Explore" in snapshot
    assert "Original Series Length: No data loaded" in snapshot
    assert "Box-Cox Lambda: None" in snapshot
    assert "Differencing Order (d): 0" in snapshot
    assert "Seasonal Differencing Order (D): 0 (period s = 1)" in snapshot
    assert "Transformation Log: None" in snapshot
    assert "Model Order: No model fitted yet" in snapshot
    assert "Model Parameters: None" in snapshot
    assert "Validation Diagnostics: Not yet run" in snapshot
    assert "Forecasting Status: Cannot generate forecasts: No data loaded." in snapshot


def test_build_context_snapshot_data_loaded():
    state = AnalysisState()
    state.original_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    snapshot = build_context_snapshot(state, current_tab_index=1)

    assert "Active Tab: 2. Transform to Stationarity" in snapshot
    assert "Original Series Length: 5" in snapshot
    assert "Model Order: No model fitted yet" in snapshot
    assert (
        "Forecasting Status: Cannot generate forecasts: No model fitted yet."
        in snapshot
    )


def test_build_context_snapshot_model_fitted_no_validation():
    state = AnalysisState()
    state.original_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    state.box_cox_lambda = 0.5
    state.diff_order_d = 1
    state.transformations = [
        {"type": "boxcox", "lambda": 0.5},
        {"type": "diff", "d": 1},
    ]
    # Simulate a fitted model
    state.fitted_model = object()
    state.model_order = (1, 1, 1)
    state.model_params = {"ar": [0.5], "ma": [-0.3], "sigma2": 1.25}

    snapshot = build_context_snapshot(state, current_tab_index=3)

    assert "Active Tab: 4. Model Identification & Selection" in snapshot
    assert "Original Series Length: 5" in snapshot
    assert "Box-Cox (lambda = 0.5)" in snapshot
    assert "Differencing (d = 1)" in snapshot
    assert "Model Order: (1, 1, 1)" in snapshot
    assert "AR Coefficients: [0.5]" in snapshot
    assert "MA Coefficients: [-0.3]" in snapshot
    assert "Sigma2 (white noise variance): 1.25" in snapshot
    assert "Validation Diagnostics: Not yet run" in snapshot
    assert (
        "Forecasting Status: Ready to generate forecasts. Model has been fitted, but validation diagnostics have not been run yet."
        in snapshot
    )


def test_build_context_snapshot_validation_run_passed():
    state = AnalysisState()
    state.original_series = pd.Series(np.random.randn(100))
    state.fitted_model = object()
    state.model_order = (0, 0, 0)
    state.model_params = {"ar": [], "ma": [], "sigma2": 1.0}
    # Simulate white noise residuals
    state.residuals = np.random.randn(100)
    state.validation_run = True
    state.validation_passed = True

    snapshot = build_context_snapshot(state, current_tab_index=4)

    assert "Active Tab: 5. Model Validation & Residual Diagnostics" in snapshot
    assert "Validation Diagnostics:" in snapshot
    assert "Ljung-Box (lag 20) p-value:" in snapshot
    assert "Jarque-Bera p-value:" in snapshot
    assert "Cumulative Periodogram Test:" in snapshot
    assert "Overall Verdict: PASS" in snapshot
    assert (
        "Forecasting Status: Ready to generate forecasts. Model fitted and successfully validated (passed diagnostic checks)."
        in snapshot
    )


def test_build_context_snapshot_validation_run_failed():
    state = AnalysisState()
    state.original_series = pd.Series(np.random.randn(100))
    state.fitted_model = object()
    state.model_order = (0, 0, 0)
    state.model_params = {"ar": [], "ma": [], "sigma2": 1.0}

    # Highly correlated residuals to guarantee failure
    residuals = np.zeros(100)
    residuals[0] = np.random.randn()
    for i in range(1, 100):
        residuals[i] = 0.95 * residuals[i - 1] + np.random.randn()
    state.residuals = residuals

    state.validation_run = True
    state.validation_passed = False

    snapshot = build_context_snapshot(state, current_tab_index=5)

    assert "Active Tab: 6. Generate & View Forecasts" in snapshot
    assert "Validation Diagnostics:" in snapshot
    # It should show FAIL on Cumulative Periodogram or Ljung-Box test
    assert "Overall Verdict: FAIL" in snapshot
    assert (
        "Forecasting Status: Ready to generate forecasts, but note that the model did not pass all validation diagnostic checks (use with caution)."
        in snapshot
    )


def test_build_context_snapshot_unknown_tab():
    state = AnalysisState()
    snapshot = build_context_snapshot(state, current_tab_index=15)
    assert "Active Tab: Unknown Tab (Index 15)" in snapshot


def test_render_latex_to_image_plain_text():
    text = "This is plain text with no math."
    result = render_latex_to_image(text)
    assert result == text


def test_render_latex_to_image_inline():
    text = "The Ljung-Box statistic $Q_{LB}$ is used."
    result = render_latex_to_image(text)
    assert "The Ljung-Box statistic " in result
    assert " is used." in result
    assert '<img src="data:image/png;base64,' in result


def test_render_latex_to_image_block():
    text = "Let us compute: $$ \\sum_{i=1}^n x_i $$"
    result = render_latex_to_image(text)
    assert "Let us compute: " in result
    assert '<img src="data:image/png;base64,' in result


def test_render_latex_to_image_multiple():
    text = "Here is $\\sigma^2$ and also $$Q_{LB}$$."
    result = render_latex_to_image(text)
    assert "Here is " in result
    assert " and also " in result
    assert result.count("<img ") == 2


def test_render_latex_to_image_invalid():
    # If the expression is invalid TeX, it should fallback to the original matched text
    text = "Invalid formula: $\\invalidcommand{foo}$"
    result = render_latex_to_image(text)
    assert result == text


# --- GeminiClient Tests ---


@patch("chat_assistant.genai.Client")
def test_gemini_client_stream_chat_success(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Hello"
    mock_chunk2 = MagicMock()
    mock_chunk2.text = " world"
    mock_chunk3 = MagicMock()
    mock_chunk3.text = "!"

    mock_client.models.generate_content_stream.return_value = [
        mock_chunk1,
        mock_chunk2,
        mock_chunk3,
    ]

    client = GeminiClient(api_key="fake-key")
    tokens = list(
        client.stream_chat([{"role": "user", "content": "hi"}], "system prompt")
    )

    assert tokens == ["Hello", " world", "!"]
    mock_client_class.assert_called_once_with(api_key="fake-key")
    mock_client.models.generate_content_stream.assert_called_once()
    args, kwargs = mock_client.models.generate_content_stream.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert len(kwargs["contents"]) == 1
    assert kwargs["contents"][0].parts[0].text == "hi"
    assert kwargs["config"].system_instruction == "system prompt"


def test_gemini_client_stream_chat_missing_key():
    client = GeminiClient(api_key=None)
    with pytest.raises(ValueError, match="API key not configured"):
        list(client.stream_chat([{"role": "user", "content": "hi"}]))


@patch("chat_assistant.genai.Client")
def test_gemini_client_stream_chat_api_error(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.models.generate_content_stream.side_effect = Exception("API Error")

    client = GeminiClient(api_key="fake-key")
    with pytest.raises(Exception, match="API Error"):
        list(client.stream_chat([{"role": "user", "content": "hi"}]))


# --- StreamWorker Tests ---


@patch("chat_assistant.genai.Client")
def test_stream_worker_success(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Hello"
    mock_chunk2 = MagicMock()
    mock_chunk2.text = "!"
    mock_client.models.generate_content_stream.return_value = [mock_chunk1, mock_chunk2]

    client = GeminiClient(api_key="fake-key")
    worker = StreamWorker(client, "hi", [], "system")

    received_tokens = []
    finished_responses = []
    errors = []

    worker.token_received.connect(received_tokens.append)
    worker.finished.connect(finished_responses.append)
    worker.error.connect(errors.append)

    worker.run()

    assert received_tokens == ["Hello", "!"]
    assert finished_responses == ["Hello!"]
    assert len(errors) == 0


@patch("chat_assistant.genai.Client")
def test_stream_worker_error(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.models.generate_content_stream.side_effect = Exception("API Error")

    client = GeminiClient(api_key="fake-key")
    worker = StreamWorker(client, "hi", [], "system")

    received_tokens = []
    finished_responses = []
    errors = []

    worker.token_received.connect(received_tokens.append)
    worker.finished.connect(finished_responses.append)
    worker.error.connect(errors.append)

    worker.run()

    assert len(received_tokens) == 0
    assert len(finished_responses) == 0
    assert len(errors) == 1
    assert "API Error" in errors[0]


# --- UI Component Tests ---


def test_api_key_dialog():
    app = QApplication.instance() or QApplication([])
    dialog = APIKeyDialog(current_key="AIzaSy-test")
    assert dialog.api_key() == "AIzaSy-test"
    assert dialog.key_input.text() == "AIzaSy-test"
    assert dialog.key_input.echoMode() == QLineEdit.EchoMode.Password


def test_chat_panel_initial():
    app = QApplication.instance() or QApplication([])
    state = AnalysisState()
    panel = ChatPanel(state=state)

    # Check that initial state shows key warning prompt if api_key not set
    panel.client.set_api_key(None)
    panel._update_chat_display()
    assert "Please configure your Gemini API key" in panel.chat_display.toHtml()

    # Set api key and verify display updates
    panel.client.set_api_key("AIzaSy-test-key")
    panel._update_chat_display()
    # Now it should be empty/prompt not present since history is empty
    assert "Please configure your Gemini API key" not in panel.chat_display.toHtml()


def test_chat_panel_clear_chat():
    app = QApplication.instance() or QApplication([])
    panel = ChatPanel()
    panel.client.set_api_key("fake")
    panel.history = [{"role": "user", "content": "hello"}]
    panel._update_chat_display()

    panel.clear_chat()
    assert len(panel.history) == 0
    assert "hello" not in panel.chat_display.toHtml()


def test_mainwindow_integration_chat_panel():
    from main import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # Verify MainWindow instantiates correctly with splitter layout
    assert hasattr(window, "splitter")
    assert window.splitter is not None
    assert window.centralWidget() == window.splitter

    # Assert that chat_panel starts hidden
    assert hasattr(window, "chat_panel")
    assert window.chat_panel.isHidden() is True

    # Verify toggle button corner widget on tab_widget
    assert hasattr(window, "chat_toggle_btn")
    assert window.tab_widget.cornerWidget() == window.chat_toggle_btn
    assert window.chat_toggle_btn.isCheckable() is True
    assert window.chat_toggle_btn.isChecked() is False

    # Simulate checking the toggle button and verify that chat_panel becomes visible
    window.chat_toggle_btn.click()
    assert window.chat_toggle_btn.isChecked() is True
    assert window.chat_panel.isHidden() is False

    # Verify splitter sizes are set (850, 350)
    sizes = window.splitter.sizes()
    assert len(sizes) == 2

    # Simulate unchecking the toggle button
    window.chat_toggle_btn.click()
    assert window.chat_toggle_btn.isChecked() is False
    assert window.chat_panel.isHidden() is True


def test_chat_panel_clear_chat_integration():
    from main import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    # Fill chat panel history
    window.chat_panel.client.set_api_key("fake-key")
    window.chat_panel.history = [
        {"role": "user", "content": "Hello Assistant"},
        {"role": "assistant", "content": "Hello user, how can I help you?"},
    ]
    window.chat_panel._update_chat_display()
    assert "Hello Assistant" in window.chat_panel.chat_display.toHtml()
    assert "Hello user" in window.chat_panel.chat_display.toHtml()

    # Verify clicking "Clear Chat" resets the conversation log in the widget
    window.chat_panel.clear_button.click()

    assert len(window.chat_panel.history) == 0
    assert "Hello Assistant" not in window.chat_panel.chat_display.toHtml()
    assert "Hello user" not in window.chat_panel.chat_display.toHtml()
