import base64
import io
import json
import os
import re
import typing

import matplotlib.mathtext as mathtext
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QTextBrowser,
                               QVBoxLayout, QWidget)


def build_context_snapshot(state, current_tab_index: int) -> str:
    """
    Generates a compact string summary of the current stage (approx. 200-500 tokens).

    Includes: active tab name, original series length, transformations applied (Box-Cox lambda,
    diff orders, transformations log), model order, parameters (AR/MA, sigma2), validation test
    results if run (Ljung-Box p-value, Jarque-Bera p-value, cumulative periodogram pass/fail),
    and forecasting status.
    """
    TAB_NAMES = [
        "1. Data Load & Explore",
        "2. Transform to Stationarity",
        "3. Spectral Exploration & Cycle Detection",
        "4. Model Identification & Selection",
        "5. Model Validation & Residual Diagnostics",
        "6. Generate & View Forecasts",
    ]

    if 0 <= current_tab_index < len(TAB_NAMES):
        active_tab = TAB_NAMES[current_tab_index]
    else:
        active_tab = f"Unknown Tab (Index {current_tab_index})"

    lines = []
    lines.append(f"Active Tab: {active_tab}")

    # Series length
    if state.original_series is not None:
        lines.append(f"Original Series Length: {len(state.original_series)}")
    else:
        lines.append("Original Series Length: No data loaded")

    # Transformations applied
    lines.append("Transformations:")
    lines.append(
        f"  Box-Cox Lambda: {state.box_cox_lambda if state.box_cox_lambda is not None else 'None'}"
    )
    lines.append(f"  Differencing Order (d): {state.diff_order_d}")
    lines.append(
        f"  Seasonal Differencing Order (D): {state.diff_order_D} (period s = {state.seasonal_period_s})"
    )

    if state.transformations:
        trans_log = []
        for t in state.transformations:
            t_type = t.get("type")
            if t_type == "boxcox":
                trans_log.append(f"Box-Cox (lambda = {t.get('lambda')})")
            elif t_type == "diff":
                trans_log.append(f"Differencing (d = {t.get('d')})")
            elif t_type == "seasonal_diff":
                trans_log.append(
                    f"Seasonal Differencing (D = {t.get('D')}, s = {t.get('s')})"
                )
            else:
                trans_log.append(str(t))
        lines.append(f"  Transformation Log: {', '.join(trans_log)}")
    else:
        lines.append("  Transformation Log: None")

    # Model order and parameters
    if state.model_order is not None:
        lines.append(f"Model Order: {state.model_order}")
    else:
        lines.append("Model Order: No model fitted yet")

    if state.model_params is not None:
        ar = state.model_params.get("ar", [])
        ma = state.model_params.get("ma", [])
        sigma2 = state.model_params.get("sigma2", None)
        lines.append("Model Parameters:")
        lines.append(f"  AR Coefficients: {ar}")
        lines.append(f"  MA Coefficients: {ma}")
        lines.append(f"  Sigma2 (white noise variance): {sigma2}")
    else:
        lines.append("Model Parameters: None")

    # Validation test results
    if state.validation_run:
        if state.residuals is not None and state.model_order is not None:
            order = state.model_order
            p_val = order[0]
            q_val = order[2]
            P_val = order[3] if len(order) == 7 else 0
            Q_val = order[5] if len(order) == 7 else 0
            p = p_val + P_val
            q = q_val + Q_val
            try:
                from axis4_validation import run_all_diagnostics

                results = run_all_diagnostics(state.residuals, p, q)
                lb_p = results.get("lb_p_value_20", float("nan"))
                jb_p = results.get("jb_pvalue", float("nan"))
                cp_pass = results.get("cp_pass", False)

                lb_p_str = f"{lb_p:.4f}" if not np.isnan(lb_p) else "NaN"
                jb_p_str = f"{jb_p:.4f}" if not np.isnan(jb_p) else "NaN"
                cp_pass_str = "PASS" if cp_pass else "FAIL"

                lines.append("Validation Diagnostics:")
                lines.append(f"  Ljung-Box (lag 20) p-value: {lb_p_str}")
                lines.append(f"  Jarque-Bera p-value: {jb_p_str}")
                lines.append(f"  Cumulative Periodogram Test: {cp_pass_str}")
                lines.append(
                    f"  Overall Verdict: {'PASS' if state.validation_passed else 'FAIL'}"
                )
            except Exception as e:
                lines.append(
                    f"Validation Diagnostics: Error running validation: {str(e)}"
                )
        else:
            lines.append(
                "Validation Diagnostics: Run flagged but residuals or model order missing"
            )
    else:
        lines.append("Validation Diagnostics: Not yet run")

    # Forecasting status
    if state.original_series is None:
        forecasting_status = "Cannot generate forecasts: No data loaded."
    elif state.fitted_model is None:
        forecasting_status = "Cannot generate forecasts: No model fitted yet."
    else:
        if not state.validation_run:
            forecasting_status = "Ready to generate forecasts. Model has been fitted, but validation diagnostics have not been run yet."
        else:
            if state.validation_passed:
                forecasting_status = "Ready to generate forecasts. Model fitted and successfully validated (passed diagnostic checks)."
            else:
                forecasting_status = "Ready to generate forecasts, but note that the model did not pass all validation diagnostic checks (use with caution)."
    lines.append(f"Forecasting Status: {forecasting_status}")

    return "\n".join(lines)


def render_latex_to_image(text: str) -> str:
    """
    Scans text for LaTeX formulas enclosed in $...$ or $$...$$.
    Uses matplotlib's TeX engine to render them to a PNG in-memory, base64 encodes it,
    and returns the text with math formulas replaced by inline HTML <img> tags.

    Normal text outside delimiters remains unmodified.
    """
    pattern = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)

    def replace_math(match):
        formula = match.group(1) or match.group(2)
        if not formula:
            return match.group(0)

        formula_stripped = formula.strip()
        math_expr = f"${formula_stripped}$"

        try:
            buf = io.BytesIO()
            # Render using matplotlib mathtext engine
            mathtext.math_to_image(math_expr, buf, format="png", dpi=120)
            img_bytes = buf.getvalue()
            base64_str = base64.b64encode(img_bytes).decode("utf-8")
            return f'<img src="data:image/png;base64,{base64_str}" />'
        except Exception:
            # Fallback to original text if rendering fails
            return match.group(0)

    return pattern.sub(replace_math, text)


class GeminiClient:
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def stream_chat(
        self, messages: list[dict], system_prompt: str = None
    ) -> typing.Generator[str, None, None]:
        if not self.api_key:
            raise ValueError("API key not configured.")

        client = genai.Client(api_key=self.api_key)

        contents = []
        for msg in messages:
            role = msg.get("role")
            if role == "error":
                continue
            api_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=api_role,
                    parts=[types.Part.from_text(text=msg.get("content", ""))],
                )
            )

        config = None
        if system_prompt:
            config = types.GenerateContentConfig(system_instruction=system_prompt)

        response = client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text


class StreamWorker(QThread):
    token_received = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        client: GeminiClient,
        prompt: str,
        history: list[dict],
        system_prompt: str,
        parent=None,
    ):
        super().__init__(parent)
        self.client = client
        self.prompt = prompt
        self.history = list(history)
        self.system_prompt = system_prompt
        self.full_response = ""

    def run(self):
        try:
            messages = self.history + [{"role": "user", "content": self.prompt}]
            for token in self.client.stream_chat(messages, self.system_prompt):
                self.full_response += token
                self.token_received.emit(token)
            self.finished.emit(self.full_response)
        except Exception as e:
            self.error.emit(str(e))


class APIKeyDialog(QDialog):
    def __init__(self, parent=None, current_key=""):
        super().__init__(parent)
        self.setWindowTitle("Configure Gemini API Key")
        self.setModal(True)
        self.resize(400, 150)

        layout = QVBoxLayout(self)

        self.label = QLabel("Enter your Gemini API Key:", self)
        layout.addWidget(self.label)

        self.key_input = QLineEdit(self)
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setText(current_key)
        self.key_input.setPlaceholderText("AIzaSy...")
        layout.addWidget(self.key_input)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def api_key(self) -> str:
        return self.key_input.text().strip()


class ChatPanel(QWidget):
    def __init__(self, parent=None, state=None, main_window=None):
        super().__init__(parent)
        self.state = state
        self.main_window = main_window
        self.client = GeminiClient(os.getenv("GEMINI_API_KEY"))
        self.history = []
        self.worker = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Header Layout
        header_layout = QHBoxLayout()
        self.title_label = QLabel("AI Chatbot Assistant", self)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        # Gear Button
        self.gear_button = QPushButton("⚙", self)
        self.gear_button.setToolTip("Configure Gemini API Key")
        self.gear_button.setFixedWidth(30)
        self.gear_button.clicked.connect(self.show_api_key_dialog)
        header_layout.addWidget(self.gear_button)

        # Clear Chat Button
        self.clear_button = QPushButton("Clear Chat", self)
        self.clear_button.clicked.connect(self.clear_chat)
        header_layout.addWidget(self.clear_button)

        layout.addLayout(header_layout)

        # Chat display
        self.chat_display = QTextBrowser(self)
        self.chat_display.setOpenExternalLinks(True)
        layout.addWidget(self.chat_display)

        # Input Layout
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Ask the statistics tutor...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)

        # Initial display update
        self._update_chat_display()

    def show_api_key_dialog(self):
        dialog = APIKeyDialog(self, current_key=self.client.api_key or "")
        if dialog.exec() == QDialog.Accepted:
            key = dialog.api_key()
            self.client.set_api_key(key)
            self._update_chat_display()

    def clear_chat(self):
        self.history = []
        self._update_chat_display()

    def send_message(self):
        if self.worker is not None:
            return

        prompt = self.input_field.text().strip()
        if not prompt:
            return

        self.input_field.clear()

        if not self.client.api_key:
            self.history.append(
                {
                    "role": "error",
                    "content": "Please configure your Gemini API key via the gear icon in the header.",
                }
            )
            self._update_chat_display()
            return

        self.history.append({"role": "user", "content": prompt})
        self._update_chat_display()

        self._set_controls_enabled(False)

        current_tab_index = 0
        if self.main_window and hasattr(self.main_window, "tab_widget"):
            current_tab_index = self.main_window.tab_widget.currentIndex()

        context_snapshot = ""
        if self.state is not None:
            context_snapshot = build_context_snapshot(self.state, current_tab_index)

        system_prompt = (
            "You are a patient time series statistics teaching assistant. "
            "You are grounded in Brockwell & Davis methodology and assist students with understanding the Box-Jenkins pipeline, "
            "stationarity transforms, spectral analysis, model identification, validation, and forecasting.\n\n"
            f"Here is the current context of the user's analysis:\n{context_snapshot}\n\n"
            "Provide helpful, accurate explanations. Use LaTeX formatting (e.g. $formula$ or $$block_formula$$) for equations where appropriate. "
            "Keep your answers focused, educational, and structured."
        )

        self.worker = StreamWorker(
            self.client, prompt, self.history[:-1], system_prompt, self
        )
        self.worker.token_received.connect(self.on_token_received)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_token_received(self, token: str):
        if not self.history or self.history[-1]["role"] != "assistant":
            self.history.append({"role": "assistant", "content": token})
        else:
            self.history[-1]["content"] += token
        self._update_chat_display()

    def on_finished(self, full_response: str):
        if not self.history or self.history[-1]["role"] != "assistant":
            self.history.append({"role": "assistant", "content": full_response})
        else:
            self.history[-1]["content"] = full_response
        self._update_chat_display()
        self._set_controls_enabled(True)
        self.worker = None

    def on_error(self, error_message: str):
        self.history.append({"role": "error", "content": f"Error: {error_message}"})
        self._update_chat_display()
        self._set_controls_enabled(True)
        self.worker = None

    def _set_controls_enabled(self, enabled: bool):
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)

    def _update_chat_display(self):
        if not self.client.api_key:
            self.chat_display.setHtml(
                '<div style="color: #6B7280; font-style: italic; padding: 10px; text-align: center; font-family: sans-serif;">'
                "Please configure your Gemini API key via the gear icon in the header..."
                "</div>"
            )
            return

        html = ""
        for msg in self.history:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                formatted_content = render_latex_to_image(content)
                formatted_content = formatted_content.replace("\n", "<br>")
                html += f"""
                <table align="right" style="margin: 4px; border-collapse: collapse;">
                    <tr>
                        <td style="background-color: #2563EB; color: white; padding: 8px; border-radius: 8px; font-family: sans-serif;">
                            {formatted_content}
                        </td>
                    </tr>
                </table>
                <div style="clear: both;"></div>
                """
            elif role == "assistant":
                formatted_content = render_latex_to_image(content)
                formatted_content = formatted_content.replace("\n", "<br>")
                html += f"""
                <table align="left" style="margin: 4px; border-collapse: collapse;">
                    <tr>
                        <td style="background-color: #F5F6F8; color: black; padding: 8px; border-radius: 8px; font-family: sans-serif;">
                            {formatted_content}
                        </td>
                    </tr>
                </table>
                <div style="clear: both;"></div>
                """
            elif role == "error":
                formatted_content = content.replace("\n", "<br>")
                html += f"""
                <table align="left" style="margin: 4px; border-collapse: collapse;">
                    <tr>
                        <td style="background-color: #FEE2E2; color: #DC2626; padding: 8px; border-radius: 8px; font-family: sans-serif; font-weight: bold;">
                            {formatted_content}
                        </td>
                    </tr>
                </table>
                <div style="clear: both;"></div>
                """

        self.chat_display.setHtml(html)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        self.chat_display.moveCursor(QTextCursor.End)
