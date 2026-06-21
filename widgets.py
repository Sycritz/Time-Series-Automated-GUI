import os
os.environ["QT_API"] = "pyside6"
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog
from PySide6.QtGui import QPainter, QColor, QBrush
from PySide6.QtCore import QSize, Qt

class StatusIndicator(QWidget):
    """
    QPainter-drawn status circle indicator (green/red/gray) for tab headers or panels.
    No emojis, professional styling.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(16, 16))
        self.color = QColor("#9CA3AF")  # Neutral default (gray)
        
    def set_status(self, status: str):
        # status can be 'neutral' (gray), 'pass' (green), 'fail' (red)
        if status == 'pass':
            self.color = QColor("#16A34A")
        elif status == 'fail':
            self.color = QColor("#DC2626")
        else:
            self.color = QColor("#9CA3AF")
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.color))
        # Draw a centered circle
        painter.drawEllipse(3, 3, 10, 10)


class PlotCanvas(FigureCanvasQTAgg):
    """
    Custom subclass of FigureCanvasQTAgg configured with clean, professional light theme styling.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor='#FFFFFF')
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Configure default plot styling
        self.axes.tick_params(colors='#1F2937')
        self.axes.xaxis.label.set_color('#1F2937')
        self.axes.yaxis.label.set_color('#1F2937')
        self.axes.title.set_color('#1F2937')


class PlotWidget(QWidget):
    """
    Reusable composite widget that packages PlotCanvas, NavigationToolbar2QT,
    and a PNG/PDF export button in a clean layout.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Style the toolbar to look clean
        self.toolbar.setStyleSheet("background-color: transparent; border: none;")
        
        # Create export button
        self.export_btn = QPushButton("Export Plot", self)
        self.export_btn.clicked.connect(self.export_plot)
        
        # Layout top control row (toolbar + export button)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.toolbar)
        top_layout.addStretch()
        top_layout.addWidget(self.export_btn)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.addLayout(top_layout)
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
    def export_plot(self):
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Plot", "", "PNG Image (*.png);;PDF Document (*.pdf)"
        )
        if file_path:
            # Save the figure, maintaining clean white background
            self.canvas.figure.savefig(
                file_path, 
                facecolor=self.canvas.figure.get_facecolor(), 
                bbox_inches='tight'
            )
