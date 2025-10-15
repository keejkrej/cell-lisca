"""Image viewer widget for displaying NPY stack frames."""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class ImageViewer(QWidget):
    """Widget for displaying NPY stack images using matplotlib."""
    
    def __init__(self):
        super().__init__()
        self.img_display = None
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the image viewer UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        
        layout.addWidget(self.canvas)
        
    def show_image(self, image: np.ndarray):
        """Display the given image"""
        if image is None:
            return
            
        if self.img_display is None:
            # First time: create the image display
            self.img_display = self.ax.imshow(image, cmap='gray', aspect='equal')
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            # Update existing image data
            self.img_display.set_data(image)
            self.img_display.set_clim(vmin=image.min(), vmax=image.max())
            
        # Force the canvas to update
        self.canvas.draw()
        
    def clear(self):
        """Clear the image display"""
        self.ax.clear()
        self.ax.axis('off')
        self.img_display = None
        self.canvas.draw()
        
    def redraw(self):
        """Redraw the canvas (useful after resize)"""
        if self.img_display is not None:
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas.draw()