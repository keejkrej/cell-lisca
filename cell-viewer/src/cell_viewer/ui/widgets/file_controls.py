"""File navigation control widget."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal


class FileControls(QWidget):
    """Widget for file navigation controls."""
    
    # Signals
    open_folder_requested = Signal()
    prev_file_requested = Signal()
    next_file_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the file controls UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Open folder button
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.setFixedWidth(100)
        self.open_folder_button.clicked.connect(self.open_folder_requested.emit)
        layout.addWidget(self.open_folder_button)
        
        # Navigation buttons
        self.prev_button = QPushButton("Previous")
        self.prev_button.setFixedWidth(100)
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self.prev_file_requested.emit)
        layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next")
        self.next_button.setFixedWidth(100)
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.next_file_requested.emit)
        layout.addWidget(self.next_button)
        
        # Current file label
        self.file_label = QLabel("No file selected")
        layout.addWidget(self.file_label)
        
        layout.addStretch()
        
    def update_navigation_state(self, can_go_prev: bool, can_go_next: bool):
        """Update the state of navigation buttons"""
        self.prev_button.setEnabled(can_go_prev)
        self.next_button.setEnabled(can_go_next)
        
    def set_file_label(self, text: str):
        """Update the file label text"""
        self.file_label.setText(text)