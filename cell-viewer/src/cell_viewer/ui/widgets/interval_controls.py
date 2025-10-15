"""Interval marking and saving control widget."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal


class IntervalControls(QWidget):
    """Widget for interval marking and saving controls."""
    
    # Signals
    mark_start_requested = Signal()
    mark_end_requested = Signal()
    set_save_folder_requested = Signal()
    save_interval_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the interval controls UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Interval marking controls
        interval_layout = QHBoxLayout()
        
        self.start_frame_button = QPushButton("Mark Start")
        self.start_frame_button.setFixedWidth(100)
        self.start_frame_button.setEnabled(False)
        self.start_frame_button.clicked.connect(self.mark_start_requested.emit)
        interval_layout.addWidget(self.start_frame_button)
        
        self.end_frame_button = QPushButton("Mark End")
        self.end_frame_button.setFixedWidth(100)
        self.end_frame_button.setEnabled(False)
        self.end_frame_button.clicked.connect(self.mark_end_requested.emit)
        interval_layout.addWidget(self.end_frame_button)
        
        self.interval_label = QLabel("Interval: Not set")
        interval_layout.addWidget(self.interval_label, stretch=1)
        
        layout.addLayout(interval_layout)
        
        # Save controls
        save_layout = QHBoxLayout()
        
        self.save_folder_button = QPushButton("Set Save Folder")
        self.save_folder_button.setFixedWidth(120)
        self.save_folder_button.clicked.connect(self.set_save_folder_requested.emit)
        save_layout.addWidget(self.save_folder_button)
        
        self.save_interval_button = QPushButton("Save Interval")
        self.save_interval_button.setFixedWidth(120)
        self.save_interval_button.setEnabled(False)
        self.save_interval_button.clicked.connect(self.save_interval_requested.emit)
        save_layout.addWidget(self.save_interval_button)
        
        save_layout.addStretch()
        
        layout.addLayout(save_layout)
        
    def set_interval_enabled(self, enabled: bool):
        """Enable or disable interval marking buttons"""
        self.start_frame_button.setEnabled(enabled)
        self.end_frame_button.setEnabled(enabled)
        
    def set_save_enabled(self, enabled: bool):
        """Enable or disable save interval button"""
        self.save_interval_button.setEnabled(enabled)
        
    def update_interval_label(self, text: str):
        """Update the interval label text"""
        self.interval_label.setText(text)
        
    def reset(self):
        """Reset controls to initial state"""
        self.start_frame_button.setEnabled(False)
        self.end_frame_button.setEnabled(False)
        self.save_interval_button.setEnabled(False)
        self.interval_label.setText("Interval: Not set")