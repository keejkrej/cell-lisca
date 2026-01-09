"""Frame control widget for navigating through stack frames."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QSlider, QLabel, QPushButton, QComboBox
from PySide6.QtCore import Qt, Signal


class FrameControls(QWidget):
    """Widget for frame navigation and channel selection controls."""
    
    # Signals
    frame_changed = Signal(int)
    autoplay_toggled = Signal(bool)
    channel_changed = Signal(int)
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the frame controls UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMaximumWidth(400)
        self.slider.valueChanged.connect(self.frame_changed.emit)
        layout.addWidget(self.slider, stretch=1)
        
        # Frame counter
        self.frame_label = QLabel("Frame: 0/0")
        layout.addWidget(self.frame_label)
        
        # Channel selector
        channel_label = QLabel("Channel:")
        layout.addWidget(channel_label)
        
        self.channel_selector = QComboBox()
        # Will be populated dynamically when file is loaded
        self.channel_selector.setFixedWidth(150)
        self.channel_selector.setEnabled(False)
        self.channel_selector.currentIndexChanged.connect(self.channel_changed.emit)
        layout.addWidget(self.channel_selector)
        
        # Autoplay button
        self.autoplay_button = QPushButton("Play")
        self.autoplay_button.setFixedWidth(100)
        self.autoplay_button.setCheckable(True)
        self.autoplay_button.setEnabled(False)
        self.autoplay_button.clicked.connect(self._handle_autoplay)
        layout.addWidget(self.autoplay_button)
        
        layout.addStretch()
        
    def _handle_autoplay(self):
        """Handle autoplay button toggle"""
        is_playing = self.autoplay_button.isChecked()
        self.autoplay_button.setText("Pause" if is_playing else "Play")
        self.autoplay_toggled.emit(is_playing)
        
    def set_frame_range(self, minimum: int, maximum: int):
        """Set the slider range"""
        self.slider.setRange(minimum, maximum)
        self.slider.setEnabled(maximum > minimum)
        self.autoplay_button.setEnabled(maximum > minimum)
        self.channel_selector.setEnabled(maximum > minimum)
        
    def set_frame(self, frame: int):
        """Set the current frame without emitting signal"""
        self.slider.blockSignals(True)
        self.slider.setValue(frame)
        self.slider.blockSignals(False)
        
    def update_frame_label(self, current: int, total: int):
        """Update the frame label"""
        self.frame_label.setText(f"Frame: {current}/{total}")
        
    def get_current_frame(self) -> int:
        """Get the current frame value"""
        return self.slider.value()
        
    def update_channel_selector(self, num_channels: int):
        """Update channel selector with available channels"""
        self.channel_selector.blockSignals(True)
        self.channel_selector.clear()
        
        # Add channel options dynamically
        for i in range(num_channels):
            self.channel_selector.addItem(f"Channel {i}")
        
        # Reset to first channel
        self.channel_selector.setCurrentIndex(0)
        self.channel_selector.blockSignals(False)
        
    def get_current_channel(self) -> int:
        """Get the current channel selection"""
        return self.channel_selector.currentIndex()
        
    def reset(self):
        """Reset controls to initial state"""
        self.slider.setEnabled(False)
        self.autoplay_button.setEnabled(False)
        self.channel_selector.setEnabled(False)
        self.slider.setRange(0, 0)
        self.frame_label.setText("Frame: 0/0")
        self.autoplay_button.setChecked(False)
        self.autoplay_button.setText("Play")