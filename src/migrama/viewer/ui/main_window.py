"""Main window for the cell viewer application."""

import os

import matplotlib
import numpy as np
import yaml
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NPY Stack Viewer")
        self.setMinimumSize(1000, 700)

        # Data state
        self.stack = None  # Normalized stack for display
        self.original_stack = None  # Original stack for saving
        self.current_frame = 0
        self.total_frames = 0
        self.file_path = None
        self.start_frame = None
        self.end_frame = None
        self.current_folder = None
        self.save_folder = None
        self.current_channel = 0
        self.current_file_index = -1
        self.npy_files = []

        # Metadata from YAML
        self.channel_names: list[str] | None = None
        self.segmentation_channel_index: int | None = None

        # UI state
        self.img_display = None

        # Initialize UI
        self._init_ui()

        # Timer for autoplay
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # 100ms between frames (10 fps)
        self.timer.timeout.connect(self._advance_frame)

    def _init_ui(self):
        """Initialize the user interface"""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # File Controls Group
        file_group = QGroupBox("File Navigation")
        file_layout = QVBoxLayout()

        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self._handle_open_folder)
        file_layout.addWidget(self.open_folder_button)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self._handle_prev_file)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next ▶")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self._handle_next_file)
        nav_layout.addWidget(self.next_button)
        file_layout.addLayout(nav_layout)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Frame Controls Group
        frame_group = QGroupBox("Frame Controls")
        frame_layout = QVBoxLayout()

        # Frame slider
        slider_label = QLabel("Frame Navigation:")
        frame_layout.addWidget(slider_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._handle_slider_change)
        frame_layout.addWidget(self.slider)

        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self.frame_label)

        # Autoplay button
        self.autoplay_button = QPushButton("▶ Play")
        self.autoplay_button.setCheckable(True)
        self.autoplay_button.setEnabled(False)
        self.autoplay_button.clicked.connect(self._handle_autoplay_toggle)
        frame_layout.addWidget(self.autoplay_button)

        # Channel selector
        channel_label = QLabel("Channel Selection:")
        frame_layout.addWidget(channel_label)

        self.channel_selector = QComboBox()
        # Will be populated dynamically when file is loaded
        self.channel_selector.setEnabled(False)
        self.channel_selector.currentIndexChanged.connect(self._handle_channel_change)
        frame_layout.addWidget(self.channel_selector)

        frame_group.setLayout(frame_layout)
        left_layout.addWidget(frame_group)

        # Interval Controls Group
        interval_group = QGroupBox("Interval Selection")
        interval_layout = QVBoxLayout()

        mark_layout = QHBoxLayout()
        self.start_frame_button = QPushButton("Mark Start")
        self.start_frame_button.setEnabled(False)
        self.start_frame_button.clicked.connect(self._handle_mark_start)
        mark_layout.addWidget(self.start_frame_button)

        self.end_frame_button = QPushButton("Mark End")
        self.end_frame_button.setEnabled(False)
        self.end_frame_button.clicked.connect(self._handle_mark_end)
        mark_layout.addWidget(self.end_frame_button)
        interval_layout.addLayout(mark_layout)

        self.interval_label = QLabel("Interval: Not set")
        self.interval_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        interval_layout.addWidget(self.interval_label)

        # Save controls
        save_label = QLabel("Save Options:")
        interval_layout.addWidget(save_label)

        self.save_folder_button = QPushButton("Set Save Folder")
        self.save_folder_button.clicked.connect(self._handle_set_save_folder)
        interval_layout.addWidget(self.save_folder_button)

        self.save_interval_button = QPushButton("💾 Save Interval")
        self.save_interval_button.setEnabled(False)
        self.save_interval_button.clicked.connect(self._handle_save_interval)
        interval_layout.addWidget(self.save_interval_button)

        interval_group.setLayout(interval_layout)
        left_layout.addWidget(interval_group)

        # Add stretch to push controls to top
        left_layout.addStretch()

        # Right panel - Image viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Create matplotlib figure and canvas
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis("off")
        right_layout.addWidget(self.canvas)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        splitter.setStretchFactor(1, 1)  # Right panel stretches

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    # File operations
    def _handle_open_folder(self):
        """Handle opening a folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "", QFileDialog.Option.ShowDirsOnly)
        if folder_path:
            self._load_folder(folder_path)

    def _load_folder(self, folder_path):
        """Load NPY files from the selected folder"""
        self.current_folder = folder_path
        self.npy_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".npy")]
        self.npy_files.sort()

        if self.npy_files:
            self.current_file_index = 0
            self._load_current_file()
            self._update_navigation_buttons()
            self.statusBar.showMessage(f"Selected folder: {folder_path}")
        else:
            self.current_file_index = -1
            self.file_label.setText("No NPY files found")
            self._update_navigation_buttons()

    def _load_current_file(self):
        """Load the current file in the folder"""
        if 0 <= self.current_file_index < len(self.npy_files):
            file_path = os.path.join(self.current_folder, self.npy_files[self.current_file_index])
            self.file_label.setText(f"File: {self.npy_files[self.current_file_index]}")
            self._load_file(file_path)

    def _normalize_stack(self, stack):
        """Normalize each channel using 1st and 99.9th percentiles of positive values.
        Segmentation channel (if any) is NOT normalized here and will be displayed from original data.
        """
        normalized_stack = np.zeros(stack.shape, dtype=np.uint8)
        num_channels = stack.shape[1]

        # Process each channel separately
        for c in range(num_channels):
            # Skip normalization for segmentation channel
            if self.segmentation_channel_index is not None and c == self.segmentation_channel_index:
                # Leave zeros; segmentation will be rendered from original_stack
                normalized_stack[:, c, :, :] = 0
                continue

            channel_data = stack[:, c, :, :]  # (t, y, x)

            # Get only positive values for percentile calculation
            positive_mask = channel_data > 0
            if positive_mask.any():
                positive_values = channel_data[positive_mask]

                # Calculate 0.1% and 99.9% percentiles
                low = np.percentile(positive_values, 0.1)
                high = np.percentile(positive_values, 99.9)

                # Normalize to 0-255 range
                if high > low:
                    normalized_channel = (channel_data - low) / (high - low) * 255
                    normalized_channel = np.clip(normalized_channel, 0, 255)
                    normalized_channel = np.round(normalized_channel).astype(np.uint8)
                else:
                    # If no range, just scale to middle gray
                    normalized_channel = np.full(channel_data.shape, 128, dtype=np.uint8)
            else:
                # If no positive values, set to zeros
                normalized_channel = np.zeros(channel_data.shape, dtype=np.uint8)

            normalized_stack[:, c, :, :] = normalized_channel

        return normalized_stack

    def _load_file(self, file_path):
        """Load a NPY file"""
        try:
            self.original_stack = np.load(file_path)
            if len(self.original_stack.shape) != 4:
                self.statusBar.showMessage("Error: Stack must be 4D (time, channels, height, width)")
                self._reset_stack_state()
                return

            # Attempt to read sidecar YAML metadata (same basename)
            self.channel_names = None
            self.segmentation_channel_index = None
            self._original_metadata = None  # Store original metadata for trimmed files
            metadata_path = os.path.splitext(file_path)[0] + ".yaml"
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path) as f:
                        meta = yaml.safe_load(f) or {}
                    self._original_metadata = meta  # Store original metadata
                    ch = meta.get("channels")
                    if isinstance(ch, list) and all(isinstance(x, str) for x in ch):
                        self.channel_names = ch
                        # Find 'segmentation' channel by exact name (case-insensitive match also)
                        for i, name in enumerate(ch):
                            if name == "segmentation" or name.lower().strip() == "segmentation":
                                self.segmentation_channel_index = i
                                break
                except Exception as e:
                    self.statusBar.showMessage(f"Warning: Failed to load YAML metadata: {e}")

            # Get number of channels and update selector
            num_channels = self.original_stack.shape[1]
            self._update_channel_selector(num_channels)

            # Normalize the stack for display (non-segmentation channels only)
            self.statusBar.showMessage(f"Normalizing stack with {num_channels} channels...")
            self.stack = self._normalize_stack(self.original_stack)

            self.total_frames = self.stack.shape[0]
            self.current_frame = 0
            self.file_path = file_path

            # Reset interval state before trying to load
            self.start_frame = None
            self.end_frame = None
            self._load_interval()

            # Enable controls
            self.slider.setRange(0, self.total_frames - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)
            self.autoplay_button.setEnabled(True)
            self.start_frame_button.setEnabled(True)
            self.end_frame_button.setEnabled(True)
            self.channel_selector.setEnabled(True)
            self.img_display = None  # Reset image display for new stack

            self.statusBar.showMessage(f"Loaded stack: {file_path} ({num_channels} channels)")
            self._update_view()

        except Exception as e:
            self.statusBar.showMessage(f"Error loading file: {str(e)}")
            self._reset_stack_state()

    def _reset_stack_state(self):
        """Reset the stack-related state"""
        self.stack = None
        self.original_stack = None
        self.channel_names = None
        self.segmentation_channel_index = None
        self.slider.setEnabled(False)
        self.autoplay_button.setEnabled(False)
        self.start_frame_button.setEnabled(False)
        self.end_frame_button.setEnabled(False)
        self.save_interval_button.setEnabled(False)
        self.channel_selector.setEnabled(False)
        self.slider.setRange(0, 0)
        self.frame_label.setText("Frame: 0/0")
        self.interval_label.setText("Interval: Not set")
        self.start_frame = None
        self.end_frame = None

    def _update_navigation_buttons(self):
        """Update the state of navigation buttons"""
        self.prev_button.setEnabled(self.current_file_index > 0)
        self.next_button.setEnabled(self.current_file_index < len(self.npy_files) - 1)

    def _handle_prev_file(self):
        """Handle loading previous file"""
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self._load_current_file()
            self._update_navigation_buttons()

    def _handle_next_file(self):
        """Handle loading next file"""
        if self.current_file_index < len(self.npy_files) - 1:
            self.current_file_index += 1
            self._load_current_file()
            self._update_navigation_buttons()

    # Frame operations
    def _handle_slider_change(self, value):
        """Handle slider value changes"""
        self.current_frame = value
        self._update_view()

    def _handle_autoplay_toggle(self, checked):
        """Handle autoplay button toggle"""
        if checked:
            self.timer.start()
            self.autoplay_button.setText("⏸ Pause")
        else:
            self.timer.stop()
            self.autoplay_button.setText("▶ Play")

    def _advance_frame(self):
        """Advance to the next frame"""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.slider.setValue(self.current_frame)
            self._update_view()
        else:
            # Loop back to start
            self.current_frame = 0
            self.slider.setValue(0)
            self._update_view()

    def _update_channel_selector(self, num_channels):
        """Update channel selector with available channels"""
        self.channel_selector.blockSignals(True)
        self.channel_selector.clear()

        # Add channel options dynamically, prefer names from YAML if available
        if self.channel_names and len(self.channel_names) == num_channels:
            for i, name in enumerate(self.channel_names):
                label = name
                if (
                    self.segmentation_channel_index is not None
                    and i == self.segmentation_channel_index
                    and name.lower().strip() != "segmentation"
                ):
                    label += " (segmentation)"
                self.channel_selector.addItem(label)
        else:
            for i in range(num_channels):
                suffix = (
                    " (segmentation)"
                    if (self.segmentation_channel_index is not None and i == self.segmentation_channel_index)
                    else ""
                )
                self.channel_selector.addItem(f"Channel {i}{suffix}")

        # Reset to first channel
        self.current_channel = 0
        self.channel_selector.setCurrentIndex(0)
        self.channel_selector.blockSignals(False)

    def _handle_channel_change(self, index):
        """Handle channel selection change"""
        self.current_channel = index
        self._update_view()
        # Status shows selected channel name if available
        if self.channel_names and 0 <= index < len(self.channel_names):
            name = self.channel_names[index]
        else:
            name = f"Channel {index}"
        self.statusBar.showMessage(f"Channel selected: {name}")

    def _update_view(self):
        """Update the display with current frame and channel"""
        if self.stack is None:
            return

        try:
            from matplotlib import colors

            is_seg = (
                self.segmentation_channel_index is not None and self.current_channel == self.segmentation_channel_index
            )

            if is_seg:
                # Render segmentation from original (unnormalized), categorical with a discrete colormap
                label_img = self.original_stack[self.current_frame, self.current_channel, :, :]
                # Ensure integer labels
                if np.issubdtype(label_img.dtype, np.floating):
                    label_img = np.round(label_img).astype(np.int64)
                else:
                    label_img = label_img.astype(np.int64)

                max_label = int(label_img.max()) if label_img.size else 0
                n_classes = max(2, max_label + 1)  # include background 0

                # Build a ListedColormap: background black, then tab20 colors cycling
                from matplotlib import cm

                base = cm.get_cmap("tab20", 20)
                colors_list = [(0.0, 0.0, 0.0, 1.0)]  # label 0 = black
                for i in range(1, n_classes):
                    colors_list.append(base((i - 1) % 20))
                cmap = matplotlib.colors.ListedColormap(colors_list, name="seg_cmap", N=n_classes)

                # Discrete boundaries for integer labels 0..max_label
                boundaries = np.arange(-0.5, max_label + 1.5, 1)
                norm = colors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

                recreate = self.img_display is None or getattr(self, "_last_mode_seg", None) is not True
                if recreate:
                    self.ax.clear()
                    self.ax.axis("off")
                    self.img_display = self.ax.imshow(label_img, cmap=cmap, norm=norm, interpolation="nearest")
                    self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
                else:
                    self.img_display.set_cmap(cmap)
                    self.img_display.set_norm(norm)
                    self.img_display.set_data(label_img)
                self._last_mode_seg = True
            else:
                # Continuous path: use normalized stack, grayscale 0..255
                current_image = self.stack[self.current_frame]  # (c, h, w)
                display_image = current_image[self.current_channel]  # (h, w)

                recreate = self.img_display is None or getattr(self, "_last_mode_seg", None) is True
                if recreate:
                    self.ax.clear()
                    self.ax.axis("off")
                    self.img_display = self.ax.imshow(display_image, cmap="gray", aspect="equal", vmin=0, vmax=255)
                    self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
                else:
                    self.img_display.set_data(display_image)
                self._last_mode_seg = False

            # Force the canvas to update
            self.canvas.draw()

            # Update frame info
            self.frame_label.setText(f"Frame: {self.current_frame + 1}/{self.total_frames}")

        except Exception as e:
            self.statusBar.showMessage(f"Error updating view: {str(e)}")

    # Interval operations
    def _handle_mark_start(self):
        """Handle marking the start frame"""
        self.start_frame = self.slider.value()
        self._update_interval_label()
        self._check_save_enabled()
        if self.end_frame is not None:
            self._save_interval_to_file()

    def _handle_mark_end(self):
        """Handle marking the end frame"""
        self.end_frame = self.slider.value()
        self._update_interval_label()
        self._check_save_enabled()
        if self.start_frame is not None:
            self._save_interval_to_file()

    def _update_interval_label(self):
        """Update the interval label with current start/end frames"""
        if self.start_frame is not None and self.end_frame is not None:
            self.interval_label.setText(f"Interval: {self.start_frame + 1} - {self.end_frame + 1}")
        elif self.start_frame is not None:
            self.interval_label.setText(f"Start: {self.start_frame + 1}")
        elif self.end_frame is not None:
            self.interval_label.setText(f"End: {self.end_frame + 1}")
        else:
            self.interval_label.setText("Interval: Not set")

    def _check_save_enabled(self):
        """Enable/disable save button based on interval state"""
        self.save_interval_button.setEnabled(
            self.start_frame is not None and self.end_frame is not None and self.start_frame != self.end_frame
        )

    def _get_interval_path(self):
        """Get the path to the interval YAML file"""
        if self.file_path is None:
            return None
        return os.path.splitext(self.file_path)[0] + "_interval.yaml"

    def _load_interval(self):
        """Load interval from YAML file"""
        try:
            interval_file = self._get_interval_path()
            if interval_file and os.path.exists(interval_file):
                with open(interval_file) as f:
                    data = yaml.safe_load(f) or {}
                    self.start_frame = data.get("start_frame")
                    self.end_frame = data.get("end_frame")
                    if self.start_frame is not None and self.end_frame is not None:
                        self._update_interval_label()
                        self._check_save_enabled()
        except Exception as e:
            self.statusBar.showMessage(f"Error loading interval: {str(e)}")

    def _save_interval_to_file(self):
        """Save interval to YAML file"""
        try:
            if self.start_frame is not None and self.end_frame is not None:
                # Ensure start < end
                start = min(self.start_frame, self.end_frame)
                end = max(self.start_frame, self.end_frame)
                self.start_frame = start
                self.end_frame = end

                interval_file = self._get_interval_path()
                if interval_file:
                    data = {"start_frame": self.start_frame, "end_frame": self.end_frame}
                    with open(interval_file, "w") as f:
                        yaml.safe_dump(data, f, sort_keys=False)
                    self.statusBar.showMessage(f"Interval set: {self.start_frame + 1} - {self.end_frame + 1}")
        except Exception as e:
            self.statusBar.showMessage(f"Error saving interval: {str(e)}")

    # Save operations
    def _handle_set_save_folder(self):
        """Handle setting the save folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Save Folder", "", QFileDialog.Option.ShowDirsOnly)
        if folder_path:
            self.save_folder = folder_path
            self.statusBar.showMessage(f"Save folder set to: {folder_path}")

    def _handle_save_interval(self):
        """Handle saving the marked interval"""
        if self.save_folder is None:
            self.statusBar.showMessage("Please set a save folder first")
            return

        if self.original_stack is None or self.start_frame is None or self.end_frame is None:
            self.statusBar.showMessage("Error: No interval marked or no stack loaded")
            return

        try:
            # Extract the interval from the ORIGINAL stack (not normalized)
            interval_stack = self.original_stack[self.start_frame : self.end_frame + 1]

            # Generate save filename
            if self.current_file_index >= 0 and self.npy_files:
                current_file = self.npy_files[self.current_file_index]
                base_name = os.path.splitext(current_file)[0]
                save_path = os.path.join(self.save_folder, f"{base_name}_trimmed.npy")

                # Save to new file
                np.save(save_path, interval_stack)

                # Save corresponding YAML metadata
                yaml_path = os.path.splitext(save_path)[0] + ".yaml"
                self._save_trimmed_yaml(yaml_path, self.start_frame, self.end_frame)

                self.statusBar.showMessage(f"Saved interval to {save_path}")
        except Exception as e:
            self.statusBar.showMessage(f"Error saving interval: {str(e)}")

    def _save_trimmed_yaml(self, yaml_path: str, start_frame: int, end_frame: int):
        """Save YAML metadata for trimmed NPY file with same structure as original"""
        try:
            metadata = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "channels": self.channel_names if self.channel_names else [],
            }

            # Add pattern_bbox if available from original metadata
            if hasattr(self, "_original_metadata") and self._original_metadata:
                if "pattern_bbox" in self._original_metadata:
                    metadata["pattern_bbox"] = self._original_metadata["pattern_bbox"]

            with open(yaml_path, "w") as f:
                yaml.safe_dump(metadata, f, sort_keys=False)
        except Exception as e:
            self.statusBar.showMessage(f"Warning: Failed to save YAML metadata: {str(e)}")

    def resizeEvent(self, event):  # noqa: N802 (Qt method override)
        """Handle window resize events"""
        super().resizeEvent(event)
        if self.img_display is not None:
            self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas.draw()
