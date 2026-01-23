"""Interrogation Tab - Model configuration and batch processing."""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QGroupBox, QLabel, QPushButton, QComboBox,
                             QCheckBox, QProgressBar, QTabWidget, QListWidget,
                             QListWidgetItem, QFileDialog, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QRadioButton, QButtonGroup, QScrollArea,
                             QLineEdit, QTextEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from pathlib import Path
from typing import Dict, Optional, List

from core import InterrogationDatabase, FileManager, TagFilterSettings, ONNXProviderSettings
from interrogators import CLIPInterrogator, WDInterrogator, CamieInterrogator
from ui.dialogs import create_clip_config_widget, create_wd_config_widget, create_camie_config_widget
from ui.dialogs_advanced import AdvancedImageInspectionDialog
from ui.workers import InterrogationWorker, DirectoryLoadWorker


class InterrogationTab(QWidget):
    """Tab for model configuration and interrogation processing."""

    # Signals
    directory_changed = pyqtSignal(str, bool)  # directory_path, recursive
    directory_loading_finished = pyqtSignal(list)  # image_paths (for cross-tab sharing)
    model_loaded = pyqtSignal(str)  # model_info
    model_unloaded = pyqtSignal()
    interrogation_started = pyqtSignal()
    interrogation_finished = pyqtSignal()

    def __init__(self, database, clip_config: Dict, wd_config: Dict,
                 camie_config: Dict, tag_filters, provider_settings=None, parent=None):
        """
        Initialize the Interrogation Tab.

        Args:
            database: InterrogationDatabase instance
            clip_config: Dictionary with CLIP configuration
            wd_config: Dictionary with WD configuration
            camie_config: Dictionary with Camie configuration
            tag_filters: TagFilterSettings instance
            provider_settings: ONNXProviderSettings instance for ONNX execution providers
            parent: Parent widget
        """
        super().__init__(parent)

        # Store shared state references
        self.database = database
        self.clip_config = clip_config
        self.wd_config = wd_config
        self.camie_config = camie_config
        self.tag_filters = tag_filters
        self.provider_settings = provider_settings

        # Internal state
        self.current_interrogator = None
        self.current_model_type = "WD"
        self.current_directory = None
        self.interrogation_worker = None
        self.directory_load_worker = None
        self.loaded_image_paths = []  # Store paths from async loading

        # Track all discovered tags during batch interrogation
        self.all_discovered_tags = {}  # tag -> (confidence, count)

        # Widget references (will be set in setup_ui)
        self.clip_config_refs = None
        self.wd_config_refs = None
        self.camie_config_refs = None

        # Setup UI
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter (Left: Config+Queue | Right: Preview+Tags+Progress)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel with scroll area
        left_panel_content = self.create_left_panel()

        # Wrap left panel in scroll area for scrolling
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel_content)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(350)

        # Right panel
        right_panel = self.create_right_panel()

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def create_left_panel(self) -> QWidget:
        """Create the left panel with config, directory, queue, and controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Directory Selection
        dir_group = QGroupBox("Directory")
        dir_layout = QVBoxLayout()

        self.dir_label = QLabel("No directory selected")
        self.dir_label.setWordWrap(True)
        dir_layout.addWidget(self.dir_label)

        self.recursive_checkbox = QCheckBox("Include subdirectories (recursive)")
        self.recursive_checkbox.setChecked(False)
        dir_layout.addWidget(self.recursive_checkbox)

        self.select_dir_button = QPushButton("Select Directory")
        dir_layout.addWidget(self.select_dir_button)

        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Model Configuration (inline, using tabs for CLIP/WD)
        model_config_group = QGroupBox("Model Configuration")
        model_config_layout = QVBoxLayout()

        # Create tabbed model config
        self.model_config_tabs = QTabWidget()

        # CLIP tab
        clip_widget, self.clip_config_refs = create_clip_config_widget(self.clip_config, self)
        self.model_config_tabs.addTab(clip_widget, "CLIP")

        # WD tab
        wd_widget, self.wd_config_refs = create_wd_config_widget(self.wd_config, self)
        self.model_config_tabs.addTab(wd_widget, "WD Tagger")

        # Camie tab
        camie_widget, self.camie_config_refs = create_camie_config_widget(self.camie_config, self)
        self.model_config_tabs.addTab(camie_widget, "Camie Tagger")

        model_config_layout.addWidget(self.model_config_tabs)
        model_config_group.setLayout(model_config_layout)
        layout.addWidget(model_config_group)

        # Model Actions
        actions_group = QGroupBox("Model Actions")
        actions_layout = QVBoxLayout()

        self.load_model_button = QPushButton("Load Model")
        actions_layout.addWidget(self.load_model_button)

        self.unload_model_button = QPushButton("Unload Model")
        self.unload_model_button.setEnabled(False)
        actions_layout.addWidget(self.unload_model_button)

        self.model_status_label = QLabel("Model: Not loaded")
        self.model_status_label.setWordWrap(True)
        actions_layout.addWidget(self.model_status_label)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # Tag Filters
        tag_filter_group = QGroupBox("Tag Filters")
        tag_filter_layout = QVBoxLayout()

        # Create tab widget for the three filter types
        self.tag_filter_tabs = QTabWidget()
        self.tag_filter_tabs.setMaximumHeight(300)

        # === Remove Filter Tab ===
        remove_tab = QWidget()
        remove_layout = QVBoxLayout(remove_tab)

        remove_info = QLabel(
            "Tags in this list will be excluded from .txt file output.\n"
            "Useful for removing unwanted tags from skewed models."
        )
        remove_info.setWordWrap(True)
        remove_info.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        remove_layout.addWidget(remove_info)

        self.remove_list = QListWidget()
        self.remove_list.setMaximumHeight(120)
        remove_layout.addWidget(self.remove_list)

        remove_controls = QHBoxLayout()
        self.remove_input = QLineEdit()
        self.remove_input.setPlaceholderText("Tag to remove...")
        remove_controls.addWidget(self.remove_input)

        add_remove_btn = QPushButton("+")
        add_remove_btn.setMaximumWidth(30)
        add_remove_btn.clicked.connect(self._add_remove_tag)
        remove_controls.addWidget(add_remove_btn)

        del_remove_btn = QPushButton("-")
        del_remove_btn.setMaximumWidth(30)
        del_remove_btn.clicked.connect(self._delete_remove_tag)
        remove_controls.addWidget(del_remove_btn)

        remove_layout.addLayout(remove_controls)
        self.tag_filter_tabs.addTab(remove_tab, "Remove")

        # === Replace Filter Tab ===
        replace_tab = QWidget()
        replace_layout = QVBoxLayout(replace_tab)

        replace_info = QLabel(
            "Replace specific tags with better alternatives.\n"
            "Example: Replace 'girl' with 'female' or fix model-specific terminology."
        )
        replace_info.setWordWrap(True)
        replace_info.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        replace_layout.addWidget(replace_info)

        self.replace_table = QTableWidget()
        self.replace_table.setColumnCount(2)
        self.replace_table.setHorizontalHeaderLabels(["Original", "Replace"])
        self.replace_table.horizontalHeader().setStretchLastSection(True)
        self.replace_table.setMaximumHeight(120)
        replace_layout.addWidget(self.replace_table)

        replace_controls = QHBoxLayout()
        self.replace_original_input = QLineEdit()
        self.replace_original_input.setPlaceholderText("Original...")
        replace_controls.addWidget(self.replace_original_input)

        self.replace_new_input = QLineEdit()
        self.replace_new_input.setPlaceholderText("Replace...")
        replace_controls.addWidget(self.replace_new_input)

        add_replace_btn = QPushButton("+")
        add_replace_btn.setMaximumWidth(30)
        add_replace_btn.clicked.connect(self._add_replace_rule)
        replace_controls.addWidget(add_replace_btn)

        del_replace_btn = QPushButton("-")
        del_replace_btn.setMaximumWidth(30)
        del_replace_btn.clicked.connect(self._delete_replace_rule)
        replace_controls.addWidget(del_replace_btn)

        replace_layout.addLayout(replace_controls)
        self.tag_filter_tabs.addTab(replace_tab, "Replace")

        # === Keep Filter Tab ===
        keep_tab = QWidget()
        keep_layout = QVBoxLayout(keep_tab)

        keep_info = QLabel(
            "Tags in this list will always be included, regardless of confidence threshold.\n"
            "Useful for WD models that assign low confidence to important tags."
        )
        keep_info.setWordWrap(True)
        keep_info.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        keep_layout.addWidget(keep_info)

        self.keep_list = QListWidget()
        self.keep_list.setMaximumHeight(120)
        keep_layout.addWidget(self.keep_list)

        keep_controls = QHBoxLayout()
        self.keep_input = QLineEdit()
        self.keep_input.setPlaceholderText("Tag to keep...")
        keep_controls.addWidget(self.keep_input)

        add_keep_btn = QPushButton("+")
        add_keep_btn.setMaximumWidth(30)
        add_keep_btn.clicked.connect(self._add_keep_tag)
        keep_controls.addWidget(add_keep_btn)

        del_keep_btn = QPushButton("-")
        del_keep_btn.setMaximumWidth(30)
        del_keep_btn.clicked.connect(self._delete_keep_tag)
        keep_controls.addWidget(del_keep_btn)

        keep_layout.addLayout(keep_controls)
        self.tag_filter_tabs.addTab(keep_tab, "Keep")

        tag_filter_layout.addWidget(self.tag_filter_tabs)

        # Filter statistics
        self.filter_stats_label = QLabel()
        self.filter_stats_label.setStyleSheet("QLabel { font-size: 9pt; color: #666; }")
        tag_filter_layout.addWidget(self.filter_stats_label)

        tag_filter_group.setLayout(tag_filter_layout)
        layout.addWidget(tag_filter_group)

        # Load current filters
        self._refresh_all_filters()

        # Image Queue (new feature)
        queue_group = QGroupBox("Image Queue")
        queue_layout = QVBoxLayout()

        self.image_queue = QListWidget()
        self.image_queue.setMaximumHeight(150)
        queue_layout.addWidget(self.image_queue)

        queue_group.setLayout(queue_layout)
        layout.addWidget(queue_group)

        # Batch Operations
        batch_group = QGroupBox("Batch Operations")
        batch_layout = QVBoxLayout()

        # .txt file output options (radio buttons)
        self.txt_output_group = QButtonGroup(self)

        self.no_txt_radio = QRadioButton("No .txt file output")
        self.txt_output_group.addButton(self.no_txt_radio, 0)
        batch_layout.addWidget(self.no_txt_radio)

        self.merge_txt_radio = QRadioButton("Write/merge .txt file output")
        self.merge_txt_radio.setChecked(True)  # Default option
        self.txt_output_group.addButton(self.merge_txt_radio, 1)
        batch_layout.addWidget(self.merge_txt_radio)

        self.overwrite_txt_radio = QRadioButton("Overwrite existing .txt files")
        self.txt_output_group.addButton(self.overwrite_txt_radio, 2)
        batch_layout.addWidget(self.overwrite_txt_radio)

        self.batch_interrogate_button = QPushButton("Start Batch")
        self.batch_interrogate_button.setEnabled(False)
        batch_layout.addWidget(self.batch_interrogate_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        batch_layout.addWidget(self.cancel_button)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        layout.addStretch()

        return widget

    def create_right_panel(self) -> QWidget:
        """Create the right panel with preview, discovered tags, and progress."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Current Image Preview
        preview_group = QGroupBox("Current Image Preview")
        preview_layout = QVBoxLayout()

        self.image_preview = QLabel("No image being processed")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumHeight(300)
        self.image_preview.setStyleSheet("QLabel { background-color: #f0f0f0; }")
        preview_layout.addWidget(self.image_preview)

        self.current_image_label = QLabel("")
        self.current_image_label.setWordWrap(True)
        preview_layout.addWidget(self.current_image_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group, 1)  # Preview grows modestly

        # Discovered Tags (real-time display)
        tags_group = QGroupBox("Discovered Tags (Real-time)")
        tags_layout = QVBoxLayout()

        self.discovered_tags_table = QTableWidget()
        self.discovered_tags_table.setColumnCount(2)
        self.discovered_tags_table.setHorizontalHeaderLabels(["Tag", "Confidence"])
        self.discovered_tags_table.horizontalHeader().setStretchLastSection(False)
        self.discovered_tags_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.discovered_tags_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.discovered_tags_table.setMinimumHeight(150)
        tags_layout.addWidget(self.discovered_tags_table)

        tags_group.setLayout(tags_layout)
        layout.addWidget(tags_group, 3)  # Tags table gets priority for extra space

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group, 0)  # Progress stays compact at minimum size

        return widget

    def setup_connections(self):
        """Setup signal/slot connections."""
        # Directory
        self.select_dir_button.clicked.connect(self.select_directory)
        self.recursive_checkbox.stateChanged.connect(self._on_recursive_changed)

        # Model
        self.load_model_button.clicked.connect(self.load_model)
        self.unload_model_button.clicked.connect(self.unload_model)

        # Image queue
        self.image_queue.itemDoubleClicked.connect(self._on_queue_item_double_click)

        # Batch operations
        self.batch_interrogate_button.clicked.connect(self.batch_interrogate)
        self.cancel_button.clicked.connect(self.cancel_operation)

    def select_directory(self):
        """Open directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            str(Path.home())
        )

        if directory:
            self.current_directory = Path(directory)
            self.dir_label.setText(str(self.current_directory))

            # Start async directory loading
            self._start_directory_load()

            # Emit signal for other tabs
            recursive = self.recursive_checkbox.isChecked()
            self.directory_changed.emit(str(self.current_directory), recursive)

    def _on_recursive_changed(self):
        """Handle recursive checkbox state change."""
        # Restart async directory loading with new recursive setting
        if self.current_directory:
            self._start_directory_load()

            # Emit signal for other tabs
            recursive = self.recursive_checkbox.isChecked()
            self.directory_changed.emit(str(self.current_directory), recursive)

    def load_image_queue(self):
        """Load images from current directory into the queue (sync, for backward compat)."""
        if not self.current_directory:
            return

        self.image_queue.clear()
        recursive = self.recursive_checkbox.isChecked()
        images = FileManager.find_images(str(self.current_directory), recursive=recursive)

        for image_path in images:
            # Show relative path if recursive, otherwise just filename
            if recursive:
                rel_path = Path(image_path).relative_to(self.current_directory)
                display_name = str(rel_path)
            else:
                display_name = Path(image_path).name

            item = QListWidgetItem(f"☐ {display_name}")
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self.image_queue.addItem(item)

        search_type = "recursively" if recursive else "in directory"
        self.progress_label.setText(f"Loaded {len(images)} images {search_type}")

    def _start_directory_load(self):
        """Start async directory loading in background thread."""
        if not self.current_directory:
            return

        # Cancel any existing directory load operation (non-blocking)
        if self.directory_load_worker and self.directory_load_worker.isRunning():
            self.directory_load_worker.cancel()
            # Disconnect signals so stale results are ignored - worker will finish in background
            try:
                self.directory_load_worker.progress.disconnect()
                self.directory_load_worker.finished.disconnect()
                self.directory_load_worker.error.disconnect()
            except TypeError:
                pass  # Signals already disconnected

        # Clear queue and show loading state
        self.image_queue.clear()
        self.loaded_image_paths = []
        recursive = self.recursive_checkbox.isChecked()

        # Show indeterminate progress during scan
        self.progress_bar.setMaximum(0)  # Indeterminate mode
        scan_type = "recursively" if recursive else ""
        self.progress_label.setText(f"Scanning directory {scan_type}...".strip())

        # Disable batch button while loading
        self.batch_interrogate_button.setEnabled(False)

        # Create and start worker
        self.directory_load_worker = DirectoryLoadWorker(
            str(self.current_directory),
            recursive=recursive
        )
        self.directory_load_worker.progress.connect(self._on_directory_load_progress)
        self.directory_load_worker.finished.connect(self._on_directory_load_finished)
        self.directory_load_worker.error.connect(self._on_directory_load_error)
        self.directory_load_worker.start()

    def _on_directory_load_progress(self, count: int, current_file: str):
        """Handle directory loading progress update."""
        recursive = self.recursive_checkbox.isChecked()
        scan_type = "Scanning recursively" if recursive else "Scanning"
        if current_file:
            self.progress_label.setText(f"{scan_type}: Found {count} images... ({current_file})")
        else:
            self.progress_label.setText(f"{scan_type}: Found {count} images...")

    def _on_directory_load_finished(self, image_paths: List[str]):
        """Handle directory loading completion - start batched population."""
        self.loaded_image_paths = image_paths
        self._batch_recursive = self.recursive_checkbox.isChecked()

        # Store batch loading ID to detect stale batches
        self._batch_load_id = id(image_paths)

        # Initialize batch state
        self._batch_index = 0
        self._batch_size = 100  # Process 100 items per batch

        # Clear and prepare queue for batch loading
        self.image_queue.clear()
        self.image_queue.setUpdatesEnabled(False)

        # Setup determinate progress bar for loading phase
        total = len(image_paths)
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"Loading 0/{total} images (0%)")
            # Start batch processing
            QTimer.singleShot(0, self._add_image_batch)
        else:
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)
            self._finish_directory_load()

    def _add_image_batch(self):
        """Add a batch of images to the queue, then schedule next batch."""
        # Check if this batch is stale (directory changed during loading)
        if not hasattr(self, '_batch_load_id') or self._batch_load_id != id(self.loaded_image_paths):
            return

        image_paths = self.loaded_image_paths
        total = len(image_paths)
        end_index = min(self._batch_index + self._batch_size, total)

        # Add batch of items
        for i in range(self._batch_index, end_index):
            image_path = image_paths[i]
            # Show relative path if recursive, otherwise just filename
            if self._batch_recursive:
                try:
                    rel_path = Path(image_path).relative_to(self.current_directory)
                    display_name = str(rel_path)
                except ValueError:
                    display_name = Path(image_path).name
            else:
                display_name = Path(image_path).name

            item = QListWidgetItem(f"☐ {display_name}")
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self.image_queue.addItem(item)

        self._batch_index = end_index

        # Update progress bar and label
        percent = int((end_index / total) * 100)
        self.progress_bar.setValue(end_index)
        self.progress_label.setText(f"Loading {end_index}/{total} images ({percent}%)")

        # Schedule next batch or finish
        if self._batch_index < total:
            QTimer.singleShot(0, self._add_image_batch)
        else:
            self._finish_directory_load()

    def _finish_directory_load(self):
        """Complete directory loading after all batches processed."""
        # Re-enable updates
        self.image_queue.setUpdatesEnabled(True)

        # Reset progress bar to idle state
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        # Update status
        search_type = "recursively" if self._batch_recursive else "in directory"
        self.progress_label.setText(f"Loaded {len(self.loaded_image_paths)} images {search_type}")

        # Enable batch button if model is loaded and images found
        if self.current_interrogator and self.current_interrogator.is_loaded and self.loaded_image_paths:
            self.batch_interrogate_button.setEnabled(True)

        # Emit signal with loaded paths for cross-tab sharing
        self.directory_loading_finished.emit(self.loaded_image_paths)

    def _on_directory_load_error(self, error_message: str):
        """Handle directory loading error."""
        # Reset progress bar
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.progress_label.setText(f"Error loading directory: {error_message}")
        QMessageBox.warning(self, "Directory Error", f"Failed to load directory:\n{error_message}")

    def get_current_model_config(self) -> Dict:
        """Get the current model configuration from the active tab."""
        current_tab_index = self.model_config_tabs.currentIndex()

        if current_tab_index == 0:  # CLIP tab
            # Get CLIP config from widget references
            caption_model = self.clip_config_refs['caption_model_combo'].currentText()
            clip_model_idx = self.clip_config_refs['clip_model_combo'].currentIndex()
            clip_model = self.clip_config_refs['clip_model_combo'].itemData(
                clip_model_idx, Qt.ItemDataRole.UserRole
            )
            if clip_model is None:
                clip_model = self.clip_config_refs['clip_model_combo'].currentText()
                clip_model = clip_model.replace(' (Default)', '').replace(' (SDXL Default)', '')

            return {
                'type': 'CLIP',
                'clip_model': clip_model,
                'caption_model': None if caption_model == 'None' else caption_model,
                'mode': self.clip_config_refs['mode_combo'].currentText(),
                'device': self.clip_config_refs['device_combo'].currentText()
            }
        elif current_tab_index == 1:  # WD tab
            return {
                'type': 'WD',
                'wd_model': self.wd_config_refs['wd_model_combo'].currentText(),
                'threshold': self.wd_config_refs['threshold_spin'].value(),
                'device': self.wd_config_refs['device_combo'].currentText()
            }
        else:  # Camie tab (index 2)
            # Get enabled categories from checkboxes
            enabled_categories = [
                cat for cat, cb in self.camie_config_refs['category_checkboxes'].items()
                if cb.isChecked()
            ]

            # Get category-specific thresholds if profile is category_specific
            threshold_profile = self.camie_config_refs['threshold_profile_combo'].currentText()
            category_thresholds = None
            if threshold_profile == 'category_specific':
                category_thresholds = {
                    cat: spin.value()
                    for cat, spin in self.camie_config_refs['category_threshold_spins'].items()
                }

            return {
                'type': 'Camie',
                'camie_model': self.camie_config_refs['camie_model_combo'].currentText(),
                'threshold': self.camie_config_refs['threshold_spin'].value(),
                'threshold_profile': threshold_profile,
                'device': self.camie_config_refs['device_combo'].currentText(),
                'category_thresholds': category_thresholds,
                'enabled_categories': enabled_categories
            }

    def load_model(self):
        """Load the selected model."""
        try:
            config = self.get_current_model_config()
            self.progress_label.setText("Loading model...")
            self.load_model_button.setEnabled(False)

            # Unload existing model
            if self.current_interrogator:
                self.current_interrogator.unload_model()

            # Create and load new interrogator
            if config['type'] == 'CLIP':
                self.current_interrogator = CLIPInterrogator(model_name=config['clip_model'])
                load_params = {
                    'mode': config['mode'],
                    'device': config['device']
                }
                if config['caption_model']:
                    load_params['caption_model'] = config['caption_model']
                self.current_interrogator.load_model(**load_params)
                model_info = f"CLIP - {config['clip_model']}"
                if config['caption_model']:
                    model_info += f"\nCaption: {config['caption_model']}"
            elif config['type'] == 'WD':
                self.current_interrogator = WDInterrogator(model_name=config['wd_model'])
                self.current_interrogator.load_model(
                    threshold=config['threshold'],
                    device=config['device'],
                    provider_settings=self.provider_settings
                )
                model_info = f"WD - {config['wd_model']}"
            else:  # Camie
                self.current_interrogator = CamieInterrogator(model_name=config['camie_model'])
                self.current_interrogator.load_model(
                    threshold=config['threshold'],
                    device=config['device'],
                    threshold_profile=config['threshold_profile'],
                    category_thresholds=config.get('category_thresholds'),
                    enabled_categories=config.get('enabled_categories'),
                    provider_settings=self.provider_settings
                )
                model_info = f"Camie - {config['camie_model']}"

            # Update status
            self.current_model_type = config['type']
            self.model_status_label.setText(f"Model: {model_info}\nLoaded ✓")
            self.progress_label.setText("Model loaded successfully")

            # Enable buttons
            if self.current_directory:
                self.batch_interrogate_button.setEnabled(True)
            self.unload_model_button.setEnabled(True)

            # Emit signal
            self.model_loaded.emit(model_info)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            self.model_status_label.setText("Model: Load failed")
        finally:
            self.load_model_button.setEnabled(True)

    def unload_model(self):
        """Unload the current model from memory."""
        if self.current_interrogator:
            try:
                self.current_interrogator.unload_model()
                self.current_interrogator = None
                self.model_status_label.setText("Model: Not loaded")
                self.progress_label.setText("Model unloaded successfully")

                # Disable buttons
                self.batch_interrogate_button.setEnabled(False)
                self.unload_model_button.setEnabled(False)

                # Emit signal
                self.model_unloaded.emit()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to unload model:\n{str(e)}")
        else:
            QMessageBox.information(self, "Info", "No model is currently loaded")

    def batch_interrogate(self):
        """Start batch interrogation."""
        if not self.current_directory or not self.current_interrogator:
            return

        # Use pre-loaded image paths from async directory scan
        images = [Path(p) for p in self.loaded_image_paths]
        if not images:
            QMessageBox.information(self, "Info", "No images found in directory")
            return

        # Confirm
        reply = QMessageBox.question(
            self,
            "Confirm Batch Interrogation",
            f"Interrogate {len(images)} images?\nThis may take some time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Clear accumulated tags from previous batch
        self.all_discovered_tags.clear()
        self.discovered_tags_table.setRowCount(0)

        # Determine write_files and overwrite_files based on radio button selection
        if self.no_txt_radio.isChecked():
            write_files = False
            overwrite_files = False
        elif self.merge_txt_radio.isChecked():
            write_files = True
            overwrite_files = False
        else:  # overwrite_txt_radio is checked
            write_files = True
            overwrite_files = True

        # Start worker
        self.interrogation_worker = InterrogationWorker(
            images,
            self.current_interrogator,
            self.database,
            write_files,
            overwrite_files,
            self.tag_filters
        )

        # Connect signals
        self.interrogation_worker.progress.connect(self.on_interrogation_progress)
        self.interrogation_worker.result.connect(self.on_interrogation_result)
        self.interrogation_worker.error.connect(self.on_interrogation_error)
        self.interrogation_worker.finished.connect(self.on_interrogation_finished)

        # Update UI
        self.batch_interrogate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setMaximum(len(images))

        # Emit signal
        self.interrogation_started.emit()

        # Start
        self.interrogation_worker.start()

    def cancel_operation(self):
        """Cancel ongoing operation."""
        if self.interrogation_worker:
            self.interrogation_worker.cancel()
            self.progress_label.setText("Cancelling...")

    def on_interrogation_progress(self, current: int, total: int, message: str):
        """Handle interrogation progress update."""
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total}: {message}")

        # Update queue item status and get image path
        image_path = None
        if current <= self.image_queue.count():
            item = self.image_queue.item(current - 1)
            if item:
                image_path = item.data(Qt.ItemDataRole.UserRole)
                item.setText(f"☑ {Path(image_path).name}")

        # Show current image preview
        if image_path:
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    400, 400,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_preview.setPixmap(scaled_pixmap)
            self.current_image_label.setText(f"Processing: {Path(image_path).name}")
        else:
            self.current_image_label.setText(f"Processing: {message}")

    def on_interrogation_result(self, image_path: str, results: Dict):
        """Handle interrogation result."""
        import json as json_module

        # Update queue status
        for i in range(self.image_queue.count()):
            item = self.image_queue.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                item.setText(f"✓ {Path(image_path).name}")
                break

        # Accumulate discovered tags from all images
        tags = results.get('tags', [])
        confidence_scores = results.get('confidence_scores', {})

        # Handle None confidence_scores (CLIP returns None)
        if confidence_scores is None:
            confidence_scores = {}

        # Extract category info from raw_output for Camie
        tag_categories = {}
        raw_output = results.get('raw_output', '')
        if raw_output and self.current_model_type == "Camie":
            try:
                raw_data = json_module.loads(raw_output)
                categories = raw_data.get('categories', {})
                for category, tag_list in categories.items():
                    for tag_info in tag_list:
                        tag_name = tag_info.get('tag', '')
                        if tag_name:
                            tag_categories[tag_name] = category
            except (json_module.JSONDecodeError, AttributeError):
                pass

        # Add new tags to accumulated list
        # Format: tag -> (confidence, count, category)
        for tag in tags:
            conf = confidence_scores.get(tag, 0.0)
            category = tag_categories.get(tag, '')
            if tag in self.all_discovered_tags:
                # Update if higher confidence or increment count
                old_conf, old_count, old_category = self.all_discovered_tags[tag]
                new_conf = max(old_conf, conf)
                # Keep existing category if new one is empty
                new_category = category if category else old_category
                self.all_discovered_tags[tag] = (new_conf, old_count + 1, new_category)
            else:
                self.all_discovered_tags[tag] = (conf, 1, category)

        # Update the discovered tags table with all accumulated tags
        self._update_discovered_tags_display()

    def _update_discovered_tags_display(self):
        """Update the discovered tags table with accumulated tags."""
        self.discovered_tags_table.setRowCount(0)

        # Configure columns based on model type
        is_camie = self.current_model_type == "Camie"
        if is_camie:
            self.discovered_tags_table.setColumnCount(3)
            self.discovered_tags_table.setHorizontalHeaderLabels(["Tag", "Category", "Confidence"])
            self.discovered_tags_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.discovered_tags_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.discovered_tags_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        else:
            self.discovered_tags_table.setColumnCount(2)
            self.discovered_tags_table.setHorizontalHeaderLabels(["Tag", "Confidence"])
            self.discovered_tags_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.discovered_tags_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        # Sort tags by count (most common first), then by confidence
        sorted_tags = sorted(
            self.all_discovered_tags.items(),
            key=lambda x: (x[1][1], x[1][0]),  # Sort by (count, confidence)
            reverse=True
        )

        # Display top 100 tags
        for tag, tag_data in sorted_tags[:100]:
            # Handle both old (conf, count) and new (conf, count, category) formats
            if len(tag_data) == 3:
                conf, count, category = tag_data
            else:
                conf, count = tag_data
                category = ''

            row = self.discovered_tags_table.rowCount()
            self.discovered_tags_table.insertRow(row)

            # Tag name with count
            tag_text = f"{tag} ({count})" if count > 1 else tag
            self.discovered_tags_table.setItem(row, 0, QTableWidgetItem(tag_text))

            if is_camie:
                # Category column
                self.discovered_tags_table.setItem(row, 1, QTableWidgetItem(category))
                # Confidence
                conf_text = f"{conf:.2f}" if conf > 0 else "N/A"
                self.discovered_tags_table.setItem(row, 2, QTableWidgetItem(conf_text))
            else:
                # Confidence
                conf_text = f"{conf:.2f}" if conf > 0 else "N/A"
                self.discovered_tags_table.setItem(row, 1, QTableWidgetItem(conf_text))

    def on_interrogation_error(self, image_path: str, error: str):
        """Handle interrogation error."""
        # Update queue status
        for i in range(self.image_queue.count()):
            item = self.image_queue.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                item.setText(f"✗ {Path(image_path).name}")
                break

    def on_interrogation_finished(self):
        """Handle interrogation completion."""
        self.progress_label.setText("Batch interrogation complete")
        self.batch_interrogate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        # Clear current image preview
        self.image_preview.clear()
        self.image_preview.setText("Batch complete")
        self.current_image_label.setText("")

        # Keep discovered tags visible so user can see all tags from batch
        # They will be cleared when starting a new batch

        # Auto-unload model if enabled in settings
        if self.parent() and hasattr(self.parent(), 'settings_tab'):
            if self.parent().settings_tab.get_auto_unload_enabled():
                if self.current_interrogator and self.current_interrogator.is_loaded:
                    try:
                        self.current_interrogator.unload_model()
                        self.current_interrogator = None
                        self.model_status_label.setText("Model: Auto-unloaded")
                        self.unload_model_button.setEnabled(False)
                        self.progress_label.setText("Batch complete - model auto-unloaded")
                    except Exception as e:
                        print(f"Error auto-unloading model: {e}")

        # Emit signal
        self.interrogation_finished.emit()

        QMessageBox.information(self, "Complete", "Batch interrogation finished")

    def on_result(self, image_path: str, results: Dict):
        """Handle interrogation result from external source."""
        # This method is called by MainWindow for cross-tab communication
        self.on_interrogation_result(image_path, results)

    # === Tag Filter Methods ===

    def _refresh_all_filters(self):
        """Refresh all filter displays."""
        self._refresh_remove_list()
        self._refresh_replace_table()
        self._refresh_keep_list()
        self._refresh_filter_stats()

    def _refresh_remove_list(self):
        """Refresh the remove list display."""
        self.remove_list.clear()
        for tag in self.tag_filters.get_remove_list():
            self.remove_list.addItem(tag)

    def _refresh_replace_table(self):
        """Refresh the replace table display."""
        self.replace_table.setRowCount(0)
        replace_dict = self.tag_filters.get_replace_dict()
        for original, replacement in replace_dict.items():
            row = self.replace_table.rowCount()
            self.replace_table.insertRow(row)
            self.replace_table.setItem(row, 0, QTableWidgetItem(original))
            self.replace_table.setItem(row, 1, QTableWidgetItem(replacement))

    def _refresh_keep_list(self):
        """Refresh the keep list display."""
        self.keep_list.clear()
        for tag in self.tag_filters.get_keep_list():
            self.keep_list.addItem(tag)

    def _refresh_filter_stats(self):
        """Refresh the filter statistics display."""
        stats = self.tag_filters.get_statistics()
        self.filter_stats_label.setText(
            f"Active: {stats['remove_count']} removed | "
            f"{stats['replace_count']} replaced | "
            f"{stats['keep_count']} kept"
        )

    def _add_remove_tag(self):
        """Add a tag to the remove list."""
        tag = self.remove_input.text().strip()
        if tag:
            self.tag_filters.add_remove_tag(tag)
            self.remove_input.clear()
            self._refresh_all_filters()

    def _delete_remove_tag(self):
        """Delete selected tag from remove list."""
        current_item = self.remove_list.currentItem()
        if current_item:
            self.tag_filters.remove_remove_tag(current_item.text())
            self._refresh_all_filters()

    def _add_replace_rule(self):
        """Add a replacement rule."""
        original = self.replace_original_input.text().strip()
        replacement = self.replace_new_input.text().strip()
        if original and replacement:
            self.tag_filters.add_replace_rule(original, replacement)
            self.replace_original_input.clear()
            self.replace_new_input.clear()
            self._refresh_all_filters()

    def _delete_replace_rule(self):
        """Delete selected replacement rule."""
        current_row = self.replace_table.currentRow()
        if current_row >= 0:
            original_item = self.replace_table.item(current_row, 0)
            if original_item:
                self.tag_filters.remove_replace_rule(original_item.text())
                self._refresh_all_filters()

    def _add_keep_tag(self):
        """Add a tag to the keep list."""
        tag = self.keep_input.text().strip()
        if tag:
            self.tag_filters.add_keep_tag(tag)
            self.keep_input.clear()
            self._refresh_all_filters()

    def _delete_keep_tag(self):
        """Delete selected tag from keep list."""
        current_item = self.keep_list.currentItem()
        if current_item:
            self.tag_filters.remove_keep_tag(current_item.text())
            self._refresh_all_filters()

    def _on_queue_item_double_click(self, item):
        """Handle double-click on queue item."""
        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path:
            self._open_advanced_inspection(image_path)

    def _open_advanced_inspection(self, image_path: str):
        """Open the advanced image inspection dialog."""
        if not self.current_directory:
            return

        try:
            # Build image list from queue
            image_list = []
            for i in range(self.image_queue.count()):
                path = self.image_queue.item(i).data(Qt.ItemDataRole.UserRole)
                if path:
                    image_list.append(path)

            dialog = AdvancedImageInspectionDialog(
                image_path=image_path,
                image_list=image_list,
                database=self.database,
                tag_filters=self.tag_filters,
                parent=self
            )
            dialog.show()  # Non-modal
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open inspection dialog:\n{str(e)}")
