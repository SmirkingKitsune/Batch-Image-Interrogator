"""Advanced dialogs for detailed image inspection and analysis."""

import logging
import json
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QComboBox, QGroupBox, QSplitter,
                             QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
                             QProgressBar, QMessageBox, QMenu, QScrollArea,
                             QCheckBox, QLineEdit, QGridLayout, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QSettings
from PyQt6.QtGui import QKeySequence, QColor, QShortcut, QBrush, QPixmap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from core import FileManager, TagFilterSettings, InterrogationDatabase, get_image_metadata
from core.hashing import hash_image_content
from interrogators import LlamaCppInterrogator
from ui.widgets import TagEditorWidget, ResultsTableWidget

logger = logging.getLogger(__name__)


class ThumbnailGridWidget(QWidget):
    """Widget displaying a grid of image thumbnails for multi-selection preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Count label
        self.count_label = QLabel("No images selected")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.count_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        layout.addWidget(self.count_label)

        # Grid container
        self.grid_frame = QFrame()
        self.grid_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.grid_layout = QGridLayout(self.grid_frame)
        self.grid_layout.setSpacing(5)
        layout.addWidget(self.grid_frame, 1)

        self.thumbnail_labels = []

    def set_images(self, image_paths: List[str], max_visible: int = 9):
        """
        Display thumbnails for selected images.

        Args:
            image_paths: List of image paths to display
            max_visible: Maximum number of thumbnails to show (default 9 for 3x3 grid)
        """
        # Clear existing thumbnails
        for label in self.thumbnail_labels:
            label.deleteLater()
        self.thumbnail_labels.clear()

        # Clear grid layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total_images = len(image_paths)

        # Update count label
        if total_images <= max_visible:
            self.count_label.setText(f"{total_images} images selected")
        else:
            self.count_label.setText(f"{total_images} images selected (showing first {max_visible})")

        # Calculate grid dimensions (aim for square-ish)
        num_to_show = min(total_images, max_visible)
        if num_to_show == 0:
            return

        cols = 3 if num_to_show > 4 else 2 if num_to_show > 1 else 1
        rows = (num_to_show + cols - 1) // cols

        # Create thumbnails
        thumb_size = 120
        for i, image_path in enumerate(image_paths[:max_visible]):
            row = i // cols
            col = i % cols

            # Create thumbnail label
            label = QLabel()
            label.setFixedSize(thumb_size, thumb_size)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFrameShape(QFrame.Shape.Box)
            label.setStyleSheet("QLabel { background-color: #f0f0f0; }")

            # Load and scale image
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    thumb_size - 4, thumb_size - 4,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                label.setPixmap(scaled)
                label.setToolTip(Path(image_path).name)
            else:
                label.setText("?")

            self.grid_layout.addWidget(label, row, col)
            self.thumbnail_labels.append(label)


class CheckboxTagSelector(QWidget):
    """Widget that displays all available tags with checkboxes for selection."""

    # Signals
    tags_changed = pyqtSignal(list)  # List of selected tags

    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_tags = {}  # tag -> QCheckBox
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter tags...")
        self.search_box.textChanged.connect(self._filter_tags)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        layout.addLayout(search_layout)

        # Tag count label
        self.tag_count_label = QLabel("No tags available")
        self.tag_count_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.tag_count_label)

        # Scroll area for tags
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.StyledPanel)

        self.tag_container = QWidget()
        self.tag_layout = QVBoxLayout(self.tag_container)
        self.tag_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.tag_container)

        layout.addWidget(scroll, 1)

        # Action buttons
        button_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        button_layout.addWidget(self.deselect_all_btn)

        button_layout.addStretch()

        self.save_btn = QPushButton("Save Tags to File")
        self.save_btn.clicked.connect(self._emit_tags_changed)
        self.save_btn.setStyleSheet("QPushButton { font-weight: bold; background-color: #4CAF50; color: white; }")
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

    def set_available_tags(self, all_tags: List[str], selected_tags: List[str]):
        """
        Set the available tags and which ones are currently selected.

        Args:
            all_tags: All possible tags from model interrogations
            selected_tags: Tags currently in the .txt file
        """
        # Clear existing checkboxes
        for checkbox in self.all_tags.values():
            checkbox.deleteLater()
        self.all_tags.clear()

        # Create checkboxes for all tags
        for tag in sorted(all_tags, key=str.lower):
            checkbox = QCheckBox(tag)
            checkbox.setChecked(tag in selected_tags)
            checkbox.stateChanged.connect(self._on_checkbox_changed)
            self.all_tags[tag] = checkbox
            self.tag_layout.addWidget(checkbox)

        # Update count
        selected_count = len(selected_tags)
        total_count = len(all_tags)
        self.tag_count_label.setText(
            f"Total tags: {total_count} | Selected: {selected_count}"
        )

    def _filter_tags(self, search_text: str):
        """Filter displayed tags based on search text."""
        search_lower = search_text.lower()
        for tag, checkbox in self.all_tags.items():
            visible = search_lower in tag.lower()
            checkbox.setVisible(visible)

    def _select_all(self):
        """Select all visible checkboxes."""
        for tag, checkbox in self.all_tags.items():
            if checkbox.isVisible():
                checkbox.setChecked(True)

    def _deselect_all(self):
        """Deselect all visible checkboxes."""
        for tag, checkbox in self.all_tags.items():
            if checkbox.isVisible():
                checkbox.setChecked(False)

    def _on_checkbox_changed(self):
        """Update count when any checkbox changes."""
        selected_count = sum(1 for cb in self.all_tags.values() if cb.isChecked())
        total_count = len(self.all_tags)
        self.tag_count_label.setText(
            f"Total tags: {total_count} | Selected: {selected_count}"
        )

    def _emit_tags_changed(self):
        """Emit the tags_changed signal with selected tags."""
        selected = self.get_selected_tags()
        self.tags_changed.emit(selected)

    def get_selected_tags(self) -> List[str]:
        """Get list of selected tags."""
        return [tag for tag, checkbox in self.all_tags.items() if checkbox.isChecked()]


class AdvancedImageInspectionDialog(QDialog):
    """
    Advanced image inspection dialog with multi-model results,
    tag comparison, and editing capabilities.

    Supports both single-image mode (with navigation) and multi-image mode
    (for batch tag editing with common tag intersection).
    """

    # Signals
    tags_saved = pyqtSignal(str, list)  # image_path, tags (single image)
    batch_tags_saved = pyqtSignal(list)  # list of (image_path, tags) tuples (multi-image)

    # Settings keys
    SETTINGS_KEY_LAST_TAB = "AdvancedImageInspectionDialog/lastTabIndex"

    def __init__(self,
                 image_path: Union[str, List[str]],
                 image_list: List[str],
                 database: InterrogationDatabase,
                 tag_filters: Optional[TagFilterSettings] = None,
                 llama_config: Optional[Dict] = None,
                 parent=None):
        """
        Initialize the advanced image inspection dialog.

        Args:
            image_path: Initial image to display, OR list of images for multi-image mode
            image_list: Full list of images for navigation (single-image mode only)
            database: Database instance for interrogation lookups
            tag_filters: Optional tag filter settings for comparison
            llama_config: Optional llama.cpp multimodal settings
            parent: Parent widget
        """
        super().__init__(parent)

        # Core state
        self.database = database
        self.tag_filters = tag_filters
        self.image_list = image_list
        self.llama_config = llama_config or {}

        # Handle both single path (str) and multiple paths (list)
        if isinstance(image_path, list):
            self.selected_images = image_path
        else:
            self.selected_images = [image_path]

        self.is_multi_mode = len(self.selected_images) > 1

        # Single-image mode state
        if not self.is_multi_mode:
            first_image = self.selected_images[0]
            self.current_index = image_list.index(first_image) if first_image in image_list else 0
            self.current_image = first_image
        else:
            self.current_index = 0
            self.current_image = None  # No single current image in multi-mode

        # Data cache
        self.current_interrogations = []  # List of dicts from database
        self.current_file_tags = []  # Tags from .txt file
        self.current_model = None  # Currently selected model name
        self.original_common_tags = set()  # For multi-mode: tracks original common tags
        self.current_image_hash = None
        self.current_multimodal_session_key = None
        self.multimodal_interrogator: Optional[LlamaCppInterrogator] = None

        # UI setup
        if self.is_multi_mode:
            self.setWindowTitle(f"Edit Tags - {len(self.selected_images)} Images")
        else:
            self.setWindowTitle("Advanced Image Inspection")
        self.setModal(False)  # Non-modal so it can be moved/resized
        self.resize(1400, 900)

        self.setup_ui()
        self.setup_keyboard_shortcuts()

        # Load initial data
        self.load_image_data()

    def setup_ui(self):
        """Setup the complete UI layout."""
        main_layout = QVBoxLayout(self)

        # Top navigation controls
        top_controls = QHBoxLayout()
        self.prev_button = QPushButton("< Previous")
        self.prev_button.clicked.connect(self._navigate_previous)
        self.image_counter = QLabel()
        self.image_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_button = QPushButton("Next >")
        self.next_button.clicked.connect(self._navigate_next)

        top_controls.addWidget(self.prev_button)
        top_controls.addStretch()
        top_controls.addWidget(self.image_counter)
        top_controls.addStretch()
        top_controls.addWidget(self.next_button)
        main_layout.addLayout(top_controls)

        # Main splitter: Left (preview) | Right (data)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT PANEL
        left_panel = self._create_left_panel()
        self.splitter.addWidget(left_panel)

        # RIGHT PANEL
        right_panel = self._create_right_panel()
        self.splitter.addWidget(right_panel)

        # Set initial sizes: 40% left, 60% right
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)

        main_layout.addWidget(self.splitter)

        # Bottom buttons
        bottom_buttons = QHBoxLayout()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        bottom_buttons.addStretch()
        bottom_buttons.addWidget(close_button)

        main_layout.addLayout(bottom_buttons)

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with image preview and info."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        if self.is_multi_mode:
            # Multi-image mode: show thumbnail grid
            preview_group = QGroupBox("Selected Images")
            preview_layout = QVBoxLayout()

            self.thumbnail_grid = ThumbnailGridWidget()
            preview_layout.addWidget(self.thumbnail_grid)

            preview_group.setLayout(preview_layout)
            layout.addWidget(preview_group, 1)

            # Placeholder for single-image widgets (not used in multi-mode)
            self.image_preview = None
            self.filename_label = None
            self.dimensions_label = None
            self.file_size_label = None
            self.file_tags_status_label = None
        else:
            # Single-image mode: show full preview
            preview_group = QGroupBox("Image Preview")
            preview_layout = QVBoxLayout()

            # Import EnhancedImagePreview here to avoid circular import
            from ui.tabs.gallery_tab import EnhancedImagePreview
            self.image_preview = EnhancedImagePreview()
            preview_layout.addWidget(self.image_preview)

            preview_group.setLayout(preview_layout)
            layout.addWidget(preview_group)

            # Image Info Group
            info_group = QGroupBox("Image Information")
            info_layout = QVBoxLayout()

            self.filename_label = QLabel()
            self.filename_label.setWordWrap(True)
            info_layout.addWidget(self.filename_label)

            self.dimensions_label = QLabel()
            info_layout.addWidget(self.dimensions_label)

            self.file_size_label = QLabel()
            info_layout.addWidget(self.file_size_label)

            self.file_tags_status_label = QLabel()
            info_layout.addWidget(self.file_tags_status_label)

            info_group.setLayout(info_layout)
            layout.addWidget(info_group)

            # Placeholder for multi-mode widget (not used in single-mode)
            self.thumbnail_grid = None

        return widget

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with tabbed interface."""
        self.main_tabs = QTabWidget()

        if self.is_multi_mode:
            # Multi-image mode: only show Tag Editor
            editor_tab = self._create_editor_tab()
            self.main_tabs.addTab(editor_tab, "Common Tags Editor")

            # Placeholders for single-mode tabs (not created in multi-mode)
            self.model_selector = None
            self.ratings_group = None
            self.results_table = None
            self.comparison_table = None
            self.comparison_info_label = None
            self.multimodal_tab = None
        else:
            # Single-image mode: show all tabs
            # Tab 1: Model Results
            model_results_tab = self._create_model_results_tab()
            self.main_tabs.addTab(model_results_tab, "Model Results")

            # Tab 2: Database vs File Comparison
            comparison_tab = self._create_comparison_tab()
            self.main_tabs.addTab(comparison_tab, "Database vs File")

            # Tab 3: Tag Editor
            editor_tab = self._create_editor_tab()
            self.main_tabs.addTab(editor_tab, "Tag Editor")

            # Tab 4: Multimodal Inquiry
            self.multimodal_tab = self._create_multimodal_tab()
            self.main_tabs.addTab(self.multimodal_tab, "Multimodal Inquiry")

            # Connect tab change signal to save preference
            self.main_tabs.currentChanged.connect(self._on_tab_changed)

            # Restore last selected tab
            settings = QSettings()
            last_tab = settings.value(self.SETTINGS_KEY_LAST_TAB, 0, type=int)
            if 0 <= last_tab < self.main_tabs.count():
                self.main_tabs.setCurrentIndex(last_tab)

        return self.main_tabs

    def _create_model_results_tab(self) -> QWidget:
        """Create the model results tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model selector
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        self.model_selector = QComboBox()
        self.model_selector.currentTextChanged.connect(self._on_model_selected)
        model_layout.addWidget(self.model_selector, 1)
        layout.addLayout(model_layout)

        # WD Sensitivity Ratings
        self.ratings_group = self._create_ratings_widget()
        layout.addWidget(self.ratings_group)

        # Tags results table
        tags_group = QGroupBox("Tags")
        tags_layout = QVBoxLayout()
        self.results_table = ResultsTableWidget()
        tags_layout.addWidget(self.results_table)
        tags_group.setLayout(tags_layout)
        layout.addWidget(tags_group)

        return widget

    def _create_ratings_widget(self) -> QGroupBox:
        """Create WD sensitivity ratings display."""
        ratings_group = QGroupBox("WD Sensitivity Ratings")
        layout = QVBoxLayout()

        # Create progress bars for each rating
        ratings = ['general', 'sensitive', 'questionable', 'explicit']
        self.rating_bars = {}
        self.rating_labels = {}

        for rating in ratings:
            row = QHBoxLayout()

            label = QLabel(f"{rating.capitalize()}:")
            label.setMinimumWidth(100)
            row.addWidget(label)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%v%")
            self.rating_bars[rating] = bar
            row.addWidget(bar)

            conf_label = QLabel("0.00")
            conf_label.setMinimumWidth(50)
            self.rating_labels[rating] = conf_label
            row.addWidget(conf_label)

            layout.addLayout(row)

        ratings_group.setLayout(layout)
        return ratings_group

    def _create_comparison_tab(self) -> QWidget:
        """Create the database vs file comparison tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info label
        self.comparison_info_label = QLabel()
        layout.addWidget(self.comparison_info_label)

        # Legend
        legend = self._create_comparison_legend()
        layout.addWidget(legend)

        # Comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(4)
        self.comparison_table.setHorizontalHeaderLabels([
            "Tag", "Confidence", "Status", "Location"
        ])
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.comparison_table)

        return widget

    def _create_comparison_legend(self) -> QGroupBox:
        """Create legend explaining color coding."""
        legend = QGroupBox("Legend")
        layout = QVBoxLayout()

        items = [
            (QColor(220, 255, 220), "In Both - Tag exists in database and file"),
            (QColor(255, 255, 220), "Database Only - Tag in DB but not written to file"),
            (QColor(220, 220, 255), "File Only - Tag in file but not in database"),
            (QColor(255, 220, 220), "Filtered Out - Tag removed by filters"),
            (QColor(255, 230, 200), "Replaced - Tag replaced by filter rule"),
            (QColor(200, 230, 255), "Manually Added - Tag in file despite being filtered")
        ]

        for color, description in items:
            row = QHBoxLayout()
            color_label = QLabel()
            color_label.setFixedSize(20, 20)
            color_label.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
                f"border: 1px solid black;"
            )
            row.addWidget(color_label)
            desc_label = QLabel(description)
            desc_label.setStyleSheet("font-size: 9pt;")
            row.addWidget(desc_label)
            row.addStretch()
            layout.addLayout(row)

        legend.setLayout(layout)
        legend.setMaximumHeight(200)
        return legend

    def _create_editor_tab(self) -> QWidget:
        """Create the tag editor tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info label
        info = QLabel(
            "All tags generated by models are shown below. "
            "Checked tags will be saved to the .txt file. "
            "Click 'Save Tags to File' to apply changes."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info)

        # Checkbox tag selector
        self.tag_selector = CheckboxTagSelector()
        self.tag_selector.tags_changed.connect(self._save_edited_tags)
        layout.addWidget(self.tag_selector)

        return widget

    def _create_multimodal_tab(self) -> QWidget:
        """Create single-image multimodal inquiry tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.mm_status_label = QLabel(
            "Configure llama.cpp model paths in Inquiry tab, then use this panel for image Q&A/OCR."
        )
        self.mm_status_label.setWordWrap(True)
        self.mm_status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.mm_status_label)

        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task:"))
        self.mm_task_combo = QComboBox()
        self.mm_task_combo.addItems(["describe", "ocr", "vqa", "custom"])
        task_row.addWidget(self.mm_task_combo)
        task_row.addStretch()
        layout.addLayout(task_row)

        layout.addWidget(QLabel("Prompt / Question:"))
        self.mm_prompt_input = QLineEdit()
        self.mm_prompt_input.setPlaceholderText("Ask a visual question or add instructions...")
        layout.addWidget(self.mm_prompt_input)

        prior_group = QGroupBox("Include Prior Interrogation Tables")
        prior_layout = QVBoxLayout()
        self.mm_prior_tables = QTableWidget()
        self.mm_prior_tables.setColumnCount(2)
        self.mm_prior_tables.setHorizontalHeaderLabels(["Include", "Model"])
        self.mm_prior_tables.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.mm_prior_tables.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.mm_prior_tables.setMinimumHeight(120)
        prior_layout.addWidget(self.mm_prior_tables)
        prior_group.setLayout(prior_layout)
        layout.addWidget(prior_group)

        button_row = QHBoxLayout()
        self.mm_send_button = QPushButton("Send Inquiry")
        self.mm_send_button.clicked.connect(self._on_send_multimodal)
        button_row.addWidget(self.mm_send_button)

        self.mm_reset_button = QPushButton("Reset Image Context")
        self.mm_reset_button.clicked.connect(self._on_reset_multimodal_context)
        button_row.addWidget(self.mm_reset_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        transcript_group = QGroupBox("Transcript")
        transcript_layout = QVBoxLayout()
        self.mm_transcript = QTableWidget()
        self.mm_transcript.setColumnCount(3)
        self.mm_transcript.setHorizontalHeaderLabels(["Turn", "Prompt", "Response Summary"])
        self.mm_transcript.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.mm_transcript.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.mm_transcript.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        transcript_layout.addWidget(self.mm_transcript)
        transcript_group.setLayout(transcript_layout)
        layout.addWidget(transcript_group, 1)

        return widget

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for dialog."""
        # Only enable navigation in single-image mode
        if not self.is_multi_mode:
            # Left arrow - previous image
            prev_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
            prev_shortcut.activated.connect(self._navigate_previous)

            # Right arrow - next image
            next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
            next_shortcut.activated.connect(self._navigate_next)

        # ESC - close dialog
        close_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        close_shortcut.activated.connect(self.close)

        # Ctrl+S - save tags (if on editor tab)
        save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        save_shortcut.activated.connect(self._quick_save_tags)

    def load_image_data(self):
        """
        Load all data for current image(s).
        Dispatches to single or multi-image loading based on mode.
        """
        if self.is_multi_mode:
            self._load_multi_image_data()
        else:
            self._load_single_image_data()

    def _load_single_image_data(self):
        """
        Load all data for current image: interrogations, file tags, display.
        This is called when navigating to a new image in single-image mode.
        """
        if not self.current_image:
            return

        # Update image counter
        self.image_counter.setText(
            f"Image {self.current_index + 1} / {len(self.image_list)}"
        )

        # Update navigation button states
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_list) - 1)

        # Load image in preview
        self.image_preview.set_image(self.current_image)

        # Update info labels
        path = Path(self.current_image)
        self.filename_label.setText(f"File: {path.name}")

        try:
            # Get image dimensions from preview
            if self.image_preview.current_pixmap:
                dimensions = (
                    f"{self.image_preview.current_pixmap.width()}x"
                    f"{self.image_preview.current_pixmap.height()}"
                )
                self.dimensions_label.setText(f"Dimensions: {dimensions}")

            # Get file size
            size_kb = path.stat().st_size / 1024
            self.file_size_label.setText(f"File Size: {size_kb:.1f} KB")
        except Exception as e:
            logger.error(f"Error reading image info: {e}")

        # Load interrogations from database
        try:
            file_hash = hash_image_content(self.current_image)
            self.current_image_hash = file_hash
            self.current_multimodal_session_key = f"single:{file_hash}"
            self.current_interrogations = self.database.get_all_interrogations_for_image(file_hash) or []
        except Exception as e:
            logger.error(f"Error loading interrogations: {e}")
            self.current_image_hash = None
            self.current_multimodal_session_key = None
            self.current_interrogations = []

        # Load tags from .txt file
        self.current_file_tags = FileManager.read_tags_from_file(Path(self.current_image))

        # Update file tags status
        if self.current_file_tags:
            self.file_tags_status_label.setText(
                f".txt file: {len(self.current_file_tags)} tags"
            )
            self.file_tags_status_label.setStyleSheet("color: green;")
        else:
            self.file_tags_status_label.setText(".txt file: No tags found")
            self.file_tags_status_label.setStyleSheet("color: orange;")

        # Populate model dropdown
        self._populate_model_selector()

        # If we have a selected model (or auto-select first), load its data
        if self.current_interrogations:
            if self.current_model is None or self.current_model not in [
                interrog['model_name'] for interrog in self.current_interrogations
            ]:
                self.current_model = self.current_interrogations[0]['model_name']

            # Find and set the model in dropdown
            for i in range(self.model_selector.count()):
                if self.model_selector.itemData(i) == self.current_model:
                    self.model_selector.setCurrentIndex(i)
                    break
        else:
            self._show_no_interrogations_message()

        # Populate tag selector with all tags from all models
        self._populate_tag_selector()
        self._refresh_multimodal_prior_tables()
        self._load_multimodal_history()

    def _load_multi_image_data(self):
        """
        Load data for multi-image mode.
        Computes common tags across all selected images.
        Only displays tags that are common to ALL images.
        """
        # Disable navigation in multi-mode
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        # Update counter to show selection info
        self.image_counter.setText(f"{len(self.selected_images)} images selected")

        # Display thumbnail grid
        self.thumbnail_grid.set_images(self.selected_images)

        # Compute common tags (only tags present in ALL images)
        common_tags = self._compute_common_tags()

        # Store original common tags for comparison during save
        self.original_common_tags = set(common_tags)

        # Update tag selector - only show common tags, all pre-selected
        if common_tags:
            sorted_common = sorted(common_tags, key=str.lower)
            # Available tags = common tags only, all selected by default
            self.tag_selector.set_available_tags(sorted_common, sorted_common)
        else:
            # No common tags found
            self.tag_selector.set_available_tags([], [])

    def _compute_common_tags(self) -> Set[str]:
        """
        Compute intersection of tags across all selected images.

        Returns:
            Set of tags that exist in ALL selected images (intersection)
        """
        tag_sets = []

        for image_path in self.selected_images:
            image_tags = set()

            # Get tags from all database interrogations for this image
            try:
                file_hash = hash_image_content(image_path)
                interrogations = self.database.get_all_interrogations_for_image(file_hash) or []
                for interrog in interrogations:
                    tags = interrog.get('tags', [])
                    if tags:
                        image_tags.update(tags)
            except Exception as e:
                logger.error(f"Error loading interrogations for {image_path}: {e}")

            # Get tags from .txt file
            file_tags = FileManager.read_tags_from_file(Path(image_path))
            image_tags.update(file_tags)

            tag_sets.append(image_tags)

        # Return intersection of all sets (common tags only)
        if tag_sets:
            return set.intersection(*tag_sets)
        else:
            return set()

    def _populate_tag_selector(self):
        """Collect all tags from all interrogations and populate the tag selector."""
        # Collect all unique tags from all model interrogations
        all_tags = set()
        for interrog in self.current_interrogations:
            tags = interrog.get('tags', [])
            if tags:
                all_tags.update(tags)

        # Also include tags from the file that may not be in the database
        if self.current_file_tags:
            all_tags.update(self.current_file_tags)

        # Convert to sorted list
        all_tags_list = sorted(all_tags, key=str.lower)

        # Set available tags and selected tags in the selector
        self.tag_selector.set_available_tags(all_tags_list, self.current_file_tags)

    def _refresh_multimodal_prior_tables(self):
        """Refresh selectable prior interrogation tables for multimodal prompt context."""
        if self.is_multi_mode or self.multimodal_tab is None:
            return

        self.mm_prior_tables.setRowCount(0)
        for interrog in self.current_interrogations:
            row = self.mm_prior_tables.rowCount()
            self.mm_prior_tables.insertRow(row)

            include_item = QTableWidgetItem()
            include_item.setFlags(
                include_item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            include_item.setCheckState(Qt.CheckState.Unchecked)
            include_item.setData(Qt.ItemDataRole.UserRole, interrog)
            self.mm_prior_tables.setItem(row, 0, include_item)

            model_label = f"{interrog.get('model_name', 'Unknown')} ({interrog.get('model_type', '')})"
            self.mm_prior_tables.setItem(row, 1, QTableWidgetItem(model_label))

    def _resolve_llama_config(self) -> Dict[str, Any]:
        """Resolve latest llama.cpp config from parent contexts and cached dialog config."""
        config = dict(self.llama_config or {})

        parent = self.parent()
        while parent:
            if hasattr(parent, "inquiry_tab") and hasattr(parent.inquiry_tab, "get_llama_config"):
                try:
                    parent_cfg = parent.inquiry_tab.get_llama_config()
                    if isinstance(parent_cfg, dict):
                        config.update(parent_cfg)
                except Exception:
                    pass
            if hasattr(parent, "get_llama_config"):
                try:
                    parent_cfg = parent.get_llama_config()
                    if isinstance(parent_cfg, dict):
                        config.update(parent_cfg)
                except Exception:
                    pass
            elif hasattr(parent, "interrogation_tab") and hasattr(parent.interrogation_tab, "_get_llama_config_from_parent"):
                try:
                    parent_cfg = parent.interrogation_tab._get_llama_config_from_parent()
                    if isinstance(parent_cfg, dict):
                        config.update(parent_cfg)
                except Exception:
                    pass
            parent = parent.parent()

        self.llama_config = config
        return config

    def _ensure_multimodal_ready(self) -> bool:
        """Ensure llama.cpp interrogator exists and is loaded with valid config."""
        if self.is_multi_mode:
            return False

        config = self._resolve_llama_config()
        binary_path = (config.get("llama_binary_path") or "").strip()
        model_path = (config.get("llama_model_path") or "").strip()
        if not binary_path or not model_path:
            self.mm_status_label.setText(
                "Set llama-server and model paths in Inquiry tab before sending inquiries."
            )
            self.mm_status_label.setStyleSheet("color: orange;")
            return False

        mmproj_path = (config.get("llama_mmproj_path") or "").strip() or None
        signature = (
            binary_path,
            model_path,
            mmproj_path or "",
            int(config.get("ctx_size", 8192)),
            int(config.get("gpu_layers", -1)),
            float(config.get("temperature", 0.2)),
            int(config.get("max_tokens", 512)),
            int(config.get("server_port", 8080)),
        )

        needs_reload = True
        if self.multimodal_interrogator and self.multimodal_interrogator.is_loaded:
            existing_cfg = self.multimodal_interrogator.get_config()
            existing_sig = (
                existing_cfg.get("llama_binary_path", ""),
                existing_cfg.get("llama_model_path", ""),
                existing_cfg.get("llama_mmproj_path", "") or "",
                int(existing_cfg.get("ctx_size", 8192)),
                int(existing_cfg.get("gpu_layers", -1)),
                float(existing_cfg.get("temperature", 0.2)),
                int(existing_cfg.get("max_tokens", 512)),
                int(existing_cfg.get("server_port", 8080)),
            )
            needs_reload = existing_sig != signature

        if needs_reload:
            if self.multimodal_interrogator:
                self.multimodal_interrogator.unload_model()
            self.multimodal_interrogator = LlamaCppInterrogator(model_name="LlamaCpp")
            self.multimodal_interrogator.load_model(
                llama_binary_path=binary_path,
                llama_model_path=model_path,
                llama_mmproj_path=mmproj_path,
                ctx_size=int(config.get("ctx_size", 8192)),
                gpu_layers=int(config.get("gpu_layers", -1)),
                temperature=float(config.get("temperature", 0.2)),
                max_tokens=int(config.get("max_tokens", 512)),
                server_port=int(config.get("server_port", 8080)),
                server_host="127.0.0.1",
            )

        # Prime in-memory context from persisted history for current image session.
        if self.current_multimodal_session_key and self.current_image_hash:
            history = self.database.get_multimodal_history(
                session_key=self.current_multimodal_session_key,
                mode="single",
                image_hash=self.current_image_hash,
                model_name=self.multimodal_interrogator.model_name,
            )
            self.multimodal_interrogator.set_session_history(self.current_multimodal_session_key, history)

        self.mm_status_label.setText(
            f"Connected: {Path(model_path).name} on 127.0.0.1:{int(config.get('server_port', 8080))}"
        )
        self.mm_status_label.setStyleSheet("color: green;")
        return True

    def _build_selected_prior_tables(self) -> List[Dict[str, Any]]:
        """Build selected prior interrogation rows for multimodal prompt context."""
        selected: List[Dict[str, Any]] = []
        for row in range(self.mm_prior_tables.rowCount()):
            include_item = self.mm_prior_tables.item(row, 0)
            if not include_item or include_item.checkState() != Qt.CheckState.Checked:
                continue
            interrog = include_item.data(Qt.ItemDataRole.UserRole) or {}
            selected.append(
                {
                    "model_name": interrog.get("model_name"),
                    "model_type": interrog.get("model_type"),
                    "tags": interrog.get("tags", []),
                    "confidence_scores": interrog.get("confidence_scores"),
                    "raw_output_summary": (interrog.get("raw_output") or "")[:1500],
                    "interrogated_at": interrog.get("interrogated_at"),
                }
            )
        return selected

    def _load_multimodal_history(self):
        """Load persisted multimodal transcript for the current single-image session."""
        if self.is_multi_mode or self.multimodal_tab is None:
            return

        self.mm_transcript.setRowCount(0)
        if not self.current_multimodal_session_key or not self.current_image_hash:
            return

        history = self.database.get_multimodal_history(
            session_key=self.current_multimodal_session_key,
            mode="single",
            image_hash=self.current_image_hash,
        )
        for turn in history:
            row = self.mm_transcript.rowCount()
            self.mm_transcript.insertRow(row)
            self.mm_transcript.setItem(row, 0, QTableWidgetItem(str(turn.get("turn_index", row))))
            self.mm_transcript.setItem(row, 1, QTableWidgetItem(turn.get("prompt_text", "")))
            response = turn.get("response_json", {}) or {}
            summary = response.get("reasoning_summary", "") if isinstance(response, dict) else ""
            self.mm_transcript.setItem(row, 2, QTableWidgetItem(summary))

    def _on_send_multimodal(self):
        """Run one multimodal inquiry turn and persist session history."""
        if self.is_multi_mode:
            return
        if not self.current_image:
            QMessageBox.warning(self, "No Image", "No image selected for multimodal inquiry.")
            return

        try:
            if not self._ensure_multimodal_ready():
                return

            task = self.mm_task_combo.currentText()
            prompt_text = self.mm_prompt_input.text().strip()
            included_tables = self._build_selected_prior_tables()

            self.mm_send_button.setEnabled(False)
            results = self.multimodal_interrogator.interrogate(
                self.current_image,
                task=task,
                prompt=prompt_text,
                session_key=self.current_multimodal_session_key,
                keep_context=True,
                included_tables=included_tables,
            )

            file_hash = self.current_image_hash or hash_image_content(self.current_image)
            metadata = get_image_metadata(self.current_image)
            image_id = self.database.register_image(
                self.current_image,
                file_hash,
                metadata["width"],
                metadata["height"],
                metadata["file_size"],
            )
            model_id = self.database.register_model(
                self.multimodal_interrogator.model_name,
                self.multimodal_interrogator.get_model_type(),
                config=self.multimodal_interrogator.get_config(),
            )
            self.database.save_interrogation(
                image_id,
                model_id,
                results["tags"],
                results.get("confidence_scores"),
                results.get("raw_output"),
            )

            session_id = self.database.create_or_get_multimodal_session(
                image_id=image_id,
                model_id=model_id,
                mode="single",
                session_key=self.current_multimodal_session_key,
            )
            response_json = results.get("multimodal_response", {})
            self.database.append_multimodal_turn(
                session_id=session_id,
                prompt_type=task,
                prompt_text=prompt_text,
                included_tables=included_tables,
                response_json=response_json,
                tags=results["tags"],
                reasoning_summary=response_json.get("reasoning_summary", ""),
            )

            self.current_interrogations = self.database.get_all_interrogations_for_image(file_hash) or []
            self._populate_model_selector()
            self._refresh_multimodal_prior_tables()
            self._load_multimodal_history()

        except Exception as e:
            QMessageBox.critical(self, "Multimodal Error", f"Multimodal inquiry failed:\n{str(e)}")
        finally:
            self.mm_send_button.setEnabled(True)

    def _on_reset_multimodal_context(self):
        """Reset single-image multimodal context and clear persisted turns for current session."""
        if self.is_multi_mode:
            return
        if not self.current_multimodal_session_key or not self.current_image_hash:
            return

        try:
            model_name = None
            if self.multimodal_interrogator:
                model_name = self.multimodal_interrogator.model_name
                self.multimodal_interrogator.reset_session(self.current_multimodal_session_key)

            self.database.clear_multimodal_session(
                session_key=self.current_multimodal_session_key,
                model_name=model_name,
                mode="single",
                image_hash=self.current_image_hash,
            )
            self.mm_transcript.setRowCount(0)
            QMessageBox.information(self, "Context Reset", "Multimodal context was reset for this image.")
        except Exception as e:
            QMessageBox.critical(self, "Reset Failed", f"Could not reset context:\n{str(e)}")

    def _populate_model_selector(self):
        """Populate model selector with available models."""
        # Skip in multi-mode (no model selector)
        if self.model_selector is None:
            return

        self.model_selector.clear()

        if not self.current_interrogations:
            self.model_selector.addItem("No models available")
            self.model_selector.setEnabled(False)
            return

        self.model_selector.setEnabled(True)
        for interrog in self.current_interrogations:
            model_name = interrog['model_name']
            model_type = interrog.get('model_type', '')
            display_name = f"{model_name} ({model_type})" if model_type else model_name
            self.model_selector.addItem(display_name, model_name)

    def _on_model_selected(self, display_name: str):
        """
        Handle model selection from dropdown.
        Loads that model's tags and updates all views.
        """
        if not self.current_interrogations:
            return

        # Get model name from item data
        idx = self.model_selector.currentIndex()
        if idx < 0:
            return

        model_name = self.model_selector.itemData(idx)
        if not model_name:
            return

        self.current_model = model_name

        # Find this model's interrogation
        model_data = None
        for interrog in self.current_interrogations:
            if interrog['model_name'] == model_name:
                model_data = interrog
                break

        if not model_data:
            return

        # Update model results tab
        self._update_ratings_display(model_data)
        self._update_results_table(model_data)

        # Update comparison tab
        self._update_comparison_view(model_data)

    def _extract_wd_ratings(self, tags: List[str], confidence_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Extract WD sensitivity ratings from tags.

        Returns:
            Dict with keys: 'general', 'sensitive', 'questionable', 'explicit'
            Values are confidence scores (0.0-1.0)
        """
        ratings = {
            'general': 0.0,
            'sensitive': 0.0,
            'questionable': 0.0,
            'explicit': 0.0
        }

        # Check for rating tags (with or without 'rating:' prefix)
        rating_mappings = [
            ('general', ['rating:safe', 'general', 'rating:general']),
            ('sensitive', ['rating:sensitive', 'sensitive']),
            ('questionable', ['rating:questionable', 'questionable']),
            ('explicit', ['rating:explicit', 'explicit'])
        ]

        for rating_name, possible_tags in rating_mappings:
            for tag in possible_tags:
                if tag in tags and tag in confidence_scores:
                    ratings[rating_name] = confidence_scores[tag]
                    break

        return ratings

    def _update_ratings_display(self, model_data: Dict):
        """Update ratings display with model data."""
        # Check if this is a WD model
        if model_data.get('model_type') != 'WD':
            self.ratings_group.setVisible(False)
            return

        self.ratings_group.setVisible(True)

        # Extract ratings
        ratings = self._extract_wd_ratings(
            model_data['tags'],
            model_data.get('confidence_scores', {}) or {}
        )

        # Update UI
        for rating_name, confidence in ratings.items():
            if rating_name in self.rating_bars:
                self.rating_bars[rating_name].setValue(int(confidence * 100))
                self.rating_labels[rating_name].setText(f"{confidence:.2f}")

    def _update_results_table(self, model_data: Dict):
        """Update the results table with model data."""
        self.results_table.set_results(model_data)

    def _build_comparison_data(self, model_data: Dict) -> List[Dict]:
        """
        Build tag comparison between database and file.

        Returns list of dicts with:
        - tag: str
        - confidence: float (from DB)
        - status: str
        - location: str
        - original_tag: str (if replaced)
        """
        db_tags = model_data['tags']
        db_confidence = model_data.get('confidence_scores', {}) or {}
        file_tags = self.current_file_tags

        # Apply filters to see what WOULD be written
        filtered_tags = []
        if self.tag_filters and db_confidence:
            threshold = 0.35  # Default, could be model-specific
            filtered_tags, _ = self.tag_filters.filter_tags_with_confidence(
                db_tags, db_confidence, threshold
            )
        else:
            filtered_tags = db_tags

        comparison = []

        # Process DB tags
        for tag in db_tags:
            tag_lower = tag.lower()
            conf = db_confidence.get(tag, 0.0)

            # Check if in file
            in_file = tag in file_tags
            in_filtered = tag in filtered_tags

            # Determine status
            if in_file and in_filtered:
                status = 'in_both'
                location = 'Both'
            elif in_file and not in_filtered:
                status = 'manually_added'
                location = 'File'
            elif not in_file and in_filtered:
                status = 'db_only'
                location = 'Database'
            else:  # not in file, not in filtered
                status = 'removed_by_filter'
                location = 'Database (filtered)'

            # Check if replaced
            if self.tag_filters and tag_lower in self.tag_filters.replace_dict:
                replacement = self.tag_filters.replace_dict[tag_lower]
                status = 'replaced'
                comparison.append({
                    'tag': replacement,
                    'confidence': conf,
                    'status': status,
                    'location': location,
                    'original_tag': tag
                })
            else:
                comparison.append({
                    'tag': tag,
                    'confidence': conf,
                    'status': status,
                    'location': location,
                    'original_tag': None
                })

        # Add file-only tags (not in DB at all)
        for tag in file_tags:
            if tag not in db_tags:
                comparison.append({
                    'tag': tag,
                    'confidence': None,
                    'status': 'file_only',
                    'location': 'File only',
                    'original_tag': None
                })

        return comparison

    def _update_comparison_view(self, model_data: Dict):
        """Update the comparison table with color-coded tags."""
        self.comparison_info_label.setText(
            f"Comparing: {model_data['model_name']} ({model_data.get('model_type', 'Unknown')})"
        )

        comparison_data = self._build_comparison_data(model_data)

        self.comparison_table.setRowCount(0)

        # Color scheme
        colors = {
            'in_both': QColor(220, 255, 220),       # Light green
            'db_only': QColor(255, 255, 220),       # Light yellow
            'file_only': QColor(220, 220, 255),     # Light blue
            'removed_by_filter': QColor(255, 220, 220),  # Light red
            'replaced': QColor(255, 230, 200),      # Light orange
            'manually_added': QColor(200, 230, 255) # Light cyan
        }

        # Black text for legibility on light pastel backgrounds (fixes dark theme)
        text_color = QBrush(QColor(0, 0, 0))

        for item in comparison_data:
            row = self.comparison_table.rowCount()
            self.comparison_table.insertRow(row)

            # Tag column
            tag_text = item['tag']
            if item['original_tag']:
                tag_text += f" (was: {item['original_tag']})"
            tag_item = QTableWidgetItem(tag_text)
            tag_item.setBackground(colors.get(item['status'], Qt.GlobalColor.white))
            tag_item.setForeground(text_color)
            self.comparison_table.setItem(row, 0, tag_item)

            # Confidence column
            conf_text = f"{item['confidence']:.4f}" if item['confidence'] is not None else "N/A"
            conf_item = QTableWidgetItem(conf_text)
            conf_item.setBackground(colors.get(item['status'], Qt.GlobalColor.white))
            conf_item.setForeground(text_color)
            self.comparison_table.setItem(row, 1, conf_item)

            # Status column
            status_labels = {
                'in_both': 'In Both',
                'db_only': 'Database Only',
                'file_only': 'File Only',
                'removed_by_filter': 'Filtered Out',
                'replaced': 'Replaced',
                'manually_added': 'Manually Added'
            }
            status_item = QTableWidgetItem(status_labels.get(item['status'], item['status']))
            status_item.setBackground(colors.get(item['status'], Qt.GlobalColor.white))
            status_item.setForeground(text_color)
            self.comparison_table.setItem(row, 2, status_item)

            # Location column
            loc_item = QTableWidgetItem(item['location'])
            loc_item.setBackground(colors.get(item['status'], Qt.GlobalColor.white))
            loc_item.setForeground(text_color)
            self.comparison_table.setItem(row, 3, loc_item)

    def _navigate_previous(self):
        """Navigate to previous image."""
        if self.current_index <= 0:
            return

        self.current_index -= 1
        self.current_image = self.image_list[self.current_index]
        self.load_image_data()

    def _navigate_next(self):
        """Navigate to next image."""
        if self.current_index >= len(self.image_list) - 1:
            return

        self.current_index += 1
        self.current_image = self.image_list[self.current_index]
        self.load_image_data()

    def _save_edited_tags(self, tags: List[str]):
        """Save edited tags - dispatches to single or multi-image save."""
        if self.is_multi_mode:
            self._save_multi_image_tags(tags)
        else:
            self._save_single_image_tags(tags)

    def _save_single_image_tags(self, tags: List[str]):
        """Save edited tags directly to file (single-image mode)."""
        try:
            FileManager.write_tags_to_file(
                Path(self.current_image),
                tags,
                overwrite=True
            )

            # Refresh file tags
            self.current_file_tags = tags

            # Update file tags status
            if tags:
                self.file_tags_status_label.setText(
                    f".txt file: {len(tags)} tags"
                )
                self.file_tags_status_label.setStyleSheet("color: green;")
            else:
                self.file_tags_status_label.setText(".txt file: No tags")
                self.file_tags_status_label.setStyleSheet("color: orange;")

            # Refresh tag selector to update checkboxes
            self._populate_tag_selector()

            # Update comparison view if we have a current model
            if self.current_model and self.current_interrogations:
                model_data = next(
                    (m for m in self.current_interrogations if m['model_name'] == self.current_model),
                    None
                )
                if model_data:
                    self._update_comparison_view(model_data)

            # Emit signal
            self.tags_saved.emit(self.current_image, tags)

            # Show success message
            QMessageBox.information(self, "Success", "Tags saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save tags:\n{str(e)}")

    def _save_multi_image_tags(self, selected_tags: List[str]):
        """
        Save tag changes to all selected images (multi-image mode).

        Only affects common tags - each image's unique tags are preserved.
        - Tags unchecked in the editor are removed from all images
        - Tags checked in the editor are ensured to exist in all images
        """
        selected_set = set(selected_tags)
        original_set = self.original_common_tags

        # Calculate what changed
        tags_to_remove = original_set - selected_set  # Were common, now unchecked
        tags_to_add = selected_set - original_set     # Shouldn't happen (can't add non-common tags)

        # Build confirmation message
        changes = []
        if tags_to_remove:
            changes.append(f"Remove {len(tags_to_remove)} tag(s): {', '.join(sorted(tags_to_remove)[:5])}"
                          + ("..." if len(tags_to_remove) > 5 else ""))
        if tags_to_add:
            changes.append(f"Add {len(tags_to_add)} tag(s): {', '.join(sorted(tags_to_add)[:5])}"
                          + ("..." if len(tags_to_add) > 5 else ""))

        if not changes:
            QMessageBox.information(self, "No Changes", "No changes to save.")
            return

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Save Tag Changes",
            f"Apply changes to {len(self.selected_images)} images?\n\n"
            + "\n".join(changes) + "\n\n"
            "Each image's unique tags will be preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Track success/failure
        success_count = 0
        failed_paths = []
        saved_results = []  # Collect results for batch signal

        # Update each image's tags
        for image_path in self.selected_images:
            try:
                # Read current tags from file
                current_tags = FileManager.read_tags_from_file(Path(image_path))
                current_set = set(current_tags)

                # Apply changes: remove unchecked common tags, add any new ones
                new_set = (current_set - tags_to_remove) | tags_to_add

                # Preserve original order for tags that weren't changed,
                # append any new tags at the end
                new_tags = [t for t in current_tags if t in new_set]
                for t in tags_to_add:
                    if t not in new_tags:
                        new_tags.append(t)

                # Write back
                FileManager.write_tags_to_file(
                    Path(image_path),
                    new_tags,
                    overwrite=True
                )
                saved_results.append((image_path, new_tags))
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to save tags to {image_path}: {e}")
                failed_paths.append(image_path)

        # Only update original common tags when all saves succeed.
        # If some failed, keep original state so retries can detect changes.
        if not failed_paths:
            self.original_common_tags = selected_set

        # Emit batch signal once (not individual signals per image)
        if saved_results:
            self.batch_tags_saved.emit(saved_results)

        # Show result message
        if failed_paths:
            QMessageBox.warning(
                self,
                "Partial Success",
                f"Tags updated for {success_count} images.\n"
                f"Failed to update {len(failed_paths)} images:\n"
                + "\n".join(Path(p).name for p in failed_paths[:5])
                + ("\n..." if len(failed_paths) > 5 else "")
            )
        else:
            QMessageBox.information(
                self,
                "Success",
                f"Tags updated for all {success_count} images!"
            )

    def _quick_save_tags(self):
        """Quick save if on editor tab."""
        # In multi-mode, there's only one tab (index 0)
        # In single-mode, editor tab is index 2
        editor_tab_index = 0 if self.is_multi_mode else 2
        if self.main_tabs.currentIndex() == editor_tab_index:
            tags = self.tag_selector.get_selected_tags()
            self._save_edited_tags(tags)

    def _show_no_interrogations_message(self):
        """Show message when image has no interrogations."""
        # Skip in multi-mode (these widgets don't exist)
        if self.is_multi_mode:
            return

        # Disable model selector
        if self.model_selector:
            self.model_selector.clear()
            self.model_selector.setEnabled(False)

        # Hide ratings section
        if self.ratings_group:
            self.ratings_group.setVisible(False)

        # Show message in results table
        if self.results_table:
            self.results_table.clear_results()

        # Comparison tab shows message
        if self.comparison_table:
            self.comparison_table.setRowCount(1)
            self.comparison_table.setColumnCount(1)
            msg_item = QTableWidgetItem("This image has not been interrogated yet.")
            msg_item.setFlags(msg_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.comparison_table.setItem(0, 0, msg_item)

        if self.comparison_info_label:
            self.comparison_info_label.setText("No interrogation data available")

    def _on_tab_changed(self, index: int):
        """Save the currently selected tab to settings (single-mode only)."""
        # Don't save tab preference in multi-mode (only one tab)
        if self.is_multi_mode:
            return
        settings = QSettings()
        settings.setValue(self.SETTINGS_KEY_LAST_TAB, index)

    def closeEvent(self, event):
        """Cleanup multimodal resources on dialog close."""
        try:
            if self.multimodal_interrogator:
                self.multimodal_interrogator.unload_model()
        except Exception:
            pass
        super().closeEvent(event)
