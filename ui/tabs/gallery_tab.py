"""Gallery Tab - Enhanced image browsing with filtering and tag editing."""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QGroupBox, QLabel, QPushButton, QComboBox,
                             QLineEdit, QCheckBox, QScrollArea, QFrame,
                             QTabWidget, QSlider, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from core import InterrogationDatabase, FileManager
from core.hashing import hash_image_content
from ui.widgets import ImageGalleryWidget, TagEditorWidget, ResultsTableWidget
from ui.dialogs import OrganizeDialog
from ui.dialogs_advanced import AdvancedImageInspectionDialog


class TagFilterWidget(QWidget):
    """Widget for filtering images by tags."""

    # Signals
    filter_changed = pyqtSignal(list)  # selected_tags
    sort_changed = pyqtSignal(str)  # sort_mode: 'name', 'date', 'size'
    show_changed = pyqtSignal(str)  # show_mode: 'all', 'tagged', 'untagged'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tag_checkboxes = {}  # tag -> QCheckBox
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Search box
        search_label = QLabel("Search Tags:")
        layout.addWidget(search_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type to search tags...")
        self.search_box.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_box)

        # Tag list in scroll area
        tag_list_label = QLabel("Filter by Tags:")
        layout.addWidget(tag_list_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)

        self.tag_list_widget = QWidget()
        self.tag_list_layout = QVBoxLayout(self.tag_list_widget)
        self.tag_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.tag_list_widget)

        layout.addWidget(scroll, 1)

        # Clear all button
        self.clear_button = QPushButton("Clear All Filters")
        self.clear_button.clicked.connect(self._on_clear_filters)
        layout.addWidget(self.clear_button)

        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator1)

        # Sort options
        sort_label = QLabel("Sort By:")
        layout.addWidget(sort_label)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Name", "Date", "Size"])
        self.sort_combo.currentTextChanged.connect(self._on_sort_changed)
        layout.addWidget(self.sort_combo)

        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator2)

        # Show options
        show_label = QLabel("Show:")
        layout.addWidget(show_label)

        self.show_combo = QComboBox()
        self.show_combo.addItems(["All Images", "Tagged Only", "Untagged Only"])
        self.show_combo.currentTextChanged.connect(self._on_show_changed)
        layout.addWidget(self.show_combo)

    def update_tag_list(self, tags_with_counts: Dict[str, int]):
        """
        Update the tag list with counts.

        Args:
            tags_with_counts: Dict of tag -> count
        """
        # Clear existing checkboxes
        for checkbox in self.tag_checkboxes.values():
            checkbox.deleteLater()
        self.tag_checkboxes.clear()

        # Sort tags by count (descending)
        sorted_tags = sorted(tags_with_counts.items(), key=lambda x: x[1], reverse=True)

        # Create checkboxes
        for tag, count in sorted_tags:
            checkbox = QCheckBox(f"{tag} ({count})")
            checkbox.setObjectName(tag)
            checkbox.stateChanged.connect(self._on_filter_changed)
            self.tag_list_layout.addWidget(checkbox)
            self.tag_checkboxes[tag] = checkbox

    def _on_search_changed(self, text: str):
        """Handle search box text change."""
        search_text = text.lower()
        for tag, checkbox in self.tag_checkboxes.items():
            if search_text in tag.lower():
                checkbox.setVisible(True)
            else:
                checkbox.setVisible(False)

    def _on_filter_changed(self):
        """Handle filter checkbox change."""
        selected_tags = [
            tag for tag, checkbox in self.tag_checkboxes.items()
            if checkbox.isChecked()
        ]
        self.filter_changed.emit(selected_tags)

    def _on_clear_filters(self):
        """Clear all tag filters."""
        for checkbox in self.tag_checkboxes.values():
            checkbox.setChecked(False)

    def _on_sort_changed(self, text: str):
        """Handle sort mode change."""
        mode_map = {
            "Name": "name",
            "Date": "date",
            "Size": "size"
        }
        self.sort_changed.emit(mode_map.get(text, "name"))

    def _on_show_changed(self, text: str):
        """Handle show mode change."""
        mode_map = {
            "All Images": "all",
            "Tagged Only": "tagged",
            "Untagged Only": "untagged"
        }
        self.show_changed.emit(mode_map.get(text, "all"))


class EnhancedImagePreview(QWidget):
    """Enhanced image preview with zoom controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_pixmap = None
        self.zoom_level = 1.0
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Image preview
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(QSize(400, 400))
        self.image_label.setScaledContents(False)
        self.image_label.setText("No image selected")
        layout.addWidget(self.image_label, 1)

        # Zoom controls
        controls_layout = QHBoxLayout()

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setMaximumWidth(40)
        self.zoom_out_btn.clicked.connect(lambda: self.set_zoom(self.zoom_level - 0.25))
        controls_layout.addWidget(self.zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.zoom_label)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setMaximumWidth(40)
        self.zoom_in_btn.clicked.connect(lambda: self.set_zoom(self.zoom_level + 0.25))
        controls_layout.addWidget(self.zoom_in_btn)

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(lambda: self.set_zoom(0))  # 0 = fit to window
        controls_layout.addWidget(self.fit_btn)

        self.actual_size_btn = QPushButton("1:1")
        self.actual_size_btn.clicked.connect(lambda: self.set_zoom(1.0))
        controls_layout.addWidget(self.actual_size_btn)

        layout.addLayout(controls_layout)

        # Image info
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

    def set_image(self, image_path: str):
        """Load and display an image."""
        self.current_pixmap = QPixmap(image_path)
        if self.current_pixmap.isNull():
            self.image_label.setText("Failed to load image")
            self.info_label.setText("")
            return

        # Update info
        path = Path(image_path)
        dimensions = f"{self.current_pixmap.width()}x{self.current_pixmap.height()}"
        size_kb = path.stat().st_size / 1024
        self.info_label.setText(f"File: {path.name}\nSize: {size_kb:.1f} KB\nDimensions: {dimensions}")

        # Reset zoom and display
        self.zoom_level = 0  # Fit to window initially
        self._update_display()

    def set_zoom(self, level: float):
        """Set zoom level (0 = fit to window, 1.0 = actual size)."""
        if level == 0:
            self.zoom_level = 0
        else:
            self.zoom_level = max(0.25, min(4.0, level))  # Clamp between 25% and 400%
        self._update_display()

    def _update_display(self):
        """Update the displayed image with current zoom."""
        if not self.current_pixmap:
            return

        if self.zoom_level == 0:
            # Fit to window
            scaled = self.current_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.zoom_label.setText("Fit")
        else:
            # Specific zoom level
            target_size = QSize(
                int(self.current_pixmap.width() * self.zoom_level),
                int(self.current_pixmap.height() * self.zoom_level)
            )
            scaled = self.current_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.zoom_label.setText(f"{int(self.zoom_level * 100)}%")

        self.image_label.setPixmap(scaled)

    def clear(self):
        """Clear the preview."""
        self.current_pixmap = None
        self.zoom_level = 1.0
        self.image_label.clear()
        self.image_label.setText("No image selected")
        self.info_label.setText("")
        self.zoom_label.setText("100%")


class MultiModelResultsWidget(QWidget):
    """Widget for displaying results from multiple models."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Tab widget for different models
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)

        # "All Models" view
        self.all_models_table = ResultsTableWidget()
        self.results_tabs.addTab(self.all_models_table, "All Models")

        # Model-specific tabs will be added dynamically
        self.model_tables = {}  # model_name -> ResultsTableWidget

        # Copy tags button
        self.copy_tags_button = QPushButton("Copy Tags to Editor")
        layout.addWidget(self.copy_tags_button)

    def set_results(self, all_results: List[Dict]):
        """
        Display results from multiple interrogations.

        Args:
            all_results: List of result dicts, each with 'tags', 'confidence_scores', 'model_name'
        """
        if not all_results:
            self.clear_results()
            return

        # Filter out None values (corrupted database entries)
        all_results = [r for r in all_results if r is not None]

        if not all_results:
            self.clear_results()
            return

        # Clear existing model tabs (keep "All Models")
        for i in range(self.results_tabs.count() - 1, 0, -1):
            self.results_tabs.removeTab(i)
        self.model_tables.clear()

        # Aggregate all results for "All Models" view
        all_tags = []
        all_confidence = {}
        all_models = {}

        for result in all_results:
            model_name = result.get('model_name', 'Unknown')
            tags = result.get('tags', [])
            confidence_scores = result.get('confidence_scores', {})

            # Handle None confidence_scores (CLIP returns None)
            if confidence_scores is None:
                confidence_scores = {}

            # Create model-specific tab
            model_table = ResultsTableWidget()
            model_table.set_results(result)
            self.results_tabs.addTab(model_table, model_name)
            self.model_tables[model_name] = model_table

            # Aggregate for "All Models"
            for tag in tags:
                if tag not in all_tags:
                    all_tags.append(tag)
                    all_confidence[tag] = confidence_scores.get(tag, 0)
                    all_models[tag] = model_name
                else:
                    # If tag appears in multiple models, keep highest confidence
                    if tag in confidence_scores:
                        if confidence_scores[tag] > all_confidence.get(tag, 0):
                            all_confidence[tag] = confidence_scores[tag]
                            all_models[tag] = model_name

        # Display aggregated results
        aggregated = {
            'tags': all_tags,
            'confidence_scores': all_confidence,
            'model_name': 'Multiple'
        }
        self.all_models_table.set_results(aggregated)

    def clear_results(self):
        """Clear all results."""
        self.all_models_table.clear_results()
        for i in range(self.results_tabs.count() - 1, 0, -1):
            self.results_tabs.removeTab(i)
        self.model_tables.clear()

    def get_current_tags(self) -> List[str]:
        """Get tags from currently active tab."""
        current_widget = self.results_tabs.currentWidget()
        if isinstance(current_widget, ResultsTableWidget):
            return current_widget.get_all_tags()
        return []


class GalleryTab(QWidget):
    """Tab for enhanced image browsing and tag editing."""

    # Signals
    image_selected = pyqtSignal(str)  # image_path
    tags_saved = pyqtSignal(str, list)  # image_path, tags

    def __init__(self, database, parent=None):
        """
        Initialize the Gallery Tab.

        Args:
            database: InterrogationDatabase instance
            parent: Parent widget
        """
        super().__init__(parent)

        # Store shared state references
        self.database = database
        self.current_directory = None
        self.current_image = None
        self.all_images = []  # List of image paths
        self.filtered_images = []  # After applying filters
        self.recursive = False  # Whether to search subdirectories

        # Filter state
        self.current_filters = {
            'tags': [],
            'sort': 'name',
            'show': 'all'
        }

        # Setup UI
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Create 3-panel splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # === Left Panel: Tag Filtering (250px) ===
        self.tag_filter = TagFilterWidget()
        self.tag_filter.setMaximumWidth(250)
        self.splitter.addWidget(self.tag_filter)

        # === Center Panel: Gallery (stretch) ===
        gallery_widget = QWidget()
        gallery_layout = QVBoxLayout(gallery_widget)

        # Gallery title
        gallery_layout.addWidget(QLabel("Image Gallery"))

        # Image gallery
        self.image_gallery = ImageGalleryWidget()
        gallery_layout.addWidget(self.image_gallery, 1)

        # Gallery controls
        controls_layout = QHBoxLayout()

        self.grid_size_label = QLabel("Grid Size:")
        controls_layout.addWidget(self.grid_size_label)

        self.grid_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_size_slider.setMinimum(100)
        self.grid_size_slider.setMaximum(400)
        self.grid_size_slider.setValue(200)
        self.grid_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.grid_size_slider.setTickInterval(50)
        self.grid_size_slider.valueChanged.connect(self._on_grid_size_changed)
        controls_layout.addWidget(self.grid_size_slider)

        self.grid_size_value_label = QLabel("200px")
        controls_layout.addWidget(self.grid_size_value_label)

        controls_layout.addStretch()

        # Multi-select checkbox
        self.multi_select_checkbox = QCheckBox("Multi-Select")
        self.multi_select_checkbox.setToolTip("Enable multi-selection mode (Ctrl+Click or Shift+Click)")
        self.multi_select_checkbox.stateChanged.connect(self._on_multi_select_changed)
        controls_layout.addWidget(self.multi_select_checkbox)

        # Organization button
        self.organize_button = QPushButton("Organize by Tags")
        self.organize_button.clicked.connect(self._open_organize_dialog)
        controls_layout.addWidget(self.organize_button)

        self.image_count_label = QLabel("0 images")
        controls_layout.addWidget(self.image_count_label)

        gallery_layout.addLayout(controls_layout)

        self.splitter.addWidget(gallery_widget)

        # === Right Panel: Preview + Results + Editor (500px) ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Image preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.image_preview = EnhancedImagePreview()
        preview_layout.addWidget(self.image_preview)
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group)

        # Multi-model results
        results_group = QGroupBox("Interrogation Results")
        results_layout = QVBoxLayout()
        self.results_widget = MultiModelResultsWidget()
        results_layout.addWidget(self.results_widget)
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)

        # Tag editor
        tag_editor_group = QGroupBox("Tag Editor")
        tag_editor_layout = QVBoxLayout()
        self.tag_editor = TagEditorWidget()
        tag_editor_layout.addWidget(self.tag_editor)
        tag_editor_group.setLayout(tag_editor_layout)
        right_layout.addWidget(tag_editor_group)

        right_panel.setMinimumWidth(500)
        self.splitter.addWidget(right_panel)

        # Set splitter proportions (1:3:2)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setStretchFactor(2, 2)

        layout.addWidget(self.splitter)

    def setup_connections(self):
        """Setup signal/slot connections."""
        # Tag filter signals
        self.tag_filter.filter_changed.connect(self._on_filter_changed)
        self.tag_filter.sort_changed.connect(self._on_sort_changed)
        self.tag_filter.show_changed.connect(self._on_show_changed)

        # Gallery signals
        self.image_gallery.image_selected.connect(self._on_image_selected)
        self.image_gallery.itemDoubleClicked.connect(self._on_gallery_double_click)
        self.image_gallery.inspection_requested.connect(self._open_advanced_inspection)
        self.image_gallery.multi_selection_changed.connect(self._on_multi_selection_changed)
        self.image_gallery.multi_inspection_requested.connect(self._open_multi_advanced_inspection)

        # Results signals
        self.results_widget.copy_tags_button.clicked.connect(self._copy_tags_to_editor)

        # Tag editor signals
        self.tag_editor.tags_changed.connect(self._save_tags)

    def set_directory(self, directory: str, recursive: bool = False):
        """
        Set the current directory and refresh gallery.

        Args:
            directory: Path to image directory
            recursive: Whether to search subdirectories
        """
        self.current_directory = Path(directory)
        self.recursive = recursive
        self.refresh_gallery()

    def set_images_direct(self, image_paths: List[str]):
        """
        Set gallery images directly from pre-scanned paths.

        This avoids duplicate directory scanning when paths are already
        loaded by another component (e.g., InterrogationTab).

        Args:
            image_paths: List of absolute image path strings
        """
        self.all_images = image_paths

        # Apply filters
        self._apply_filters()

        # Update tag filter with available tags
        self._update_tag_filter()

    def refresh_gallery(self):
        """Refresh the image gallery from current directory."""
        if not self.current_directory:
            return

        # Find all images
        self.all_images = [str(p) for p in FileManager.find_images(str(self.current_directory), recursive=self.recursive)]

        # Apply filters
        self._apply_filters()

        # Update tag filter with available tags
        self._update_tag_filter()

    def _update_tag_filter(self):
        """Update the tag filter widget with tags from current directory."""
        if not self.current_directory:
            return

        # Collect all tags from all images in directory
        tag_counts = {}
        for image_path in self.all_images:
            tags = FileManager.read_tags_from_file(Path(image_path))
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        self.tag_filter.update_tag_list(tag_counts)

    def _apply_filters(self):
        """Apply current filters and update gallery display."""
        # Start with all images
        filtered = self.all_images.copy()

        # Apply tag filter
        if self.current_filters['tags']:
            filtered = [
                img for img in filtered
                if self._image_has_tags(img, self.current_filters['tags'])
            ]

        # Apply show filter (all/tagged/untagged)
        if self.current_filters['show'] == 'tagged':
            filtered = [img for img in filtered if FileManager.has_text_file(Path(img))]
        elif self.current_filters['show'] == 'untagged':
            filtered = [img for img in filtered if not FileManager.has_text_file(Path(img))]

        # Apply sort
        filtered = self._sort_images(filtered, self.current_filters['sort'])

        # Update gallery
        self.filtered_images = filtered
        self._update_gallery_display()

    def _image_has_tags(self, image_path: str, required_tags: List[str]) -> bool:
        """Check if image has all required tags."""
        image_tags = FileManager.read_tags_from_file(Path(image_path))
        return all(tag in image_tags for tag in required_tags)

    def _sort_images(self, images: List[str], sort_mode: str) -> List[str]:
        """Sort images by specified mode."""
        if sort_mode == 'name':
            return sorted(images, key=lambda p: Path(p).name.lower())
        elif sort_mode == 'date':
            return sorted(images, key=lambda p: Path(p).stat().st_mtime, reverse=True)
        elif sort_mode == 'size':
            return sorted(images, key=lambda p: Path(p).stat().st_size, reverse=True)
        return images

    def _update_gallery_display(self):
        """Update the gallery widget with filtered images - start batched loading."""
        self.image_gallery.clear_gallery()

        # Store batch loading ID to detect stale batches
        self._gallery_batch_id = id(self.filtered_images)

        # Initialize batch state
        self._gallery_batch_index = 0
        self._gallery_batch_size = 50  # Smaller batches since image loading is expensive

        total = len(self.filtered_images)
        if total > 0:
            self.image_gallery.setUpdatesEnabled(False)
            self.image_count_label.setText(f"Loading 0/{total} images (0%)")
            # Start batch processing
            QTimer.singleShot(0, self._add_gallery_batch)
        else:
            self.image_count_label.setText("0 images")

    def _add_gallery_batch(self):
        """Add a batch of images to the gallery, then schedule next batch."""
        # Check if this batch is stale (filters changed during loading)
        if not hasattr(self, '_gallery_batch_id') or self._gallery_batch_id != id(self.filtered_images):
            return

        images = self.filtered_images
        total = len(images)
        end_index = min(self._gallery_batch_index + self._gallery_batch_size, total)

        # Add batch of images
        for i in range(self._gallery_batch_index, end_index):
            image_path = images[i]
            has_tags = FileManager.has_text_file(Path(image_path))
            self.image_gallery.add_image(image_path, has_tags)

        self._gallery_batch_index = end_index

        # Update progress with percentage
        percent = int((end_index / total) * 100)
        self.image_count_label.setText(f"Loading {end_index}/{total} images ({percent}%)")

        # Schedule next batch or finish
        if self._gallery_batch_index < total:
            QTimer.singleShot(0, self._add_gallery_batch)
        else:
            self._finish_gallery_display()

    def _finish_gallery_display(self):
        """Complete gallery display after all batches processed."""
        self.image_gallery.setUpdatesEnabled(True)
        self.image_count_label.setText(f"{len(self.filtered_images)} images")

    def _on_filter_changed(self, selected_tags: List[str]):
        """Handle tag filter change."""
        self.current_filters['tags'] = selected_tags
        self._apply_filters()

    def _on_sort_changed(self, sort_mode: str):
        """Handle sort mode change."""
        self.current_filters['sort'] = sort_mode
        self._apply_filters()

    def _on_show_changed(self, show_mode: str):
        """Handle show mode change."""
        self.current_filters['show'] = show_mode
        self._apply_filters()

    def _on_grid_size_changed(self, size: int):
        """Handle grid size slider change."""
        self.image_gallery.setIconSize(QSize(size, size))
        self.grid_size_value_label.setText(f"{size}px")

    def _on_image_selected(self, image_path: str):
        """Handle image selection from gallery."""
        self.current_image = image_path

        # Update preview
        self.image_preview.set_image(image_path)

        # Load existing tags
        existing_tags = FileManager.read_tags_from_file(Path(image_path))
        self.tag_editor.set_tags(existing_tags)

        # Load cached interrogations (all models)
        try:
            file_hash = hash_image_content(image_path)
            cached_results = self.database.get_all_interrogations_for_image(file_hash)

            if cached_results:
                self.results_widget.set_results(cached_results)
            else:
                self.results_widget.clear_results()
        except Exception as e:
            print(f"Error loading cached results: {e}")
            self.results_widget.clear_results()

        # Emit signal
        self.image_selected.emit(image_path)

    def _copy_tags_to_editor(self):
        """Copy tags from current results tab to editor."""
        tags = self.results_widget.get_current_tags()
        if tags:
            self.tag_editor.set_tags(tags)

    def _save_tags(self, tags: List[str]):
        """Save edited tags to file."""
        if not self.current_image:
            return

        try:
            FileManager.write_tags_to_file(Path(self.current_image), tags, overwrite=True)
            self.image_gallery.update_image_status(self.current_image, len(tags) > 0)

            # Update tag filter counts
            self._update_tag_filter()

            # Emit signal
            self.tags_saved.emit(self.current_image, tags)

            # Show success message in parent's status bar if available
            parent = self.parent()
            while parent and not hasattr(parent, 'statusBar'):
                parent = parent.parent()
            if parent and hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage("Tags saved", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save tags:\n{str(e)}")

    def refresh_current_image(self):
        """Refresh the display of the current image (e.g., after new interrogation)."""
        if self.current_image:
            self._on_image_selected(self.current_image)

    def _on_gallery_double_click(self, item):
        """Handle double-click on gallery image."""
        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path:
            self._open_advanced_inspection(image_path)

    def _open_advanced_inspection(self, image_path: str):
        """Open the advanced image inspection dialog."""
        try:
            dialog = AdvancedImageInspectionDialog(
                image_path=image_path,
                image_list=self.filtered_images,  # For navigation
                database=self.database,
                tag_filters=None,  # Optional - not passed from gallery
                parent=self
            )
            # Connect signal to refresh gallery when tags are saved
            dialog.tags_saved.connect(self._on_advanced_dialog_tags_saved)
            dialog.show()  # Non-modal
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open inspection dialog:\n{str(e)}")

    def _on_advanced_dialog_tags_saved(self, image_path: str, tags: List[str]):
        """Handle tags saved from advanced inspection dialog."""
        # Update gallery image status
        self.image_gallery.update_image_status(image_path, len(tags) > 0)

        # Update tag filter counts
        self._update_tag_filter()

        # If it's the current image, refresh the display
        if self.current_image == image_path:
            self.refresh_current_image()

    def _open_organize_dialog(self):
        """Open the organization dialog."""
        if not self.current_directory:
            QMessageBox.warning(self, "Warning", "Please select a directory first")
            return

        dialog = OrganizeDialog(str(self.current_directory), self)
        dialog.exec()

        # Refresh gallery after organization
        self.refresh_gallery()

    def _on_multi_select_changed(self, state: int):
        """Handle multi-select checkbox toggle."""
        multi_mode = state == Qt.CheckState.Checked.value
        self.image_gallery.set_selection_mode(multi_mode)

        # Update UI hint
        if multi_mode:
            self.image_count_label.setText(
                f"{len(self.filtered_images)} images (multi-select enabled)"
            )
        else:
            self.image_count_label.setText(f"{len(self.filtered_images)} images")

    def _on_multi_selection_changed(self, image_paths: List[str]):
        """Handle multiple image selection."""
        # Update count label with selection info
        self.image_count_label.setText(
            f"{len(image_paths)} of {len(self.filtered_images)} images selected"
        )

        # Clear preview when multiple images selected (show first one as hint)
        if image_paths:
            self.image_preview.set_image(image_paths[0])
            self.image_preview.info_label.setText(
                f"Selected: {len(image_paths)} images\n"
                f"Right-click for batch tag editing"
            )

    def _open_multi_advanced_inspection(self, image_paths: List[str]):
        """Open the advanced inspection dialog for multiple images."""
        if not image_paths:
            return

        try:
            dialog = AdvancedImageInspectionDialog(
                image_path=image_paths,  # Pass list for multi-mode
                image_list=self.filtered_images,
                database=self.database,
                tag_filters=None,
                parent=self
            )
            # Connect signals to refresh gallery when tags are saved
            dialog.tags_saved.connect(self._on_advanced_dialog_tags_saved)
            dialog.batch_tags_saved.connect(self._on_batch_tags_saved)
            dialog.show()  # Non-modal
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open multi-image dialog:\n{str(e)}")

    def _on_batch_tags_saved(self, saved_results: List[tuple]):
        """
        Handle batch tags saved from multi-image dialog.

        Args:
            saved_results: List of (image_path, tags) tuples
        """
        # Update gallery status for each image (lightweight)
        for image_path, tags in saved_results:
            self.image_gallery.update_image_status(image_path, len(tags) > 0)

        # Update tag filter counts ONCE (expensive operation)
        self._update_tag_filter()

        # If current image was in the batch, refresh its display
        if self.current_image:
            for image_path, tags in saved_results:
                if self.current_image == image_path:
                    self.refresh_current_image()
                    break
