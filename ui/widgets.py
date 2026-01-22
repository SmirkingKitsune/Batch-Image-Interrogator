"""Custom PyQt6 widgets for image interrogator UI."""

from PyQt6.QtWidgets import (QListWidget, QListWidgetItem, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QMenu)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon
from pathlib import Path
from typing import Dict, List, Optional


class ImageGalleryWidget(QListWidget):
    """Custom image gallery widget with thumbnail display."""

    image_selected = pyqtSignal(str)  # Emits image path when selected
    inspection_requested = pyqtSignal(str)  # Emits image path for advanced inspection

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(200, 200))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setSpacing(10)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Connect selection signal
        self.itemSelectionChanged.connect(self._on_selection_changed)

        self.image_items = {}  # path -> QListWidgetItem
    
    def add_image(self, image_path: str, has_tags: bool = False):
        """Add an image to the gallery."""
        path = Path(image_path)
        
        # Create thumbnail
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        
        # Scale to fit (use FastTransformation for bulk loading performance)
        scaled_pixmap = pixmap.scaled(
            200, 200,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        
        # Create item
        item = QListWidgetItem(QIcon(scaled_pixmap), path.name)
        item.setData(Qt.ItemDataRole.UserRole, image_path)
        
        # Visual indicator for tagged images
        if has_tags:
            item.setBackground(Qt.GlobalColor.lightGray)
        
        self.addItem(item)
        self.image_items[image_path] = item
    
    def update_image_status(self, image_path: str, has_tags: bool):
        """Update visual status of an image."""
        if image_path in self.image_items:
            item = self.image_items[image_path]
            if has_tags:
                item.setBackground(Qt.GlobalColor.lightGray)
            else:
                item.setBackground(Qt.GlobalColor.white)
    
    def clear_gallery(self):
        """Clear all images from gallery."""
        self.clear()
        self.image_items.clear()
    
    def _on_selection_changed(self):
        """Handle selection change."""
        items = self.selectedItems()
        if items:
            image_path = items[0].data(Qt.ItemDataRole.UserRole)
            self.image_selected.emit(image_path)

    def _show_context_menu(self, position):
        """Show context menu for gallery items."""
        item = self.itemAt(position)
        if not item:
            return

        image_path = item.data(Qt.ItemDataRole.UserRole)
        if not image_path:
            return

        menu = QMenu(self)

        # Advanced inspection action
        inspect_action = menu.addAction("Advanced Inspection...")

        # Open folder action
        open_folder_action = menu.addAction("Open Folder")

        # Show menu and handle action
        action = menu.exec(self.mapToGlobal(position))

        if action == inspect_action:
            self.inspection_requested.emit(image_path)
        elif action == open_folder_action:
            # Open folder in file explorer
            import subprocess
            import sys
            folder_path = str(Path(image_path).parent)
            if sys.platform == 'win32':
                subprocess.run(['explorer', folder_path])
            elif sys.platform == 'darwin':
                subprocess.run(['open', folder_path])
            else:
                subprocess.run(['xdg-open', folder_path])


class TagEditorWidget(QWidget):
    """Widget for viewing and editing tags."""
    
    tags_changed = pyqtSignal(list)  # Emits new tag list
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        
        # Tag display/edit area
        self.tag_edit = QTextEdit()
        self.tag_edit.setPlaceholderText("Tags (comma-separated)")
        self.tag_edit.setMaximumHeight(150)
        layout.addWidget(QLabel("Tags:"))
        layout.addWidget(self.tag_edit)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Tags")
        self.save_button.clicked.connect(self._on_save_clicked)
        button_layout.addWidget(self.save_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
    
    def set_tags(self, tags: List[str]):
        """Set the displayed tags."""
        self.tag_edit.setPlainText(', '.join(tags))
    
    def get_tags(self) -> List[str]:
        """Get current tags as list."""
        text = self.tag_edit.toPlainText().strip()
        if not text:
            return []
        return [tag.strip() for tag in text.split(',') if tag.strip()]
    
    def clear_tags(self):
        """Clear all tags."""
        self.tag_edit.clear()
    
    def _on_save_clicked(self):
        """Handle save button click."""
        tags = self.get_tags()
        self.tags_changed.emit(tags)
    
    def _on_clear_clicked(self):
        """Handle clear button click."""
        self.tag_edit.clear()


class ResultsTableWidget(QTableWidget):
    """Table widget for displaying interrogation results with confidence scores."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Tag", "Confidence", "Model"])
        
        # Configure table
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(True)
    
    def set_results(self, results: Dict):
        """
        Display interrogation results.
        
        Args:
            results: Dict with 'tags', 'confidence_scores', 'model_name'
        """
        self.setRowCount(0)
        
        tags = results.get('tags', [])
        confidence_scores = results.get('confidence_scores', {})
        model_name = results.get('model_name', 'Unknown')
        
        for tag in tags:
            row = self.rowCount()
            self.insertRow(row)
            
            # Tag
            tag_item = QTableWidgetItem(tag)
            self.setItem(row, 0, tag_item)
            
            # Confidence (if available)
            if confidence_scores and tag in confidence_scores:
                conf_value = confidence_scores[tag]
                conf_item = QTableWidgetItem(f"{conf_value:.4f}")
                conf_item.setData(Qt.ItemDataRole.UserRole, conf_value)
            else:
                conf_item = QTableWidgetItem("N/A")
                conf_item.setData(Qt.ItemDataRole.UserRole, -1)
            
            self.setItem(row, 1, conf_item)
            
            # Model
            model_item = QTableWidgetItem(model_name)
            self.setItem(row, 2, model_item)
    
    def clear_results(self):
        """Clear all results."""
        self.setRowCount(0)
    
    def get_all_tags(self) -> List[str]:
        """Get all tags currently displayed."""
        tags = []
        for row in range(self.rowCount()):
            tag_item = self.item(row, 0)
            if tag_item:
                tags.append(tag_item.text())
        return tags
