"""Custom PyQt6 widgets for image interrogator UI."""

from PyQt6.QtWidgets import (QListWidget, QListWidgetItem, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QMenu, QFrame,
                             QInputDialog, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap, QIcon, QPalette
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from interrogators import LlamaCppInterrogator


class ImageGalleryWidget(QListWidget):
    """Custom image gallery widget with thumbnail display."""

    image_selected = pyqtSignal(str)  # Emits image path when selected (single selection)
    multi_selection_changed = pyqtSignal(list)  # Emits list of image paths (multi selection)
    inspection_requested = pyqtSignal(str)  # Emits image path for advanced inspection
    multi_inspection_requested = pyqtSignal(list)  # Emits list of paths for multi-image inspection

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
        self._multi_select_mode = False

    def set_selection_mode(self, multi: bool):
        """Toggle between single and multi-selection mode."""
        self._multi_select_mode = multi
        if multi:
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            # Clear selection when switching back to single mode
            self.clearSelection()
    
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
        if not items:
            return

        if len(items) == 1:
            # Single selection - emit image_selected for backward compatibility
            image_path = items[0].data(Qt.ItemDataRole.UserRole)
            self.image_selected.emit(image_path)
        else:
            # Multiple selection - emit list of paths
            image_paths = [item.data(Qt.ItemDataRole.UserRole) for item in items]
            self.multi_selection_changed.emit(image_paths)

    def get_selected_paths(self) -> List[str]:
        """Get list of currently selected image paths."""
        items = self.selectedItems()
        return [item.data(Qt.ItemDataRole.UserRole) for item in items if item.data(Qt.ItemDataRole.UserRole)]

    def _show_context_menu(self, position):
        """Show context menu for gallery items."""
        item = self.itemAt(position)
        if not item:
            return

        image_path = item.data(Qt.ItemDataRole.UserRole)
        if not image_path:
            return

        # Get all selected items
        selected_items = self.selectedItems()
        selected_paths = [i.data(Qt.ItemDataRole.UserRole) for i in selected_items if i.data(Qt.ItemDataRole.UserRole)]

        menu = QMenu(self)

        # Multi-selection context menu
        if len(selected_paths) > 1:
            # Multi-image inspection action
            multi_inspect_action = menu.addAction(f"Edit Selected Tags ({len(selected_paths)} images)...")
            menu.addSeparator()

            # Open folder action (opens folder of clicked item)
            open_folder_action = menu.addAction("Open Folder")

            # Show menu and handle action
            action = menu.exec(self.mapToGlobal(position))

            if action == multi_inspect_action:
                self.multi_inspection_requested.emit(selected_paths)
            elif action == open_folder_action:
                self._open_folder(image_path)
        else:
            # Single selection context menu
            inspect_action = menu.addAction("Advanced Inspection...")
            open_folder_action = menu.addAction("Open Folder")

            # Show menu and handle action
            action = menu.exec(self.mapToGlobal(position))

            if action == inspect_action:
                self.inspection_requested.emit(image_path)
            elif action == open_folder_action:
                self._open_folder(image_path)

    def _open_folder(self, image_path: str):
        """Open the folder containing the image in the system file explorer."""
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


class InquiryTranscriptWidget(QListWidget):
    """Word-wrapped transcript list rendered as prompt/response cards."""

    def __init__(
        self,
        parent=None,
        display_name_func: Optional[Callable[[str], str]] = None,
        model_name_func: Optional[Callable[[], Optional[str]]] = None,
    ):
        super().__init__(parent)
        self.display_name_func = display_name_func
        self.model_name_func = model_name_func

        self.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSpacing(8)
        self.setWordWrap(True)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(520)
        self.verticalScrollBar().setSingleStep(18)

    def append_turn_card(self, turn: Dict[str, Any], image_path: Optional[str] = None) -> None:
        """Render one transcript turn using a card layout."""
        palette = self.palette()
        text_hex = "#111111"
        neutral_text_hex = palette.color(QPalette.ColorRole.Text).name()
        prompt_border_hex = "#5A8FD8"
        prompt_bg_hex = "#DCEBFF"
        normal_border_hex = "#4D9B63"
        normal_bg_hex = "#DEF6E3"
        unusual_border_hex = "#C85D5D"
        unusual_bg_hex = "#FCE1E1"
        chip_border_hex = palette.color(QPalette.ColorRole.Mid).name()
        chip_bg_hex = palette.color(QPalette.ColorRole.Button).name()

        card = QWidget()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        prompt_type = turn.get("prompt_type") or "describe"
        user_prompt_text = turn.get("prompt_text") or ""
        included_tables = turn.get("included_tables") or []
        included_transcripts = turn.get("included_transcripts") or []
        sidecar_tags = turn.get("sidecar_tags") or []
        prompt_text = LlamaCppInterrogator.build_prompt_display_summary(
            prompt_type,
            user_prompt_text,
            included_tables,
            included_transcripts=included_transcripts,
            sidecar_tags=sidecar_tags,
        )
        full_prompt_text = LlamaCppInterrogator.build_user_prompt_from_turn(
            {
                "prompt_type": prompt_type,
                "prompt_text": user_prompt_text,
                "included_tables": included_tables,
                "included_transcripts": included_transcripts,
                "sidecar_tags": sidecar_tags,
            }
        )
        prompt_frame = QFrame()
        prompt_frame.setFrameShape(QFrame.Shape.StyledPanel)
        prompt_frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {prompt_border_hex}; border-radius: 6px; background-color: {prompt_bg_hex}; }}"
        )
        prompt_layout = QVBoxLayout(prompt_frame)
        prompt_layout.setContentsMargins(8, 6, 8, 6)
        prompt_label = QLabel(prompt_text)
        prompt_label.setWordWrap(True)
        prompt_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        prompt_label.setStyleSheet(f"color: {text_hex};")
        prompt_layout.addWidget(prompt_label)

        details_button = QPushButton("Show Prompt Details")
        details_button.setCheckable(True)
        details_button.setStyleSheet(f"color: {text_hex};")
        prompt_layout.addWidget(details_button)

        details_view = QTextEdit()
        details_view.setReadOnly(True)
        details_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        details_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        details_view.setPlainText(full_prompt_text)
        details_view.setMaximumHeight(220)
        details_view.setVisible(False)
        prompt_layout.addWidget(details_view)

        item = QListWidgetItem()

        def toggle_prompt_details(visible: bool):
            details_view.setVisible(visible)
            details_button.setText("Hide Prompt Details" if visible else "Show Prompt Details")
            self._sync_item_size(item, card)

        details_button.toggled.connect(toggle_prompt_details)

        turn_image_path = image_path or turn.get("image_path")
        if turn_image_path:
            path_label = QLabel(f"[{self._to_display_name(turn_image_path)}]")
            path_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            path_label.setWordWrap(True)
            path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            path_label.setStyleSheet(f"color: {text_hex};")
            prompt_layout.addWidget(path_label)
        card_layout.addWidget(prompt_frame)

        image_label = QLabel("[image]")
        image_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if turn_image_path and Path(turn_image_path).exists():
            pixmap = QPixmap(turn_image_path)
            if not pixmap.isNull():
                thumb = pixmap.scaled(
                    220,
                    160,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                image_label.setPixmap(thumb)
        card_layout.addWidget(image_label)

        response_json = turn.get("response_json", {}) or {}
        comment = ""
        if isinstance(response_json, dict):
            comment = (
                response_json.get("comment")
                or response_json.get("answer")
                or response_json.get("reasoning_summary")
                or ""
            )
        warnings = response_json.get("warnings", []) if isinstance(response_json, dict) else []
        parse_mode = response_json.get("_parse_mode", "") if isinstance(response_json, dict) else ""
        unusual = bool(
            "model_returned_non_json_response" in warnings
            or parse_mode == "non_json_fallback"
        )
        raw_text = ""
        if isinstance(response_json, dict):
            raw_text = (
                response_json.get("_debug_raw_response")
                or response_json.get("comment")
                or response_json.get("answer")
                or ""
            )
        if unusual:
            raw_payload = raw_text.strip()
            display_text = "[Raw]" if not raw_payload else f"[Raw]\n{raw_payload}"
        else:
            display_text = (comment or "").strip() or "[no comment]"
        model_name = turn.get("model_name") or self._current_model_name() or "LlamaCpp"

        response_frame = QFrame()
        response_frame.setFrameShape(QFrame.Shape.StyledPanel)
        response_border_hex = unusual_border_hex if unusual else normal_border_hex
        response_bg_hex = unusual_bg_hex if unusual else normal_bg_hex
        response_frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {response_border_hex}; border-radius: 6px; background-color: {response_bg_hex}; }}"
        )
        response_layout = QVBoxLayout(response_frame)
        response_layout.setContentsMargins(8, 6, 8, 6)
        comment_label = QLabel(display_text)
        comment_label.setWordWrap(True)
        comment_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        comment_label.setStyleSheet(f"color: {text_hex};")
        model_label = QLabel(f"[{model_name}]")
        model_label.setWordWrap(True)
        model_label.setStyleSheet(f"color: {text_hex};")
        model_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        response_layout.addWidget(comment_label)
        response_layout.addWidget(model_label)
        card_layout.addWidget(response_frame)

        tags_row = QHBoxLayout()
        tags_row.setSpacing(4)
        tags = list(turn.get("tags") or [])

        def refresh_tags_row():
            while tags_row.count():
                child = tags_row.takeAt(0)
                widget_obj = child.widget()
                if widget_obj:
                    widget_obj.deleteLater()

            for tag in tags:
                tag_button = QPushButton(f"[{tag}]")
                tag_button.setStyleSheet(
                    f"QPushButton {{ border: 1px solid {chip_border_hex}; border-radius: 10px; padding: 2px 8px; background: {chip_bg_hex}; color: {neutral_text_hex}; }}"
                )
                tag_button.setFlat(False)

                def remove_tag(_: bool = False, tag_value: str = tag):
                    if tag_value in tags:
                        tags.remove(tag_value)
                        turn["tags"] = list(tags)
                        refresh_tags_row()
                        self._sync_item_size(item, card)

                tag_button.clicked.connect(remove_tag)
                tags_row.addWidget(tag_button)

            add_button = QPushButton("[+]")
            add_button.setStyleSheet(
                f"QPushButton {{ border: 1px solid {chip_border_hex}; border-radius: 10px; padding: 2px 8px; background: {chip_bg_hex}; color: {neutral_text_hex}; }}"
            )

            def add_tag(_: bool = False):
                new_tag, ok = QInputDialog.getText(self, "Add Tag", "New tag:")
                if not ok:
                    return
                clean = new_tag.strip()
                if not clean:
                    return
                if clean not in tags:
                    tags.append(clean)
                    turn["tags"] = list(tags)
                    refresh_tags_row()
                    self._sync_item_size(item, card)

            add_button.clicked.connect(add_tag)
            tags_row.addWidget(add_button)
            tags_row.addStretch()

        refresh_tags_row()
        card_layout.addLayout(tags_row)

        self.addItem(item)
        self.setItemWidget(item, card)
        self._sync_item_size(item, card)
        QTimer.singleShot(0, lambda: self._sync_item_size(item, card))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_all_item_sizes()

    def _sync_all_item_sizes(self) -> None:
        for row in range(self.count()):
            item = self.item(row)
            widget = self.itemWidget(item)
            if widget:
                self._sync_item_size(item, widget)

    def _sync_item_size(self, item: QListWidgetItem, widget: QWidget) -> None:
        width = max(120, self.viewport().width() - 4)
        widget.setFixedWidth(width)
        widget.updateGeometry()
        widget.adjustSize()
        item.setSizeHint(QSize(width, widget.sizeHint().height()))

    def _to_display_name(self, image_path: str) -> str:
        if self.display_name_func:
            return self.display_name_func(image_path)
        return Path(image_path).name

    def _current_model_name(self) -> Optional[str]:
        if self.model_name_func:
            return self.model_name_func()
        return None


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
