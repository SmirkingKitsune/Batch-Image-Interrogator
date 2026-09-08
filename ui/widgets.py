"""Custom PyQt6 widgets for image interrogator UI."""

from PyQt6.QtWidgets import (QListWidget, QListWidgetItem, QLabel,
                             QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QMenu, QFrame,
                             QInputDialog, QScrollArea, QSizePolicy, QStyle)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QPixmap, QIcon, QPalette
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from interrogators import LlamaCppInterrogator
from ui.workers import decode_thumbnail


class ImageGalleryWidget(QListWidget):
    """Custom image gallery widget with thumbnail display."""

    THUMBNAIL_SIZE = QSize(200, 200)
    _placeholder = None  # Lazily built; QPixmap needs a live QApplication.

    image_selected = pyqtSignal(str)  # Emits image path when selected (single selection)
    multi_selection_changed = pyqtSignal(list)  # Emits list of image paths (multi selection)
    inspection_requested = pyqtSignal(str)  # Emits image path for advanced inspection
    multi_inspection_requested = pyqtSignal(list)  # Emits list of paths for multi-image inspection

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(self.THUMBNAIL_SIZE)
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
        """Add an image to the gallery, decoding its thumbnail inline."""
        image = decode_thumbnail(image_path, self.THUMBNAIL_SIZE)
        if image is None:
            return

        item = self._create_item(image_path, has_tags)
        item.setIcon(QIcon(QPixmap.fromImage(image)))
        self.addItem(item)
        self.image_items[image_path] = item

    def add_image_placeholder(self, image_path: str, has_tags: bool = False):
        """Add an image with a blank thumbnail, to be filled in later.

        Lets the gallery become scrollable immediately while a worker decodes
        thumbnails; see set_thumbnail.
        """
        item = self._create_item(image_path, has_tags)
        item.setIcon(self._placeholder_icon())
        self.addItem(item)
        self.image_items[image_path] = item

    def set_thumbnail(self, image_path: str, image):
        """Attach a decoded thumbnail to an already-added image.

        Silently ignores paths that are no longer in the gallery, so results
        arriving from a superseded load are harmless.
        """
        item = self.image_items.get(image_path)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(image)))

    def remove_image(self, image_path: str):
        """Remove an image from the gallery."""
        item = self.image_items.pop(image_path, None)
        if item is not None:
            self.takeItem(self.row(item))

    def _create_item(self, image_path: str, has_tags: bool) -> QListWidgetItem:
        """Build a gallery item without its thumbnail."""
        item = QListWidgetItem(Path(image_path).name)
        item.setData(Qt.ItemDataRole.UserRole, image_path)

        # Visual indicator for tagged images
        if has_tags:
            item.setBackground(Qt.GlobalColor.lightGray)
        return item

    @classmethod
    def _placeholder_icon(cls) -> QIcon:
        """Transparent icon that reserves the thumbnail's space."""
        if cls._placeholder is None:
            pixmap = QPixmap(cls.THUMBNAIL_SIZE)
            pixmap.fill(Qt.GlobalColor.transparent)
            cls._placeholder = QIcon(pixmap)
        return cls._placeholder

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


class TranscriptSpinner(QLabel):
    """Text spinner shown while a transcript turn waits on the model."""

    FRAMES = ("|", "/", "-", "\\")
    INTERVAL_MS = 120

    def __init__(self, parent=None):
        super().__init__(parent)
        self._index = 0
        self._timer = QTimer(self)
        self._timer.setInterval(self.INTERVAL_MS)
        self._timer.timeout.connect(self._advance)
        self._render()

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        self.start()

    def hideEvent(self, event):
        # A card scrolled out of view still repaints on every tick otherwise.
        self.stop()
        super().hideEvent(event)

    def _advance(self):
        self._index = (self._index + 1) % len(self.FRAMES)
        self._render()

    def _render(self):
        self.setText(f"[{self.FRAMES[self._index]}]")


class TranscriptTurnHandle:
    """Live reference to a transcript row that is still awaiting a response.

    The row is a plain QListWidgetItem, so it is destroyed whenever the
    transcript is cleared (image switch, mode switch, context reset). Every
    method therefore tolerates a row that no longer exists.
    """

    def __init__(
        self,
        transcript: "InquiryTranscriptWidget",
        item: QListWidgetItem,
        turn: Dict[str, Any],
        refs: Dict[str, Any],
        image_path: Optional[str],
    ):
        self._transcript = transcript
        self._item = item
        self._turn = turn
        self._refs = refs
        self._image_path = image_path
        self._finished = False

    @property
    def is_active(self) -> bool:
        """True while the row still exists and has not been finalized."""
        return not self._finished and self._live_item() is not None

    def set_status(self, text: str) -> None:
        """Replace the spinner caption, e.g. while a retry is in flight."""
        item = self._live_item()
        if item is None or self._finished:
            return
        status_label = self._refs.get("status_label")
        if status_label is not None:
            status_label.setText(text)
        self._resync(item)

    def set_stream_text(self, text: str) -> None:
        """Show partial response text streamed back from llama-server."""
        item = self._live_item()
        if item is None or self._finished:
            return
        stream_label = self._refs.get("stream_label")
        if stream_label is None:
            return

        clean = (text or "").strip()
        stream_label.setText(clean)
        stream_label.setVisible(bool(clean))
        self._resync(item)

    def complete(self, turn: Dict[str, Any], image_path: Optional[str] = None) -> None:
        """Swap the pending row for the finished prompt/response card."""
        item = self._live_item()
        self._finished = True
        if item is None:
            return
        self._transcript.replace_turn_card(item, turn, image_path or self._image_path)

    def fail(self, message: str) -> None:
        """Swap the pending row for an error card that keeps the request visible."""
        item = self._live_item()
        self._finished = True
        if item is None:
            return
        failed_turn = dict(self._turn)
        failed_turn["tags"] = []
        failed_turn["error"] = message or "Inquiry failed."
        self._transcript.replace_turn_card(item, failed_turn, self._image_path)

    def _live_item(self) -> Optional[QListWidgetItem]:
        try:
            if self._transcript.row(self._item) < 0:
                return None
        except RuntimeError:
            # The underlying C++ item was deleted by clear().
            return None
        return self._item

    def _resync(self, item: QListWidgetItem) -> None:
        widget = self._transcript.itemWidget(item)
        if widget is not None:
            self._transcript.sync_item_size(item, widget)
            self._transcript.follow_tail()


class InquiryTranscriptWidget(QListWidget):
    """Word-wrapped transcript list rendered as prompt/response cards."""

    # Tag chips scroll horizontally inside this band. Keeping the band a fixed
    # height is what makes the row size hints reliable: a plain tag row grows
    # the card's natural width without bound (thousands of pixels for a large
    # tag set), and Qt then computes the card's sizeHint at that width, where
    # every word-wrapped label reports a single line. Rows ended up ~200px tall
    # and everything below the prompt was clipped.
    TAG_ROW_HEIGHT = 30
    TAIL_FOLLOW_SLACK_PX = 48

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

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def append_turn_card(self, turn: Dict[str, Any], image_path: Optional[str] = None) -> QListWidgetItem:
        """Render one completed transcript turn using a card layout."""
        item = QListWidgetItem()
        self.addItem(item)
        self.replace_turn_card(item, turn, image_path)
        return item

    def begin_turn_card(
        self,
        turn: Dict[str, Any],
        image_path: Optional[str] = None,
    ) -> TranscriptTurnHandle:
        """Render the request half of a turn immediately, before the model answers.

        The prompt and image are already known when the request is sent, so
        they are shown right away and a spinner stands in for the response
        until the first streamed text arrives.
        """
        item = QListWidgetItem()
        self.addItem(item)
        card, refs = self._build_pending_card(turn, image_path)
        self._install_card(item, card)
        return TranscriptTurnHandle(self, item, dict(turn), refs, image_path)

    def replace_turn_card(
        self,
        item: QListWidgetItem,
        turn: Dict[str, Any],
        image_path: Optional[str] = None,
    ) -> None:
        """Render a finished card into an existing row, replacing its widget."""
        card = self._build_turn_card(turn, image_path)
        self._install_card(item, card)

    def _install_card(self, item: QListWidgetItem, card: QWidget) -> None:
        # setItemWidget deletes any widget already installed on the row.
        self.setItemWidget(item, card)
        self.sync_item_size(item, card)
        QTimer.singleShot(0, lambda: self._sync_row_later(item))

    def _sync_row_later(self, item: QListWidgetItem) -> None:
        """Re-measure a row once Qt has laid the card out for real."""
        try:
            if self.row(item) < 0:
                return
        except RuntimeError:
            # Row was cleared before the deferred pass ran.
            return
        widget = self.itemWidget(item)
        if widget is not None:
            self.sync_item_size(item, widget)

    def _build_turn_card(self, turn: Dict[str, Any], image_path: Optional[str]) -> QWidget:
        card, card_layout = self._build_card_shell()
        theme = self._card_theme()

        card_layout.addWidget(self._build_prompt_frame(turn, image_path, theme, card))
        card_layout.addWidget(self._build_image_label(turn, image_path))
        card_layout.addWidget(self._build_response_frame(turn, theme))
        card_layout.addWidget(self._build_tags_area(turn, theme, card))
        return card

    def _build_pending_card(
        self,
        turn: Dict[str, Any],
        image_path: Optional[str],
    ) -> Tuple[QWidget, Dict[str, Any]]:
        card, card_layout = self._build_card_shell()
        theme = self._card_theme()

        card_layout.addWidget(self._build_prompt_frame(turn, image_path, theme, card))
        card_layout.addWidget(self._build_image_label(turn, image_path))

        response_frame = QFrame()
        response_frame.setObjectName("transcriptResponseFrame")
        response_frame.setFrameShape(QFrame.Shape.StyledPanel)
        # Scoped by object name: QLabel derives from QFrame, so a bare "QFrame"
        # selector would draw a border around every label inside the card too.
        response_frame.setStyleSheet(
            f"QFrame#transcriptResponseFrame {{ border: 1px solid {theme['pending_border']}; "
            f"border-radius: 6px; background-color: {theme['pending_bg']}; }}"
        )
        response_layout = QVBoxLayout(response_frame)
        response_layout.setContentsMargins(8, 6, 8, 6)
        response_layout.setSpacing(4)

        spinner_row = QHBoxLayout()
        spinner_row.setSpacing(6)
        spinner = TranscriptSpinner()
        spinner.setStyleSheet(f"color: {theme['text']}; font-family: monospace;")
        status_label = QLabel("Waiting for model response...")
        status_label.setWordWrap(True)
        status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        status_label.setStyleSheet(f"color: {theme['text']};")
        spinner_row.addWidget(spinner)
        spinner_row.addWidget(status_label, 1)
        response_layout.addLayout(spinner_row)

        stream_label = QLabel("")
        stream_label.setWordWrap(True)
        stream_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        stream_label.setStyleSheet(f"color: {theme['text']};")
        stream_label.setVisible(False)
        response_layout.addWidget(stream_label)

        model_label = QLabel(f"[{self._turn_model_name(turn)}]")
        model_label.setWordWrap(True)
        model_label.setStyleSheet(f"color: {theme['text']};")
        model_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        response_layout.addWidget(model_label)

        card_layout.addWidget(response_frame)
        spinner.start()

        refs = {
            "spinner": spinner,
            "status_label": status_label,
            "stream_label": stream_label,
        }
        return card, refs

    @staticmethod
    def _build_card_shell() -> Tuple[QWidget, QVBoxLayout]:
        card = QWidget()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)
        return card, card_layout

    def _card_theme(self) -> Dict[str, str]:
        palette = self.palette()
        return {
            "text": "#111111",
            "neutral_text": palette.color(QPalette.ColorRole.Text).name(),
            "prompt_border": "#5A8FD8",
            "prompt_bg": "#DCEBFF",
            "normal_border": "#4D9B63",
            "normal_bg": "#DEF6E3",
            "unusual_border": "#C85D5D",
            "unusual_bg": "#FCE1E1",
            "pending_border": "#9A8FD8",
            "pending_bg": "#ECE8FB",
            "chip_border": palette.color(QPalette.ColorRole.Mid).name(),
            "chip_bg": palette.color(QPalette.ColorRole.Button).name(),
        }

    def _build_prompt_frame(
        self,
        turn: Dict[str, Any],
        image_path: Optional[str],
        theme: Dict[str, str],
        card: QWidget,
    ) -> QFrame:
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
        prompt_frame.setObjectName("transcriptPromptFrame")
        prompt_frame.setFrameShape(QFrame.Shape.StyledPanel)
        prompt_frame.setStyleSheet(
            f"QFrame#transcriptPromptFrame {{ border: 1px solid {theme['prompt_border']}; "
            f"border-radius: 6px; background-color: {theme['prompt_bg']}; }}"
        )
        prompt_layout = QVBoxLayout(prompt_frame)
        prompt_layout.setContentsMargins(8, 6, 8, 6)
        prompt_label = QLabel(prompt_text)
        prompt_label.setWordWrap(True)
        prompt_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        prompt_label.setStyleSheet(f"color: {theme['text']};")
        prompt_layout.addWidget(prompt_label)

        details_button = QPushButton("Show Prompt Details")
        details_button.setCheckable(True)
        details_button.setStyleSheet(f"color: {theme['text']};")
        prompt_layout.addWidget(details_button)

        details_view = QTextEdit()
        details_view.setReadOnly(True)
        details_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        details_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        details_view.setPlainText(full_prompt_text)
        details_view.setMaximumHeight(220)
        details_view.setVisible(False)
        prompt_layout.addWidget(details_view)

        def toggle_prompt_details(visible: bool):
            details_view.setVisible(visible)
            details_button.setText("Hide Prompt Details" if visible else "Show Prompt Details")
            self._resync_card(card)

        details_button.toggled.connect(toggle_prompt_details)

        turn_image_path = image_path or turn.get("image_path")
        if turn_image_path:
            path_label = QLabel(f"[{self._to_display_name(turn_image_path)}]")
            path_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            path_label.setWordWrap(True)
            path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            path_label.setStyleSheet(f"color: {theme['text']};")
            prompt_layout.addWidget(path_label)

        return prompt_frame

    def _build_image_label(self, turn: Dict[str, Any], image_path: Optional[str]) -> QLabel:
        turn_image_path = image_path or turn.get("image_path")
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
        return image_label

    def _build_response_frame(self, turn: Dict[str, Any], theme: Dict[str, str]) -> QFrame:
        error_text = (turn.get("error") or "").strip()
        response_json = turn.get("response_json", {}) or {}
        if not isinstance(response_json, dict):
            response_json = {}

        comment = (
            response_json.get("comment")
            or response_json.get("answer")
            or response_json.get("reasoning_summary")
            or ""
        )
        warnings = response_json.get("warnings", [])
        parse_mode = response_json.get("_parse_mode", "")
        unusual = bool(
            error_text
            or "model_returned_non_json_response" in warnings
            or parse_mode == "non_json_fallback"
        )
        raw_text = (
            response_json.get("_debug_raw_response")
            or response_json.get("comment")
            or response_json.get("answer")
            or ""
        )

        if error_text:
            display_text = f"[Error]\n{error_text}"
        elif unusual:
            raw_payload = raw_text.strip()
            display_text = "[Raw]" if not raw_payload else f"[Raw]\n{raw_payload}"
        else:
            display_text = (comment or "").strip() or "[no comment]"

        response_frame = QFrame()
        response_frame.setObjectName("transcriptResponseFrame")
        response_frame.setFrameShape(QFrame.Shape.StyledPanel)
        response_border_hex = theme["unusual_border"] if unusual else theme["normal_border"]
        response_bg_hex = theme["unusual_bg"] if unusual else theme["normal_bg"]
        response_frame.setStyleSheet(
            f"QFrame#transcriptResponseFrame {{ border: 1px solid {response_border_hex}; "
            f"border-radius: 6px; background-color: {response_bg_hex}; }}"
        )
        response_layout = QVBoxLayout(response_frame)
        response_layout.setContentsMargins(8, 6, 8, 6)
        comment_label = QLabel(display_text)
        comment_label.setWordWrap(True)
        comment_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        comment_label.setStyleSheet(f"color: {theme['text']};")
        model_label = QLabel(f"[{self._turn_model_name(turn)}]")
        model_label.setWordWrap(True)
        model_label.setStyleSheet(f"color: {theme['text']};")
        model_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        response_layout.addWidget(comment_label)
        response_layout.addWidget(model_label)
        return response_frame

    def _build_tags_area(
        self,
        turn: Dict[str, Any],
        theme: Dict[str, str],
        card: QWidget,
    ) -> QScrollArea:
        """Build the horizontally scrolling chip band for a turn's tags."""
        chip_style = (
            f"QPushButton {{ border: 1px solid {theme['chip_border']}; border-radius: 10px; "
            f"padding: 2px 8px; background: {theme['chip_bg']}; color: {theme['neutral_text']}; }}"
        )

        tags_area = QScrollArea()
        tags_area.setWidgetResizable(True)
        tags_area.setFrameShape(QFrame.Shape.NoFrame)
        tags_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tags_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Ignored width keeps the chip count from driving the card's width, and
        # a fixed height keeps the row measurable no matter how many chips fit.
        tags_area.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        scrollbar_extent = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        tags_area.setFixedHeight(self.TAG_ROW_HEIGHT + scrollbar_extent)

        tags_container = QWidget()
        tags_row = QHBoxLayout(tags_container)
        tags_row.setContentsMargins(0, 0, 0, 0)
        tags_row.setSpacing(4)
        tags_area.setWidget(tags_container)

        tags = list(turn.get("tags") or [])

        def refresh_tags_row():
            while tags_row.count():
                child = tags_row.takeAt(0)
                widget_obj = child.widget()
                if widget_obj:
                    widget_obj.deleteLater()

            for tag in tags:
                tag_button = QPushButton(f"[{tag}]")
                tag_button.setStyleSheet(chip_style)
                tag_button.setFlat(False)
                tag_button.setToolTip(f"Remove tag: {tag}")

                def remove_tag(_: bool = False, tag_value: str = tag):
                    if tag_value in tags:
                        tags.remove(tag_value)
                        turn["tags"] = list(tags)
                        refresh_tags_row()

                tag_button.clicked.connect(remove_tag)
                tags_row.addWidget(tag_button)

            add_button = QPushButton("[+]")
            add_button.setStyleSheet(chip_style)
            add_button.setToolTip("Add a tag to this turn")

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

            add_button.clicked.connect(add_tag)
            tags_row.addWidget(add_button)
            tags_row.addStretch()

        refresh_tags_row()
        return tags_area

    def _turn_model_name(self, turn: Dict[str, Any]) -> str:
        return turn.get("model_name") or self._current_model_name() or "LlamaCpp"

    # ------------------------------------------------------------------
    # Sizing and scrolling
    # ------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_all_item_sizes()

    def follow_tail(self) -> None:
        """Scroll to the newest row, but only when already parked at the end.

        Streaming updates the last row continuously; yanking the view back down
        while the user is reading an earlier turn would make it unusable.
        """
        scrollbar = self.verticalScrollBar()
        if scrollbar.value() >= scrollbar.maximum() - self.TAIL_FOLLOW_SLACK_PX:
            self.scrollToBottom()

    def _sync_all_item_sizes(self) -> None:
        for row in range(self.count()):
            item = self.item(row)
            widget = self.itemWidget(item)
            if widget:
                self.sync_item_size(item, widget)

    def _resync_card(self, card: QWidget) -> None:
        """Re-measure the row that owns `card` after its contents changed."""
        for row in range(self.count()):
            item = self.item(row)
            if self.itemWidget(item) is card:
                self.sync_item_size(item, card)
                return

    def sync_item_size(self, item: QListWidgetItem, widget: QWidget) -> None:
        """Size a row to the card's real height at the current viewport width."""
        width = max(120, self.viewport().width() - 4)
        widget.setFixedWidth(width)
        widget.updateGeometry()
        # sizeHint() measures the card at its *natural* width, which is never
        # the width the row actually gets. For word-wrapped content that is
        # wrong in both directions, so heightForWidth is the measurement to
        # trust and sizeHint is only a fallback for cards without one.
        height = widget.heightForWidth(width)
        if height <= 0:
            widget.adjustSize()
            height = widget.sizeHint().height()
        item.setSizeHint(QSize(width, height))

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
