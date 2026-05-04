"""Inquiry Tab - dedicated llama.cpp multimodal workflows."""

import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPalette, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import (
    InquirySettings,
    InterrogationDatabase,
    TagFilterSettings,
    get_image_metadata,
    hash_image_content,
)
from core.device_detector import get_device_detector
from core.llama_cpp_runtime import is_llama_timeout_error
from interrogators import LlamaCppInterrogator
from ui.dialogs import create_llama_config_widget
from ui.dialogs_advanced import AdvancedImageInspectionDialog
from ui.workers import MultimodalInterrogationWorker


class InquiryTab(QWidget):
    """Dedicated tab for llama.cpp multimodal image inquiry workflows."""

    DGX_TIMEOUT_HINT = (
        "Hint: On NVIDIA ARM64 systems (for example DGX Spark), prebuilt llama.cpp "
        "binaries can be unstable for multimodal inference. Build llama.cpp from source "
        "with CUDA support and set Inquiry -> llama-server Path to that compiled binary."
    )

    model_loaded = pyqtSignal(str)
    model_unloaded = pyqtSignal()
    inquiry_started = pyqtSignal()
    inquiry_finished = pyqtSignal()

    def __init__(
        self,
        database: InterrogationDatabase,
        llama_config: Dict[str, Any],
        tag_filters: TagFilterSettings,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.database = database
        self.llama_config = llama_config
        self.tag_filters = tag_filters
        self.inquiry_settings = InquirySettings()
        self.llama_config.update(self.inquiry_settings.get_llama_config())

        self.current_interrogator: Optional[LlamaCppInterrogator] = None
        self.interrogation_worker: Optional[MultimodalInterrogationWorker] = None
        self.loaded_image_paths: List[str] = []
        self.current_directory: Optional[Path] = None
        self.current_recursive = False
        self.current_image_path: Optional[str] = None
        self.current_image_hash: Optional[str] = None
        self.current_session_key: Optional[str] = None
        self.current_interrogations: List[Dict[str, Any]] = []
        self.all_discovered_tags: Dict[str, int] = {}
        self.batch_error_count = 0
        self.last_batch_error = ""
        self.batch_transcript_entries: List[Dict[str, Any]] = []
        self.active_batch_task = "describe"
        self.active_batch_prompt = ""

        self.llama_config_refs: Optional[Dict[str, Any]] = None

        self.setup_ui()
        self._apply_saved_inquiry_options()
        self.setup_connections()

    def setup_ui(self):
        """Create Inquiry tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()

        left_scroll = self._wrap_scroll_area(left_panel)
        left_scroll.setMinimumWidth(420)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    @staticmethod
    def _wrap_scroll_area(content: QWidget) -> QScrollArea:
        """Wrap a panel in a vertical scroll area to avoid clipped controls."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _create_left_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        source_group = QGroupBox("Shared Image Source")
        source_layout = QVBoxLayout()
        self.source_dir_label = QLabel("Directory: Not selected")
        self.source_dir_label.setWordWrap(True)
        source_layout.addWidget(self.source_dir_label)
        self.source_mode_label = QLabel("Mode: Non-recursive")
        source_layout.addWidget(self.source_mode_label)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        config_group = QGroupBox("LlamaCpp Configuration")
        config_layout = QVBoxLayout()
        config_widget, self.llama_config_refs = create_llama_config_widget(self.llama_config, self)
        config_layout.addWidget(config_widget)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        model_group = QGroupBox("Model Actions")
        model_layout = QVBoxLayout()
        self.load_model_button = QPushButton("Load Llama Model")
        self.unload_model_button = QPushButton("Unload Model")
        self.unload_model_button.setEnabled(False)
        self.model_status_label = QLabel("Model: Not loaded")
        self.model_status_label.setWordWrap(True)
        model_layout.addWidget(self.load_model_button)
        model_layout.addWidget(self.unload_model_button)
        model_layout.addWidget(self.model_status_label)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.addTab(self._create_single_controls_tab(), "Single Inquiry")
        self.mode_tabs.addTab(self._create_batch_controls_tab(), "Batch Inquiry")
        layout.addWidget(self.mode_tabs, 1)

        self.inquiry_status_label = QLabel("Configure and load llama model to begin inquiries.")
        self.inquiry_status_label.setWordWrap(True)
        self.inquiry_status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.inquiry_status_label)

        return widget

    def _create_single_controls_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        preview_group = QGroupBox("Current Image Preview")
        preview_layout = QVBoxLayout()
        self.image_preview = QLabel("No image selected")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumHeight(260)
        self.image_preview.setStyleSheet("QLabel { background-color: #f0f0f0; }")
        self.current_image_label = QLabel("")
        self.current_image_label.setWordWrap(True)
        preview_layout.addWidget(self.image_preview)
        preview_layout.addWidget(self.current_image_label)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        single_group = QGroupBox("Single-Image Inquiry")
        single_layout = QVBoxLayout()

        image_row = QHBoxLayout()
        image_row.addWidget(QLabel("Image:"))
        self.image_selector = QComboBox()
        self.image_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        image_row.addWidget(self.image_selector, 1)
        self.open_advanced_button = QPushButton("Open Advanced")
        image_row.addWidget(self.open_advanced_button)
        single_layout.addLayout(image_row)

        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("Task:"))
        self.single_task_combo = QComboBox()
        self.single_task_combo.addItems(["describe", "ocr", "vqa", "custom"])
        task_row.addWidget(self.single_task_combo)
        task_row.addStretch()
        single_layout.addLayout(task_row)

        single_layout.addWidget(QLabel("Prompt / Question:"))
        self.single_prompt_input = QLineEdit()
        self.single_prompt_input.setPlaceholderText("Ask a visual question or provide custom guidance...")
        single_layout.addWidget(self.single_prompt_input)

        prior_group = QGroupBox("Include Prior Interrogation Tables")
        prior_layout = QVBoxLayout()
        self.single_prior_tables = QTableWidget()
        self.single_prior_tables.setColumnCount(2)
        self.single_prior_tables.setHorizontalHeaderLabels(["Include", "Model"])
        self.single_prior_tables.horizontalHeader().setStretchLastSection(True)
        prior_layout.addWidget(self.single_prior_tables)
        prior_group.setLayout(prior_layout)
        single_layout.addWidget(prior_group)

        controls_row = QHBoxLayout()
        self.send_single_button = QPushButton("Send Inquiry")
        self.reset_context_button = QPushButton("Reset Image Context")
        controls_row.addWidget(self.send_single_button)
        controls_row.addWidget(self.reset_context_button)
        controls_row.addStretch()
        single_layout.addLayout(controls_row)

        single_group.setLayout(single_layout)
        layout.addWidget(single_group)
        layout.addStretch()
        return widget

    def _create_batch_controls_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        queue_group = QGroupBox("Shared Queue")
        queue_layout = QVBoxLayout()
        self.image_queue = QListWidget()
        self.image_queue.setMinimumHeight(180)
        queue_layout.addWidget(self.image_queue)
        queue_group.setLayout(queue_layout)
        layout.addWidget(queue_group)

        batch_group = QGroupBox("Batch Inquiry")
        batch_layout = QVBoxLayout()

        self.batch_task_combo = QComboBox()
        self.batch_task_combo.addItems(["describe", "ocr", "vqa", "custom"])
        batch_layout.addWidget(QLabel("Task:"))
        batch_layout.addWidget(self.batch_task_combo)

        self.batch_prompt_input = QTextEdit()
        self.batch_prompt_input.setPlaceholderText(
            "Optional prompt/question for all images in batch."
        )
        self.batch_prompt_input.setMaximumHeight(110)
        batch_layout.addWidget(QLabel("Prompt:"))
        batch_layout.addWidget(self.batch_prompt_input)

        self.txt_output_group = QButtonGroup(self)
        self.no_txt_radio = QRadioButton("No .txt file output")
        self.merge_txt_radio = QRadioButton("Write/merge .txt file output")
        self.overwrite_txt_radio = QRadioButton("Overwrite existing .txt files")
        self.merge_txt_radio.setChecked(True)
        self.txt_output_group.addButton(self.no_txt_radio, 0)
        self.txt_output_group.addButton(self.merge_txt_radio, 1)
        self.txt_output_group.addButton(self.overwrite_txt_radio, 2)
        batch_layout.addWidget(self.no_txt_radio)
        batch_layout.addWidget(self.merge_txt_radio)
        batch_layout.addWidget(self.overwrite_txt_radio)

        self.batch_inquiry_button = QPushButton("Start Batch Inquiry")
        self.batch_inquiry_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        batch_layout.addWidget(self.batch_inquiry_button)
        batch_layout.addWidget(self.cancel_button)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        tags_group = QGroupBox("Batch Tags (Real-time)")
        tags_layout = QVBoxLayout()
        self.discovered_tags_table = QTableWidget()
        self.discovered_tags_table.setColumnCount(2)
        self.discovered_tags_table.setHorizontalHeaderLabels(["Tag", "Count"])
        self.discovered_tags_table.horizontalHeader().setStretchLastSection(False)
        self.discovered_tags_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.discovered_tags_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        tags_layout.addWidget(self.discovered_tags_table)
        tags_group.setLayout(tags_layout)
        layout.addWidget(tags_group)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        layout.addStretch()
        return widget

    def _create_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        transcript_group = QGroupBox("Transcript")
        transcript_layout = QVBoxLayout()
        self.single_transcript = QListWidget()
        self.single_transcript.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.single_transcript.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.single_transcript.setSpacing(8)
        self.single_transcript.setMinimumHeight(520)
        transcript_layout.addWidget(self.single_transcript)
        transcript_group.setLayout(transcript_layout)
        layout.addWidget(transcript_group, 1)

        raw_group = QGroupBox("Raw Model Response")
        raw_layout = QVBoxLayout()
        self.raw_toggle_button = QPushButton("Show Raw Response")
        self.raw_toggle_button.setCheckable(True)
        self.raw_toggle_button.setChecked(False)
        self.raw_toggle_button.toggled.connect(self._toggle_raw_response_visibility)
        raw_layout.addWidget(self.raw_toggle_button)

        self.raw_response_view = QTextEdit()
        self.raw_response_view.setReadOnly(True)
        self.raw_response_view.setPlaceholderText("Raw llama response will appear here.")
        self.raw_response_view.setMinimumHeight(180)
        self.raw_response_view.setVisible(False)
        raw_layout.addWidget(self.raw_response_view)
        raw_group.setLayout(raw_layout)
        layout.addWidget(raw_group)

        return widget

    def setup_connections(self):
        self.load_model_button.clicked.connect(self.load_model)
        self.unload_model_button.clicked.connect(self.unload_model)
        self.batch_inquiry_button.clicked.connect(self.batch_interrogate)
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.send_single_button.clicked.connect(self.send_single_inquiry)
        self.reset_context_button.clicked.connect(self.reset_single_context)
        self.image_selector.currentIndexChanged.connect(self._on_selected_image_changed)
        self.image_queue.itemDoubleClicked.connect(self._on_queue_item_double_click)
        self.open_advanced_button.clicked.connect(self._open_selected_in_advanced)
        self.mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        self.mode_tabs.currentChanged.connect(lambda _: self._save_inquiry_options())

        for key in ("binary_path_edit", "model_path_edit", "mmproj_path_edit"):
            self.llama_config_refs[key].textChanged.connect(lambda *_: self._save_inquiry_options())
        for key in ("ctx_size_spin", "gpu_layers_spin", "temperature_spin", "max_tokens_spin", "server_port_spin"):
            self.llama_config_refs[key].valueChanged.connect(lambda _: self._save_inquiry_options())
        for key in (
            "include_prior_tables_check",
            "carry_batch_context_check",
            "include_clip_check",
            "include_wd_check",
            "include_camie_check",
        ):
            self.llama_config_refs[key].toggled.connect(lambda *_: self._save_inquiry_options())

        self.single_task_combo.currentTextChanged.connect(lambda *_: self._save_inquiry_options())
        self.single_prompt_input.textChanged.connect(lambda *_: self._save_inquiry_options())
        self.batch_task_combo.currentTextChanged.connect(lambda *_: self._save_inquiry_options())
        self.batch_prompt_input.textChanged.connect(self._save_inquiry_options)
        self.txt_output_group.idClicked.connect(lambda _: self._save_inquiry_options())

    def _on_mode_tab_changed(self, index: int):
        """Switch transcript source based on active mode tab."""
        self._refresh_transcript_for_active_mode()

    def _toggle_raw_response_visibility(self, visible: bool):
        """Show/hide raw response panel content."""
        self.raw_response_view.setVisible(visible)
        self.raw_toggle_button.setText("Hide Raw Response" if visible else "Show Raw Response")

    def _apply_saved_inquiry_options(self):
        """Restore persisted Inquiry controls that are not part of llama config."""
        options = self.inquiry_settings.get_options()
        self._set_combo_text(self.single_task_combo, options.get("single_task", "describe"))
        self.single_prompt_input.setText(options.get("single_prompt", ""))
        self._set_combo_text(self.batch_task_combo, options.get("batch_task", "describe"))
        self.batch_prompt_input.setPlainText(options.get("batch_prompt", ""))
        self._set_txt_output_mode(options.get("txt_output_mode", "merge"))

        active_tab = int(options.get("active_tab", 0))
        if 0 <= active_tab < self.mode_tabs.count():
            self.mode_tabs.setCurrentIndex(active_tab)

    @staticmethod
    def _set_combo_text(combo: QComboBox, value: str):
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_txt_output_mode(self, mode: str):
        if mode == "none":
            self.no_txt_radio.setChecked(True)
        elif mode == "overwrite":
            self.overwrite_txt_radio.setChecked(True)
        else:
            self.merge_txt_radio.setChecked(True)

    def _get_txt_output_mode(self) -> str:
        if self.no_txt_radio.isChecked():
            return "none"
        if self.overwrite_txt_radio.isChecked():
            return "overwrite"
        return "merge"

    def _save_inquiry_options(self):
        """Persist current Inquiry options."""
        if not self.llama_config_refs:
            return

        self.inquiry_settings.update_options(
            {
                "llama_config": self.get_llama_config(),
                "single_task": self.single_task_combo.currentText(),
                "single_prompt": self.single_prompt_input.text(),
                "batch_task": self.batch_task_combo.currentText(),
                "batch_prompt": self.batch_prompt_input.toPlainText(),
                "txt_output_mode": self._get_txt_output_mode(),
                "active_tab": self.mode_tabs.currentIndex(),
            }
        )

    def get_llama_config(self) -> Dict[str, Any]:
        """Return latest llama.cpp configuration from UI controls."""
        if not self.llama_config_refs:
            return dict(self.llama_config)

        included_types = []
        if self.llama_config_refs["include_clip_check"].isChecked():
            included_types.append("CLIP")
        if self.llama_config_refs["include_wd_check"].isChecked():
            included_types.append("WD")
        if self.llama_config_refs["include_camie_check"].isChecked():
            included_types.append("Camie")

        self.llama_config.update(
            {
                "llama_binary_path": self.llama_config_refs["binary_path_edit"].text().strip(),
                "llama_model_path": self.llama_config_refs["model_path_edit"].text().strip(),
                "llama_mmproj_path": self.llama_config_refs["mmproj_path_edit"].text().strip() or None,
                "ctx_size": self.llama_config_refs["ctx_size_spin"].value(),
                "gpu_layers": self.llama_config_refs["gpu_layers_spin"].value(),
                "temperature": self.llama_config_refs["temperature_spin"].value(),
                "max_tokens": self.llama_config_refs["max_tokens_spin"].value(),
                "server_port": self.llama_config_refs["server_port_spin"].value(),
                "include_prior_tables": self.llama_config_refs["include_prior_tables_check"].isChecked(),
                "carry_batch_context": self.llama_config_refs["carry_batch_context_check"].isChecked(),
                "included_model_types": included_types,
            }
        )
        return dict(self.llama_config)

    def set_directory_context(self, directory: str, recursive: bool):
        """Receive shared source directory metadata from Interrogation tab."""
        self.current_directory = Path(directory) if directory else None
        self.current_recursive = bool(recursive)
        if self.current_directory:
            self.source_dir_label.setText(f"Directory: {self.current_directory}")
        else:
            self.source_dir_label.setText("Directory: Not selected")
        self.source_mode_label.setText(f"Mode: {'Recursive' if recursive else 'Non-recursive'}")

    def set_images_from_interrogation(self, image_paths: List[str]):
        """Receive shared queue images from Interrogation tab."""
        previous_selection = self.current_image_path
        self.loaded_image_paths = list(image_paths or [])

        self.image_queue.clear()
        self.image_selector.blockSignals(True)
        self.image_selector.clear()

        for image_path in self.loaded_image_paths:
            display_name = self._to_display_name(image_path)
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, image_path)
            self.image_queue.addItem(item)
            self.image_selector.addItem(display_name, image_path)

        if previous_selection and previous_selection in self.loaded_image_paths:
            idx = self.image_selector.findData(previous_selection)
            if idx >= 0:
                self.image_selector.setCurrentIndex(idx)
        elif self.image_selector.count() > 0:
            self.image_selector.setCurrentIndex(0)

        self.image_selector.blockSignals(False)
        self._on_selected_image_changed()

        if self.current_interrogator and self.current_interrogator.is_loaded:
            self.batch_inquiry_button.setEnabled(bool(self.loaded_image_paths))
        else:
            self.batch_inquiry_button.setEnabled(False)

    def _to_display_name(self, image_path: str) -> str:
        path = Path(image_path)
        if self.current_directory:
            try:
                return str(path.relative_to(self.current_directory))
            except ValueError:
                return path.name
        return path.name

    def _append_timeout_hint_if_needed(self, error_text: str) -> str:
        """Append DGX Spark guidance for timeout failures on ARM64 NVIDIA systems."""
        if not is_llama_timeout_error(error_text):
            return error_text

        arch = platform.machine().lower()
        if arch not in ("arm64", "aarch64"):
            return error_text

        try:
            if not get_device_detector().is_pytorch_cuda_available():
                return error_text
        except Exception:
            return error_text

        if self.DGX_TIMEOUT_HINT in error_text:
            return error_text
        return f"{error_text}\n\n{self.DGX_TIMEOUT_HINT}"

    def load_model(self):
        """Load llama.cpp interrogator with current config."""
        try:
            config = self.get_llama_config()
            if not config.get("llama_binary_path"):
                raise ValueError("llama-server path is required")
            if not config.get("llama_model_path"):
                raise ValueError("Multimodal model path is required")
            self._save_inquiry_options()

            self.load_model_button.setEnabled(False)
            self.progress_label.setText("Loading model...")

            if self.current_interrogator:
                self.current_interrogator.unload_model()

            self.current_interrogator = LlamaCppInterrogator(model_name="LlamaCpp")
            self.current_interrogator.load_model(
                llama_binary_path=config["llama_binary_path"],
                llama_model_path=config["llama_model_path"],
                llama_mmproj_path=config.get("llama_mmproj_path"),
                ctx_size=config["ctx_size"],
                gpu_layers=config["gpu_layers"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                server_port=config["server_port"],
                server_host="127.0.0.1",
            )
            requested_port = int(config.get("server_port", 8080))
            resolved_port = int(
                self.current_interrogator.get_config().get("server_port", requested_port)
            )
            if resolved_port != requested_port:
                self.llama_config["server_port"] = resolved_port
                if self.llama_config_refs:
                    self.llama_config_refs["server_port_spin"].setValue(resolved_port)
                self.inquiry_settings.update_llama_config(self.llama_config)

            model_info = f"LlamaCpp - {Path(config['llama_model_path']).name}"
            runtime_meta = self.current_interrogator.runtime.get_runtime_metadata()
            runtime_url = runtime_meta.get("base_url") or "http://127.0.0.1:unknown"
            runtime_pid = runtime_meta.get("pid")
            runtime_log = runtime_meta.get("log_path")
            self.model_status_label.setText(f"Model: {model_info}\nLoaded")
            self.inquiry_status_label.setText(
                f"Connected: {Path(config['llama_model_path']).name} at {runtime_url} (pid={runtime_pid})"
            )
            self.inquiry_status_label.setStyleSheet("color: green;")
            if runtime_log:
                if resolved_port != requested_port:
                    self.progress_label.setText(
                        "Model loaded successfully. "
                        f"Requested port {requested_port} was unavailable; using {resolved_port}. "
                        f"Runtime logs: {runtime_log}"
                    )
                else:
                    self.progress_label.setText(
                        f"Model loaded successfully. Runtime logs: {runtime_log}"
                    )
            else:
                if resolved_port != requested_port:
                    self.progress_label.setText(
                        "Model loaded successfully. "
                        f"Requested port {requested_port} was unavailable; using {resolved_port}."
                    )
                else:
                    self.progress_label.setText("Model loaded successfully")
            self.unload_model_button.setEnabled(True)
            self.batch_inquiry_button.setEnabled(bool(self.loaded_image_paths))
            self._prime_current_session_history()
            self.model_loaded.emit(model_info)
        except Exception as exc:
            recent_logs = ""
            if self.current_interrogator:
                recent_logs = self.current_interrogator.runtime.get_recent_logs(max_lines=40)
            details = f"\n\nRecent llama logs:\n{recent_logs}" if recent_logs else ""
            QMessageBox.critical(self, "Error", f"Failed to load llama model:\n{exc}{details}")
            self.model_status_label.setText("Model: Load failed")
            self.inquiry_status_label.setText("Model load failed.")
            self.inquiry_status_label.setStyleSheet("color: red;")
        finally:
            self.load_model_button.setEnabled(True)

    def unload_model(self):
        """Unload current llama model."""
        if not self.current_interrogator:
            QMessageBox.information(self, "Info", "No model is currently loaded")
            return

        try:
            self.current_interrogator.unload_model()
            self.current_interrogator = None
            self.model_status_label.setText("Model: Not loaded")
            self.inquiry_status_label.setText("Model unloaded.")
            self.inquiry_status_label.setStyleSheet("color: #666;")
            self.progress_label.setText("Model unloaded successfully")
            self.raw_response_view.clear()
            self.unload_model_button.setEnabled(False)
            self.batch_inquiry_button.setEnabled(False)
            self.model_unloaded.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to unload model:\n{exc}")

    def _on_selected_image_changed(self):
        """Refresh preview + single-image history when selected image changes."""
        image_path = self.image_selector.currentData()
        self.current_image_path = image_path if isinstance(image_path, str) else None
        self.current_image_hash = None
        self.current_session_key = None
        self.current_interrogations = []

        if not self.current_image_path:
            self.image_preview.setText("No image selected")
            self.current_image_label.setText("")
            self.single_prior_tables.setRowCount(0)
            self._refresh_transcript_for_active_mode()
            self.raw_response_view.clear()
            return

        pixmap = QPixmap(self.current_image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                420,
                300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_preview.setPixmap(scaled)
        else:
            self.image_preview.setText("Failed to load preview")

        self.current_image_label.setText(f"Selected: {Path(self.current_image_path).name}")
        self.current_image_hash = hash_image_content(self.current_image_path)
        self.current_session_key = f"single:{self.current_image_hash}"
        self.current_interrogations = self.database.get_all_interrogations_for_image(self.current_image_hash) or []
        self._refresh_single_prior_tables()
        self._refresh_transcript_for_active_mode()
        self._load_latest_raw_response_for_image()
        self._prime_current_session_history()

    def _load_latest_raw_response_for_image(self):
        """Load latest saved llama raw output for the selected image, when available."""
        if not self.current_interrogations:
            self.raw_response_view.clear()
            return

        latest_raw = ""
        for interrog in self.current_interrogations:
            if (interrog.get("model_type") or "") == "LlamaCpp" and (interrog.get("raw_output") or "").strip():
                latest_raw = interrog.get("raw_output") or ""
                break
        self.raw_response_view.setPlainText(latest_raw)

    def _refresh_single_prior_tables(self):
        """Rebuild selectable prior interrogation table rows for the selected image."""
        self.single_prior_tables.setRowCount(0)
        if not self.current_interrogations:
            return

        for interrog in self.current_interrogations:
            row = self.single_prior_tables.rowCount()
            self.single_prior_tables.insertRow(row)

            include_item = QTableWidgetItem()
            include_item.setFlags(
                include_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            include_item.setCheckState(Qt.CheckState.Unchecked)
            include_item.setData(Qt.ItemDataRole.UserRole, interrog)
            self.single_prior_tables.setItem(row, 0, include_item)

            model_label = f"{interrog.get('model_name', 'Unknown')} ({interrog.get('model_type', '')})"
            self.single_prior_tables.setItem(row, 1, QTableWidgetItem(model_label))

    def _refresh_transcript_for_active_mode(self):
        """Render transcript according to selected mode tab."""
        if self.mode_tabs.currentIndex() == 1:
            self._load_batch_transcript_history()
        else:
            self._load_transcript_history()

    def _load_batch_transcript_history(self):
        """Render in-memory batch transcript entries (multi-image)."""
        self.single_transcript.clear()
        for entry in self.batch_transcript_entries:
            self._append_transcript_turn_card(entry, image_path=entry.get("image_path"))
        if self.single_transcript.count() > 0:
            self.single_transcript.scrollToBottom()

    def _load_transcript_history(self):
        """Load transcript for selected image, including single and batch turns."""
        self.single_transcript.clear()
        if not self.current_image_hash:
            return

        model_name = self.current_interrogator.model_name if self.current_interrogator else None
        history = self.database.get_multimodal_history(
            image_hash=self.current_image_hash,
            model_name=model_name,
        )

        for turn in history:
            self._append_transcript_turn_card(turn, image_path=self.current_image_path)

        if self.single_transcript.count() > 0:
            self.single_transcript.scrollToBottom()

    def _append_transcript_turn_card(self, turn: Dict[str, Any], image_path: Optional[str] = None) -> None:
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
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        prompt_text = (turn.get("prompt_text") or "").strip() or "[no prompt]"
        prompt_frame = QFrame()
        prompt_frame.setFrameShape(QFrame.Shape.StyledPanel)
        prompt_frame.setStyleSheet(
            f"QFrame {{ border: 1px solid {prompt_border_hex}; border-radius: 6px; background-color: {prompt_bg_hex}; }}"
        )
        prompt_layout = QVBoxLayout(prompt_frame)
        prompt_layout.setContentsMargins(8, 6, 8, 6)
        prompt_label = QLabel(prompt_text)
        prompt_label.setWordWrap(True)
        prompt_label.setStyleSheet(f"color: {text_hex};")
        prompt_layout.addWidget(prompt_label)
        turn_image_path = image_path or turn.get("image_path") or self.current_image_path
        if turn_image_path:
            prompt_path_row = QHBoxLayout()
            prompt_path_row.addStretch()
            path_label = QLabel(f"[{self._to_display_name(turn_image_path)}]")
            path_label.setStyleSheet(f"color: {text_hex};")
            prompt_path_row.addWidget(path_label)
            prompt_layout.addLayout(prompt_path_row)
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
        model_name = turn.get("model_name") or (
            self.current_interrogator.model_name if self.current_interrogator else "LlamaCpp"
        )

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
        comment_label.setStyleSheet(f"color: {text_hex};")
        model_label = QLabel(f"[{model_name}]")
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

            add_button.clicked.connect(add_tag)
            tags_row.addWidget(add_button)
            tags_row.addStretch()

        refresh_tags_row()
        card_layout.addLayout(tags_row)

        item = QListWidgetItem()
        item.setSizeHint(card.sizeHint())
        self.single_transcript.addItem(item)
        self.single_transcript.setItemWidget(item, card)

    def _prime_current_session_history(self):
        """Prime in-memory context for the currently selected image session."""
        if not self.current_interrogator or not self.current_interrogator.is_loaded:
            return
        if not self.current_session_key or not self.current_image_hash:
            return

        history = self.database.get_multimodal_history(
            session_key=self.current_session_key,
            mode="single",
            image_hash=self.current_image_hash,
            model_name=self.current_interrogator.model_name,
        )
        self.current_interrogator.set_session_history(self.current_session_key, history)

    def _build_selected_prior_tables(self) -> List[Dict[str, Any]]:
        """Extract selected prior table rows as prompt context."""
        selected: List[Dict[str, Any]] = []
        for row in range(self.single_prior_tables.rowCount()):
            include_item = self.single_prior_tables.item(row, 0)
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

    def send_single_inquiry(self):
        """Send one multimodal inquiry turn for selected image."""
        if not self.current_interrogator or not self.current_interrogator.is_loaded:
            QMessageBox.warning(self, "Model Required", "Load a llama model before sending inquiries.")
            return
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Select an image from the shared queue.")
            return

        try:
            task = self.single_task_combo.currentText()
            prompt_text = self.single_prompt_input.text().strip()
            included_tables = self._build_selected_prior_tables()
            self._save_inquiry_options()
            self.send_single_button.setEnabled(False)

            results = self.current_interrogator.interrogate(
                self.current_image_path,
                task=task,
                prompt=prompt_text,
                session_key=self.current_session_key,
                keep_context=True,
                included_tables=included_tables,
            )

            file_hash = self.current_image_hash or hash_image_content(self.current_image_path)
            metadata = get_image_metadata(self.current_image_path)
            image_id = self.database.register_image(
                self.current_image_path,
                file_hash,
                metadata["width"],
                metadata["height"],
                metadata["file_size"],
            )
            model_id = self.database.register_model(
                self.current_interrogator.model_name,
                self.current_interrogator.get_model_type(),
                config=self.current_interrogator.get_config(),
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
                session_key=self.current_session_key,
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
            self._refresh_single_prior_tables()
            self._refresh_transcript_for_active_mode()
            self.raw_response_view.setPlainText(results.get("raw_output", ""))

            warnings = (results.get("multimodal_response") or {}).get("warnings", [])
            if warnings:
                self.progress_label.setText(f"Single-image inquiry completed with warnings: {', '.join(warnings[:2])}")
            else:
                self.progress_label.setText("Single-image inquiry completed.")
        except Exception as exc:
            recent_logs = self.current_interrogator.runtime.get_recent_logs(max_lines=60)
            details = f"\n\nRecent llama logs:\n{recent_logs}" if recent_logs else ""
            error_text = self._append_timeout_hint_if_needed(str(exc))
            QMessageBox.critical(
                self,
                "Inquiry Error",
                f"Single-image inquiry failed:\n{error_text}{details}",
            )
        finally:
            self.send_single_button.setEnabled(True)

    def reset_single_context(self):
        """Reset current image multimodal session history."""
        if not self.current_session_key or not self.current_image_hash:
            return

        try:
            model_name = None
            if self.current_interrogator:
                model_name = self.current_interrogator.model_name
                self.current_interrogator.reset_session(self.current_session_key)

            self.database.clear_multimodal_session(
                session_key=self.current_session_key,
                model_name=model_name,
                mode="single",
                image_hash=self.current_image_hash,
            )
            self.single_transcript.clear()
            self.progress_label.setText("Image context reset.")
        except Exception as exc:
            QMessageBox.critical(self, "Reset Failed", f"Could not reset session:\n{exc}")

    def batch_interrogate(self):
        """Start multimodal batch inquiry with shared queue images."""
        if not self.current_interrogator or not self.current_interrogator.is_loaded:
            QMessageBox.warning(self, "Model Required", "Load a llama model before batch inquiry.")
            return
        if not self.loaded_image_paths:
            QMessageBox.information(self, "No Images", "No images available from Interrogation queue.")
            return

        images = [Path(p) for p in self.loaded_image_paths]
        reply = QMessageBox.question(
            self,
            "Confirm Batch Inquiry",
            f"Run multimodal inquiry on {len(images)} images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        config = self.get_llama_config()
        task = self.batch_task_combo.currentText()
        prompt_text = self.batch_prompt_input.toPlainText().strip()
        self._save_inquiry_options()
        self.active_batch_task = task
        self.active_batch_prompt = prompt_text

        if self.no_txt_radio.isChecked():
            write_files = False
            overwrite_files = False
        elif self.merge_txt_radio.isChecked():
            write_files = True
            overwrite_files = False
        else:
            write_files = True
            overwrite_files = True

        self.all_discovered_tags.clear()
        self.discovered_tags_table.setRowCount(0)
        self.batch_error_count = 0
        self.last_batch_error = ""
        self.batch_transcript_entries.clear()
        if self.mode_tabs.currentIndex() == 1:
            self.single_transcript.clear()

        self.interrogation_worker = MultimodalInterrogationWorker(
            image_paths=images,
            interrogator=self.current_interrogator,
            database=self.database,
            task=task,
            prompt=prompt_text,
            write_files=write_files,
            overwrite_files=overwrite_files,
            tag_filters=self.tag_filters,
            include_prior_tables=config.get("include_prior_tables", False),
            included_model_types=config.get("included_model_types", []),
            carry_context_across_batch=config.get("carry_batch_context", False),
        )
        self.interrogation_worker.progress.connect(self.on_batch_progress)
        self.interrogation_worker.result.connect(self.on_batch_result)
        self.interrogation_worker.error.connect(self.on_batch_error)
        self.interrogation_worker.finished.connect(self.on_batch_finished)

        self.batch_inquiry_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setMaximum(len(images))
        self.inquiry_started.emit()
        self.interrogation_worker.start()

    def cancel_operation(self):
        """Cancel active batch inquiry."""
        if self.interrogation_worker:
            self.interrogation_worker.cancel()
            self.progress_label.setText("Cancelling...")

    def on_batch_progress(self, current: int, total: int, message: str):
        """Update progress UI for batch inquiry."""
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total}: {message}")

        if current <= self.image_queue.count():
            item = self.image_queue.item(current - 1)
            if item:
                image_path = item.data(Qt.ItemDataRole.UserRole)
                item.setText(f"Processing - {self._to_display_name(image_path)}")

    def on_batch_result(self, image_path: str, results: Dict[str, Any]):
        """Track batch result and refresh rolling tag frequency table."""
        for i in range(self.image_queue.count()):
            item = self.image_queue.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                item.setText(f"Done - {self._to_display_name(image_path)}")
                break

        self.raw_response_view.setPlainText(results.get("raw_output", ""))

        batch_entry = {
            "prompt_type": self.active_batch_task,
            "prompt_text": self.active_batch_prompt,
            "response_json": results.get("multimodal_response", {}) or {},
            "tags": results.get("tags", []) or [],
            "model_name": self.current_interrogator.model_name if self.current_interrogator else "LlamaCpp",
            "image_path": image_path,
        }
        self.batch_transcript_entries.append(batch_entry)

        if self.mode_tabs.currentIndex() == 1:
            self._append_transcript_turn_card(batch_entry, image_path=image_path)
            self.single_transcript.scrollToBottom()

        if (
            self.current_image_path
            and Path(self.current_image_path).resolve() == Path(image_path).resolve()
        ):
            self.current_image_hash = hash_image_content(self.current_image_path)
            self.current_interrogations = self.database.get_all_interrogations_for_image(
                self.current_image_hash
            ) or []
            self._refresh_single_prior_tables()
            if self.mode_tabs.currentIndex() == 0:
                self._load_transcript_history()

        for tag in results.get("tags", []):
            self.all_discovered_tags[tag] = self.all_discovered_tags.get(tag, 0) + 1
        self._refresh_batch_tags_table()

    def on_batch_error(self, image_path: str, error: str):
        """Mark failed batch image rows."""
        for i in range(self.image_queue.count()):
            item = self.image_queue.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                item.setText(f"Error - {self._to_display_name(image_path)}")
                break
        self.batch_error_count += 1
        self.last_batch_error = self._append_timeout_hint_if_needed(error)
        self.progress_label.setText(error or "Batch inquiry error")

    def on_batch_finished(self):
        """Reset controls when batch processing completes."""
        self.batch_inquiry_button.setEnabled(bool(self.current_interrogator and self.loaded_image_paths))
        self.cancel_button.setEnabled(False)
        self.progress_label.setText("Batch inquiry complete")
        self.inquiry_finished.emit()
        if self.batch_error_count > 0:
            recent_logs = ""
            if self.current_interrogator:
                recent_logs = self.current_interrogator.runtime.get_recent_logs(max_lines=60)
            details = f"\n\nRecent llama logs:\n{recent_logs}" if recent_logs else ""
            QMessageBox.warning(
                self,
                "Batch Inquiry Finished With Errors",
                f"Completed with {self.batch_error_count} error(s).\n"
                f"Last error: {self.last_batch_error or 'Unknown error'}{details}",
            )
        else:
            QMessageBox.information(self, "Complete", "Batch inquiry finished")

    def _refresh_batch_tags_table(self):
        self.discovered_tags_table.setRowCount(0)
        sorted_tags = sorted(self.all_discovered_tags.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:100]:
            row = self.discovered_tags_table.rowCount()
            self.discovered_tags_table.insertRow(row)
            self.discovered_tags_table.setItem(row, 0, QTableWidgetItem(tag))
            self.discovered_tags_table.setItem(row, 1, QTableWidgetItem(str(count)))

    def _on_queue_item_double_click(self, item: QListWidgetItem):
        image_path = item.data(Qt.ItemDataRole.UserRole)
        if image_path:
            self._open_advanced_inspection(image_path)

    def _open_selected_in_advanced(self):
        image_path = self.image_selector.currentData()
        if image_path:
            self._open_advanced_inspection(image_path)

    def _open_advanced_inspection(self, image_path: str):
        """Open advanced inspection dialog using Inquiry llama config."""
        try:
            dialog = AdvancedImageInspectionDialog(
                image_path=image_path,
                image_list=list(self.loaded_image_paths),
                database=self.database,
                tag_filters=self.tag_filters,
                llama_config=self.get_llama_config(),
                parent=self,
            )
            dialog.show()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open inspection dialog:\n{exc}")
