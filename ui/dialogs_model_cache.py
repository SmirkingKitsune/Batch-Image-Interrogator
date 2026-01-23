"""Model Cache Manager Dialog for managing cached ML models."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QWidget,
    QCheckBox, QProgressBar, QMessageBox, QHeaderView,
    QGroupBox, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, List, Optional

from core.model_cache import ModelCacheManager, ModelCacheInfo
from ui.workers import CacheScanWorker, CacheDeleteWorker, TensorRTConversionWorker


class ModelCacheDialog(QDialog):
    """Dialog for managing model caches (HuggingFace and TensorRT)."""

    # Signal emitted when cache changes occur
    cache_changed = pyqtSignal()

    def __init__(self, provider_settings=None, parent=None):
        """Initialize the Model Cache Manager dialog.

        Args:
            provider_settings: Optional ONNXProviderSettings for TensorRT conversion
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Model Cache Manager")
        self.setModal(True)
        self.resize(800, 600)

        self.provider_settings = provider_settings
        self.cache_manager = ModelCacheManager()
        self.models_data: Dict[str, List[ModelCacheInfo]] = {}

        # Workers
        self.scan_worker: Optional[CacheScanWorker] = None
        self.delete_worker: Optional[CacheDeleteWorker] = None
        self.trt_worker: Optional[TensorRTConversionWorker] = None

        self.setup_ui()

        # Start initial scan
        self._start_scan()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Top bar with filter and total size
        top_bar = QHBoxLayout()

        self.cached_only_check = QCheckBox("Show cached models only")
        self.cached_only_check.setChecked(False)
        self.cached_only_check.stateChanged.connect(self._apply_filter)
        top_bar.addWidget(self.cached_only_check)

        top_bar.addStretch()

        self.total_size_label = QLabel("Total: calculating...")
        self.total_size_label.setStyleSheet("font-weight: bold;")
        top_bar.addWidget(self.total_size_label)

        layout.addLayout(top_bar)

        # Tab widget for model types
        self.tabs = QTabWidget()

        # WD Tagger tab
        self.wd_table = self._create_model_table()
        wd_tab = QWidget()
        wd_layout = QVBoxLayout(wd_tab)
        wd_layout.addWidget(self.wd_table)
        self.tabs.addTab(wd_tab, "WD Tagger")

        # Camie tab
        self.camie_table = self._create_model_table()
        camie_tab = QWidget()
        camie_layout = QVBoxLayout(camie_tab)
        camie_layout.addWidget(self.camie_table)
        self.tabs.addTab(camie_tab, "Camie")

        layout.addWidget(self.tabs)

        # Selection summary
        self.selection_label = QLabel("Selected: 0 models (0 B)")
        layout.addWidget(self.selection_label)

        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()

        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.clicked.connect(self._delete_selected)
        self.delete_selected_btn.setEnabled(False)
        action_layout.addWidget(self.delete_selected_btn)

        self.convert_trt_btn = QPushButton("Convert to TensorRT")
        self.convert_trt_btn.clicked.connect(self._convert_selected_to_trt)
        self.convert_trt_btn.setEnabled(False)
        self.convert_trt_btn.setToolTip("Pre-compile selected ONNX models to TensorRT engines")
        action_layout.addWidget(self.convert_trt_btn)

        action_layout.addStretch()

        self.delete_all_btn = QPushButton("Delete All Cached")
        self.delete_all_btn.clicked.connect(self._delete_all_cached)
        self.delete_all_btn.setStyleSheet("color: #c00;")
        action_layout.addWidget(self.delete_all_btn)

        self.convert_all_trt_btn = QPushButton("Convert All to TRT")
        self.convert_all_trt_btn.clicked.connect(self._convert_all_to_trt)
        self.convert_all_trt_btn.setToolTip("Pre-compile all cached ONNX models to TensorRT engines")
        action_layout.addWidget(self.convert_all_trt_btn)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        # Bottom buttons
        bottom_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._start_scan)
        bottom_layout.addWidget(self.refresh_btn)

        bottom_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(self.close_btn)

        layout.addLayout(bottom_layout)

    def _create_model_table(self) -> QTableWidget:
        """Create a model table widget."""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["", "Model Name", "Status", "Size", "TRT"])

        # Configure columns
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Checkbox
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Model name
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)  # Status
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)  # Size
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)  # TRT

        table.setColumnWidth(0, 30)
        table.setColumnWidth(2, 80)
        table.setColumnWidth(3, 80)
        table.setColumnWidth(4, 60)

        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.itemChanged.connect(self._on_item_changed)

        return table

    def _populate_table(self, table: QTableWidget, models: List[ModelCacheInfo]):
        """Populate a table with model information."""
        # Disconnect signal temporarily to avoid triggering during population
        table.blockSignals(True)

        table.setRowCount(0)
        show_cached_only = self.cached_only_check.isChecked()

        for model in models:
            # Skip uncached models if filter is enabled
            if show_cached_only and not model.is_cached:
                continue

            row = table.rowCount()
            table.insertRow(row)

            # Checkbox column
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.CheckState.Unchecked)
            checkbox_item.setData(Qt.ItemDataRole.UserRole, model.model_id)
            table.setItem(row, 0, checkbox_item)

            # Model name
            name_item = QTableWidgetItem(model.model_id)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 1, name_item)

            # Status
            status_text = "Cached" if model.is_cached else "--"
            status_item = QTableWidgetItem(status_text)
            status_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            if model.is_cached:
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            else:
                status_item.setForeground(Qt.GlobalColor.gray)
            table.setItem(row, 2, status_item)

            # Size
            if model.is_cached:
                size_text = ModelCacheManager.format_size(model.cache_size_bytes)
            else:
                size_text = "--"
            size_item = QTableWidgetItem(size_text)
            size_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 3, size_item)

            # TensorRT status
            if model.has_tensorrt_engine:
                trt_text = "Yes"
                trt_item = QTableWidgetItem(trt_text)
                trt_item.setForeground(Qt.GlobalColor.darkGreen)
            else:
                trt_text = "--"
                trt_item = QTableWidgetItem(trt_text)
                trt_item.setForeground(Qt.GlobalColor.gray)
            trt_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            table.setItem(row, 4, trt_item)

        table.blockSignals(False)

    def _get_current_table(self) -> QTableWidget:
        """Get the currently visible table widget."""
        current_index = self.tabs.currentIndex()
        if current_index == 0:
            return self.wd_table
        elif current_index == 1:
            return self.camie_table
        return self.wd_table

    def _get_selected_models(self, table: Optional[QTableWidget] = None) -> List[str]:
        """Get list of selected model IDs from a table."""
        if table is None:
            # Get from all tables
            selected = []
            selected.extend(self._get_selected_from_table(self.wd_table))
            selected.extend(self._get_selected_from_table(self.camie_table))
            return selected
        return self._get_selected_from_table(table)

    def _get_selected_from_table(self, table: QTableWidget) -> List[str]:
        """Get selected model IDs from a specific table."""
        selected = []
        for row in range(table.rowCount()):
            checkbox_item = table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                model_id = checkbox_item.data(Qt.ItemDataRole.UserRole)
                if model_id:
                    selected.append(model_id)
        return selected

    def _get_cached_models(self) -> List[str]:
        """Get list of all cached model IDs."""
        cached = []
        for model_type, models in self.models_data.items():
            for model in models:
                if model.is_cached:
                    cached.append(model.model_id)
        return cached

    def _get_model_info(self, model_id: str) -> Optional[ModelCacheInfo]:
        """Get ModelCacheInfo for a given model ID."""
        for model_type, models in self.models_data.items():
            for model in models:
                if model.model_id == model_id:
                    return model
        return None

    def _on_item_changed(self, item: QTableWidgetItem):
        """Handle table item changes (checkbox state)."""
        if item.column() == 0:  # Checkbox column
            self._update_selection_summary()

    def _update_selection_summary(self):
        """Update the selection summary label."""
        selected = self._get_selected_models()
        total_size = 0

        for model_id in selected:
            info = self._get_model_info(model_id)
            if info:
                total_size += info.cache_size_bytes

        size_str = ModelCacheManager.format_size(total_size)
        self.selection_label.setText(f"Selected: {len(selected)} model(s) ({size_str})")

        # Enable/disable buttons
        has_selection = len(selected) > 0
        self.delete_selected_btn.setEnabled(has_selection)
        self.convert_trt_btn.setEnabled(has_selection)

    def _update_total_size(self):
        """Update the total cache size display."""
        sizes = self.cache_manager.get_total_cache_size()
        total_str = ModelCacheManager.format_size(sizes['total'])
        hf_str = ModelCacheManager.format_size(sizes['huggingface'])
        trt_str = ModelCacheManager.format_size(sizes['tensorrt'])

        self.total_size_label.setText(f"Total: {total_str} (HF: {hf_str}, TRT: {trt_str})")

    def _apply_filter(self):
        """Apply the cached-only filter to tables."""
        if 'WD' in self.models_data:
            self._populate_table(self.wd_table, self.models_data['WD'])
        if 'Camie' in self.models_data:
            self._populate_table(self.camie_table, self.models_data['Camie'])
        self._update_selection_summary()

    def _start_scan(self):
        """Start scanning model caches."""
        if self.scan_worker and self.scan_worker.isRunning():
            return

        self.refresh_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_label.setVisible(True)
        self.progress_label.setText("Scanning model caches...")

        self.scan_worker = CacheScanWorker(self.cache_manager)
        self.scan_worker.progress.connect(self._on_scan_progress)
        self.scan_worker.finished.connect(self._on_scan_finished)
        self.scan_worker.start()

    def _on_scan_progress(self, current: int, total: int, message: str):
        """Handle scan progress updates."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        self.progress_label.setText(message)

    def _on_scan_finished(self, models_data: Dict[str, List[ModelCacheInfo]]):
        """Handle scan completion."""
        self.models_data = models_data

        # Populate tables
        if 'WD' in models_data:
            self._populate_table(self.wd_table, models_data['WD'])
        if 'Camie' in models_data:
            self._populate_table(self.camie_table, models_data['Camie'])

        # Update UI
        self._update_total_size()
        self._update_selection_summary()

        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.refresh_btn.setEnabled(True)

    def _delete_selected(self):
        """Delete selected model caches."""
        selected = self._get_selected_models()
        if not selected:
            return

        # Calculate total size
        total_size = 0
        for model_id in selected:
            info = self._get_model_info(model_id)
            if info:
                total_size += info.cache_size_bytes

        size_str = ModelCacheManager.format_size(total_size)

        reply = QMessageBox.warning(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete cache for {len(selected)} model(s)?\n\n"
            f"This will free approximately {size_str} of disk space.\n\n"
            f"Models will be re-downloaded when needed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self._start_delete(selected)

    def _delete_all_cached(self):
        """Delete all cached models."""
        cached = self._get_cached_models()
        if not cached:
            QMessageBox.information(self, "No Cached Models", "No cached models to delete.")
            return

        # Calculate total size
        sizes = self.cache_manager.get_total_cache_size()
        size_str = ModelCacheManager.format_size(sizes['huggingface'])

        reply = QMessageBox.warning(
            self,
            "Confirm Delete All",
            f"Are you sure you want to delete ALL cached models?\n\n"
            f"This will delete {len(cached)} model(s) and free approximately {size_str}.\n\n"
            f"Models will be re-downloaded when needed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self._start_delete(cached)

    def _start_delete(self, model_ids: List[str]):
        """Start the delete worker."""
        if self.delete_worker and self.delete_worker.isRunning():
            return

        self._disable_actions()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(model_ids))
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Deleting caches...")

        self.delete_worker = CacheDeleteWorker(
            self.cache_manager, model_ids, delete_tensorrt=True
        )
        self.delete_worker.progress.connect(self._on_delete_progress)
        self.delete_worker.finished.connect(self._on_delete_finished)
        self.delete_worker.start()

    def _on_delete_progress(self, current: int, total: int, message: str):
        """Handle delete progress updates."""
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)

    def _on_delete_finished(self, success_count: int, error_count: int):
        """Handle delete completion."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._enable_actions()

        if error_count > 0:
            QMessageBox.warning(
                self,
                "Delete Complete",
                f"Deleted {success_count} cache(s) with {error_count} error(s)."
            )
        else:
            QMessageBox.information(
                self,
                "Delete Complete",
                f"Successfully deleted {success_count} model cache(s)."
            )

        # Refresh the view
        self._start_scan()
        self.cache_changed.emit()

    def _convert_selected_to_trt(self):
        """Convert selected models to TensorRT."""
        selected = self._get_selected_models()
        if not selected:
            return

        # Filter to only cached models
        cached_selected = []
        for model_id in selected:
            info = self._get_model_info(model_id)
            if info and info.is_cached:
                cached_selected.append(model_id)

        if not cached_selected:
            QMessageBox.warning(
                self,
                "No Cached Models",
                "Selected models are not cached. Download them first by using them in interrogation."
            )
            return

        self._start_trt_conversion(cached_selected)

    def _convert_all_to_trt(self):
        """Convert all cached models to TensorRT."""
        cached = self._get_cached_models()
        if not cached:
            QMessageBox.information(
                self,
                "No Cached Models",
                "No cached models available for TensorRT conversion."
            )
            return

        # Filter out models that already have TensorRT engines
        to_convert = []
        for model_id in cached:
            info = self._get_model_info(model_id)
            if info and not info.has_tensorrt_engine:
                to_convert.append(model_id)

        if not to_convert:
            QMessageBox.information(
                self,
                "All Converted",
                "All cached models already have TensorRT engines."
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm TensorRT Conversion",
            f"Convert {len(to_convert)} model(s) to TensorRT?\n\n"
            f"This may take several minutes per model.\n"
            f"TensorRT engines improve inference speed significantly.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self._start_trt_conversion(to_convert)

    def _start_trt_conversion(self, model_ids: List[str]):
        """Start the TensorRT conversion worker."""
        if self.trt_worker and self.trt_worker.isRunning():
            return

        self._disable_actions()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(model_ids))
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Starting TensorRT conversion...")

        self.trt_worker = TensorRTConversionWorker(
            self.cache_manager, model_ids, self.provider_settings
        )
        self.trt_worker.progress.connect(self._on_trt_progress)
        self.trt_worker.conversion_complete.connect(self._on_trt_model_complete)
        self.trt_worker.conversion_failed.connect(self._on_trt_model_failed)
        self.trt_worker.finished.connect(self._on_trt_finished)
        self.trt_worker.start()

    def _on_trt_progress(self, current: int, total: int, message: str):
        """Handle TensorRT conversion progress."""
        self.progress_bar.setValue(current)
        self.progress_label.setText(message)

    def _on_trt_model_complete(self, model_id: str):
        """Handle successful model conversion."""
        pass  # Progress updates handle UI

    def _on_trt_model_failed(self, model_id: str, error: str):
        """Handle failed model conversion."""
        print(f"TensorRT conversion failed for {model_id}: {error}")

    def _on_trt_finished(self, success_count: int, error_count: int):
        """Handle TensorRT conversion completion."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._enable_actions()

        if error_count > 0:
            QMessageBox.warning(
                self,
                "Conversion Complete",
                f"Converted {success_count} model(s) with {error_count} error(s).\n\n"
                f"Some models may not support TensorRT conversion."
            )
        elif success_count > 0:
            QMessageBox.information(
                self,
                "Conversion Complete",
                f"Successfully converted {success_count} model(s) to TensorRT."
            )
        else:
            QMessageBox.warning(
                self,
                "Conversion Failed",
                "No models were converted. TensorRT may not be available."
            )

        # Refresh the view
        self._start_scan()
        self.cache_changed.emit()

    def _disable_actions(self):
        """Disable action buttons during operations."""
        self.delete_selected_btn.setEnabled(False)
        self.delete_all_btn.setEnabled(False)
        self.convert_trt_btn.setEnabled(False)
        self.convert_all_trt_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.close_btn.setEnabled(False)

    def _enable_actions(self):
        """Re-enable action buttons after operations."""
        self.delete_all_btn.setEnabled(True)
        self.convert_all_trt_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.close_btn.setEnabled(True)
        self._update_selection_summary()  # This enables selected buttons if needed

    def closeEvent(self, event):
        """Handle dialog close event."""
        # Cancel any running workers
        if self.scan_worker and self.scan_worker.isRunning():
            self.scan_worker.cancel()
            self.scan_worker.wait()
        if self.delete_worker and self.delete_worker.isRunning():
            self.delete_worker.cancel()
            self.delete_worker.wait()
        if self.trt_worker and self.trt_worker.isRunning():
            self.trt_worker.cancel()
            self.trt_worker.wait()

        super().closeEvent(event)
