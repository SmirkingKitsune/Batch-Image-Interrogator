"""Main application window for Image Interrogator."""

import threading
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QFileDialog, QMessageBox, QStatusBar, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction
from pathlib import Path
from typing import Optional, Dict

from core import InterrogationDatabase, TagFilterSettings, ONNXProviderSettings
from ui.tabs import InterrogationTab, GalleryTab, SettingsTab
from ui.dialogs_database import DatabaseBusyDialog, QueueProcessingDialog
from ui.workers import DatabaseQueueWorker


class MainWindow(QMainWindow):
    """Main application window."""

    # Internal signal for thread-safe dialog display
    _database_busy = pyqtSignal(str, dict, int)  # operation, params, retry_count

    def __init__(self, device_status: Optional[Dict] = None):
        super().__init__()
        self.setWindowTitle("Image Interrogator - Batch Tagging Tool")
        self.setGeometry(100, 100, 1600, 900)
        self.setMinimumSize(1024, 600)

        # Store device status
        self.device_status = device_status or {}

        # Auto-detect default device based on availability
        from core.device_detector import get_device_detector
        detector = get_device_detector()
        default_pytorch_device = detector.get_pytorch_device()
        default_onnx_device = detector.get_onnx_device()

        # Core components
        self.database = InterrogationDatabase()
        self.tag_filters = TagFilterSettings()
        self.provider_settings = ONNXProviderSettings()
        self.current_interrogator = None
        self.current_model_type = "WD"
        self.current_directory = None
        self.current_image = None

        # Workers
        self.interrogation_worker = None
        self.organization_worker = None
        self.queue_worker = None

        # Database busy dialog state
        self._busy_response_event = threading.Event()
        self._busy_response_value = "abort"

        # Model configs with AUTO-DETECTED device
        self.clip_config = {
            'clip_model': 'ViT-L-14/openai',
            'caption_model': None,
            'mode': 'best',
            'device': default_pytorch_device  # AUTO-DETECTED
        }
        self.wd_config = {
            'wd_model': 'SmilingWolf/wd-v1-4-moat-tagger-v2',
            'threshold': 0.35,
            'device': default_onnx_device  # AUTO-DETECTED
        }
        self.camie_config = {
            'camie_model': 'Camais03/camie-tagger-v2',
            'threshold': 0.5,
            'threshold_profile': 'overall',
            'device': default_onnx_device,  # AUTO-DETECTED
            'enabled_categories': ['general', 'character', 'copyright', 'artist', 'meta', 'rating', 'year']
        }

        # Setup UI
        self.setup_ui()
        self.setup_menubar()
        self.setup_connections()
        self._setup_database_signals()

        self.statusBar().showMessage("Ready - Select a directory to begin")

        # Check for pending queue operations after UI is ready
        QTimer.singleShot(1000, self._check_startup_queue)

    def populate_model_lists(self):
        """
        Populate model dropdown lists with available models.

        This is called AFTER the window is shown to avoid blocking UI initialization
        and to prevent CUDA initialization conflicts.

        Called via QTimer.singleShot() from main.py after window.show().
        """
        try:
            self.statusBar().showMessage("Loading model lists...")

            # Get reference to CLIP model combo box
            clip_combo = self.interrogation_tab.clip_config_refs['clip_model_combo']

            # Clear placeholder
            clip_combo.clear()
            clip_combo.addItem("Loading...")
            clip_combo.setEnabled(False)

            # Force UI update to show loading state
            from PyQt6.QtWidgets import QApplication as QApp
            QApp.processEvents()

            # Populate CLIP models (this triggers open_clip import)
            from ui.dialogs import _populate_clip_models_combo
            clip_combo.clear()
            _populate_clip_models_combo(clip_combo)

            # Restore selected CLIP model
            current_clip_model = self.clip_config.get('clip_model', 'ViT-L-14/openai')
            index = clip_combo.findText(current_clip_model)
            if index >= 0:
                clip_combo.setCurrentIndex(index)
            else:
                # Select first valid model
                from ui.dialogs import _select_first_valid_clip_model
                _select_first_valid_clip_model(clip_combo)

            # Enable combo box
            clip_combo.setEnabled(True)
            self.statusBar().showMessage("✓ Model lists loaded successfully", 3000)

        except Exception as e:
            import logging
            logging.error(f"Error populating model lists: {e}")

            # Show error in status bar
            self.statusBar().showMessage(
                f"⚠ Warning: Could not load CLIP models - {str(e)}",
                10000
            )

            # Re-enable combo box with fallback list
            clip_combo.setEnabled(True)
            if clip_combo.count() == 0:
                # Add minimal fallback list
                clip_combo.addItems([
                    'ViT-L-14/openai',
                    'ViT-H-14/laion2b_s32b_b79k',
                    'ViT-B-32/openai'
                ])

    def setup_ui(self):
        """Setup the main UI with tabs."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tabs = QTabWidget()

        # Create tabs (pass shared state)
        self.interrogation_tab = InterrogationTab(
            database=self.database,
            clip_config=self.clip_config,
            wd_config=self.wd_config,
            camie_config=self.camie_config,
            tag_filters=self.tag_filters,
            provider_settings=self.provider_settings,
            parent=self
        )
        self.gallery_tab = GalleryTab(
            database=self.database,
            parent=self
        )
        self.settings_tab = SettingsTab(
            database=self.database,
            tag_filters=self.tag_filters,
            provider_settings=self.provider_settings,
            parent=self
        )

        # Add tabs
        self.tabs.addTab(self.interrogation_tab, "Interrogation")
        self.tabs.addTab(self.gallery_tab, "Gallery")
        self.tabs.addTab(self.settings_tab, "Database/Settings")

        layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Background process indicator in status bar
        self.background_indicator = QLabel("")
        self.background_indicator.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        self.status_bar.addPermanentWidget(self.background_indicator)


    def setup_menubar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_connections(self):
        """Setup signal/slot connections between tabs."""
        self.setup_cross_tab_signals()

    def _setup_database_signals(self):
        """Setup database busy callback and signal connections."""
        # Connect internal signal to slot (for thread-safe dialog display)
        self._database_busy.connect(self._on_database_busy)

        # Set the callback on the database
        self.database.set_busy_callback(self._emit_database_busy)

    def _emit_database_busy(self, operation: str, params: Dict, retry_count: int) -> str:
        """
        Called from worker thread when database is busy.

        Emits signal to show dialog on main thread and waits for response.
        """
        # Reset the event
        self._busy_response_event.clear()
        self._busy_response_value = "abort"

        # Emit signal to main thread
        self._database_busy.emit(operation, params, retry_count)

        # Wait for response from main thread (with timeout)
        # Max wait of 5 minutes - if no response, treat as abort
        if self._busy_response_event.wait(timeout=300):
            return self._busy_response_value
        else:
            return "abort"

    def _on_database_busy(self, operation: str, params: Dict, retry_count: int):
        """Handle database busy signal on main thread - show dialog."""
        # Get queue status for display
        queue_status = self.database.get_queue_status()
        queued_operations = queue_status.get('operations', [])

        # Check if operation is queueable
        is_queueable = self.database._operation_queue.is_queueable(operation)

        # Create and show dialog
        dialog = DatabaseBusyDialog(
            operation=operation,
            params=params,
            retry_count=retry_count,
            queued_operations=queued_operations,
            is_queueable=is_queueable,
            parent=self
        )

        # Connect dialog signals
        dialog.retry_requested.connect(lambda: self._set_busy_response("retry"))
        dialog.queue_and_continue.connect(lambda: self._set_busy_response("queue"))
        dialog.force_close_requested.connect(lambda: self._set_busy_response("abort"))

        # Show dialog (non-blocking for retry button)
        dialog.show()

    def _set_busy_response(self, response: str):
        """Set the response value and signal the waiting thread."""
        self._busy_response_value = response
        self._busy_response_event.set()

    def _check_startup_queue(self):
        """Check for pending queue operations at startup."""
        status = self.database.get_queue_status()
        pending_count = status.get('pending_count', 0)

        if pending_count > 0:
            operations = status.get('operations', [])

            dialog = QueueProcessingDialog(
                pending_count=pending_count,
                operations=operations,
                parent=self
            )

            dialog.processing_requested.connect(self._process_queue_now)
            dialog.skip_requested.connect(
                lambda: self.statusBar().showMessage(
                    f"{pending_count} queued operations will be processed later", 5000
                )
            )
            dialog.clear_requested.connect(self._clear_queue)

            dialog.exec()

    def _process_queue_now(self):
        """Process queued operations."""
        self.queue_worker = DatabaseQueueWorker(self.database)
        self.queue_worker.progress.connect(
            lambda cur, total, msg: self.statusBar().showMessage(
                f"Processing queue: {cur}/{total} - {msg}"
            )
        )
        self.queue_worker.finished.connect(self._on_queue_processing_finished)
        self.queue_worker.start()

    def _on_queue_processing_finished(self, success_count: int, failed_count: int):
        """Handle queue processing completion."""
        total = success_count + failed_count
        if failed_count == 0:
            self.statusBar().showMessage(
                f"Queue processed: {success_count} operations completed successfully", 5000
            )
        else:
            QMessageBox.warning(
                self,
                "Queue Processing Complete",
                f"Processed {total} operations:\n"
                f"  - {success_count} succeeded\n"
                f"  - {failed_count} failed\n\n"
                f"Failed operations remain in queue."
            )

        # Update settings tab if visible
        self.settings_tab.update_stats()

    def _clear_queue(self):
        """Clear all queued operations."""
        count = self.database.clear_operation_queue()
        self.statusBar().showMessage(f"Cleared {count} queued operations", 5000)

    def setup_cross_tab_signals(self):
        """Connect signals between tabs for cross-tab communication."""
        # === Directory Changes ===
        # When directory is selected in Interrogation tab, update other tabs
        # Note: gallery_tab.set_directory would trigger sync scanning, so we use
        # directory_loading_finished -> set_images_direct instead for efficiency
        self.interrogation_tab.directory_changed.connect(self._on_directory_changed)
        self.interrogation_tab.directory_changed.connect(self.settings_tab.set_directory)
        self.interrogation_tab.directory_changed.connect(
            lambda dir, recursive: self.statusBar().showMessage(
                f"Directory: {dir}{' (recursive)' if recursive else ''}", 5000
            )
        )

        # When directory loading finishes, pass pre-scanned paths to gallery
        # This avoids duplicate scanning and keeps UI responsive
        self.interrogation_tab.directory_loading_finished.connect(
            self.gallery_tab.set_images_direct
        )

        # === Model Loading ===
        # Update status bar when model is loaded/unloaded
        self.interrogation_tab.model_loaded.connect(
            lambda info: self.statusBar().showMessage(f"Model loaded: {info}", 5000)
        )
        self.interrogation_tab.model_unloaded.connect(
            lambda: self.statusBar().showMessage("Model unloaded", 3000)
        )

        # === Interrogation Process ===
        # Update status bar and indicator when interrogation starts/finishes
        self.interrogation_tab.interrogation_started.connect(
            lambda: self._on_interrogation_started()
        )
        self.interrogation_tab.interrogation_finished.connect(
            lambda: self._on_interrogation_finished()
        )

        # === Tag Editing ===
        # Update stats when tags are saved in gallery
        self.gallery_tab.tags_saved.connect(
            lambda path, tags: self.settings_tab.update_stats()
        )

    def _on_directory_changed(self, directory: str, recursive: bool):
        """Handle directory change - update gallery state without sync scanning."""
        # Update gallery's directory reference without triggering a sync scan
        # (the actual image list will come from directory_loading_finished signal)
        self.gallery_tab.current_directory = Path(directory)
        self.gallery_tab.recursive = recursive

    def _on_interrogation_started(self):
        """Handle interrogation start across tabs."""
        # Show background process indicator
        self.background_indicator.setText("⏳ Interrogating...")
        self.statusBar().showMessage("Batch interrogation started...")

    def _on_interrogation_finished(self):
        """Handle interrogation completion across tabs."""
        # Hide background process indicator
        self.background_indicator.setText("")

        # Update database stats in Settings tab
        self.settings_tab.update_stats()

        # Refresh gallery to show updated tag status
        self.gallery_tab.refresh_gallery()

        # Update status bar
        self.statusBar().showMessage("Batch interrogation complete", 5000)

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Image Interrogator",
            "Image Interrogator v1.0\n\n"
            "Batch image tagging tool using CLIP and Waifu Diffusion models.\n\n"
            "Features:\n"
            "• Batch interrogation with caching\n"
            "• Multiple model support\n"
            "• Tag-based organization\n"
            "• SQLite database for efficiency"
        )
    
    def closeEvent(self, event):
        """Handle application close."""
        # Cleanup
        if self.current_interrogator:
            self.current_interrogator.unload_model()
        
        if self.database:
            self.database.close()
        
        event.accept()
