"""Settings Tab - Database statistics and application settings."""

from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
                             QGroupBox, QLabel, QPushButton, QFormLayout,
                             QLineEdit, QComboBox, QCheckBox, QRadioButton,
                             QSlider, QProgressBar, QMessageBox, QTextEdit,
                             QTabWidget, QListWidget, QListWidgetItem,
                             QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path
from typing import Dict, List

from core import FileManager, TagFilterSettings


class DatabaseStatsWidget(QWidget):
    """Widget for displaying database statistics."""

    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Stats display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(120)
        layout.addWidget(self.stats_text)

        # Buttons
        button_layout = QHBoxLayout()

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.update_stats)
        button_layout.addWidget(self.refresh_button)

        self.vacuum_button = QPushButton("Vacuum Database")
        self.vacuum_button.clicked.connect(self.vacuum_database)
        button_layout.addWidget(self.vacuum_button)

        button_layout.addStretch()

        layout.addLayout(button_layout)

        # Initial update
        self.update_stats()

    def update_stats(self):
        """Update database statistics display."""
        stats = self.database.get_statistics()
        db_location = self.database.get_db_location()
        db_mode = "Local (per-directory)" if self.database.use_local_db else "Global (shared)"

        text = (
            f"Total Images: {stats['total_images']}\n"
            f"Total Interrogations: {stats['total_interrogations']}\n"
            f"Unique Models Used: {stats['unique_models_used']}\n"
            f"\n"
            f"Database Mode: {db_mode}\n"
            f"Location: {db_location}\n"
            f"\n"
            f"Performance:\n"
            f"• All interrogation results are cached\n"
            f"• Re-interrogating cached images is instant"
        )
        self.stats_text.setText(text)

    def vacuum_database(self):
        """Vacuum the database to reclaim space."""
        try:
            # Get database file size before
            db_path = Path(self.database.db_path)
            size_before = db_path.stat().st_size / 1024 / 1024  # MB

            # Vacuum
            self.database.vacuum()

            # Get size after
            size_after = db_path.stat().st_size / 1024 / 1024  # MB
            saved = size_before - size_after

            QMessageBox.information(
                self,
                "Database Vacuumed",
                f"Database optimized successfully!\n\n"
                f"Size before: {size_before:.2f} MB\n"
                f"Size after: {size_after:.2f} MB\n"
                f"Space saved: {saved:.2f} MB"
            )
            self.update_stats()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to vacuum database:\n{str(e)}")


class QueueStatusWidget(QWidget):
    """Widget for displaying and managing the database operation queue."""

    # Signals
    process_queue_requested = pyqtSignal()
    clear_queue_requested = pyqtSignal()

    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Status label
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Queue list
        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(100)
        self.queue_list.setVisible(False)
        layout.addWidget(self.queue_list)

        # Buttons
        button_layout = QHBoxLayout()

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.update_status)
        button_layout.addWidget(self.refresh_button)

        self.process_button = QPushButton("Process Queue")
        self.process_button.clicked.connect(self._on_process_clicked)
        self.process_button.setEnabled(False)
        button_layout.addWidget(self.process_button)

        self.clear_button = QPushButton("Clear Queue")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.clear_button.setEnabled(False)
        button_layout.addWidget(self.clear_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Initial update
        self.update_status()

    def update_status(self):
        """Update the queue status display."""
        status = self.database.get_queue_status()
        pending_count = status.get('pending_count', 0)
        operations = status.get('operations', [])

        if pending_count == 0:
            self.status_label.setText(
                "<b>Queue Status:</b> Empty\n\n"
                "No pending operations. Database writes are completing successfully."
            )
            self.queue_list.setVisible(False)
            self.process_button.setEnabled(False)
            self.clear_button.setEnabled(False)
        else:
            near_capacity = status.get('near_capacity', False)
            at_capacity = status.get('at_capacity', False)

            warning = ""
            if at_capacity:
                warning = "\n\n<font color='red'><b>Warning:</b> Queue is at capacity!</font>"
            elif near_capacity:
                warning = "\n\n<font color='orange'><b>Warning:</b> Queue is approaching capacity.</font>"

            self.status_label.setText(
                f"<b>Queue Status:</b> {pending_count} pending operation(s){warning}\n\n"
                f"These operations will be retried when the database becomes available."
            )

            # Populate queue list
            self.queue_list.clear()
            for op in operations[:10]:
                summary = op.get('summary', op.get('operation', 'Unknown'))
                timestamp = op.get('timestamp', '')
                retry_count = op.get('retry_count', 0)

                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime('%H:%M:%S')
                    except ValueError:
                        time_str = ''
                else:
                    time_str = ''

                item_text = f"\u2022 {summary}"
                if time_str:
                    item_text += f" ({time_str})"
                if retry_count > 0:
                    item_text += f" [retried {retry_count}x]"

                item = QListWidgetItem(item_text)
                self.queue_list.addItem(item)

            if len(operations) > 10:
                item = QListWidgetItem(f"... and {len(operations) - 10} more")
                item.setForeground(Qt.GlobalColor.gray)
                self.queue_list.addItem(item)

            self.queue_list.setVisible(True)
            self.process_button.setEnabled(True)
            self.clear_button.setEnabled(True)

    def _on_process_clicked(self):
        """Handle Process Queue button click."""
        self.process_queue_requested.emit()

    def _on_clear_clicked(self):
        """Handle Clear Queue button click."""
        status = self.database.get_queue_status()
        pending_count = status.get('pending_count', 0)

        reply = QMessageBox.warning(
            self,
            "Clear Queue",
            f"Are you sure you want to clear all {pending_count} queued operations?\n\n"
            "This will permanently discard the data and cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.clear_queue_requested.emit()


class TagFilterWidget(QWidget):
    """Widget for managing tag filters (remove, replace, keep)."""

    def __init__(self, tag_filters: TagFilterSettings, parent=None):
        super().__init__(parent)
        self.tag_filters = tag_filters
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)

        # Statistics
        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

        # Load current filters
        self.refresh_all()

    def refresh_all(self):
        """Refresh all filter displays."""
        self._refresh_remove_list()
        self._refresh_replace_table()
        self._refresh_keep_list()
        self._refresh_stats()

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

    def _refresh_stats(self):
        """Refresh the statistics display."""
        stats = self.tag_filters.get_statistics()
        self.stats_label.setText(
            f"Active Filters: {stats['remove_count']} removed | "
            f"{stats['replace_count']} replaced | "
            f"{stats['keep_count']} force-included"
        )

    # === Remove Filter Methods ===

    def _add_remove_tag(self):
        """Add a tag to the remove list."""
        tag = self.remove_input.text().strip()
        if tag:
            self.tag_filters.add_remove_tag(tag)
            self.remove_input.clear()
            self.refresh_all()

    def _delete_remove_tag(self):
        """Delete selected tag from remove list."""
        current_item = self.remove_list.currentItem()
        if current_item:
            self.tag_filters.remove_remove_tag(current_item.text())
            self.refresh_all()

    def _clear_remove_list(self):
        """Clear all tags from remove list."""
        reply = QMessageBox.question(
            self, "Clear Remove List",
            "Are you sure you want to clear all tags from the remove list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.tag_filters.clear_remove_list()
            self.refresh_all()

    # === Replace Filter Methods ===

    def _add_replace_rule(self):
        """Add a replacement rule."""
        original = self.replace_original_input.text().strip()
        replacement = self.replace_new_input.text().strip()
        if original and replacement:
            self.tag_filters.add_replace_rule(original, replacement)
            self.replace_original_input.clear()
            self.replace_new_input.clear()
            self.refresh_all()

    def _delete_replace_rule(self):
        """Delete selected replacement rule."""
        current_row = self.replace_table.currentRow()
        if current_row >= 0:
            original_item = self.replace_table.item(current_row, 0)
            if original_item:
                self.tag_filters.remove_replace_rule(original_item.text())
                self.refresh_all()

    def _clear_replace_dict(self):
        """Clear all replacement rules."""
        reply = QMessageBox.question(
            self, "Clear Replace Rules",
            "Are you sure you want to clear all replacement rules?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.tag_filters.clear_replace_dict()
            self.refresh_all()

    # === Keep Filter Methods ===

    def _add_keep_tag(self):
        """Add a tag to the keep list."""
        tag = self.keep_input.text().strip()
        if tag:
            self.tag_filters.add_keep_tag(tag)
            self.keep_input.clear()
            self.refresh_all()

    def _delete_keep_tag(self):
        """Delete selected tag from keep list."""
        current_item = self.keep_list.currentItem()
        if current_item:
            self.tag_filters.remove_keep_tag(current_item.text())
            self.refresh_all()

    def _clear_keep_list(self):
        """Clear all tags from keep list."""
        reply = QMessageBox.question(
            self, "Clear Keep List",
            "Are you sure you want to clear all tags from the keep list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.tag_filters.clear_keep_list()
            self.refresh_all()


class SettingsTab(QWidget):
    """Tab for database management and application settings."""

    # Signals
    settings_changed = pyqtSignal(dict)  # settings_dict

    def __init__(self, database, tag_filters, parent=None):
        """
        Initialize the Settings Tab.

        Args:
            database: InterrogationDatabase instance
            tag_filters: TagFilterSettings instance
            parent: Parent widget
        """
        super().__init__(parent)

        # Store shared state references
        self.database = database
        self.tag_filters = tag_filters
        self.current_directory = None

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Scroll area for all settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # Container widget for scroll area
        container = QWidget()
        layout = QVBoxLayout(container)

        # === Database Statistics ===
        stats_group = QGroupBox("Database Statistics")
        stats_layout = QVBoxLayout()
        self.stats_widget = DatabaseStatsWidget(self.database)
        stats_layout.addWidget(self.stats_widget)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # === Operation Queue Status ===
        queue_group = QGroupBox("Operation Queue")
        queue_layout = QVBoxLayout()
        self.queue_widget = QueueStatusWidget(self.database)
        self.queue_widget.process_queue_requested.connect(self._on_process_queue)
        self.queue_widget.clear_queue_requested.connect(self._on_clear_queue)
        queue_layout.addWidget(self.queue_widget)
        queue_group.setLayout(queue_layout)
        layout.addWidget(queue_group)

        # === Application Settings ===
        settings_group = QGroupBox("Application Settings")
        settings_layout = QVBoxLayout()

        # Database Mode
        db_mode_label = QLabel("<b>Database Mode:</b>")
        settings_layout.addWidget(db_mode_label)

        self.global_db_radio = QRadioButton("Global Database (shared across all directories)")
        self.global_db_radio.setChecked(not self.database.use_local_db)
        self.global_db_radio.toggled.connect(self._on_db_mode_changed)
        settings_layout.addWidget(self.global_db_radio)

        self.local_db_radio = QRadioButton("Local Database (saved in each image directory)")
        self.local_db_radio.setChecked(self.database.use_local_db)
        settings_layout.addWidget(self.local_db_radio)

        # Database mode description
        db_mode_desc = QLabel(
            "\n<b>Global:</b> One database for all images. Efficient if interrogating diverse images.\n"
            "<b>Local:</b> Separate database per directory. Keeps results with images for easy backup/sharing.\n"
            "\nNote: Changing mode will switch database when you select a new directory."
        )
        db_mode_desc.setWordWrap(True)
        db_mode_desc.setStyleSheet("QLabel { color: #666; font-size: 10pt; }")
        settings_layout.addWidget(db_mode_desc)

        # Performance Settings
        perf_label = QLabel("\n<b>Performance Settings:</b>")
        settings_layout.addWidget(perf_label)

        self.auto_unload_check = QCheckBox("Auto-unload model after batch interrogation")
        self.auto_unload_check.setChecked(True)
        self.auto_unload_check.setToolTip("Automatically unload the model from GPU memory after batch interrogation completes")
        settings_layout.addWidget(self.auto_unload_check)

        perf_desc = QLabel(
            "Auto-unloading frees GPU memory when not actively interrogating images."
        )
        perf_desc.setWordWrap(True)
        perf_desc.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        settings_layout.addWidget(perf_desc)

        # Future settings placeholder
        future_label = QLabel(
            "\n<b>Future Settings:</b>\n"
            "• Default model selection\n"
            "• Default device (CUDA/CPU)\n"
            "• Auto-write .txt files option\n"
            "• Thumbnail size\n"
            "• UI theme"
        )
        future_label.setWordWrap(True)
        future_label.setStyleSheet("QLabel { color: #999; font-size: 9pt; }")
        settings_layout.addWidget(future_label)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # === Database Management ===
        db_mgmt_group = QGroupBox("Database Management")
        db_mgmt_layout = QVBoxLayout()

        db_mgmt_info = QLabel(
            "Database management features:\n\n"
            "• Export database to JSON\n"
            "• Import database from JSON\n"
            "• Clear model history\n"
            "• Reset database\n\n"
            "These features will be added in future updates."
        )
        db_mgmt_info.setWordWrap(True)
        db_mgmt_layout.addWidget(db_mgmt_info)

        db_mgmt_group.setLayout(db_mgmt_layout)
        layout.addWidget(db_mgmt_group)

        # Add stretch to push everything to top
        layout.addStretch()

        # Set container as scroll widget
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def set_directory(self, directory: str, recursive: bool = False):
        """Set the current directory (for database switching)."""
        self.current_directory = directory

        # Switch database based on mode
        self.database.switch_to_directory(directory)

        # Update stats to show new database location
        self.update_stats()

    def update_stats(self):
        """Update database statistics and queue status."""
        self.stats_widget.update_stats()
        self.queue_widget.update_status()

    def get_auto_unload_enabled(self) -> bool:
        """Get whether auto-unload is enabled."""
        return self.auto_unload_check.isChecked()

    def _on_db_mode_changed(self, checked: bool):
        """Handle database mode radio button change."""
        if not checked:
            return  # Only respond to the newly checked radio button

        # Update database mode
        use_local = self.local_db_radio.isChecked()
        self.database.set_local_mode(use_local)

        # If we have a current directory, switch database immediately
        if self.current_directory:
            self.database.switch_to_directory(self.current_directory)
            self.update_stats()

            # Show info message
            mode_name = "Local (per-directory)" if use_local else "Global (shared)"
            QMessageBox.information(
                self,
                "Database Mode Changed",
                f"Database mode changed to: {mode_name}\n\n"
                f"New database location:\n{self.database.get_db_location()}\n\n"
                f"The database will switch when you select a new directory."
            )

    def _on_process_queue(self):
        """Handle process queue request from queue widget."""
        # Get the main window to trigger queue processing
        main_window = self.window()
        if hasattr(main_window, '_process_queue_now'):
            main_window._process_queue_now()

    def _on_clear_queue(self):
        """Handle clear queue request from queue widget."""
        count = self.database.clear_operation_queue()
        QMessageBox.information(
            self,
            "Queue Cleared",
            f"Cleared {count} queued operations."
        )
        self.queue_widget.update_status()

    def update_queue_status(self):
        """Update the queue status display."""
        self.queue_widget.update_status()
