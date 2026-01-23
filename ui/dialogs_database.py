"""Database-related dialogs for handling locked database scenarios."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QGroupBox, QCheckBox,
    QMessageBox, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from typing import Dict, List, Any, Optional
from datetime import datetime


class DatabaseBusyDialog(QDialog):
    """
    Dialog shown when the database is locked by another process.

    Provides options to:
    - Wait and retry (continues attempting to access database)
    - Queue operation and continue (queues for later, if queueable)
    - Force close (abort the operation)
    """

    # Signals for user actions
    retry_requested = pyqtSignal()
    queue_and_continue = pyqtSignal()
    force_close_requested = pyqtSignal()

    def __init__(
        self,
        operation: str,
        params: Dict[str, Any],
        retry_count: int,
        queued_operations: Optional[List[Dict]] = None,
        is_queueable: bool = True,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the dialog.

        Args:
            operation: Name of the database operation being attempted.
            params: Parameters for the operation (for display purposes).
            retry_count: Number of retries attempted so far.
            queued_operations: List of already queued operations (for display).
            is_queueable: Whether this operation can be queued.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.operation = operation
        self.params = params
        self.retry_count = retry_count
        self.queued_operations = queued_operations or []
        self.is_queueable = is_queueable

        self._start_time = datetime.now()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_wait_time)

        self.setWindowTitle("Database Locked")
        self.setModal(True)
        self.setMinimumWidth(450)
        self.setMinimumHeight(300)

        self._setup_ui()
        self._timer.start(1000)  # Update every second

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header with icon and message
        header_layout = QHBoxLayout()
        icon_label = QLabel("\u23f3")  # Hourglass emoji
        icon_label.setFont(QFont("Segoe UI Emoji", 24))
        header_layout.addWidget(icon_label)

        header_text = QLabel(
            "<b>Database Locked</b><br>"
            "Another process is using the database.<br>"
            "Waiting for access..."
        )
        header_text.setWordWrap(True)
        header_layout.addWidget(header_text, 1)
        layout.addLayout(header_layout)

        # Status info
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.operation_label = QLabel(f"<b>Current Operation:</b> {self._format_operation()}")
        status_layout.addWidget(self.operation_label)

        self.wait_time_label = QLabel("<b>Time Waiting:</b> 00:00")
        status_layout.addWidget(self.wait_time_label)

        self.retry_label = QLabel(f"<b>Retry Attempt:</b> {self.retry_count}")
        status_layout.addWidget(self.retry_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Queued operations list (if any)
        if self.queued_operations:
            queue_group = QGroupBox(f"Queued Operations ({len(self.queued_operations)} pending)")
            queue_layout = QVBoxLayout()

            self.queue_list = QListWidget()
            self.queue_list.setMaximumHeight(100)
            for op in self.queued_operations[:10]:  # Show max 10
                summary = op.get('summary', op.get('operation', 'Unknown'))
                item = QListWidgetItem(f"\u2022 {summary}")
                self.queue_list.addItem(item)

            if len(self.queued_operations) > 10:
                item = QListWidgetItem(f"... and {len(self.queued_operations) - 10} more")
                item.setForeground(Qt.GlobalColor.gray)
                self.queue_list.addItem(item)

            queue_layout.addWidget(self.queue_list)
            queue_group.setLayout(queue_layout)
            layout.addWidget(queue_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.queue_button = QPushButton("Queue && Continue")
        self.queue_button.setToolTip("Queue this operation for later and continue processing")
        self.queue_button.clicked.connect(self._on_queue_clicked)
        self.queue_button.setEnabled(self.is_queueable)
        button_layout.addWidget(self.queue_button)

        self.retry_button = QPushButton("Retry Now")
        self.retry_button.setToolTip("Try to access the database again immediately")
        self.retry_button.clicked.connect(self._on_retry_clicked)
        button_layout.addWidget(self.retry_button)

        self.force_close_button = QPushButton("Force Close")
        self.force_close_button.setToolTip("Abort this operation (may lose data)")
        self.force_close_button.clicked.connect(self._on_force_close_clicked)
        button_layout.addWidget(self.force_close_button)

        layout.addLayout(button_layout)

        # Warning label
        warning_label = QLabel(
            "\u26a0 <i>Force Close may result in data loss for this operation</i>"
        )
        warning_label.setStyleSheet("QLabel { color: #856404; }")
        warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(warning_label)

        # Disable queue button if not queueable
        if not self.is_queueable:
            self.queue_button.setToolTip(
                "This operation cannot be queued because it returns a required value"
            )

    def _format_operation(self) -> str:
        """Format the operation name for display."""
        # Make operation name more readable
        op_display = self.operation.replace('_', ' ').title()

        # Add context from params if available
        if self.operation == 'save_interrogation':
            tag_count = len(self.params.get('tags', []))
            return f"{op_display} ({tag_count} tags)"
        elif self.operation == 'vacuum':
            return "Optimize Database"
        else:
            return op_display

    def _update_wait_time(self):
        """Update the wait time display."""
        elapsed = datetime.now() - self._start_time
        minutes, seconds = divmod(int(elapsed.total_seconds()), 60)
        self.wait_time_label.setText(f"<b>Time Waiting:</b> {minutes:02d}:{seconds:02d}")

    def update_retry_count(self, count: int):
        """Update the displayed retry count."""
        self.retry_count = count
        self.retry_label.setText(f"<b>Retry Attempt:</b> {count}")

    def _on_queue_clicked(self):
        """Handle Queue & Continue button click."""
        self._timer.stop()
        self.queue_and_continue.emit()
        self.accept()

    def _on_retry_clicked(self):
        """Handle Retry Now button click."""
        self.retry_requested.emit()
        # Don't close dialog - let the caller handle it

    def _on_force_close_clicked(self):
        """Handle Force Close button click."""
        # Show confirmation dialog
        confirm = QMessageBox.warning(
            self,
            "Confirm Force Close",
            "Are you sure you want to force close?\n\n"
            "The current operation will be aborted and its data will be lost.\n"
            "Any already-queued operations will be preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if confirm == QMessageBox.StandardButton.Yes:
            self._timer.stop()
            self.force_close_requested.emit()
            self.reject()

    def closeEvent(self, event):
        """Handle dialog close (X button)."""
        # Treat closing as force close
        self._on_force_close_clicked()
        event.ignore()  # Let _on_force_close_clicked handle accept/reject


class QueueProcessingDialog(QDialog):
    """Dialog shown when processing queued operations at startup."""

    processing_requested = pyqtSignal()
    skip_requested = pyqtSignal()
    clear_requested = pyqtSignal()

    def __init__(
        self,
        pending_count: int,
        operations: List[Dict],
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the dialog.

        Args:
            pending_count: Number of pending operations.
            operations: List of operation details for display.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.pending_count = pending_count
        self.operations = operations

        self.setWindowTitle("Pending Database Operations")
        self.setModal(True)
        self.setMinimumWidth(450)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header_label = QLabel(
            f"<b>{self.pending_count} database operation(s) were queued</b><br>"
            "from a previous session and are waiting to be processed."
        )
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        # Operations list
        if self.operations:
            ops_group = QGroupBox("Pending Operations")
            ops_layout = QVBoxLayout()

            self.ops_list = QListWidget()
            self.ops_list.setMaximumHeight(150)

            for op in self.operations[:20]:  # Show max 20
                summary = op.get('summary', op.get('operation', 'Unknown'))
                timestamp = op.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime('%Y-%m-%d %H:%M')
                    except ValueError:
                        time_str = timestamp
                else:
                    time_str = 'Unknown time'

                item = QListWidgetItem(f"\u2022 {summary} ({time_str})")
                self.ops_list.addItem(item)

            if len(self.operations) > 20:
                item = QListWidgetItem(f"... and {len(self.operations) - 20} more")
                item.setForeground(Qt.GlobalColor.gray)
                self.ops_list.addItem(item)

            ops_layout.addWidget(self.ops_list)
            ops_group.setLayout(ops_layout)
            layout.addWidget(ops_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.process_button = QPushButton("Process Now")
        self.process_button.setToolTip("Attempt to save all queued operations to the database")
        self.process_button.clicked.connect(self._on_process_clicked)
        button_layout.addWidget(self.process_button)

        self.skip_button = QPushButton("Skip for Now")
        self.skip_button.setToolTip("Keep in queue and try again later")
        self.skip_button.clicked.connect(self._on_skip_clicked)
        button_layout.addWidget(self.skip_button)

        self.clear_button = QPushButton("Clear Queue")
        self.clear_button.setToolTip("Discard all queued operations (data loss)")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        button_layout.addWidget(self.clear_button)

        layout.addLayout(button_layout)

    def _on_process_clicked(self):
        """Handle Process Now button click."""
        self.processing_requested.emit()
        self.accept()

    def _on_skip_clicked(self):
        """Handle Skip button click."""
        self.skip_requested.emit()
        self.accept()

    def _on_clear_clicked(self):
        """Handle Clear Queue button click."""
        confirm = QMessageBox.warning(
            self,
            "Confirm Clear Queue",
            f"Are you sure you want to clear all {self.pending_count} queued operations?\n\n"
            "This will permanently discard the data and cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if confirm == QMessageBox.StandardButton.Yes:
            self.clear_requested.emit()
            self.accept()
