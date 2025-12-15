"""Main application window for Image Interrogator."""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget,
                             QFileDialog, QMessageBox, QStatusBar, QLabel)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from pathlib import Path
from typing import Optional, Dict

from core import InterrogationDatabase, TagFilterSettings
from ui.tabs import InterrogationTab, GalleryTab, SettingsTab


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Interrogator - Batch Tagging Tool")
        self.setGeometry(100, 100, 1600, 900)
        
        # Core components
        self.database = InterrogationDatabase()
        self.tag_filters = TagFilterSettings()
        self.current_interrogator = None
        self.current_model_type = "WD"
        self.current_directory = None
        self.current_image = None

        # Workers
        self.interrogation_worker = None
        self.organization_worker = None

        # Model configs
        self.clip_config = {
            'clip_model': 'ViT-L-14/openai',
            'caption_model': None,
            'mode': 'best',
            'device': 'cuda'
        }
        self.wd_config = {
            'wd_model': 'SmilingWolf/wd-v1-4-moat-tagger-v2',
            'threshold': 0.35,
            'device': 'cuda'
        }
        
        # Setup UI
        self.setup_ui()
        self.setup_menubar()
        self.setup_connections()
        
        self.statusBar().showMessage("Ready - Select a directory to begin")
    
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
            tag_filters=self.tag_filters,
            parent=self
        )
        self.gallery_tab = GalleryTab(
            database=self.database,
            parent=self
        )
        self.settings_tab = SettingsTab(
            database=self.database,
            tag_filters=self.tag_filters,
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

    def setup_cross_tab_signals(self):
        """Connect signals between tabs for cross-tab communication."""
        # === Directory Changes ===
        # When directory is selected in Interrogation tab, update other tabs
        self.interrogation_tab.directory_changed.connect(self.gallery_tab.set_directory)
        self.interrogation_tab.directory_changed.connect(self.settings_tab.set_directory)
        self.interrogation_tab.directory_changed.connect(
            lambda dir, recursive: self.statusBar().showMessage(
                f"Directory: {dir}{' (recursive)' if recursive else ''}", 5000
            )
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
