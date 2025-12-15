"""Main application window for Image Interrogator."""

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSplitter, QComboBox,
                             QProgressBar, QFileDialog, QMessageBox, QGroupBox,
                             QCheckBox, QStatusBar, QMenuBar, QMenu, QTextEdit,
                             QDockWidget)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QAction
from pathlib import Path
from typing import Optional, Dict

from core import InterrogationDatabase, FileManager
from interrogators import CLIPInterrogator, WDInterrogator
from ui.widgets import ImageGalleryWidget, TagEditorWidget, ResultsTableWidget
from ui.dialogs import ModelConfigDialog, CLIPConfigDialog, WDConfigDialog, OrganizeDialog
from ui.workers import InterrogationWorker, OrganizationWorker


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Interrogator - Batch Tagging Tool")
        self.setGeometry(100, 100, 1600, 900)
        
        # Core components
        self.database = InterrogationDatabase()
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
        """Setup the main UI components."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        left_panel.setMaximumWidth(350)
        
        # Center - Image gallery
        gallery_panel = self.create_gallery_panel()
        
        # Right - Results and tag editor
        right_panel = self.create_results_panel()
        right_panel.setMaximumWidth(500)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(gallery_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def create_control_panel(self) -> QWidget:
        """Create the left control panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Directory selection
        dir_group = QGroupBox("Directory")
        dir_layout = QVBoxLayout()
        
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setWordWrap(True)
        dir_layout.addWidget(self.dir_label)
        
        self.select_dir_button = QPushButton("Select Directory")
        dir_layout.addWidget(self.select_dir_button)
        
        self.refresh_button = QPushButton("Refresh Gallery")
        self.refresh_button.setEnabled(False)
        dir_layout.addWidget(self.refresh_button)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        model_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["WD Tagger", "CLIP"])
        model_layout.addWidget(self.model_type_combo)
        
        self.config_button = QPushButton("Configure Models")
        model_layout.addWidget(self.config_button)

        self.load_model_button = QPushButton("Load Model")
        model_layout.addWidget(self.load_model_button)

        self.unload_model_button = QPushButton("Unload Model")
        self.unload_model_button.setEnabled(False)
        model_layout.addWidget(self.unload_model_button)

        self.model_status_label = QLabel("Model: Not loaded")
        self.model_status_label.setWordWrap(True)
        model_layout.addWidget(self.model_status_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Batch operations
        batch_group = QGroupBox("Batch Operations")
        batch_layout = QVBoxLayout()
        
        self.write_files_check = QCheckBox("Write .txt files")
        self.write_files_check.setChecked(True)
        batch_layout.addWidget(self.write_files_check)
        
        self.overwrite_check = QCheckBox("Overwrite existing .txt")
        self.overwrite_check.setChecked(False)
        batch_layout.addWidget(self.overwrite_check)
        
        self.batch_interrogate_button = QPushButton("Batch Interrogate")
        self.batch_interrogate_button.setEnabled(False)
        batch_layout.addWidget(self.batch_interrogate_button)
        
        self.interrogate_single_button = QPushButton("Interrogate Selected")
        self.interrogate_single_button.setEnabled(False)
        batch_layout.addWidget(self.interrogate_single_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        batch_layout.addWidget(self.cancel_button)
        
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)
        
        # Organization
        org_group = QGroupBox("Organization")
        org_layout = QVBoxLayout()
        
        self.organize_button = QPushButton("Organize by Tags")
        self.organize_button.setEnabled(False)
        org_layout.addWidget(self.organize_button)
        
        org_group.setLayout(org_layout)
        layout.addWidget(org_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        layout.addStretch()
        
        # Database stats
        stats_label = QLabel("Database Statistics")
        layout.addWidget(stats_label)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
        layout.addWidget(self.stats_text)
        
        return widget
    
    def create_gallery_panel(self) -> QWidget:
        """Create the center gallery panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Image Gallery"))
        
        self.image_gallery = ImageGalleryWidget()
        layout.addWidget(self.image_gallery)
        
        return widget
    
    def create_results_panel(self) -> QWidget:
        """Create the right results panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Image preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setMinimumSize(QSize(300, 300))
        self.image_preview.setScaledContents(False)
        self.image_preview.setText("No image selected")
        preview_layout.addWidget(self.image_preview)
        
        self.image_info_label = QLabel("")
        self.image_info_label.setWordWrap(True)
        preview_layout.addWidget(self.image_info_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Results table
        results_group = QGroupBox("Interrogation Results")
        results_layout = QVBoxLayout()
        
        self.results_table = ResultsTableWidget()
        results_layout.addWidget(self.results_table)
        
        self.copy_tags_button = QPushButton("Copy Tags to Editor")
        results_layout.addWidget(self.copy_tags_button)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Tag editor
        tag_editor_group = QGroupBox("Tag Editor")
        tag_editor_layout = QVBoxLayout()
        
        self.tag_editor = TagEditorWidget()
        tag_editor_layout.addWidget(self.tag_editor)
        
        tag_editor_group.setLayout(tag_editor_layout)
        layout.addWidget(tag_editor_group)
        
        return widget
    
    def setup_menubar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        select_dir_action = QAction("Select Directory", self)
        select_dir_action.triggered.connect(self.select_directory)
        file_menu.addAction(select_dir_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Database menu
        db_menu = menubar.addMenu("Database")
        
        update_stats_action = QAction("Update Statistics", self)
        update_stats_action.triggered.connect(self.update_database_stats)
        db_menu.addAction(update_stats_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Directory
        self.select_dir_button.clicked.connect(self.select_directory)
        self.refresh_button.clicked.connect(self.refresh_gallery)
        
        # Model
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        self.config_button.clicked.connect(self.configure_model)
        self.load_model_button.clicked.connect(self.load_model)
        self.unload_model_button.clicked.connect(self.unload_model)
        
        # Batch operations
        self.batch_interrogate_button.clicked.connect(self.batch_interrogate)
        self.interrogate_single_button.clicked.connect(self.interrogate_single)
        self.cancel_button.clicked.connect(self.cancel_operation)
        self.organize_button.clicked.connect(self.organize_images)
        
        # Gallery
        self.image_gallery.image_selected.connect(self.on_image_selected)
        
        # Results and tags
        self.copy_tags_button.clicked.connect(self.copy_tags_to_editor)
        self.tag_editor.tags_changed.connect(self.save_tags)
    
    def select_directory(self):
        """Open directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Image Directory",
            str(Path.home())
        )
        
        if directory:
            self.current_directory = Path(directory)
            self.dir_label.setText(str(self.current_directory))
            self.refresh_gallery()
            self.refresh_button.setEnabled(True)
            self.organize_button.setEnabled(True)
            
            # Update buttons state
            if self.current_interrogator and self.current_interrogator.is_loaded:
                self.batch_interrogate_button.setEnabled(True)
    
    def refresh_gallery(self):
        """Refresh the image gallery."""
        if not self.current_directory:
            return
        
        self.image_gallery.clear_gallery()
        images = FileManager.find_images(str(self.current_directory))
        
        for image_path in images:
            has_tags = FileManager.has_text_file(image_path)
            self.image_gallery.add_image(str(image_path), has_tags)
        
        self.statusBar().showMessage(f"Loaded {len(images)} images")
        self.update_database_stats()
    
    def on_model_type_changed(self, model_type: str):
        """Handle model type change."""
        if model_type == "CLIP":
            self.current_model_type = "CLIP"
        else:
            self.current_model_type = "WD"
        
        # Update model status
        if self.current_interrogator and self.current_interrogator.is_loaded:
            if self.current_interrogator.get_model_type() != self.current_model_type:
                self.current_interrogator.unload_model()
                self.current_interrogator = None
                self.model_status_label.setText("Model: Not loaded")
                self.batch_interrogate_button.setEnabled(False)
                self.interrogate_single_button.setEnabled(False)
    
    def configure_model(self):
        """Open unified model configuration dialog."""
        dialog = ModelConfigDialog(self, self.clip_config, self.wd_config)
        if dialog.exec():
            configs = dialog.get_config()
            self.clip_config = configs['clip']
            self.wd_config = configs['wd']
            QMessageBox.information(self, "Success", "Model configurations updated")
    
    def load_model(self):
        """Load the selected model."""
        try:
            self.statusBar().showMessage("Loading model...")
            self.load_model_button.setEnabled(False)

            # Unload existing model
            if self.current_interrogator:
                self.current_interrogator.unload_model()

            # Create and load new interrogator
            if self.current_model_type == "CLIP":
                clip_model = self.clip_config.get('clip_model', 'ViT-L-14/openai')
                self.current_interrogator = CLIPInterrogator(model_name=clip_model)

                # Build load parameters
                load_params = {
                    'mode': self.clip_config.get('mode', 'best'),
                    'device': self.clip_config.get('device', 'cuda')
                }

                # Add caption model if specified
                caption_model = self.clip_config.get('caption_model')
                if caption_model:
                    load_params['caption_model'] = caption_model

                self.current_interrogator.load_model(**load_params)

            else:
                wd_model = self.wd_config.get('wd_model', 'SmilingWolf/wd-v1-4-moat-tagger-v2')
                self.current_interrogator = WDInterrogator(model_name=wd_model)
                self.current_interrogator.load_model(
                    threshold=self.wd_config.get('threshold', 0.35),
                    device=self.wd_config.get('device', 'cuda')
                )

            # Update status
            model_info = f"Model: {self.current_model_type} - {self.current_interrogator.model_name}"
            if self.current_model_type == "CLIP" and self.clip_config.get('caption_model'):
                model_info += f"\nCaption: {self.clip_config['caption_model']}"
            model_info += "\nLoaded"

            self.model_status_label.setText(model_info)
            self.statusBar().showMessage("Model loaded successfully")

            # Enable buttons
            if self.current_directory:
                self.batch_interrogate_button.setEnabled(True)
            self.interrogate_single_button.setEnabled(True)
            self.unload_model_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            self.model_status_label.setText("Model: Load failed")
        finally:
            self.load_model_button.setEnabled(True)

    def unload_model(self):
        """Unload the current model from memory."""
        if self.current_interrogator:
            try:
                self.current_interrogator.unload_model()
                self.current_interrogator = None
                self.model_status_label.setText("Model: Not loaded")
                self.statusBar().showMessage("Model unloaded successfully")

                # Disable buttons
                self.batch_interrogate_button.setEnabled(False)
                self.interrogate_single_button.setEnabled(False)
                self.unload_model_button.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to unload model:\n{str(e)}")
        else:
            QMessageBox.information(self, "Info", "No model is currently loaded")

    def batch_interrogate(self):
        """Start batch interrogation."""
        if not self.current_directory or not self.current_interrogator:
            return
        
        # Get all images
        images = FileManager.find_images(str(self.current_directory))
        if not images:
            QMessageBox.information(self, "Info", "No images found in directory")
            return
        
        # Confirm
        reply = QMessageBox.question(
            self,
            "Confirm Batch Interrogation",
            f"Interrogate {len(images)} images?\nThis may take some time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Start worker
        self.interrogation_worker = InterrogationWorker(
            images,
            self.current_interrogator,
            self.database,
            self.write_files_check.isChecked(),
            self.overwrite_check.isChecked()
        )
        
        # Connect signals
        self.interrogation_worker.progress.connect(self.on_interrogation_progress)
        self.interrogation_worker.result.connect(self.on_interrogation_result)
        self.interrogation_worker.error.connect(self.on_interrogation_error)
        self.interrogation_worker.finished.connect(self.on_interrogation_finished)
        
        # Update UI
        self.batch_interrogate_button.setEnabled(False)
        self.interrogate_single_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setMaximum(len(images))
        
        # Start
        self.interrogation_worker.start()
    
    def interrogate_single(self):
        """Interrogate currently selected image."""
        if not self.current_image or not self.current_interrogator:
            return
        
        try:
            self.statusBar().showMessage(f"Interrogating {Path(self.current_image).name}...")
            
            # Run interrogation
            results = self.current_interrogator.interrogate(self.current_image)
            
            # Save to database
            from core.hashing import hash_image_content, get_image_metadata
            file_hash = hash_image_content(self.current_image)
            metadata = get_image_metadata(self.current_image)
            
            image_id = self.database.register_image(
                self.current_image,
                file_hash,
                metadata['width'],
                metadata['height'],
                metadata['file_size']
            )
            
            model_id = self.database.register_model(
                self.current_interrogator.model_name,
                self.current_interrogator.get_model_type(),
                config=self.current_interrogator.get_config()
            )
            
            self.database.save_interrogation(
                image_id,
                model_id,
                results['tags'],
                results.get('confidence_scores'),
                results.get('raw_output')
            )
            
            # Update display
            results['model_name'] = self.current_interrogator.model_name
            self.results_table.set_results(results)
            
            # Write file if requested
            if self.write_files_check.isChecked():
                FileManager.write_tags_to_file(
                    Path(self.current_image),
                    results['tags'],
                    overwrite=self.overwrite_check.isChecked()
                )
                self.image_gallery.update_image_status(self.current_image, True)
            
            self.statusBar().showMessage("Interrogation complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Interrogation failed:\n{str(e)}")
    
    def cancel_operation(self):
        """Cancel ongoing operation."""
        if self.interrogation_worker:
            self.interrogation_worker.cancel()
            self.progress_label.setText("Cancelling...")
        
        if self.organization_worker:
            self.organization_worker.cancel()
            self.progress_label.setText("Cancelling...")
    
    def on_interrogation_progress(self, current: int, total: int, message: str):
        """Handle interrogation progress update."""
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total}: {message}")
    
    def on_interrogation_result(self, image_path: str, results: Dict):
        """Handle interrogation result."""
        # Update gallery status
        self.image_gallery.update_image_status(image_path, True)
    
    def on_interrogation_error(self, image_path: str, error: str):
        """Handle interrogation error."""
        print(f"Error interrogating {image_path}: {error}")
    
    def on_interrogation_finished(self):
        """Handle interrogation completion."""
        self.progress_label.setText("Batch interrogation complete")
        self.batch_interrogate_button.setEnabled(True)
        self.interrogate_single_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.update_database_stats()
        QMessageBox.information(self, "Complete", "Batch interrogation finished")
    
    def organize_images(self):
        """Organize images by tags."""
        if not self.current_directory:
            return
        
        # Get all unique tags in directory
        all_tags = FileManager.get_all_tags_in_directory(str(self.current_directory))
        
        dialog = OrganizeDialog(self, list(all_tags))
        if not dialog.exec():
            return
        
        config = dialog.get_config()
        
        if not config['tags']:
            QMessageBox.warning(self, "Warning", "No tags specified")
            return
        
        # Get images
        images = FileManager.find_images(str(self.current_directory))
        
        # Start worker
        self.organization_worker = OrganizationWorker(
            images,
            config['tags'],
            config['subdir'],
            config['match_mode'],
            config['move_text']
        )
        
        # Connect signals
        self.organization_worker.progress.connect(self.on_organization_progress)
        self.organization_worker.moved.connect(self.on_image_moved)
        self.organization_worker.error.connect(self.on_organization_error)
        self.organization_worker.finished.connect(self.on_organization_finished)
        
        # Update UI
        self.organize_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setMaximum(len(images))
        
        # Start
        self.organization_worker.start()
    
    def on_organization_progress(self, current: int, total: int, message: str):
        """Handle organization progress."""
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total}: {message}")
    
    def on_image_moved(self, source: str, dest: str):
        """Handle image moved."""
        pass
    
    def on_organization_error(self, image_path: str, error: str):
        """Handle organization error."""
        print(f"Error organizing {image_path}: {error}")
    
    def on_organization_finished(self, moved_count: int):
        """Handle organization completion."""
        self.progress_label.setText(f"Organization complete - moved {moved_count} images")
        self.organize_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.refresh_gallery()
        QMessageBox.information(self, "Complete", f"Moved {moved_count} images")
    
    def on_image_selected(self, image_path: str):
        """Handle image selection from gallery."""
        self.current_image = image_path
        
        # Load and display image
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_preview.setPixmap(scaled_pixmap)
        
        # Update info
        path = Path(image_path)
        self.image_info_label.setText(f"File: {path.name}\nSize: {path.stat().st_size / 1024:.1f} KB")
        
        # Load existing tags
        existing_tags = FileManager.read_tags_from_file(path)
        self.tag_editor.set_tags(existing_tags)
        
        # Load cached interrogations
        from core.hashing import hash_image_content
        try:
            file_hash = hash_image_content(image_path)
            cached_results = self.database.get_all_interrogations_for_image(file_hash)
            
            if cached_results:
                # Display most recent
                latest = cached_results[0]
                latest['model_name'] = latest['model_name']
                self.results_table.set_results(latest)
            else:
                self.results_table.clear_results()
        except:
            self.results_table.clear_results()
    
    def copy_tags_to_editor(self):
        """Copy tags from results table to editor."""
        tags = self.results_table.get_all_tags()
        if tags:
            self.tag_editor.set_tags(tags)
    
    def save_tags(self, tags: list):
        """Save edited tags to file."""
        if not self.current_image:
            return
        
        try:
            FileManager.write_tags_to_file(Path(self.current_image), tags, overwrite=True)
            self.image_gallery.update_image_status(self.current_image, len(tags) > 0)
            self.statusBar().showMessage("Tags saved")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save tags:\n{str(e)}")
    
    def update_database_stats(self):
        """Update database statistics display."""
        stats = self.database.get_statistics()
        text = (
            f"Images: {stats['total_images']}\n"
            f"Interrogations: {stats['total_interrogations']}\n"
            f"Models used: {stats['unique_models_used']}"
        )
        self.stats_text.setText(text)
    
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
