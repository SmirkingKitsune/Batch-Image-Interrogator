"""Worker threads for background processing in PyQt6."""

import os
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from typing import List, Dict, Any, Optional
from core import InterrogationDatabase, hash_image_content, get_image_metadata, FileManager, TagFilterSettings
from core.base_interrogator import BaseInterrogator


class InterrogationWorker(QThread):
    """Worker thread for batch image interrogation."""
    
    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    result = pyqtSignal(str, dict)  # image_path, results
    error = pyqtSignal(str, str)  # image_path, error_message
    finished = pyqtSignal()
    
    def __init__(self, image_paths: List[Path], interrogator: BaseInterrogator,
                 database: InterrogationDatabase, write_files: bool = True,
                 overwrite_files: bool = False, tag_filters: Optional[TagFilterSettings] = None):
        super().__init__()
        self.image_paths = image_paths
        self.interrogator = interrogator
        self.database = database
        self.write_files = write_files
        self.overwrite_files = overwrite_files
        self.tag_filters = tag_filters
        self.is_cancelled = False
    
    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True
    
    def run(self):
        """Execute batch interrogation."""
        total = len(self.image_paths)
        model_id = None
        
        # Register model once
        try:
            model_id = self.database.register_model(
                self.interrogator.model_name,
                self.interrogator.get_model_type(),
                config=self.interrogator.get_config()
            )
        except Exception as e:
            self.error.emit("", f"Failed to register model: {e}")
            self.finished.emit()
            return
        
        for idx, image_path in enumerate(self.image_paths):
            if self.is_cancelled:
                break
            
            try:
                image_path_str = str(image_path)
                self.progress.emit(idx + 1, total, f"Processing: {image_path.name}")
                
                # Hash the image
                file_hash = hash_image_content(image_path_str)
                
                # Check cache first
                cached = self.database.get_interrogation(
                    file_hash, 
                    self.interrogator.model_name
                )
                
                if cached:
                    # Use cached results
                    results = cached
                    self.progress.emit(
                        idx + 1, total, 
                        f"Using cached: {image_path.name}"
                    )
                else:
                    # Interrogate image
                    results = self.interrogator.interrogate(image_path_str)
                    
                    # Register image and save to database
                    metadata = get_image_metadata(image_path_str)
                    image_id = self.database.register_image(
                        image_path_str,
                        file_hash,
                        metadata['width'],
                        metadata['height'],
                        metadata['file_size']
                    )
                    
                    self.database.save_interrogation(
                        image_id,
                        model_id,
                        results['tags'],
                        results.get('confidence_scores'),
                        results.get('raw_output')
                    )
                
                # Write to text file if requested
                if self.write_files:
                    # Apply tag filters if configured
                    tags_to_write = results['tags']
                    if self.tag_filters:
                        confidence_scores = results.get('confidence_scores')

                        # Use confidence-based filtering if scores are available
                        if confidence_scores is not None:
                            # Get threshold from interrogator config
                            threshold = self.interrogator.get_config().get('threshold', 0.35)
                            tags_to_write, _ = self.tag_filters.filter_tags_with_confidence(
                                tags_to_write,
                                confidence_scores,
                                threshold
                            )
                        else:
                            # No confidence scores (CLIP), use simple filtering
                            tags_to_write = self.tag_filters.apply_filters(tags_to_write)

                    # Check if file exists and handle overwrite
                    txt_path = FileManager.get_text_file_path(image_path)
                    if not txt_path.exists() or self.overwrite_files:
                        FileManager.write_tags_to_file(
                            image_path,
                            tags_to_write,
                            overwrite=self.overwrite_files
                        )
                
                # Emit result
                self.result.emit(image_path_str, results)
                
            except Exception as e:
                self.error.emit(str(image_path), str(e))
        
        self.finished.emit()


class OrganizationWorker(QThread):
    """Worker thread for organizing images by tags."""
    
    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    moved = pyqtSignal(str, str)  # source_path, destination_path
    error = pyqtSignal(str, str)  # image_path, error_message
    finished = pyqtSignal(int)  # total_moved
    
    def __init__(self, image_paths: List[Path], tag_criteria: List[str],
                 target_subdir: str, match_mode: str = 'any', move_text: bool = True):
        super().__init__()
        self.image_paths = image_paths
        self.tag_criteria = tag_criteria
        self.target_subdir = target_subdir
        self.match_mode = match_mode
        self.move_text = move_text
        self.is_cancelled = False
    
    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True
    
    def run(self):
        """Execute batch organization."""
        total = len(self.image_paths)
        moved_count = 0
        
        for idx, image_path in enumerate(self.image_paths):
            if self.is_cancelled:
                break
            
            try:
                self.progress.emit(idx + 1, total, f"Checking: {image_path.name}")
                
                # Try to organize
                was_moved = FileManager.organize_by_tags(
                    image_path,
                    self.tag_criteria,
                    self.target_subdir,
                    self.move_text,
                    self.match_mode
                )
                
                if was_moved:
                    moved_count += 1
                    dest_path = str(image_path.parent / self.target_subdir / image_path.name)
                    self.moved.emit(str(image_path), dest_path)
                
            except Exception as e:
                self.error.emit(str(image_path), str(e))
        
        self.finished.emit(moved_count)


class DirectoryLoadWorker(QThread):
    """Worker thread for scanning directories for images without blocking UI."""

    # Signals
    progress = pyqtSignal(int, str)   # count_so_far, current_file
    finished = pyqtSignal(list)        # complete list of image paths (strings)
    error = pyqtSignal(str)            # error_message

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

    def __init__(self, directory: str, recursive: bool = False):
        super().__init__()
        self.directory = directory
        self.recursive = recursive
        self.is_cancelled = False

    def cancel(self):
        """Cancel the directory scan."""
        self.is_cancelled = True

    def run(self):
        """Execute directory scan using os.scandir for cancellable iteration."""
        try:
            dir_path = Path(self.directory)
            if not dir_path.exists() or not dir_path.is_dir():
                self.error.emit(f"Invalid directory: {self.directory}")
                self.finished.emit([])
                return

            image_paths = []
            count = 0

            if self.recursive:
                # Use os.walk for recursive scanning (cancellable)
                for root, dirs, files in os.walk(self.directory):
                    if self.is_cancelled:
                        break

                    for filename in files:
                        if self.is_cancelled:
                            break

                        ext = os.path.splitext(filename)[1].lower()
                        if ext in self.SUPPORTED_EXTENSIONS:
                            full_path = os.path.join(root, filename)
                            image_paths.append(full_path)
                            count += 1

                            # Emit progress every 50 files
                            if count % 50 == 0:
                                self.progress.emit(count, filename)
            else:
                # Use os.scandir for non-recursive scanning (faster than glob)
                with os.scandir(self.directory) as entries:
                    for entry in entries:
                        if self.is_cancelled:
                            break

                        if entry.is_file():
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in self.SUPPORTED_EXTENSIONS:
                                image_paths.append(entry.path)
                                count += 1

                                # Emit progress every 50 files
                                if count % 50 == 0:
                                    self.progress.emit(count, entry.name)

            if self.is_cancelled:
                self.finished.emit([])
                return

            # Sort and deduplicate
            image_paths = sorted(set(image_paths))

            # Emit final progress
            self.progress.emit(len(image_paths), "")
            self.finished.emit(image_paths)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([])
