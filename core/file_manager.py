"""File management for images and associated text files."""

from pathlib import Path
from typing import List, Set, Optional
import shutil


class FileManager:
    """Manages image files and associated text files."""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    
    @staticmethod
    def find_images(directory: str, recursive: bool = False) -> List[Path]:
        """
        Find all supported images in directory.

        Args:
            directory: Directory path to search
            recursive: If True, search subdirectories recursively

        Returns:
            Sorted list of image paths
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        images = []
        glob_pattern = '**/*' if recursive else '*'

        for ext in FileManager.SUPPORTED_EXTENSIONS:
            images.extend(dir_path.glob(f'{glob_pattern}{ext}'))
            images.extend(dir_path.glob(f'{glob_pattern}{ext.upper()}'))

        return sorted(set(images))  # Remove duplicates and sort
    
    @staticmethod
    def get_text_file_path(image_path: Path) -> Path:
        """Get the corresponding .txt file path for an image."""
        return image_path.with_suffix('.txt')
    
    @staticmethod
    def write_tags_to_file(image_path: Path, tags: List[str], 
                          overwrite: bool = True, separator: str = ', '):
        """Write tags to text file."""
        txt_path = FileManager.get_text_file_path(image_path)
        
        if overwrite:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(separator.join(tags))
        else:
            # Append mode
            existing_tags = FileManager.read_tags_from_file(image_path)
            all_tags = list(dict.fromkeys(existing_tags + tags))  # Remove duplicates, preserve order
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(separator.join(all_tags))
    
    @staticmethod
    def read_tags_from_file(image_path: Path) -> List[str]:
        """Read existing tags from text file."""
        txt_path = FileManager.get_text_file_path(image_path)
        
        if not txt_path.exists():
            return []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return []
                return [tag.strip() for tag in content.split(',') if tag.strip()]
        except Exception:
            return []
    
    @staticmethod
    def has_text_file(image_path: Path) -> bool:
        """Check if image has an associated text file."""
        return FileManager.get_text_file_path(image_path).exists()
    
    @staticmethod
    def organize_by_tags(image_path: Path, tag_criteria: List[str], 
                        target_subdir: str, move_text: bool = True,
                        match_mode: str = 'any') -> bool:
        """
        Move image (and text file) to subdirectory if it has matching tags.
        
        Args:
            image_path: Path to the image
            tag_criteria: List of tags to match against
            target_subdir: Subdirectory name to move to
            move_text: Whether to also move the text file
            match_mode: 'any' or 'all' - whether to match any tag or all tags
            
        Returns:
            True if file was moved, False otherwise
        """
        existing_tags = FileManager.read_tags_from_file(image_path)
        existing_tags_lower = [tag.lower() for tag in existing_tags]
        tag_criteria_lower = [tag.lower() for tag in tag_criteria]
        
        # Check matching
        if match_mode == 'any':
            matches = any(tag in existing_tags_lower for tag in tag_criteria_lower)
        else:  # 'all'
            matches = all(tag in existing_tags_lower for tag in tag_criteria_lower)
        
        if not matches:
            return False
        
        # Create target directory
        target_dir = image_path.parent / target_subdir
        target_dir.mkdir(exist_ok=True)
        
        # Move image
        new_image_path = target_dir / image_path.name
        if new_image_path.exists():
            # Handle duplicate names
            stem = image_path.stem
            suffix = image_path.suffix
            counter = 1
            while new_image_path.exists():
                new_image_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        
        shutil.move(str(image_path), str(new_image_path))
        
        # Move text file if exists
        if move_text:
            txt_path = FileManager.get_text_file_path(image_path)
            if txt_path.exists():
                new_txt_path = FileManager.get_text_file_path(new_image_path)
                shutil.move(str(txt_path), str(new_txt_path))
        
        return True
    
    @staticmethod
    def get_all_tags_in_directory(directory: str) -> Set[str]:
        """Get all unique tags used in text files in directory."""
        all_tags = set()
        images = FileManager.find_images(directory)
        
        for image_path in images:
            tags = FileManager.read_tags_from_file(image_path)
            all_tags.update(tags)
        
        return all_tags
