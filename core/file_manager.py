"""File management for images and associated text files."""

from pathlib import Path
from typing import List, Set, Optional, Tuple
import os
import shutil

from core.atomic_write import write_text_atomic


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

        # One traversal with an extension test, rather than a glob pass per
        # extension per case. This also matches mixed-case suffixes such as
        # ".JpG", which globbing only the lower and upper forms misses.
        images = []
        if recursive:
            for root, _dirs, files in os.walk(dir_path):
                root_path = Path(root)
                for name in files:
                    if os.path.splitext(name)[1].lower() in FileManager.SUPPORTED_EXTENSIONS:
                        images.append(root_path / name)
        else:
            with os.scandir(dir_path) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    if os.path.splitext(entry.name)[1].lower() in FileManager.SUPPORTED_EXTENSIONS:
                        images.append(Path(entry.path))

        return sorted(set(images))  # Remove duplicates and sort
    
    @staticmethod
    def get_text_file_path(image_path: Path) -> Path:
        """Get the corresponding .txt file path for an image."""
        return image_path.with_suffix('.txt')

    @staticmethod
    def _read_tags_strict(image_path: Path) -> List[str]:
        """Read sidecar tags, raising on a genuine read error.

        Returns [] only for a missing or empty file. Unlike read_tags_from_file
        this does NOT swallow read errors, so callers that merge-then-overwrite
        do not silently discard existing content when the read fails.
        """
        txt_path = FileManager.get_text_file_path(image_path)
        if not txt_path.exists():
            return []
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return []
        return [tag.strip() for tag in content.split(',') if tag.strip()]

    @staticmethod
    def _dedupe_preserve_order(tags: List[str]) -> List[str]:
        """Case-insensitive de-dupe keeping first-seen spelling/order."""
        seen: Set[str] = set()
        result: List[str] = []
        for tag in tags:
            key = tag.strip().casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(tag.strip())
        return result

    @staticmethod
    def write_tags_to_file(image_path: Path, tags: List[str],
                          overwrite: bool = True, separator: str = ', '):
        """Write tags to text file (atomically)."""
        txt_path = FileManager.get_text_file_path(image_path)

        if overwrite:
            write_text_atomic(txt_path, separator.join(tags))
        else:
            # Append mode: merge with existing. Use a strict read so a failed
            # read raises instead of truncating the file to only the new tags.
            existing_tags = FileManager._read_tags_strict(image_path)
            # Case-insensitive de-dupe to match delete/organize semantics.
            all_tags = FileManager._dedupe_preserve_order(existing_tags + tags)
            write_text_atomic(txt_path, separator.join(all_tags))

    @staticmethod
    def delete_tags_from_file(
        image_path: Path,
        tags_to_delete: List[str],
        separator: str = ', ',
    ) -> Tuple[List[str], List[str]]:
        """Remove selected tags from an image sidecar text file.

        Returns:
            Tuple of (removed_tags, remaining_tags), preserving sidecar order and spelling.
        """
        existing_tags = FileManager._read_tags_strict(image_path)
        if not existing_tags or not tags_to_delete:
            return [], existing_tags

        delete_lookup = {
            tag.strip().casefold()
            for tag in tags_to_delete
            if isinstance(tag, str) and tag.strip()
        }
        if not delete_lookup:
            return [], existing_tags

        removed_tags = []
        remaining_tags = []
        for tag in existing_tags:
            if tag.strip().casefold() in delete_lookup:
                removed_tags.append(tag)
            else:
                remaining_tags.append(tag)

        if removed_tags:
            txt_path = FileManager.get_text_file_path(image_path)
            write_text_atomic(txt_path, separator.join(remaining_tags))

        return removed_tags, remaining_tags
    
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
    def organize_by_tags(
        image_path: Path,
        tag_criteria: List[str],
        target_subdir: str,
        move_text: bool = True,
        match_mode: str = 'any',
        target_root_dir: Optional[Path] = None,
    ) -> bool:
        """
        Move image (and text file) to subdirectory if it has matching tags.
        
        Args:
            image_path: Path to the image
            tag_criteria: List of tags to match against
            target_subdir: Subdirectory name to move to
            move_text: Whether to also move the text file
            match_mode: 'any' or 'all' - whether to match any tag or all tags
            target_root_dir: Optional root directory for destination. If None,
                destination is created under image_path.parent (legacy behavior).
            
        Returns:
            True if file was moved, False otherwise
        """
        return FileManager.organize_by_tags_destination(
            image_path=image_path,
            tag_criteria=tag_criteria,
            target_subdir=target_subdir,
            move_text=move_text,
            match_mode=match_mode,
            target_root_dir=target_root_dir,
        ) is not None

    @staticmethod
    def organize_by_tags_destination(
        image_path: Path,
        tag_criteria: List[str],
        target_subdir: str,
        move_text: bool = True,
        match_mode: str = 'any',
        target_root_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Move a matching image and return its actual destination.

        Returns None when the image does not match. Invalid organization rules
        raise ValueError instead of silently broadening their meaning.
        """
        image_path = Path(image_path)
        if match_mode not in {'any', 'all'}:
            raise ValueError("match_mode must be 'any' or 'all'")

        normalized_criteria = [
            tag.strip().casefold()
            for tag in tag_criteria
            if isinstance(tag, str) and tag.strip()
        ]
        if not normalized_criteria:
            return None

        existing_tags = {
            tag.casefold() for tag in FileManager._read_tags_strict(image_path)
        }
        matches = (
            any(tag in existing_tags for tag in normalized_criteria)
            if match_mode == 'any'
            else all(tag in existing_tags for tag in normalized_criteria)
        )
        if not matches:
            return None

        destination_root = Path(
            target_root_dir if target_root_dir is not None else image_path.parent
        ).resolve(strict=False)
        subdir_path = Path(target_subdir)
        if not target_subdir.strip() or subdir_path.is_absolute():
            raise ValueError("target_subdir must be a non-empty relative path")
        target_dir = (destination_root / subdir_path).resolve(strict=False)
        try:
            relative_target = target_dir.relative_to(destination_root)
        except ValueError as exc:
            raise ValueError("target_subdir must remain inside the destination root") from exc
        if not relative_target.parts:
            raise ValueError("target_subdir must name a child directory")
        target_dir.mkdir(parents=True, exist_ok=True)

        source_sidecar = FileManager.get_text_file_path(image_path)
        move_existing_sidecar = move_text and source_sidecar.exists()
        stem = image_path.stem
        suffix = image_path.suffix
        counter = 0
        while True:
            candidate_name = image_path.name if counter == 0 else f"{stem}_{counter}{suffix}"
            new_image_path = target_dir / candidate_name
            new_sidecar_path = FileManager.get_text_file_path(new_image_path)
            if not new_image_path.exists() and not (
                move_existing_sidecar and new_sidecar_path.exists()
            ):
                break
            counter += 1

        shutil.move(str(image_path), str(new_image_path))
        if move_existing_sidecar:
            try:
                shutil.move(str(source_sidecar), str(new_sidecar_path))
            except Exception as sidecar_error:
                try:
                    shutil.move(str(new_image_path), str(image_path))
                except Exception as rollback_error:
                    raise RuntimeError(
                        "Sidecar move failed and image rollback also failed; "
                        f"image remains at {new_image_path}, sidecar remains at {source_sidecar}"
                    ) from rollback_error
                raise sidecar_error

        return new_image_path
    
    @staticmethod
    def get_all_tags_in_directory(directory: str) -> Set[str]:
        """Get all unique tags used in text files in directory."""
        all_tags = set()
        images = FileManager.find_images(directory)
        
        for image_path in images:
            tags = FileManager.read_tags_from_file(image_path)
            all_tags.update(tags)
        
        return all_tags
