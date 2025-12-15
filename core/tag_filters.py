"""Tag filtering system for customizing interrogation output."""

import json
from pathlib import Path
from typing import List, Dict, Set, Tuple


class TagFilterSettings:
    """Manages tag filtering rules for customizing output."""

    def __init__(self, settings_file: str = "tag_filters.json"):
        self.settings_file = Path(settings_file)

        # Filter lists
        self.remove_list: Set[str] = set()  # Tags to exclude
        self.replace_dict: Dict[str, str] = {}  # tag -> replacement
        self.keep_list: Set[str] = set()  # Tags to always include (ignore confidence)

        # Load saved settings
        self.load_settings()

    def load_settings(self):
        """Load filter settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.remove_list = set(data.get('remove_list', []))
                    self.replace_dict = data.get('replace_dict', {})
                    self.keep_list = set(data.get('keep_list', []))
            except Exception as e:
                print(f"Error loading tag filter settings: {e}")

    def save_settings(self):
        """Save filter settings to file."""
        try:
            data = {
                'remove_list': list(self.remove_list),
                'replace_dict': self.replace_dict,
                'keep_list': list(self.keep_list)
            }
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving tag filter settings: {e}")

    # === Remove Filter (Blacklist) ===

    def add_remove_tag(self, tag: str):
        """Add a tag to the remove list."""
        self.remove_list.add(tag.strip().lower())
        self.save_settings()

    def remove_remove_tag(self, tag: str):
        """Remove a tag from the remove list."""
        self.remove_list.discard(tag.strip().lower())
        self.save_settings()

    def clear_remove_list(self):
        """Clear all tags from remove list."""
        self.remove_list.clear()
        self.save_settings()

    # === Replace Filter ===

    def add_replace_rule(self, original: str, replacement: str):
        """Add a tag replacement rule."""
        self.replace_dict[original.strip().lower()] = replacement.strip()
        self.save_settings()

    def remove_replace_rule(self, original: str):
        """Remove a tag replacement rule."""
        self.replace_dict.pop(original.strip().lower(), None)
        self.save_settings()

    def clear_replace_dict(self):
        """Clear all replacement rules."""
        self.replace_dict.clear()
        self.save_settings()

    # === Keep Filter (Force Include) ===

    def add_keep_tag(self, tag: str):
        """Add a tag to the keep list (always include)."""
        self.keep_list.add(tag.strip().lower())
        self.save_settings()

    def remove_keep_tag(self, tag: str):
        """Remove a tag from the keep list."""
        self.keep_list.discard(tag.strip().lower())
        self.save_settings()

    def clear_keep_list(self):
        """Clear all tags from keep list."""
        self.keep_list.clear()
        self.save_settings()

    # === Filtering Logic ===

    def should_keep_tag(self, tag: str, confidence: float, threshold: float) -> bool:
        """
        Determine if a tag should be kept based on confidence and keep list.

        Args:
            tag: The tag to check
            confidence: The tag's confidence score
            threshold: The normal confidence threshold

        Returns:
            True if tag should be kept, False otherwise
        """
        tag_lower = tag.lower()

        # Always keep if in keep list (overrides confidence threshold)
        if tag_lower in self.keep_list:
            return True

        # Otherwise, use normal confidence threshold
        return confidence >= threshold

    def apply_filters(self, tags: List[str]) -> List[str]:
        """
        Apply all filters to a list of tags.

        Args:
            tags: List of tags to filter

        Returns:
            Filtered list of tags
        """
        filtered_tags = []

        for tag in tags:
            tag_lower = tag.lower()

            # Skip if in remove list
            if tag_lower in self.remove_list:
                continue

            # Apply replacement if exists
            if tag_lower in self.replace_dict:
                filtered_tags.append(self.replace_dict[tag_lower])
            else:
                filtered_tags.append(tag)

        return filtered_tags

    def filter_tags_with_confidence(
        self,
        tags: List[str],
        confidence_scores: Dict[str, float],
        threshold: float
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Apply all filters including confidence-based filtering.

        Args:
            tags: List of tags
            confidence_scores: Dict of tag -> confidence score
            threshold: Confidence threshold

        Returns:
            Tuple of (filtered_tags, filtered_confidence_scores)
        """
        filtered_tags = []
        filtered_scores = {}

        for tag in tags:
            tag_lower = tag.lower()
            confidence = confidence_scores.get(tag, 0.0)

            # Check if should keep based on confidence and keep list
            if not self.should_keep_tag(tag, confidence, threshold):
                continue

            # Skip if in remove list
            if tag_lower in self.remove_list:
                continue

            # Apply replacement if exists
            final_tag = self.replace_dict.get(tag_lower, tag)
            filtered_tags.append(final_tag)
            filtered_scores[final_tag] = confidence

        return filtered_tags, filtered_scores

    # === Utility Methods ===

    def get_remove_list(self) -> List[str]:
        """Get all tags in remove list."""
        return sorted(list(self.remove_list))

    def get_replace_dict(self) -> Dict[str, str]:
        """Get all replacement rules."""
        return dict(self.replace_dict)

    def get_keep_list(self) -> List[str]:
        """Get all tags in keep list."""
        return sorted(list(self.keep_list))

    def get_statistics(self) -> Dict:
        """Get filter statistics."""
        return {
            'remove_count': len(self.remove_list),
            'replace_count': len(self.replace_dict),
            'keep_count': len(self.keep_list)
        }
