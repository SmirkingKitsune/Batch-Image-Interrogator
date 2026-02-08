"""Tag filtering system for customizing interrogation output."""

import json
from pathlib import Path
from typing import List, Dict, Set, Tuple


class TagFilterSettings:
    """Manages tag filtering rules for customizing output."""

    # Default emoji-style tags to skip when replacing underscores
    DEFAULT_UNDERSCORE_SKIP_LIST = {
        "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>",
        "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o",
        "u_u", "x_x", "|_|", "||_||"
    }

    def __init__(self, settings_file: str = "tag_filters.json"):
        self.settings_file = Path(settings_file)

        # Filter lists
        self.remove_list: Set[str] = set()  # Tags to exclude
        self.replace_dict: Dict[str, str] = {}  # tag -> replacement
        self.keep_list: Set[str] = set()  # Tags to always include (ignore confidence)

        # Prefix tags (prepended to all output)
        self.prefix_tags: List[str] = []

        # Underscore replacement settings
        self.replace_underscores: bool = False  # Toggle underscore→space
        self.underscore_skip_list: Set[str] = set(self.DEFAULT_UNDERSCORE_SKIP_LIST)

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
                    self.prefix_tags = data.get('prefix_tags', [])
                    self.replace_underscores = data.get('replace_underscores', False)
                    # Load custom skip list or use defaults
                    skip_list = data.get('underscore_skip_list')
                    if skip_list is not None:
                        self.underscore_skip_list = set(skip_list)
                    else:
                        self.underscore_skip_list = set(self.DEFAULT_UNDERSCORE_SKIP_LIST)
            except Exception as e:
                print(f"Error loading tag filter settings: {e}")

    def save_settings(self):
        """Save filter settings to file."""
        try:
            data = {
                'remove_list': list(self.remove_list),
                'replace_dict': self.replace_dict,
                'keep_list': list(self.keep_list),
                'prefix_tags': self.prefix_tags,
                'replace_underscores': self.replace_underscores,
                'underscore_skip_list': list(self.underscore_skip_list)
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

    # === Prefix Tags Filter ===

    def set_prefix_tags(self, tags: List[str]):
        """Set the list of prefix tags."""
        self.prefix_tags = [tag.strip() for tag in tags if tag.strip()]
        self.save_settings()

    def get_prefix_tags(self) -> List[str]:
        """Get the list of prefix tags."""
        return list(self.prefix_tags)

    def add_prefix_tag(self, tag: str):
        """Add a tag to the prefix list."""
        tag = tag.strip()
        if tag and tag not in self.prefix_tags:
            self.prefix_tags.append(tag)
            self.save_settings()

    def remove_prefix_tag(self, tag: str):
        """Remove a tag from the prefix list."""
        tag = tag.strip()
        if tag in self.prefix_tags:
            self.prefix_tags.remove(tag)
            self.save_settings()

    def clear_prefix_tags(self):
        """Clear all prefix tags."""
        self.prefix_tags.clear()
        self.save_settings()

    # === Underscore Replacement ===

    def set_replace_underscores(self, enabled: bool):
        """Set whether to replace underscores with spaces."""
        self.replace_underscores = enabled
        self.save_settings()

    def get_replace_underscores(self) -> bool:
        """Get whether underscore replacement is enabled."""
        return self.replace_underscores

    def get_underscore_skip_list(self) -> List[str]:
        """Get the list of tags to skip when replacing underscores."""
        return sorted(list(self.underscore_skip_list))

    def add_underscore_skip_tag(self, tag: str):
        """Add a tag to the underscore skip list."""
        self.underscore_skip_list.add(tag.strip())
        self.save_settings()

    def remove_underscore_skip_tag(self, tag: str):
        """Remove a tag from the underscore skip list."""
        self.underscore_skip_list.discard(tag.strip())
        self.save_settings()

    def reset_underscore_skip_list(self):
        """Reset the underscore skip list to defaults."""
        self.underscore_skip_list = set(self.DEFAULT_UNDERSCORE_SKIP_LIST)
        self.save_settings()

    def replace_underscore_in_tag(self, tag: str) -> str:
        """
        Replace underscores with spaces in a tag, unless it's in the skip list.

        Args:
            tag: The tag to process

        Returns:
            Tag with underscores replaced (if not in skip list)
        """
        if not self.replace_underscores:
            return tag

        # Check if tag should be skipped (preserve emoji-style tags)
        if tag.lower() in {t.lower() for t in self.underscore_skip_list}:
            return tag

        return tag.replace('_', ' ')

    def normalize_tag_for_comparison(self, tag: str) -> str:
        """
        Normalize a tag for comparison purposes.
        Applies underscore replacement if enabled, and lowercases.

        Args:
            tag: The tag to normalize

        Returns:
            Normalized tag for comparison (lowercase, underscores replaced if enabled)
        """
        normalized = tag.lower()
        if self.replace_underscores:
            # Check if tag should be skipped (preserve emoji-style tags)
            if normalized not in {t.lower() for t in self.underscore_skip_list}:
                normalized = normalized.replace('_', ' ')
        return normalized

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
            Filtered list of tags (with prefix tags prepended)
        """
        filtered_tags = []

        for tag in tags:
            tag_lower = tag.lower()

            # Skip if in remove list
            if tag_lower in self.remove_list:
                continue

            # Apply replacement if exists
            if tag_lower in self.replace_dict:
                final_tag = self.replace_dict[tag_lower]
            else:
                final_tag = tag

            # Apply underscore replacement
            final_tag = self.replace_underscore_in_tag(final_tag)

            filtered_tags.append(final_tag)

        # Prepend prefix tags
        if self.prefix_tags:
            filtered_tags = list(self.prefix_tags) + filtered_tags

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

            # Apply underscore replacement
            final_tag = self.replace_underscore_in_tag(final_tag)

            filtered_tags.append(final_tag)
            filtered_scores[final_tag] = confidence

        # Prepend prefix tags (with confidence 1.0)
        if self.prefix_tags:
            prefix_scores = {tag: 1.0 for tag in self.prefix_tags}
            filtered_tags = list(self.prefix_tags) + filtered_tags
            filtered_scores = {**prefix_scores, **filtered_scores}

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
            'keep_count': len(self.keep_list),
            'prefix_count': len(self.prefix_tags),
            'replace_underscores': self.replace_underscores
        }
