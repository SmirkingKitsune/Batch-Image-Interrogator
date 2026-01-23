"""Camie Tagger interrogator implementation."""

import json
from core.base_interrogator import BaseInterrogator
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np


class CamieInterrogator(BaseInterrogator):
    """Camie Tagger interrogator for anime/manga image tagging."""

    # Available models
    MODELS = {
        'Camais03/camie-tagger': 'v1',
        'Camais03/camie-tagger-v2': 'v2'
    }

    # Tag categories
    CATEGORIES = ['artist', 'character', 'copyright', 'general', 'meta', 'rating', 'year']

    # Threshold profiles with optimized values
    THRESHOLD_PROFILES = {
        'overall': {
            'description': 'Single threshold for all categories',
            'default': 0.5
        },
        'micro_optimized': {
            'description': 'Optimized for micro F1 score (more tags)',
            'default': 0.614
        },
        'macro_optimized': {
            'description': 'Optimized for macro F1 score (balanced)',
            'default': 0.492
        },
        'balanced': {
            'description': 'Balance between precision and recall',
            'default': 0.55
        },
        'category_specific': {
            'description': 'Different threshold per category',
            'defaults': {
                'artist': 0.5,
                'character': 0.5,
                'copyright': 0.5,
                'general': 0.35,
                'meta': 0.5,
                'rating': 0.5,
                'year': 0.5
            }
        }
    }

    def __init__(self, model_name: str = "Camais03/camie-tagger-v2"):
        """
        Initialize CamieInterrogator.

        Args:
            model_name: HuggingFace model name (v1 or v2)
        """
        super().__init__(model_name)
        self.threshold = 0.5
        self.threshold_profile = 'overall'
        self.category_thresholds = {}
        self.enabled_categories = list(self.CATEGORIES)
        self.model = None
        self.tags_data = None
        self.target_size = 512
        # Camie uses specific padding color (124, 116, 104) - grey-brown
        self.pad_color = (124, 116, 104)
        # ImageNet normalization values
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def load_model(self, threshold: float = 0.5, device: str = 'cuda',
                   threshold_profile: str = 'overall',
                   category_thresholds: Dict[str, float] = None,
                   enabled_categories: List[str] = None, **kwargs):
        """
        Load Camie Tagger model.

        Args:
            threshold: Base confidence threshold for tag inclusion (0.0-1.0)
            device: Device to use ('cuda' or 'cpu')
            threshold_profile: One of 'overall', 'micro_optimized', 'macro_optimized',
                             'balanced', or 'category_specific'
            category_thresholds: Dict of category -> threshold (for category_specific profile)
            enabled_categories: List of categories to include in output
            **kwargs: Additional configuration
        """
        self.threshold = threshold
        self.threshold_profile = threshold_profile
        self.enabled_categories = enabled_categories or list(self.CATEGORIES)

        # Set up category thresholds based on profile
        if threshold_profile == 'category_specific':
            default_cat_thresholds = self.THRESHOLD_PROFILES['category_specific']['defaults']
            self.category_thresholds = category_thresholds or default_cat_thresholds.copy()
        else:
            # Use profile default or user-specified threshold for all categories
            profile_threshold = self.THRESHOLD_PROFILES.get(
                threshold_profile, {}
            ).get('default', threshold)
            self.category_thresholds = {cat: profile_threshold for cat in self.CATEGORIES}

        self.config = {
            'threshold': threshold,
            'threshold_profile': threshold_profile,
            'category_thresholds': self.category_thresholds,
            'enabled_categories': self.enabled_categories,
            'device': device,
            **kwargs
        }

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            # Download model and metadata
            model_path = hf_hub_download(self.model_name, "model.onnx")

            # Try to get metadata file (may be named differently)
            try:
                metadata_path = hf_hub_download(self.model_name, "metadata.json")
                with open(metadata_path, 'r') as f:
                    self.tags_data = json.load(f)
            except Exception:
                # Try alternative name
                try:
                    metadata_path = hf_hub_download(self.model_name, "tags.json")
                    with open(metadata_path, 'r') as f:
                        self.tags_data = json.load(f)
                except Exception:
                    # Fall back to tag_to_index.json
                    try:
                        metadata_path = hf_hub_download(self.model_name, "tag_to_index.json")
                        with open(metadata_path, 'r') as f:
                            tag_to_index = json.load(f)
                            # Convert to expected format
                            self.tags_data = self._convert_tag_index_format(tag_to_index)
                    except Exception as e:
                        raise RuntimeError(f"Failed to load tag metadata: {e}")

            # Load ONNX model with appropriate provider
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(model_path, providers=providers)

            self.is_loaded = True

        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}\n"
                "Install with: pip install onnxruntime huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Camie model: {e}")

    def _convert_tag_index_format(self, tag_to_index: Dict) -> Dict:
        """
        Convert tag_to_index.json format to expected metadata format.

        Args:
            tag_to_index: Dict mapping tag names to indices

        Returns:
            Dict with tags organized by category
        """
        # Create structure with tags list and category info
        result = {
            'tags': [],
            'tag_to_category': {}
        }

        # If tag_to_index is already in expected format with categories
        if isinstance(tag_to_index, dict):
            if 'tags' in tag_to_index:
                return tag_to_index

            # Check if it's organized by category
            if any(cat in tag_to_index for cat in self.CATEGORIES):
                all_tags = []
                tag_to_category = {}
                for category in self.CATEGORIES:
                    if category in tag_to_index:
                        cat_tags = tag_to_index[category]
                        if isinstance(cat_tags, dict):
                            # Format: {tag: index}
                            for tag in cat_tags.keys():
                                all_tags.append({'name': tag, 'category': category})
                                tag_to_category[tag] = category
                        elif isinstance(cat_tags, list):
                            # Format: [tag1, tag2, ...]
                            for tag in cat_tags:
                                all_tags.append({'name': tag, 'category': category})
                                tag_to_category[tag] = category
                result['tags'] = all_tags
                result['tag_to_category'] = tag_to_category
                return result

            # Simple format: {tag: index} - assume all general tags
            for tag, idx in tag_to_index.items():
                result['tags'].append({'name': tag, 'category': 'general', 'index': idx})
                result['tag_to_category'][tag] = 'general'

        return result

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for Camie Tagger.

        Camie uses 512x512 input with specific preprocessing:
        - Maintain aspect ratio
        - Pad with (124, 116, 104) color
        - ImageNet normalization

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array in NCHW format
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")

        # Calculate scaling to maintain aspect ratio
        width, height = image.size
        scale = min(self.target_size / width, self.target_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize with high quality
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create padded canvas with Camie-specific padding color
        canvas = Image.new('RGB', (self.target_size, self.target_size), self.pad_color)

        # Center the image on canvas
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        canvas.paste(image, (paste_x, paste_y))

        # Convert to numpy and normalize
        image_array = np.array(canvas).astype(np.float32) / 255.0

        # Apply ImageNet normalization
        image_array = (image_array - self.mean) / self.std

        # Convert from HWC to NCHW format (batch, channels, height, width)
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        return image_array

    def interrogate(self, image_path: str, threshold: float = None) -> Dict[str, Any]:
        """
        Interrogate image using Camie Tagger.

        Args:
            image_path: Path to image file
            threshold: Optional override for base confidence threshold

        Returns:
            Dict with 'tags', 'confidence_scores', and 'raw_output' (JSON with category breakdown)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess image
        try:
            image_array = self.preprocess_image(image_path)
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {e}")

        # Run inference
        try:
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: image_array})

            # Use refined_logits (outputs[1]) if available, otherwise use outputs[0]
            if len(outputs) > 1:
                logits = outputs[1][0]  # Use refined predictions
            else:
                logits = outputs[0][0]

            # Apply sigmoid to convert logits to probabilities
            probs = 1 / (1 + np.exp(-logits))

        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")

        # Process results
        tags_with_scores = {}
        category_breakdown = {cat: [] for cat in self.CATEGORIES}

        # Get tags based on metadata format
        tags_list = self.tags_data.get('tags', [])
        tag_to_category = self.tags_data.get('tag_to_category', {})

        for i, prob in enumerate(probs):
            if i >= len(tags_list):
                break

            tag_info = tags_list[i]
            if isinstance(tag_info, dict):
                tag_name = tag_info.get('name', f'tag_{i}')
                category = tag_info.get('category', 'general')
            else:
                tag_name = str(tag_info)
                category = tag_to_category.get(tag_name, 'general')

            # Get threshold for this category
            cat_threshold = self.category_thresholds.get(category, self.threshold)

            # Check if category is enabled and tag passes threshold
            if category in self.enabled_categories and prob >= cat_threshold:
                tags_with_scores[tag_name] = float(prob)
                category_breakdown[category].append({
                    'tag': tag_name,
                    'confidence': float(prob)
                })

        # Sort by confidence (descending)
        sorted_items = sorted(tags_with_scores.items(), key=lambda x: x[1], reverse=True)
        tags = [tag for tag, _ in sorted_items]

        # Sort category breakdown by confidence
        for category in category_breakdown:
            category_breakdown[category].sort(key=lambda x: x['confidence'], reverse=True)

        # Create detailed raw output as JSON
        raw_output = {
            'model': self.model_name,
            'threshold_profile': self.threshold_profile,
            'enabled_categories': self.enabled_categories,
            'total_tags': len(tags),
            'categories': category_breakdown
        }

        return {
            'tags': tags,
            'confidence_scores': tags_with_scores,
            'raw_output': json.dumps(raw_output, indent=2)
        }

    def get_model_type(self) -> str:
        """Return model type identifier."""
        return "Camie"

    def unload_model(self):
        """Unload model from memory."""
        self.model = None
        self.tags_data = None
        self.is_loaded = False

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available Camie models."""
        return list(cls.MODELS.keys())

    @classmethod
    def get_threshold_profiles(cls) -> Dict[str, Dict]:
        """Get available threshold profiles with descriptions."""
        return cls.THRESHOLD_PROFILES.copy()

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of tag categories."""
        return cls.CATEGORIES.copy()
