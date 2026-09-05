"""Waifu Diffusion Tagger interrogator implementation."""

from core.base_interrogator import BaseInterrogator
from typing import Dict, Any, List
from PIL import Image
import numpy as np


class WDInterrogator(BaseInterrogator):
    """Waifu Diffusion Tagger interrogator."""
    
    def __init__(self, model_name: str = "SmilingWolf/wd-v1-4-moat-tagger-v2"):
        super().__init__(model_name)
        self.threshold = 0.35
        self.model = None
        self.tags = None
        self.target_size = 448
        self.input_name = None
        # Tag-name column cached from self.tags (see _get_tag_names).
        self._tag_names_source = None
        self._tag_names = []
    
    def load_model(self, threshold: float = 0.35, device: str = 'cuda',
                   provider_settings=None, **kwargs):
        """
        Load WD Tagger model with automatic CPU fallback.

        Args:
            threshold: Confidence threshold for tag inclusion (0.0-1.0)
            device: Requested device ('cuda' or 'cpu')
            provider_settings: ONNXProviderSettings instance for provider configuration
            **kwargs: Additional configuration
        """
        # Auto-detect and validate device
        from core.device_detector import get_device_detector
        detector = get_device_detector()

        # If CUDA requested but not available, fall back to CPU
        if device == 'cuda' and not detector.is_onnx_cuda_available():
            import logging
            logging.warning(
                f"CUDA requested but ONNX Runtime CUDA not available. "
                f"Falling back to CPU."
            )
            device = 'cpu'

        self.threshold = threshold
        self.config = {
            'threshold': threshold,
            'device': device,
            **kwargs
        }

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            import pandas as pd

            # Download model and tags
            model_path = hf_hub_download(self.model_name, "model.onnx")
            tags_path = hf_hub_download(self.model_name, "selected_tags.csv")

            # Load tags
            self.tags = pd.read_csv(tags_path)

            # Load ONNX model with provider settings if available
            if provider_settings:
                self.model = provider_settings.create_inference_session(model_path, device)
            else:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
                self.model = ort.InferenceSession(model_path, providers=providers)

            # Drop the cached input name so it is re-read from the new session.
            self.input_name = None
            self.is_loaded = True

            import logging
            logging.info(f"WD model loaded successfully on {device}")

        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}\n"
                "Install with: pip install onnxruntime huggingface-hub pandas"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load WD model: {e}")
    
    def _get_tag_names(self) -> List[str]:
        """
        Resolve the tag-name column for the loaded tag table.

        WD models carry ~10k tags and a DataFrame lookup per tag dominates
        interrogate(), so the column is materialized once and cached until
        self.tags is replaced.
        """
        if self._tag_names_source is not self.tags:
            self._tag_names = self.tags['name'].tolist()
            self._tag_names_source = self.tags
        return self._tag_names

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for WD tagger.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")
        
        # Resize while maintaining aspect ratio
        image.thumbnail((self.target_size, self.target_size), Image.LANCZOS)
        
        # Pad to square
        canvas = Image.new('RGB', (self.target_size, self.target_size), (255, 255, 255))
        canvas.paste(
            image, 
            ((self.target_size - image.width) // 2, (self.target_size - image.height) // 2)
        )
        
        # WD tagger ONNX models expect NHWC float32 in 0-255 BGR order.
        image_array = np.array(canvas).astype(np.float32)
        image_array = image_array[:, :, ::-1]
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def interrogate(self, image_path: str, threshold: float = None) -> Dict[str, Any]:
        """
        Interrogate image using WD Tagger.
        
        Args:
            image_path: Path to image file
            threshold: Optional override for confidence threshold
            
        Returns:
            Dict with 'tags', 'confidence_scores', and 'raw_output'
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        threshold = threshold if threshold is not None else self.threshold
        
        # Preprocess image
        try:
            image_array = self.preprocess_image(image_path)
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {e}")
        
        # Run inference
        try:
            if self.input_name is None:
                self.input_name = self.model.get_inputs()[0].name
            probs = self.model.run(None, {self.input_name: image_array})[0][0]
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")

        # Filter by threshold and create results. The threshold test is
        # vectorized so only tags that actually pass cost Python time.
        tag_names = self._get_tag_names()
        limit = min(len(probs), len(tag_names))

        tags_with_scores = {
            tag_names[i]: float(probs[i])
            for i in np.flatnonzero(probs[:limit] >= threshold)
        }

        # Sort by confidence (descending)
        sorted_items = sorted(tags_with_scores.items(), key=lambda x: x[1], reverse=True)
        tags = [tag for tag, _ in sorted_items]
        
        return {
            'tags': tags,
            'confidence_scores': tags_with_scores,
            'raw_output': ', '.join(tags)
        }
    
    def get_model_type(self) -> str:
        """Return model type identifier."""
        return "WD"

    def unload_model(self):
        """Unload model from memory."""
        self.model = None
        self.tags = None
        self.input_name = None
        self._tag_names_source = None
        self._tag_names = []
        self.is_loaded = False
