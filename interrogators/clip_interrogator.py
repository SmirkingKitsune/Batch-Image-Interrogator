"""CLIP-based image interrogator implementation."""

from core.base_interrogator import BaseInterrogator
from typing import Dict, Any, Optional
from PIL import Image


class CLIPInterrogator(BaseInterrogator):
    """CLIP-based image interrogator with BLIP2 support."""

    MODES = ['best', 'fast', 'classic', 'negative']
    CAPTION_MODELS = ['blip-base', 'blip-large', 'blip2-2.7b', 'blip2-flan-t5-xl', 'git-large-coco']

    def __init__(self, model_name: str = "ViT-L-14/openai"):
        super().__init__(model_name)
        self.mode = 'best'
        self.ci = None
        self.caption_model = None
    
    def load_model(self, mode: str = 'best', device: str = 'cuda', caption_model: Optional[str] = None, **kwargs):
        """
        Load CLIP Interrogator model with automatic CPU fallback.

        Args:
            mode: Interrogation mode ('best', 'fast', 'classic', 'negative')
            device: Requested device ('cuda' or 'cpu')
            caption_model: Caption model to use (optional). Options: 'blip-base', 'blip-large',
                          'blip2-2.7b', 'blip2-flan-t5-xl', 'git-large-coco'.
                          BLIP2 models require clip-interrogator >= 0.6.0
            **kwargs: Additional config for CLIP Interrogator
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.MODES}")

        if caption_model and caption_model not in self.CAPTION_MODELS:
            raise ValueError(f"Invalid caption_model '{caption_model}'. Choose from {self.CAPTION_MODELS}")

        # Auto-detect and validate device
        from core.device_detector import get_device_detector
        detector = get_device_detector()

        # If CUDA requested but not available, fall back to CPU
        if device == 'cuda' and not detector.is_pytorch_cuda_available():
            import logging
            logging.warning(
                f"CUDA requested but PyTorch CUDA not available. "
                f"Falling back to CPU. "
                f"Reason: {detector.get_status()['pytorch_error']}"
            )
            device = 'cpu'

        self.mode = mode
        self.caption_model = caption_model
        self.config = {
            'mode': mode,
            'device': device,
            'caption_model': caption_model,
            **kwargs
        }

        try:
            from clip_interrogator import Config, Interrogator

            config_params = {
                'clip_model_name': self.model_name,
                'device': device,
                **kwargs
            }

            if caption_model:
                config_params['caption_model_name'] = caption_model

            config = Config(**config_params)
            self.ci = Interrogator(config)
            self.is_loaded = True

            import logging
            logging.info(f"CLIP model loaded successfully on {device}")

        except ImportError:
            raise ImportError(
                "clip-interrogator package not installed. "
                "Install with: pip install clip-interrogator torch torchvision"
            )
        except RuntimeError as e:
            # Handle CUDA initialization errors
            if 'CUDA' in str(e) or 'c10' in str(e):
                if device == 'cuda':
                    import logging
                    logging.error(f"CUDA initialization failed: {e}")
                    logging.info("Attempting to fall back to CPU...")

                    # Retry with CPU
                    return self.load_model(
                        mode=mode, device='cpu',
                        caption_model=caption_model, **kwargs
                    )
                else:
                    raise RuntimeError(f"Failed to load CLIP model on CPU: {e}")
            else:
                raise RuntimeError(f"Failed to load CLIP model: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading CLIP model: {e}")
    
    def interrogate(self, image_path: str, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Interrogate image using CLIP.
        
        Args:
            image_path: Path to image file
            mode: Optional override for interrogation mode
            
        Returns:
            Dict with 'tags', 'confidence_scores', and 'raw_output'
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        mode = mode or self.mode
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")
        
        # Run interrogation based on mode
        try:
            if mode == 'best':
                caption = self.ci.interrogate(image)
            elif mode == 'fast':
                caption = self.ci.interrogate_fast(image)
            elif mode == 'classic':
                caption = self.ci.interrogate_classic(image)
            elif mode == 'negative':
                caption = self.ci.interrogate_negative(image)
            else:
                caption = self.ci.interrogate(image)
        except Exception as e:
            raise RuntimeError(f"Interrogation failed: {e}")
        
        # CLIP returns a caption, convert to tags
        tags = [tag.strip() for tag in caption.split(',') if tag.strip()]
        
        return {
            'tags': tags,
            'confidence_scores': None,  # CLIP doesn't provide per-tag confidence
            'raw_output': caption
        }
    
    def get_model_type(self) -> str:
        """Return model type identifier."""
        if self.caption_model:
            return f"CLIP+{self.caption_model}"
        return "CLIP"

    def unload_model(self):
        """Unload model from memory."""
        self.ci = None
        self.is_loaded = False
