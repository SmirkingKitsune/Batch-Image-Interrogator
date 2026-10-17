"""CLIP-based image interrogator implementation."""

import warnings

from core.base_interrogator import BaseInterrogator
from typing import Dict, Any, Optional
from PIL import Image

# clip-interrogator still calls torch.cuda.amp.autocast(), which PyTorch now
# deprecates with a FutureWarning. Only the library could act on it, and each
# one printed breaks the library's progress bar onto a new line.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated",
    category=FutureWarning,
    module=r"clip_interrogator\.clip_interrogator",
)


def _release_cuda_cache() -> None:
    """Hand VRAM that PyTorch has cached but no longer uses back to the driver.

    Other GPU users (the ONNX taggers, llama-server, the VRAM meter) see
    PyTorch's cache as used memory until it is emptied.
    """
    import gc
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    # Never create a CUDA context just to empty it.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _build_clip_on_cpu(config):
    """Create the CLIP model in its final precision, on the CPU.

    Left to clip-interrogator, open_clip creates the model on the GPU in fp32
    and only then casts it to fp16. Each fp16 tensor lands inside a freed fp32
    block, so PyTorch keeps the whole fp32 footprint reserved (about 10 GB for
    ViT-bigG rather than 5) and cannot give the difference back. Built here,
    only the fp16 weights ever reach the GPU. Arguments match what
    clip-interrogator's own load passes, apart from the device.
    """
    import open_clip

    name, pretrained = config.clip_model_name.split('/', 2)
    if not config.quiet:
        print(f"Loading CLIP model {config.clip_model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        name,
        pretrained=pretrained,
        precision='fp16' if config.device == 'cuda' else 'fp32',
        device='cpu',
        jit=False,
        cache_dir=config.clip_model_path,
    )
    model.eval()
    return model, preprocess


def _create_interrogator(config_params):
    from clip_interrogator import Config, Interrogator

    config = Config(**config_params)
    config.clip_model, config.clip_preprocess = _build_clip_on_cpu(config)
    # Moves the model to the device itself.
    return Interrogator(config)


def _is_out_of_memory(error: BaseException) -> bool:
    # torch.OutOfMemoryError ("CUDA out of memory") and the plain CUDA error
    # ("CUDA error: out of memory") both say so.
    return 'out of memory' in str(error).lower()


def _gpu_memory_summary() -> str:
    """Free GPU memory, as in " (3.1 GiB of 15.6 GiB free)".

    Empty when CUDA has not been started, so asking never starts it.
    """
    try:
        import torch
        if not torch.cuda.is_initialized():
            return ""
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return ""
    return f" ({free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB free)"


class CLIPInterrogator(BaseInterrogator):
    """CLIP-based image interrogator with BLIP2 support."""

    MODES = ['best', 'fast', 'classic', 'negative']
    CAPTION_MODELS = ['blip-base', 'blip-large', 'blip2-2.7b', 'blip2-flan-t5-xl', 'git-large-coco']

    # 1: no caption model means CLIP only. Earlier rows from a run without a
    # caption model were silently captioned by clip-interrogator's default
    # (blip-large).
    RESULT_VERSION = 1

    def __init__(self, model_name: str = "ViT-L-14/openai"):
        super().__init__(model_name)
        self.mode = 'best'
        self.ci = None
        self.caption_model = None
    
    def load_model(self, mode: str = 'best', device: str = 'cuda', caption_model: Optional[str] = None, **kwargs):
        """
        Load CLIP Interrogator model, falling back to the CPU when CUDA is
        unavailable or fails to start. Running out of GPU memory is a load
        error instead: on the CPU a large model would quietly take minutes
        per image.

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

        out_of_memory = False
        try:
            config_params = {
                'clip_model_name': self.model_name,
                'device': device,
                **kwargs,
                # Always explicit: clip-interrogator defaults to blip-large
                # when the name is left out, so "no caption model" would
                # otherwise still download and run one.
                'caption_model_name': caption_model,
            }

            self.ci = _create_interrogator(config_params)
            self.is_loaded = True
            # Label tables built during the load (when their cache files are
            # missing) leave text-encoder batches in PyTorch's cache.
            _release_cuda_cache()

            import logging
            logging.info(f"CLIP model loaded successfully on {device}")

        except ImportError:
            raise ImportError(
                "clip-interrogator package not installed. "
                "Install with: pip install clip-interrogator torch torchvision"
            )
        except RuntimeError as e:
            if device == 'cuda' and _is_out_of_memory(e):
                # Raised below: this exception's frames hold the half-loaded
                # model, and its VRAM can only be released once they are gone.
                out_of_memory = True
            # Handle CUDA initialization errors
            elif 'CUDA' in str(e) or 'c10' in str(e):
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

        if out_of_memory:
            _release_cuda_cache()
            name = self.model_name + (f" with {caption_model}" if caption_model else "")
            raise RuntimeError(
                f"Not enough GPU memory to load {name}{_gpu_memory_summary()}. "
                "Unload the Inquiry model or close other programs using the GPU, "
                "or choose a smaller CLIP model or the CPU device."
            )

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
            if not self.caption_model and mode != 'negative':
                caption = self._interrogate_without_caption(image, mode)
            elif mode == 'best':
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
    
    def _interrogate_without_caption(self, image: Image.Image, mode: str,
                                     min_flavors: int = 8, max_flavors: int = 32) -> str:
        """CLIP-only versions of the best/fast/classic modes.

        clip-interrogator's own methods open every prompt with a generated
        caption and assert when no caption model is loaded. These follow the
        same steps with the caption left out, so the prompt is built purely
        from CLIP-ranked terms.
        """
        # The library's own helpers for its interrogate_* methods.
        from clip_interrogator.clip_interrogator import _merge_tables, _truncate_to_fit

        ci = self.ci
        image_features = ci.image_to_features(image)

        def classic(flavor_count: int) -> str:
            medium = ci.mediums.rank(image_features, 1)[0]
            artist = ci.artists.rank(image_features, 1)[0]
            trending = ci.trendings.rank(image_features, 1)[0]
            movement = ci.movements.rank(image_features, 1)[0]
            flaves = ", ".join(ci.flavors.rank(image_features, flavor_count))
            return _truncate_to_fit(f"{medium} {artist}, {trending}, {movement}, {flaves}", ci.tokenize)

        merged = _merge_tables([ci.artists, ci.flavors, ci.mediums, ci.movements, ci.trendings], ci)

        def fast(flavor_count: int) -> str:
            return _truncate_to_fit(", ".join(merged.rank(image_features, flavor_count)), ci.tokenize)

        if mode == 'classic':
            return classic(3)
        if mode == 'fast':
            return fast(max_flavors)

        # best: with no starting prompt, chain() seeds itself from the
        # top-ranked term, then the closest of the candidates wins.
        flaves = merged.rank(image_features, ci.config.flavor_intermediate_count)
        chained = ci.chain(image_features, flaves, min_count=min_flavors,
                           max_count=max_flavors, desc="Flavor chain")
        candidates = [classic(max_flavors), fast(max_flavors), chained]
        similarities = ci.similarities(image_features, candidates)
        return candidates[similarities.index(max(similarities))]

    def result_signature(self) -> Dict[str, Any]:
        return {'mode': self.mode, 'caption_model': self.caption_model}

    def get_model_type(self) -> str:
        """Return model type identifier."""
        if self.caption_model:
            return f"CLIP+{self.caption_model}"
        return "CLIP"

    def unload_model(self):
        """Unload model from memory and release cached GPU/VRAM."""
        self.ci = None
        self.is_loaded = False
        _release_cuda_cache()
