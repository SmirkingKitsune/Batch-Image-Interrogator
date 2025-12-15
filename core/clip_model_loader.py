"""
CLIP Model Loader Utility

Dynamically loads and categorizes all available OpenCLIP models,
organized by Stable Diffusion compatibility (SD 1.x / SD 2.0 / SDXL).
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_available_clip_models() -> Optional[List[Tuple[str, str]]]:
    """
    Get all available CLIP models from OpenCLIP.

    Returns:
        List of (model_name, pretrained_dataset) tuples, or None if import fails
    """
    try:
        import open_clip
        models = open_clip.list_pretrained()
        logger.info(f"Successfully loaded {len(models)} CLIP models from OpenCLIP")
        return models
    except ImportError as e:
        logger.warning(f"OpenCLIP not available: {e}. Using fallback model list.")
        return None
    except Exception as e:
        logger.error(f"Error loading CLIP models: {e}. Using fallback model list.")
        return None


def format_model_string(model_name: str, pretrained: str) -> str:
    """
    Format model tuple as string.

    Args:
        model_name: Model architecture name (e.g., 'ViT-L-14')
        pretrained: Pretrained dataset identifier (e.g., 'openai')

    Returns:
        Formatted string (e.g., 'ViT-L-14/openai')
    """
    return f"{model_name}/{pretrained}"


def categorize_clip_models(models: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Categorize CLIP models by Stable Diffusion compatibility.

    Categories:
    - sd_1x: Models recommended for Stable Diffusion 1.x
    - sd_20: Models recommended for Stable Diffusion 2.0
    - sdxl: Models recommended for Stable Diffusion XL
    - other: All other models (ResNet, experimental, etc.)

    Args:
        models: List of (model_name, pretrained_dataset) tuples

    Returns:
        Dict with categorized model strings
    """
    categorized = {
        'sd_1x': [],
        'sd_20': [],
        'sdxl': [],
        'other': []
    }

    # Define defaults to place first in each category
    sd_1x_default = 'ViT-L-14/openai'
    sdxl_default = 'ViT-bigG-14/laion2b_s39b_b160k'

    for model_name, pretrained in models:
        model_str = format_model_string(model_name, pretrained)

        # SD 1.x: ViT-L-14, ViT-B-32, ViT-B-16 with specific pretraining
        if model_name in ['ViT-L-14', 'ViT-B-32', 'ViT-B-16']:
            if pretrained in ['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_s32b_b79k']:
                categorized['sd_1x'].append(model_str)
            else:
                categorized['other'].append(model_str)

        # SDXL: ViT-bigG-14, EVA models, large ViT-g-14
        elif 'bigG' in model_name or model_name.startswith('EVA'):
            categorized['sdxl'].append(model_str)
        elif model_name == 'ViT-g-14':
            # Large datasets go to SDXL, others to SD 2.0
            if any(marker in pretrained for marker in ['s34b', 's39b', 's11b']):
                categorized['sdxl'].append(model_str)
            else:
                categorized['sd_20'].append(model_str)

        # SD 2.0: ViT-H-14 and medium ViT-g-14
        elif model_name == 'ViT-H-14':
            categorized['sd_20'].append(model_str)

        # Everything else (ResNet, etc.)
        else:
            categorized['other'].append(model_str)

    # Sort each category alphabetically
    for category in categorized:
        categorized[category].sort()

    # Move defaults to front
    if sd_1x_default in categorized['sd_1x']:
        categorized['sd_1x'].remove(sd_1x_default)
        categorized['sd_1x'].insert(0, sd_1x_default)

    if sdxl_default in categorized['sdxl']:
        categorized['sdxl'].remove(sdxl_default)
        categorized['sdxl'].insert(0, sdxl_default)

    logger.info(
        f"Categorized models: SD1.x={len(categorized['sd_1x'])}, "
        f"SD2.0={len(categorized['sd_20'])}, "
        f"SDXL={len(categorized['sdxl'])}, "
        f"Other={len(categorized['other'])}"
    )

    return categorized


def get_fallback_models() -> Dict[str, List[str]]:
    """
    Get fallback model list when OpenCLIP is unavailable.

    Returns:
        Dict with original 5 hardcoded models, categorized
    """
    logger.info("Using fallback CLIP model list (5 models)")
    return {
        'sd_1x': [
            'ViT-L-14/openai',
            'ViT-B-32/openai',
            'ViT-B-16/openai'
        ],
        'sd_20': [
            'ViT-H-14/laion2b_s32b_b79k',
            'ViT-g-14/laion2b_s12b_b42k'
        ],
        'sdxl': [],
        'other': []
    }


def get_categorized_models() -> Dict[str, List[str]]:
    """
    Main entry point for getting categorized CLIP models.

    Attempts to load all models from OpenCLIP, falls back to
    hardcoded list if unavailable.

    Returns:
        Dict with keys: 'sd_1x', 'sd_20', 'sdxl', 'other'
        Each containing a list of model strings
    """
    try:
        # Try to get models from OpenCLIP
        models = get_available_clip_models()

        if models is None or len(models) == 0:
            # OpenCLIP unavailable or no models found
            return get_fallback_models()

        # Categorize the models
        categorized = categorize_clip_models(models)

        # Validate that we have at least some models
        total_models = sum(len(v) for v in categorized.values())
        if total_models == 0:
            logger.warning("No models in any category, using fallback")
            return get_fallback_models()

        return categorized

    except Exception as e:
        logger.error(f"Unexpected error in get_categorized_models: {e}")
        return get_fallback_models()
