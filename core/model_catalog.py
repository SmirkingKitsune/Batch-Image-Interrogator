"""Choices offered for the classic taggers, shared by both front ends.

The PyQt6 config widgets (ui/dialogs.py) and the Electron bridge list the same
models, modes and profiles from here, in the same order.
"""

from typing import Dict, List

WD_MODELS: List[str] = [
    # V1.4 models
    'SmilingWolf/wd-v1-4-moat-tagger-v2',
    'SmilingWolf/wd-v1-4-vit-tagger-v2',
    'SmilingWolf/wd-v1-4-vit-tagger',
    'SmilingWolf/wd-v1-4-convnext-tagger-v2',
    'SmilingWolf/wd-v1-4-convnext-tagger',
    'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
    # V3 models (latest)
    'SmilingWolf/wd-vit-tagger-v3',
    'SmilingWolf/wd-vit-large-tagger-v3',
    'SmilingWolf/wd-convnext-tagger-v3',
    'SmilingWolf/wd-swinv2-tagger-v3',
    'SmilingWolf/wd-eva02-large-tagger-v3',
]
DEFAULT_WD_MODEL = 'SmilingWolf/wd-v1-4-moat-tagger-v2'

# One-line notes shown beside the selected model.
WD_MODEL_NOTES: Dict[str, str] = {
    'SmilingWolf/wd-v1-4-moat-tagger-v2': "V1.4 · MOAT architecture, recommended for v1.4.",
    'SmilingWolf/wd-v1-4-vit-tagger-v2': "V1.4 · Vision Transformer.",
    'SmilingWolf/wd-v1-4-vit-tagger': "V1.4 · Vision Transformer (original).",
    'SmilingWolf/wd-v1-4-convnext-tagger-v2': "V1.4 · ConvNeXt.",
    'SmilingWolf/wd-v1-4-convnext-tagger': "V1.4 · ConvNeXt (original).",
    'SmilingWolf/wd-v1-4-convnextv2-tagger-v2': "V1.4 · ConvNeXt V2.",
    'SmilingWolf/wd-v1-4-swinv2-tagger-v2': "V1.4 · Swin Transformer V2.",
    'SmilingWolf/wd-vit-tagger-v3': "V3 · balanced. eva02-large is highest quality, convnext is fastest.",
    'SmilingWolf/wd-vit-large-tagger-v3': "V3 · large ViT, best quality after eva02-large.",
    'SmilingWolf/wd-convnext-tagger-v3': "V3 · ConvNeXt, the fastest V3 tagger.",
    'SmilingWolf/wd-swinv2-tagger-v3': "V3 · Swin Transformer V2.",
    'SmilingWolf/wd-eva02-large-tagger-v3': "V3 · EVA-02 Large, highest quality.",
}

CAMIE_MODELS: List[str] = [
    'Camais03/camie-tagger-v2',
    'Camais03/camie-tagger',
]
DEFAULT_CAMIE_MODEL = 'Camais03/camie-tagger-v2'

CAMIE_THRESHOLD_PROFILES: List[str] = [
    'overall',
    'micro_optimized',
    'macro_optimized',
    'balanced',
    'category_specific',
]

CAMIE_CATEGORIES: List[str] = ['general', 'character', 'copyright', 'artist', 'meta', 'rating', 'year']

CAMIE_DEFAULT_CATEGORY_THRESHOLDS: Dict[str, float] = {
    'artist': 0.5, 'character': 0.5, 'copyright': 0.5,
    'general': 0.35, 'meta': 0.5, 'rating': 0.5, 'year': 0.5,
}

CAPTION_MODELS: List[str] = [
    'None',
    'blip-base',
    'blip-large',
    'blip2-2.7b',
    'blip2-flan-t5-xl',
    'git-large-coco',
]

CLIP_MODES: List[str] = ['best', 'fast', 'classic', 'negative']
DEFAULT_CLIP_MODEL = 'ViT-L-14/openai'

# Shown when the open_clip model list cannot be loaded.
FALLBACK_CLIP_MODELS: List[str] = [
    'ViT-L-14/openai',
    'ViT-H-14/laion2b_s32b_b79k',
    'ViT-g-14/laion2b_s12b_b42k',
    'ViT-B-32/openai',
    'ViT-B-16/openai',
]
