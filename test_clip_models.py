"""
Test script for CLIP model loader
"""
from core.clip_model_loader import get_categorized_models

print("=" * 60)
print("CLIP MODEL LOADER TEST")
print("=" * 60)
print()

# Get categorized models
models = get_categorized_models()

print("Model Categories:")
print(f"  SD 1.x Models: {len(models['sd_1x'])} models")
print(f"  SD 2.0 Models: {len(models['sd_20'])} models")
print(f"  SDXL Models: {len(models['sdxl'])} models")
print(f"  Other Models: {len(models['other'])} models")
print(f"  TOTAL: {sum(len(v) for v in models.values())} models")
print()

print("Sample SD 1.x Models:")
for model in models['sd_1x'][:5]:
    print(f"  - {model}")
if len(models['sd_1x']) > 5:
    print(f"  ... and {len(models['sd_1x']) - 5} more")
print()

print("Sample SD 2.0 Models:")
for model in models['sd_20'][:5]:
    print(f"  - {model}")
if len(models['sd_20']) > 5:
    print(f"  ... and {len(models['sd_20']) - 5} more")
print()

print("Sample SDXL Models:")
for model in models['sdxl'][:5]:
    print(f"  - {model}")
if len(models['sdxl']) > 5:
    print(f"  ... and {len(models['sdxl']) - 5} more")
print()

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
