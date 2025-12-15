# Waifu Diffusion Tagger Models

This document describes all available WD Tagger models in the Image Interrogator application.

## Overview

The application now supports **12 different WD Tagger models** from the SmilingWolf repository on HuggingFace. These models are specialized for anime/manga image tagging using the Danbooru tag set.

## Available Models

### V1.4 Models (Stable)

These are the original stable models, well-tested and reliable:

1. **SmilingWolf/wd-v1-4-moat-tagger-v2** ⭐ *Recommended for V1.4*
   - Architecture: MOAT (Mixture of Attention Transformers)
   - Best overall performance in the V1.4 series
   - Good balance of speed and accuracy

2. **SmilingWolf/wd-v1-4-vit-tagger-v2**
   - Architecture: Vision Transformer (ViT)
   - Updated version with improved accuracy

3. **SmilingWolf/wd-v1-4-vit-tagger**
   - Architecture: Vision Transformer (ViT)
   - Original ViT variant

4. **SmilingWolf/wd-v1-4-convnext-tagger-v2**
   - Architecture: ConvNeXt
   - Modernized ConvNet architecture
   - Updated version

5. **SmilingWolf/wd-v1-4-convnext-tagger**
   - Architecture: ConvNeXt
   - Original ConvNeXt variant

6. **SmilingWolf/wd-v1-4-convnextv2-tagger-v2**
   - Architecture: ConvNeXt V2
   - Improved version of ConvNeXt with better feature learning

7. **SmilingWolf/wd-v1-4-swinv2-tagger-v2**
   - Architecture: Swin Transformer V2
   - Hierarchical vision transformer

### V3 Models (Latest - Recommended) ✨

These are the newest models with improved accuracy and expanded tag vocabulary:

8. **SmilingWolf/wd-vit-tagger-v3**
   - Architecture: Vision Transformer
   - Latest ViT implementation
   - Improved tagging accuracy over V1.4

9. **SmilingWolf/wd-vit-large-tagger-v3** ⭐ *Best Quality*
   - Architecture: Large Vision Transformer
   - Highest accuracy in ViT family
   - More parameters = better tag recognition
   - Recommended for highest quality tagging

10. **SmilingWolf/wd-convnext-tagger-v3**
    - Architecture: ConvNeXt
    - Latest ConvNeXt implementation
    - Improved over V1.4 variants

11. **SmilingWolf/wd-swinv2-tagger-v3**
    - Architecture: Swin Transformer V2
    - Latest Swin implementation
    - Better hierarchical feature extraction

12. **SmilingWolf/wd-eva02-large-tagger-v3** ⭐⭐ *Highest Overall Quality*
    - Architecture: EVA-02 Large
    - State-of-the-art vision foundation model
    - Best overall tagging performance
    - Recommended for professional use
    - Largest model with highest accuracy

## Model Comparison

### By Performance (Accuracy)
1. **EVA-02 Large V3** - Highest
2. **ViT Large V3** - Very High
3. **MOAT V1.4 V2** - High
4. **Other V3 models** - High
5. **Other V1.4 models** - Good

### By Speed (Inference Time)
1. **ConvNeXt variants** - Fast
2. **ViT variants** - Medium
3. **Swin variants** - Medium
4. **ViT Large V3** - Slower (larger model)
5. **EVA-02 Large V3** - Slowest (largest model)

### By VRAM Usage
- **Small models** (~200-400MB): V1.4 base variants
- **Medium models** (~400-800MB): V3 base variants, ViT Large V3
- **Large models** (~800MB-1.2GB): EVA-02 Large V3

## Recommendations

### For General Use
**SmilingWolf/wd-v1-4-moat-tagger-v2** or **SmilingWolf/wd-vit-tagger-v3**
- Good balance of speed and accuracy
- Reliable and well-tested
- Suitable for batch processing

### For Best Quality
**SmilingWolf/wd-eva02-large-tagger-v3**
- Highest accuracy available
- Best for professional/production use
- Ideal for curating datasets

### For Speed
**SmilingWolf/wd-v1-4-convnext-tagger-v2** or **SmilingWolf/wd-convnext-tagger-v3**
- Fastest inference
- Good for large batch processing
- Still maintains good accuracy

### For Limited VRAM
**SmilingWolf/wd-v1-4-vit-tagger-v2**
- Smaller memory footprint
- Still provides good accuracy
- Works well on GPUs with 4-6GB VRAM

## Usage

1. Click **"Configure Model"** button
2. Switch to the **"WD Tagger"** tab
3. Select your desired model from the **"WD Model"** dropdown
4. Adjust the **Confidence Threshold** (0.35 recommended)
5. Select **Device** (cuda for GPU, cpu for CPU)
6. Click **OK** to save configuration
7. Click **"Load Model"** to download and load the selected model

## Technical Details

### Model Format
- All models use **ONNX** format for inference
- Downloaded from HuggingFace Hub automatically
- Models are cached locally after first download

### Tag Set
- V1.4 models: ~6,000 tags
- V3 models: ~9,000+ tags (expanded vocabulary)

### Input Size
- All models use **448x448** pixel input
- Images are automatically resized and padded

### Threshold Recommendations
- **0.35** (default): Balanced, recommended for most use cases
- **0.25-0.30**: More tags, good for exploratory tagging
- **0.40-0.50**: Fewer but more confident tags
- **0.50+**: Only very confident tags

## Performance Notes

### GPU Acceleration
All models support CUDA acceleration for significantly faster inference:
- **CPU**: 1-3 seconds per image
- **GPU**: 0.1-0.5 seconds per image (10-30x faster)

### First Run
The first time you use a model, it will be downloaded from HuggingFace:
- Model download: ~200MB-1.2GB depending on model
- Models are cached in `~/.cache/huggingface/hub/`
- Subsequent runs use the cached model

## Credits

All models are created and maintained by **SmilingWolf** on HuggingFace:
- [SmilingWolf's HuggingFace Profile](https://huggingface.co/SmilingWolf)
- Models are trained on Danbooru image dataset
- Licensed for research and commercial use

## Troubleshooting

### Model Download Fails
- Check internet connection
- Verify HuggingFace is accessible
- Check disk space (need ~1-2GB free)

### Out of Memory
- Try a smaller model (V1.4 variants)
- Use CPU mode instead of GPU
- Close other applications
- Reduce batch size

### Poor Tagging Quality
- Try a V3 model (improved accuracy)
- Adjust confidence threshold
- Ensure GPU acceleration is enabled
- Try EVA-02 Large for best results
