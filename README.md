# Image Interrogator - Batch Image Tagging Tool

A professional PyQt6-based application for batch image interrogation and tagging using CLIP and Waifu Diffusion models with SQLite caching for efficiency.

## Features

- **Multiple Model Support**: Use CLIP or Waifu Diffusion (WD) Tagger models
  - **12 WD Tagger Models**: V1.4 (stable) and V3 (latest) variants including EVA-02, ViT, ConvNeXt, and Swin transformers
  - **5 CLIP Models**: Multiple CLIP architectures with optional caption models
- **Intelligent Caching**: SQLite database stores previous interrogations to avoid reprocessing
- **Batch Processing**: Interrogate entire directories efficiently
- **Tag Management**: Edit, save, and organize tags with an intuitive UI
- **Advanced Image Inspection**: Detailed dialog with multi-model comparison, WD sensitivity ratings, and tag filtering visualization
- **Tag Filtering System**: Remove, replace, or force-include tags with customizable rules
- **File Organization**: Automatically organize images into subdirectories based on tags
- **Confidence Scores**: WD Tagger provides confidence scores for each tag
- **Gallery View**: Thumbnail-based image browser with visual indicators for tagged images
- **GPU Acceleration**: CUDA support for 10-50x faster processing

## Architecture

```
image_interrogator/
├── core/                      # Core business logic
│   ├── database.py           # SQLite database management
│   ├── hashing.py            # Image hashing utilities
│   ├── file_manager.py       # File I/O operations
│   └── base_interrogator.py # Abstract interrogator class
├── interrogators/            # Model implementations
│   ├── clip_interrogator.py # CLIP model
│   └── wd_interrogator.py   # Waifu Diffusion Tagger
├── ui/                       # PyQt6 UI components
│   ├── main_window.py       # Main application window
│   ├── widgets.py           # Custom widgets
│   ├── dialogs.py           # Configuration dialogs
│   └── workers.py           # Background worker threads
├── main.py                  # Application entry point
└── requirements.txt         # Dependencies
```

## Installation

### Prerequisites

- Python 3.10+

**Optional:**

- CUDA-capable NVIDIA GPU (recommended for optimal performance)
- NVIDIA drivers installed (if you have a GPU)

### Quick Start (Recommended)

The easiest way to set up the project is using the automated setup script:

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Check Python installation and version
2. Create a virtual environment
3. Detect your NVIDIA GPU and CUDA version automatically
4. Install PyTorch with the correct CUDA support for your system
5. Install all other dependencies
6. Verify the installation

**If you have an NVIDIA GPU**, the script will automatically detect your CUDA version and ask for permission to install PyTorch with GPU support. Just answer 'y' when prompted!

### Manual Installation (Advanced)

If you prefer manual setup:

**Step 1: Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 2: Install PyTorch**

For CPU-only installation:
```bash
pip install torch torchvision
```

For GPU support (recommended):
```bash
# Install PyTorch with CUDA support first
# Choose the appropriate CUDA version based on your GPU driver:

# For CUDA 12.6 (recommended for most modern GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.8 (newer GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# For CUDA 13.0 (latest)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**Step 3: Install Other Dependencies**
```bash
pip install -r requirements.txt

# For ONNX GPU support (WD Tagger)
pip install onnxruntime-gpu
```

**Important Note**: PyTorch bundles the necessary CUDA runtime libraries, so you only need compatible NVIDIA drivers installed. A separate CUDA Toolkit installation is not required unless you're compiling custom CUDA code.

## Usage

### Starting the Application

**Quick Launch (Recommended):**

The run scripts will perform a quick health check and alert you if GPU acceleration is not enabled:

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
./run.sh
```

If the script detects you have an NVIDIA GPU but PyTorch is running in CPU mode, it will warn you and suggest running the setup script to fix it.

**Direct Launch:**
```bash
python main.py
```

### Basic Workflow

1. **Select Directory**: Click "Select Directory" to choose a folder containing images
2. **Configure Model**:
   - Choose model type (WD Tagger or CLIP)
   - Click "Configure Model" to set parameters
   - Click "Load Model" to load into memory
3. **Interrogate Images**:
   - **Batch**: Click "Batch Interrogate" to process all images
   - **Single**: Select an image and click "Interrogate Selected"
4. **Review Results**: View tags in the results table with confidence scores (WD only)
5. **Edit Tags**: Use the tag editor to modify tags manually
6. **Advanced Inspection**: Double-click any image (Gallery or Queue) for detailed analysis
7. **Organize**: Use "Organize by Tags" to move images into subdirectories

### Model Configuration

#### CLIP Configuration
- **Mode**: 
  - `best`: Highest quality, slowest (recommended for final tagging)
  - `fast`: Quick processing, good quality (recommended for testing)
  - `classic`: Traditional CLIP approach
  - `negative`: Generate negative prompts for Stable Diffusion
- **Device**: `cuda` (GPU) or `cpu`

#### WD Tagger Configuration
- **Model Selection**: Choose from 12 available models
  - V1.4 models: moat, vit, convnext, swinv2 variants (stable)
  - V3 models: vit, vit-large, convnext, swinv2, eva02-large (latest, recommended)
  - See [WD_MODELS.md](WD_MODELS.md) for detailed model comparison
- **Confidence Threshold**: 0.0 - 1.0
  - Lower (0.2-0.3): More tags, less precise
  - Medium (0.35-0.5): Balanced (recommended: 0.35)
  - Higher (0.5-0.8): Fewer tags, more confident
- **Device**: `cuda` (GPU) or `cpu`

**Recommended Models:**
- Best Quality: `SmilingWolf/wd-eva02-large-tagger-v3`
- Balanced: `SmilingWolf/wd-vit-tagger-v3` or `SmilingWolf/wd-v1-4-moat-tagger-v2`
- Fastest: `SmilingWolf/wd-convnext-tagger-v3`

### Database Caching

The application uses SQLite to cache interrogation results:
- Images are hashed using SHA256
- Previous interrogations are retrieved from cache when available
- Different models maintain separate cached results
- Database file: `interrogations.db` in the application directory

### Tag-Based Organization

Organize images into subdirectories based on tags:

1. Click "Organize by Tags"
2. Enter tags to match (comma-separated)
3. Specify target subdirectory name
4. Choose match mode:
   - **any**: Move if image has at least one matching tag
   - **all**: Move only if image has all specified tags
5. Optionally move .txt files with images

Example: Tags "landscape, sunset" with mode "any" will move all images tagged with either "landscape" OR "sunset" to the specified subdirectory.

### Advanced Image Inspection Dialog

Access detailed image analysis by:
- **Double-clicking** any image thumbnail in the Gallery tab
- **Right-clicking** an image and selecting "Advanced Inspection..."
- **Double-clicking** an item in the Interrogation queue

**Features:**

**Model Results Tab:**
- Switch between different interrogation results (CLIP/WD models)
- **WD Sensitivity Ratings**: Visual display of content ratings
  - General, Sensitive, Questionable, Explicit (with confidence bars)
- Complete tag list with confidence scores

**Database vs File Comparison Tab:**
- Visual comparison of database tags vs .txt file tags
- Color-coded status indicators:
  - 🟢 **Green**: Tag in both database and file
  - 🟡 **Yellow**: Tag in database only (would be written with current filters)
  - 🔴 **Red**: Tag filtered out by tag removal rules
  - 🟠 **Orange**: Tag replaced by substitution rules
  - 🔵 **Blue**: Tag in file only (manually added)
- Understand exactly what tag filters are doing

**Tag Editor Tab:**
- Direct tag editing with instant save
- Copy tags from any model result
- Reload tags from file
- **Saves bypass all filters** - complete user control

**Navigation:**
- Browse through images with Prev/Next buttons
- Arrow keys (←/→) for quick navigation
- ESC to close dialog
- Ctrl+S to save tags

### Tag Filtering System

Configure tag filters in the Interrogation tab to automatically clean up interrogation results:

**Remove List:**
- Blacklist unwanted tags (e.g., "letterboxed", "watermark")
- Tags are removed before writing to .txt files
- Database keeps all original tags

**Replace Rules:**
- Substitute tags with better alternatives
- Example: "1girl" → "solo female character"
- Applied before writing to .txt files

**Keep List:**
- Force-include specific tags even if below confidence threshold
- Useful for important but low-confidence tags

**Important:** Tag filters only affect .txt file output. The database always stores complete, unfiltered results for every model.

### Keyboard Shortcuts

**Main Window:**
- `Ctrl+O`: Select directory
- `Ctrl+Q`: Quit application

**Advanced Inspection Dialog:**
- `←/→`: Navigate prev/next image
- `ESC`: Close dialog
- `Ctrl+S`: Save tags

## File Format

Tags are saved as comma-separated values in .txt files:
```
landscape, sunset, mountains, scenic, nature, beautiful sky
```

## Database Schema

### Images Table
- `id`: Primary key
- `file_path`: Current file path
- `file_hash`: SHA256 hash (unique identifier)
- `width`, `height`: Image dimensions
- `file_size`: File size in bytes
- `created_at`, `updated_at`: Timestamps

### Models Table
- `id`: Primary key
- `model_name`: Model identifier
- `model_type`: CLIP or WD
- `version`: Model version
- `config`: JSON configuration

### Interrogations Table
- `id`: Primary key
- `image_id`: Foreign key to images
- `model_id`: Foreign key to models
- `tags`: JSON array of tags
- `confidence_scores`: JSON object of tag:score pairs
- `raw_output`: Raw model output
- `interrogated_at`: Timestamp

## Performance Tips

1. **GPU Usage**: Always use CUDA if available for 10-50x speedup
2. **Batch Size**: Process images in batches for optimal performance
3. **Cache Utilization**: The database cache eliminates redundant processing
4. **Model Selection**: 
   - WD Tagger is faster and provides confidence scores
   - CLIP provides more natural language descriptions

## Troubleshooting

### CUDA Not Detected (CPU Mode Warning)

If you see "CUDA not available (CPU mode will be used)" but you have an NVIDIA GPU:

1. **Run the setup script**: `setup.bat` (Windows) or `./setup.sh` (Linux/Mac)
2. The script will detect your GPU and CUDA version
3. Allow it to reinstall PyTorch with CUDA support when prompted
4. Verify with: `python verify_cuda.py`

**Common causes:**
- PyTorch CPU-only version is installed (check with: `python -c "import torch; print(torch.__version__)"`)
  - Look for `+cpu` in the version (bad) vs `+cu130` (good)
- NVIDIA drivers not installed or outdated
- Incompatible CUDA version between PyTorch and drivers

**Quick fix:**
```bash
# Windows
setup.bat

# Linux/Mac
./setup.sh
```

### Model Loading Fails
- Ensure CUDA is properly configured (see above)
- Check that all dependencies are installed
- Try CPU mode if GPU fails (slower but works)

### Out of Memory
- Reduce batch size (process fewer images)
- Use CPU mode (slower but less memory)
- Close other applications
- Check GPU memory usage with `nvidia-smi`

### Images Not Showing
- Ensure images are in supported formats: .jpg, .jpeg, .png, .webp, .bmp, .gif
- Check file permissions

### Tags Not Saving
- Verify write permissions in the image directory
- Check that image paths don't contain special characters

### Setup Script Issues

**"Python not found":**
- Install Python 3.10+ from python.org
- Make sure Python is in your PATH

**"Failed to install PyTorch":**
- Check your internet connection
- Try running the script again
- For slow connections, the download may timeout (PyTorch is ~2GB)

## Advanced Usage

### Custom Model Names

Modify the model names in the interrogator classes:

```python
# For WD Tagger
wd_interrogator = WDInterrogator("SmilingWolf/wd-v1-4-swinv2-tagger-v2")

# For CLIP
clip_interrogator = CLIPInterrogator("ViT-H-14/laion2b_s32b_b79k")
```

### Programmatic Usage

```python
from core import InterrogationDatabase, FileManager
from interrogators import WDInterrogator

# Initialize
db = InterrogationDatabase()
interrogator = WDInterrogator()
interrogator.load_model(threshold=0.35, device='cuda')

# Interrogate
results = interrogator.interrogate("path/to/image.jpg")

# Save to database
from core.hashing import hash_image_content, get_image_metadata

file_hash = hash_image_content("path/to/image.jpg")
metadata = get_image_metadata("path/to/image.jpg")

image_id = db.register_image(
    "path/to/image.jpg",
    file_hash,
    metadata['width'],
    metadata['height'],
    metadata['file_size']
)

model_id = db.register_model(
    interrogator.model_name,
    interrogator.get_model_type()
)

db.save_interrogation(
    image_id,
    model_id,
    results['tags'],
    results['confidence_scores'],
    results['raw_output']
)

# Write to file
from pathlib import Path
FileManager.write_tags_to_file(Path("path/to/image.jpg"), results['tags'])
```

## Credits

- **CLIP Interrogator**: Uses the clip-interrogator library
- **Waifu Diffusion Tagger**: Uses SmilingWolf's WD Tagger models
- **UI Framework**: PyQt6

## Support

For issues or questions regarding upstream modules, refer to the documentation of the underlying libraries:
- [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator)
- [WD Tagger Models](https://huggingface.co/SmilingWolf)
- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
