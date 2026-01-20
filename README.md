# Image Interrogator - Batch Image Tagging Tool

A professional PyQt6-based application for batch image interrogation and tagging using CLIP and Waifu Diffusion models with SQLite caching for efficiency.

## Features

- **Multiple Model Support**: Use CLIP or Waifu Diffusion (WD) Tagger models
  - **12 WD Tagger Models**: V1.4 (stable) and V3 (latest) variants including EVA-02, ViT, ConvNeXt, and Swin transformers
  - **5 CLIP Models**: Multiple CLIP architectures with optional caption models
- **Intelligent Caching**: SQLite database stores previous interrogations to avoid reprocessing
- **Batch Processing**: Interrogate entire directories efficiently with optional recursive subdirectory search
- **Tag Management**: Edit, save, and organize tags with an intuitive UI
- **Advanced Image Inspection**: Detailed dialog with multi-model comparison, WD sensitivity ratings, and tag filtering visualization
- **Checkbox Tag Selector**: Visual tag editor showing all model-generated tags with checkboxes for easy selection
- **Tag Filtering System**: Remove, replace, or force-include tags with customizable rules
- **Smart File Organization**: Organize images by tags with recursive search and directory selection to prevent re-organizing
- **Confidence Scores**: WD Tagger provides confidence scores for each tag
- **Gallery View**: Thumbnail-based image browser with visual indicators for tagged images
- **GPU Acceleration**: CUDA (NVIDIA) and ROCm (AMD) support for 10-50x faster processing

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

- **NVIDIA GPU**: CUDA-capable GPU with drivers installed (recommended for optimal performance)
- **AMD GPU (Linux only)**: ROCm-capable GPU with ROCm installed (see ROCm Support section)

### Quick Start (Recommended)

The easiest way to set up the project is using the automated setup script:

**Windows:**
```cmd
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

If you prefer manual setup or the automated script fails:

**Step 1: Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 2: Install PyTorch**

Choose the installation that matches your GPU:

**Option A: NVIDIA GPU (CUDA)**
```bash
# Recommended: CUDA 12.6 (RTX 30/40 series, GTX 16 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Or CUDA 13.0 (latest, RTX 40 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Or CUDA 12.8 (newer GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Or CUDA 11.8 (older GPUs: GTX 10 series, RTX 20 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Option B: AMD GPU (ROCm) - Linux only**
```bash
# ROCm 6.0 (RX 7000 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# ROCm 5.7 (RX 6000 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

**Option C: CPU Only (no GPU)**
```bash
pip install torch torchvision
```

**Step 3: Install ONNX Runtime**

**For NVIDIA GPU:**
```bash
pip install onnxruntime-gpu>=1.16.0
```

**For AMD GPU or CPU:**
```bash
pip install onnxruntime>=1.16.0
```

**Step 4: Install Other Dependencies**
```bash
pip install -r requirements.txt
```

**Important Notes**:
- PyTorch bundles CUDA runtime libraries - no separate CUDA Toolkit installation needed
- ONNX Runtime GPU package is CUDA-only (AMD users get CPU version)
- Check your NVIDIA driver supports the CUDA version you're installing

## Usage

### Starting the Application

**Quick Launch (Recommended):**

The run scripts will perform a quick health check and alert you if GPU acceleration is not enabled:

**Windows:**
```cmd
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

If the script detects you have an NVIDIA GPU but PyTorch is running in CPU mode, it will warn you and suggest running the setup script to fix it.

**Direct Launch:**
```bash
python main.py
```

### Basic Workflow

1. **Select Directory**: Click "Select Directory" to choose a folder containing images
   - Optional: Check "Include subdirectories (recursive)" to process images in all subdirectories
2. **Configure Model**:
   - Choose model type (WD Tagger or CLIP)
   - Click "Configure Model" to set parameters
   - Click "Load Model" to load into memory
3. **Interrogate Images**:
   - **Batch**: Click "Start Batch" to process all images
   - **Single**: Select an image and click "Interrogate Selected"
   - Recursive mode will process images from all selected subdirectories
4. **Review Results**: View tags in the results table with confidence scores (WD only)
5. **Edit Tags**: Use the checkbox tag selector to enable/disable tags from all models
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

Organize images into subdirectories based on tags with smart recursive search:

1. Click "Organize by Tags"
2. **Review the warning**: Operation will MOVE (not copy) files
3. Enter tags to match (comma-separated)
4. Specify target subdirectory name
5. Choose match mode:
   - **any**: Move if image has at least one matching tag
   - **all**: Move only if image has all specified tags
6. **Optional - Recursive Search**:
   - Check "Include subdirectories (recursive)" to search all subdirectories
   - Select which subdirectories to include as sources (prevents re-organizing already-organized files)
   - Example: Uncheck "organized" folder to avoid moving previously organized images
7. Optionally move .txt files with images
8. Click "Move Images" and confirm the operation

**Important Safety Features:**
- ⚠️ Bold warning at top of dialog indicates this is a MOVE operation
- Confirmation dialog shows all operation details before proceeding
- Default answer is "No" for safety
- Directory selection prevents accidentally reorganizing already-organized files

Example: Tags "landscape, sunset" with mode "any" will move all images tagged with either "landscape" OR "sunset" from selected source directories to the specified subdirectory.

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
- **Checkbox Tag Selector**: Visual interface showing ALL tags generated by ALL models
- **Search/Filter**: Quickly find specific tags with the search box
- **Visual Selection**: Checkboxes show which tags are currently in the .txt file
  - ☑ Checked = Tag is saved in .txt file
  - ☐ Unchecked = Tag was generated but not currently saved
- **Tag Count Display**: Shows "Total tags: X | Selected: Y"
- **Quick Controls**:
  - Select All / Deselect All buttons
  - Search filter for finding specific tags
- **Save Tags to File**: Green button applies checkbox selections to .txt file
- **Saves bypass all filters** - complete user control over which tags to keep

**Navigation:**
- Browse through images with Prev/Next buttons
- Arrow keys (←/→) for quick navigation
- ESC to close dialog
- Ctrl+S to save tags

### Recursive Directory Processing

The application supports recursive subdirectory search for both interrogation and organization:

**During Interrogation:**
1. Check "Include subdirectories (recursive)" in the Directory section
2. All subdirectories are automatically included in the image queue
3. Queue displays relative paths (e.g., "subfolder/image.jpg") when recursive
4. Status bar shows "(recursive)" to confirm mode
5. Gallery tab automatically syncs with the same recursive setting

**During Organization:**
1. Enable "Include subdirectories (recursive)" in the Organize dialog
2. Select which subdirectories to use as sources:
   - All subdirectories are shown with checkboxes
   - Includes "(Root directory)" option for images in the main folder
   - Default: All directories selected
3. **Smart Organization**: Deselect already-organized folders to prevent re-organizing:
   ```
   my_images/
   ├── raw/           ← ☑ Include as source
   ├── processed/     ← ☑ Include as source
   └── organized/     ← ☐ Skip (already organized)
   ```
4. Only images from selected directories will be organized

**Benefits:**
- Process entire folder hierarchies in one operation
- Maintain organized structure by excluding certain directories
- Visual feedback shows which directories are being processed
- Prevents accidental re-organization of already-sorted files

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

1. **GPU Usage**: Always use CUDA (NVIDIA) or ROCm (AMD) if available for 10-50x speedup
2. **Batch Size**: Process images in batches for optimal performance
3. **Cache Utilization**: The database cache eliminates redundant processing
4. **Model Selection**:
   - WD Tagger is faster and provides confidence scores
   - CLIP provides more natural language descriptions

## GPU Acceleration

### CUDA Support (NVIDIA GPUs)

The application provides full GPU acceleration for NVIDIA GPUs on Windows, Linux, and macOS (deprecated):

#### Requirements
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- NVIDIA drivers installed (CUDA Toolkit not required - PyTorch bundles CUDA runtime)
- Windows: GeForce/RTX/Quadro series
- Linux: GeForce/RTX/Quadro/Tesla series
- macOS: Legacy support only (Apple deprecated NVIDIA GPU support)

#### Installation

1. **Install NVIDIA drivers** (if not already installed):
   - Windows: https://www.nvidia.com/download/index.aspx
   - Linux: Use your distribution's package manager or NVIDIA's official installer

2. **Run the setup script**:

   **Windows:**
   ```cmd
   setup.bat
   ```

   **Linux:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   The script will:
   - Automatically detect your NVIDIA GPU
   - Detect CUDA version from `nvidia-smi`
   - Install PyTorch with matching CUDA support (12.6, 12.8, 13.0, or 11.8)
   - Install ONNX Runtime GPU for WD Tagger acceleration

#### GPU Acceleration Status

✅ **CLIP Models**: Full CUDA acceleration via PyTorch
✅ **WD Tagger Models**: Full CUDA acceleration via ONNX Runtime GPU

#### Performance

- **CLIP interrogation**: 10-50x faster than CPU (depending on GPU)
- **WD Tagger interrogation**: 10-50x faster than CPU (depending on GPU)
- **Batch processing**: Optimal performance with GPU acceleration

#### Supported CUDA Versions

The setup script automatically detects and installs the correct version:
- **CUDA 13.0**: Latest (RTX 40 series recommended)
- **CUDA 12.8**: Recent GPUs
- **CUDA 12.6**: Most common (RTX 30/40 series)
- **CUDA 11.8**: Older GPUs (GTX 10 series, RTX 20 series)

#### Verification

After running the setup script, you should see:

**Windows:**
```
PyTorch:
  Version: 2.x.x+cu126
  CUDA Available: True
  GPU: NVIDIA GeForce RTX 4090

ONNX Runtime:
  Version: 1.x.x
  CUDA Provider: Yes

[SUCCESS] GPU acceleration is fully enabled!
  - PyTorch: CUDA enabled (for CLIP models)
  - ONNX Runtime: CUDA enabled (for WD Tagger models)
```

**Linux:**
```
PyTorch:
  Version: 2.x.x+cu126
  CUDA Available: True
  GPU: NVIDIA GeForce RTX 4090

ONNX Runtime:
  Version: 1.x.x
  CUDA Provider: Yes

[SUCCESS] GPU acceleration is fully enabled!
  - PyTorch: CUDA enabled (for CLIP models)
  - ONNX Runtime: CUDA enabled (for WD Tagger models)
```

#### Manual Installation

If you prefer to install manually or the setup script fails:

```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux: source venv/bin/activate

# Install PyTorch with CUDA 12.6 (most common)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install ONNX Runtime GPU
pip install onnxruntime-gpu>=1.16.0

# Install other dependencies
pip install -r requirements.txt
```

For other CUDA versions, change `cu126` to:
- `cu130` for CUDA 13.0
- `cu128` for CUDA 12.8
- `cu118` for CUDA 11.8

### ROCm Support (AMD GPUs on Linux)

The application supports AMD GPUs on Linux through ROCm:

### Requirements
- **Linux only** (ROCm is not available on Windows/Mac)
- AMD GPU with ROCm support (RX 6000/7000 series, Radeon VII, Vega, etc.)
- ROCm 5.7 or later installed

### Installation

1. **Install ROCm** (if not already installed):
   ```bash
   # Ubuntu/Debian
   # Visit https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   The script will:
   - Automatically detect your AMD GPU
   - Detect your ROCm version
   - Install PyTorch with ROCm support
   - Install ONNX Runtime (CPU version, see note below)

### GPU Acceleration Status

✅ **CLIP Models**: Full ROCm acceleration via PyTorch
⚠️ **WD Tagger Models**: CPU-only (ONNX Runtime ROCm pip package not available)

### Performance Notes

- **CLIP interrogation**: Full GPU acceleration with ROCm (10-50x faster than CPU)
- **WD Tagger interrogation**: Runs on CPU (no pre-built ROCm package for ONNX Runtime)
- For ONNX Runtime ROCm support, you must build from source: https://onnxruntime.ai/docs/build/eps.html#migraphx

### Verification

After running the setup script, you should see:
```
PyTorch:
  Version: 2.x.x+rocmX.X
  CUDA Available: True
  GPU: AMD Radeon RX 7900 XTX

ONNX Runtime:
  Version: 1.x.x
  CUDA Provider: No

[SUCCESS] ROCm GPU acceleration is enabled!
  - PyTorch: ROCm enabled (for CLIP models)
  - ONNX Runtime: CPU mode (no ROCm pip package available)
```

### ARM64 Support (NVIDIA Spark, Jetson, Grace Blackwell)

The application supports ARM64 Linux systems with NVIDIA GPUs (e.g., Jetson Orin, Grace Hopper, Grace Blackwell):

#### GPU Acceleration Status

✅ **CLIP Models**: Full CUDA acceleration via PyTorch
⚠️ **WD Tagger Models**: CPU-only by default (no pre-built ARM64 wheels on PyPI)

#### Installation

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Detect ARM64 architecture automatically
- Install PyTorch with CUDA support
- Install CPU-only ONNX Runtime (GPU wheels not available on PyPI for ARM64)

#### Building ONNX Runtime GPU for ARM64

To enable GPU acceleration for WD Tagger on ARM64, you can build ONNX Runtime from source:

```bash
chmod +x build_onnx_arm64.sh
./build_onnx_arm64.sh
```

**Requirements:**
- CUDA Toolkit installed (`nvcc` in PATH)
- cuDNN installed
- CMake 3.26+
- Build tools (gcc, g++, make)
- 10GB+ free disk space
- 30-60+ minutes build time

The build script will:
1. Check all prerequisites
2. Clone ONNX Runtime source
3. Build with CUDA support for ARM64
4. Install the wheel into your virtual environment

#### Verification

After setup (without building ONNX Runtime):
```
[SUCCESS] GPU acceleration is enabled!
  - PyTorch: CUDA enabled (for CLIP models)
  - ONNX Runtime: CPU mode (no ARM64 GPU wheels on PyPI)
```

After building ONNX Runtime:
```
[SUCCESS] GPU acceleration is fully enabled!
  - PyTorch: CUDA enabled (for CLIP models)
  - ONNX Runtime: CUDA enabled (for WD Tagger models)
```

### GPU Acceleration Comparison

| Feature | CUDA (NVIDIA x86) | CUDA (NVIDIA ARM64) | ROCm (AMD) | CPU Only |
|---------|-------------------|---------------------|------------|----------|
| **Platforms** | Windows, Linux | Linux only | Linux only | All |
| **CLIP Models** | ✅ GPU (10-50x) | ✅ GPU (10-50x) | ✅ GPU (10-50x) | ❌ CPU (1x) |
| **WD Tagger** | ✅ GPU (10-50x) | ⚠️ CPU (1x)* | ❌ CPU (1x)** | ❌ CPU (1x) |
| **Setup** | Automatic | Automatic | Automatic | Automatic |
| **Driver Required** | NVIDIA | NVIDIA | ROCm | None |
| **Best For** | Maximum performance | ARM64 NVIDIA users | AMD Linux users | Testing/compatibility |

\* WD Tagger on ARM64 can use GPU by running `./build_onnx_arm64.sh`
\*\* WD Tagger on ROCm requires building ONNX Runtime from source

### Recommended GPU Configuration

**For best performance:**
1. **NVIDIA GPU (preferred)**: Full acceleration for both CLIP and WD Tagger
2. **AMD GPU on Linux**: Full acceleration for CLIP, CPU for WD Tagger
3. **CPU only**: Works but significantly slower (use for testing or compatibility)

**Minimum recommended:**
- NVIDIA: GTX 1060 6GB or better
- AMD: RX 6600 or better
- VRAM: 6GB minimum, 8GB+ recommended for larger models

## Troubleshooting

### GPU Not Detected (NVIDIA/CUDA)

If you see "CUDA not available (CPU mode will be used)" but you have an NVIDIA GPU:

#### Quick Fix
```cmd
# Windows
setup.bat
```

```bash
# Linux/Mac
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Detect your GPU and CUDA version
2. Offer to reinstall PyTorch with CUDA support
3. Install ONNX Runtime GPU for WD Tagger acceleration
4. Verify the installation

#### Manual Verification

**Check PyTorch version:**
```bash
python -c "import torch; print(torch.__version__)"
```
- ❌ Bad: `2.x.x+cpu` (CPU-only)
- ✅ Good: `2.x.x+cu126` (CUDA 12.6)

**Check CUDA availability:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Check ONNX Runtime:**
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
- ✅ Should include: `CUDAExecutionProvider`
- ❌ Missing: Only shows `CPUExecutionProvider`

#### Common Causes

1. **PyTorch CPU-only installed**: Reinstall with CUDA support
2. **NVIDIA drivers missing/outdated**:
   - Windows: Download from https://www.nvidia.com/download/index.aspx
   - Linux: `sudo apt install nvidia-driver-XXX` or use your package manager
3. **Wrong CUDA version**: Setup script will detect and fix
4. **ONNX Runtime CPU-only**: Run `pip install onnxruntime-gpu --force-reinstall`

#### Verify Drivers

```bash
nvidia-smi
```
Should show your GPU and CUDA version. If this fails, install/update NVIDIA drivers.

### GPU Not Detected (AMD/ROCm)

If you have an AMD GPU on Linux but it's not being detected:

#### Quick Fix
```bash
chmod +x setup.sh
./setup.sh
```

#### Manual Verification

**Check if ROCm is installed:**
```bash
rocm-smi
# or
rocminfo
```

**Check PyTorch version:**
```bash
python -c "import torch; print(torch.__version__)"
```
- ✅ Good: `2.x.x+rocm6.0` (ROCm 6.0)
- ❌ Bad: `2.x.x+cpu` (CPU-only)

**Check GPU availability:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```
Note: ROCm uses the CUDA API in PyTorch, so `cuda.is_available()` returns True for AMD GPUs.

#### Common Causes

1. **ROCm not installed**:
   - Install from https://rocm.docs.amd.com/
   - Ubuntu: `sudo apt install rocm-hip-runtime`
2. **PyTorch CPU-only installed**: Reinstall with ROCm support
3. **Unsupported GPU**: Check ROCm compatibility at https://rocm.docs.amd.com/
4. **Wrong ROCm version**: Setup script will detect and match

#### ROCm Version Check
```bash
# Check installed ROCm version
cat /opt/rocm/.info/version

# Should match PyTorch ROCm version
python -c "import torch; print(torch.__version__)"
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
