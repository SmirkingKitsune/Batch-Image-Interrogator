"""Image hashing and metadata extraction utilities."""

import hashlib
from pathlib import Path
from PIL import Image
from typing import Dict


def hash_image_content(image_path: str) -> str:
    """Generate SHA256 hash of image file content."""
    sha256_hash = hashlib.sha256()
    
    with open(image_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def get_image_metadata(image_path: str) -> Dict:
    """Extract image metadata (dimensions, file size)."""
    path = Path(image_path)
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        raise ValueError(f"Failed to read image metadata: {e}")
    
    file_size = path.stat().st_size
    
    return {
        'width': width,
        'height': height,
        'file_size': file_size
    }
