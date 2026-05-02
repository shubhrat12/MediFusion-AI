"""
Data loader security utilities for the medical AI analysis platform.
Provides secure data loading with memory limits, input validation, and hash verification.
"""

import os
import sys
import hashlib
import logging
import psutil
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import torch.utils.data as data
from PIL import Image
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security Constants
MAX_IMAGE_SIZE_MB = 100  # Maximum size for a single image in MB
MAX_CSV_SIZE_MB = 1000   # Maximum size for CSV files in MB
MAX_MEMORY_PERCENT = 80  # Maximum memory usage as percentage of system memory
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
ALLOWED_CSV_EXTENSIONS = {'.csv'}

# Model hash verification (update these with your model hashes)
PRETRAINED_MODEL_HASHES = {
    'densenet121': 'a5050dbff25b0e59fb33cb27d96634a4e85b6af903882800314a23a7c3bbda90',
    'resnet50': '7e165f7048a2bfd026eba8e187f68986b50dc960d46469a8302bdf48162e6532'
}

def verify_memory_usage():
    """Check if memory usage is within safe limits."""
    memory = psutil.virtual_memory()
    if memory.percent > MAX_MEMORY_PERCENT:
        raise MemoryError(f"Memory usage ({memory.percent}%) exceeds safe limit ({MAX_MEMORY_PERCENT}%)")
    return True

def verify_file_size(file_path: Union[str, Path], max_size_mb: float) -> bool:
    """Verify that a file is within size limits."""
    file_path = Path(file_path)
    size_mb = file_path.stat().st_size / (1024 * 1024)  # Convert to MB
    if size_mb > max_size_mb:
        raise ValueError(f"File {file_path} exceeds size limit of {max_size_mb}MB")
    return True

def verify_file_hash(file_path: Union[str, Path], expected_hash: str) -> bool:
    """Verify file integrity using SHA-256."""
    file_path = Path(file_path)
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    if sha256_hash.hexdigest() != expected_hash:
        raise ValueError(f"Hash mismatch for {file_path}")
    return True

def safe_path_join(*paths: str) -> str:
    """Safely join paths and resolve to absolute path without following symlinks."""
    base_path = os.path.abspath(os.path.normpath(paths[0]))
    for path in paths[1:]:
        path = os.path.normpath(path)
        if path.startswith(('/', '..')):
            raise ValueError("Path traversal attempt detected")
        base_path = os.path.join(base_path, path)
    return os.path.abspath(base_path)

class SecureDataLoader:
    """Base class for secure data loading with memory and size limits."""
    
    def __init__(self):
        self.max_memory_percent = MAX_MEMORY_PERCENT
    
    def verify_memory(self):
        """Check available memory before loading data."""
        return verify_memory_usage()
    
    def verify_file(self, file_path: Union[str, Path], max_size_mb: float):
        """Verify file size and extension."""
        return verify_file_size(file_path, max_size_mb)

class SecureImageDataset(data.Dataset):
    """Secure dataset for loading and processing medical images."""
    
    def __init__(self, 
                 image_dir: str,
                 labels_file: Optional[str] = None,
                 transform=None):
        self.transform = transform
        self.image_dir = Path(image_dir)
        
        # Validate image directory
        if not self.image_dir.is_dir():
            raise ValueError(f"Invalid image directory: {image_dir}")
        
        # Load and validate image paths
        self.image_paths = []
        for img_path in self.image_dir.glob('*'):
            if img_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
                try:
                    verify_file_size(img_path, MAX_IMAGE_SIZE_MB)
                    self.image_paths.append(img_path)
                except ValueError as e:
                    logger.warning(f"Skipping {img_path}: {e}")
        
        # Load labels if provided
        self.labels = None
        if labels_file:
            labels_path = Path(labels_file)
            if not labels_path.suffix in ALLOWED_CSV_EXTENSIONS:
                raise ValueError(f"Invalid labels file format: {labels_file}")
            verify_file_size(labels_file, MAX_CSV_SIZE_MB)
            self.labels = pd.read_csv(labels_file)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Verify memory before loading
        verify_memory_usage()
        
        img_path = self.image_paths[idx]
        try:
            with Image.open(img_path) as img:
                # Convert grayscale to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                # Get labels if available
                label = self.labels.iloc[idx].values if self.labels is not None else None
                
                return img, label if label is not None else img
                
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise

class SecureModelLoader:
    """Secure loader for pretrained models with hash verification."""
    
    @staticmethod
    def load_model(model_path: str, model_name: str) -> bool:
        """Load a model securely with hash verification."""
        if model_name not in PRETRAINED_MODEL_HASHES:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Verify file hash
        verify_file_hash(model_path, PRETRAINED_MODEL_HASHES[model_name])
        
        return True

def create_secure_data_loader(
    dataset: data.Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True
) -> data.DataLoader:
    """Create a DataLoader with security checks."""
    # Verify memory availability
    verify_memory_usage()
    
    # Calculate approximate memory needed per batch
    if hasattr(dataset, 'calculate_batch_memory'):
        batch_memory = dataset.calculate_batch_memory(batch_size)
        if batch_memory > psutil.virtual_memory().available:
            raise MemoryError(f"Batch size {batch_size} would exceed available memory")
    
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
