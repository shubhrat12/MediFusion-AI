"""
Data loaders for the medical AI analysis platform.
Provides secure data loading functionality for different data types.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

from .data_loader_security import (
    SecureDataLoader,
    SecureImageDataset,
    SecureModelLoader,
    create_secure_data_loader,
    verify_memory_usage,
    verify_file_size,
    MAX_CSV_SIZE_MB,
    MAX_IMAGE_SIZE_MB
)

logger = logging.getLogger(__name__)

class ChestXrayDataset(SecureImageDataset):
    """Secure dataset for chest X-ray images with additional validations."""
    
    def __init__(self, image_dir: str, labels_file: str, transform=None):
        super().__init__(image_dir, labels_file, transform)
        
        # Additional validation for medical image specifics
        self.validate_medical_images()
    
    def validate_medical_images(self):
        """Validate medical image properties."""
        for img_path in self.image_paths:
            try:
                with Image.open(img_path) as img:
                    # Check image mode (should be L for X-rays)
                    if img.mode not in ['L', 'RGB']:
                        logger.warning(f"Unexpected image mode {img.mode} for {img_path}")
                    
                    # Check image dimensions
                    if any(dim < 100 for dim in img.size):
                        logger.warning(f"Image {img_path} may be too small: {img.size}")
                    
                    # Check bit depth
                    if img.mode == 'L' and img.getextrema()[1] < 100:
                        logger.warning(f"Image {img_path} may have insufficient bit depth")
                        
            except Exception as e:
                logger.error(f"Error validating {img_path}: {e}")
                self.image_paths.remove(img_path)

def load_chest_xray_data(
    image_dir: str,
    labels_file: str,
    batch_size: int = 32,
    num_workers: int = 0,
    transform = None
) -> Tuple[DataLoader, List[str]]:
    """
    Securely load chest X-ray images and labels.
    
    Args:
        image_dir: Directory containing the X-ray images
        labels_file: Path to CSV file containing labels
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        transform: Optional transforms to apply to images
    
    Returns:
        DataLoader for the dataset and list of class names
    """
    try:
        # Verify memory available
        verify_memory_usage()
        
        # Create secure dataset
        dataset = ChestXrayDataset(
            image_dir=image_dir,
            labels_file=labels_file,
            transform=transform
        )
        
        # Create secure data loader
        data_loader = create_secure_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        
        # Extract class names from labels
        with open(labels_file, 'r') as f:
            header = f.readline().strip().split(',')
            class_names = [col for col in header if col not in ['Path', 'Patient', 'Study']]
        
        return data_loader, class_names
        
    except Exception as e:
        logger.error(f"Error loading chest X-ray data: {e}")
        raise

def load_tabular_data(file_path: str) -> pd.DataFrame:
    """
    Securely load tabular data from CSV file.
    
    Args:
        file_path: Path to CSV file containing tabular data
    
    Returns:
        Pandas DataFrame with the loaded data
    """
    try:
        # Verify file size
        verify_file_size(file_path, MAX_CSV_SIZE_MB)
        
        # Verify memory
        verify_memory_usage()
        
        # Read CSV in chunks to control memory usage
        chunk_size = 1000  # Adjust based on your needs
        chunks = []
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Validate data types and values
            for col in chunk.select_dtypes(include=['float64', 'int64']):
                if chunk[col].isnull().sum() > 0:
                    logger.warning(f"Column {col} contains null values")
                if chunk[col].min() < 0 and not col.endswith('_id'):
                    logger.warning(f"Column {col} contains negative values")
            
            chunks.append(chunk)
            
            # Check memory after each chunk
            verify_memory_usage()
        
        return pd.concat(chunks, ignore_index=True)
        
    except Exception as e:
        logger.error(f"Error loading tabular data: {e}")
        raise

def load_pretrained_model(model_path: str, model_name: str):
    """
    Securely load a pretrained model with hash verification.
    
    Args:
        model_path: Path to the model file
        model_name: Name of the model for hash verification
    """
    try:
        # Verify model file
        loader = SecureModelLoader()
        if loader.load_model(model_path, model_name):
            return torch.load(model_path)
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        raise