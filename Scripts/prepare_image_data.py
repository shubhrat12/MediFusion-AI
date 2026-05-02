#!/usr/bin/env python
"""
Prepare image data for the 13 common classes.

This script:
1. Loads the fixed 13-class data
2. Locates corresponding images
3. Creates train/test splits with image paths
4. Saves metadata files for training
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the 13 common classes
COMMON_CLASSES = [
    'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices', 'No Finding'
]

def extract_image_id(report_text):
    """Extract image ID from report text using regex."""
    # Look for patterns like "image123.png" or similar
    match = re.search(r'[a-zA-Z0-9_]+\.(png|jpg|jpeg|dcm)', report_text, re.IGNORECASE)
    if match:
        return match.group(0)
    return None

def find_image(image_id, base_dirs):
    """Find image file in possible base directories with security checks."""
    if not image_id or not isinstance(image_id, str):
        logger.warning("Invalid image_id provided")
        return None
        
    # Sanitize image_id to prevent path traversal
    image_id = os.path.basename(image_id)
    
    # Validate file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.dcm'}
    if not any(image_id.lower().endswith(ext) for ext in valid_extensions):
        logger.warning(f"Invalid file extension for image: {image_id}")
        return None
        
    for base_dir in base_dirs:
        try:
            # Resolve base directory to absolute path
            base_dir = os.path.abspath(base_dir)
            if not os.path.exists(base_dir):
                continue
                
            # Try different possible paths
            paths = [
                os.path.join(base_dir, image_id),
                os.path.join(base_dir, "images", image_id)
            ]
            
            # Add paths for each numbered directory
            for i in range(1, 13):  # images_001 to images_012
                dir_name = f"images_{i:03d}"
                paths.append(os.path.join(base_dir, dir_name, "images", image_id))
            
            for path in paths:
                # Validate the resolved path is under base_dir
                resolved_path = os.path.abspath(path)
                if not resolved_path.startswith(base_dir):
                    logger.warning(f"Path traversal attempt detected: {path}")
                    continue
                    
                if os.path.exists(resolved_path):
                    # Check file size
                    if os.path.getsize(resolved_path) > 50 * 1024 * 1024:  # 50MB limit
                        logger.warning(f"Image too large: {resolved_path}")
                        continue
                        
                    return resolved_path
                    
        except (OSError, IOError) as e:
            logger.error(f"Error accessing path {base_dir}: {str(e)}")
            continue
            
    return None

def prepare_data(train_csv, test_csv, image_dirs, output_dir):
    """
    Prepare image data for training.
    
    Args:
        train_csv: Path to training CSV file
        test_csv: Path to test CSV file
        image_dirs: List of directories containing images
        output_dir: Directory to save prepared data
    """
    logger.info("Loading data files...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Create output directories
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Load original image metadata
    chestxray_meta = pd.read_csv("data/raw/images/archive/Data_Entry_2017.csv")
    image_to_findings = dict(zip(chestxray_meta['Image Index'], chestxray_meta['Finding Labels']))
    
    # Process training data
    logger.info("Processing training data...")
    train_data = []
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
        # Try to find image ID in the findings dictionary
        for img_id, findings in image_to_findings.items():
            if findings in row['report_text']:
                image_path = find_image(img_id, image_dirs)
                if image_path:
                    data = {'image_path': image_path}
                    for class_name in COMMON_CLASSES:
                        data[class_name] = row[class_name]
                    train_data.append(data)
                    break
    
    # Process test data
    logger.info("Processing test data...")
    test_data = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test data"):
        # Try to find image ID in the findings dictionary
        for img_id, findings in image_to_findings.items():
            if findings in row['report_text']:
                image_path = find_image(img_id, image_dirs)
                if image_path:
                    data = {'image_path': image_path}
                    for class_name in COMMON_CLASSES:
                        data[class_name] = row[class_name]
                    test_data.append(data)
                    break
    
    # Convert to DataFrames
    train_metadata = pd.DataFrame(train_data)
    test_metadata = pd.DataFrame(test_data)
    
    # Save metadata
    logger.info("Saving metadata...")
    
    train_metadata.to_csv(os.path.join(metadata_dir, 'train_metadata.csv'), index=False)
    test_metadata.to_csv(os.path.join(metadata_dir, 'test_metadata.csv'), index=False)
    
    # Save class names
    with open(os.path.join(metadata_dir, 'class_names.txt'), 'w') as f:
        for class_name in COMMON_CLASSES:
            f.write(f"{class_name}\n")
    
    logger.info(f"Prepared {len(train_data)} training samples and {len(test_data)} test samples")
    logger.info(f"Metadata saved to {metadata_dir}")

def main():
    # Input files
    train_csv = "data/processed/13class_train_fixed.csv"
    test_csv = "data/processed/13class_test_fixed.csv"
    
    # Image directories to search
    image_dirs = [
        "data/raw/images/archive",
        "data/raw/images",
        "data/images"
    ]
    
    # Output directory
    output_dir = "data/processed"
    
    # Prepare data
    prepare_data(train_csv, test_csv, image_dirs, output_dir)

if __name__ == "__main__":
    main()