#!/usr/bin/env python
"""
Create a small test dataset for fusion inference.

This script:
1. Takes a few samples from the 13-class test dataset
2. Adds path to corresponding images
3. Saves a CSV file for fusion testing
"""

import os
import pandas as pd
import logging
import random
from pathlib import Path

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

def main():
    # Input and output paths
    input_file = 'data/processed/text/13class_test_fixed.csv'
    output_file = 'data/processed/fusion_test_data.csv'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the test data
    logger.info(f"Loading test data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} samples")
    
    # Select a small number of samples
    num_samples = 10
    if len(df) > num_samples:
        logger.info(f"Selecting {num_samples} random samples")
        df = df.sample(num_samples, random_state=42)
    
    # Add placeholder image paths
    # In a real scenario, you would map report_text to actual X-ray images
    # Here we just create placeholder paths
    logger.info("Adding placeholder image paths")
    df['image_path'] = [f"data/raw/images/sample_{i}.png" for i in range(len(df))]
    
    # Check if we have any example images in the repository
    image_dir = "data/raw/images"
    if os.path.exists(image_dir):
        logger.info(f"Found image directory: {image_dir}")
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            logger.info(f"Found {len(image_files)} images")
            # For each sample, assign a random image from the directory
            df['image_path'] = [os.path.join(image_dir, random.choice(image_files)) for _ in range(len(df))]
    
    # For demonstration purposes, if no actual images exist:
    # Create a directory with placeholder images if needed
    if not os.path.exists(image_dir) or not any(f.endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(image_dir)):
        logger.warning("No image files found. Creating placeholder directory and dummy image")
        os.makedirs(image_dir, exist_ok=True)
        
        # We can create a dummy image if PIL is available
        try:
            from PIL import Image, ImageDraw
            
            # Create a black and white image with text
            dummy_image_path = os.path.join(image_dir, "dummy_xray.png")
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((40, 100), "Dummy X-ray Image", fill=(255, 255, 255))
            img.save(dummy_image_path)
            logger.info(f"Created dummy image at {dummy_image_path}")
            
            # Update all image paths to the dummy image
            df['image_path'] = dummy_image_path
            
        except ImportError:
            logger.warning("PIL not available to create dummy image")
    
    # Add patient_id column if it doesn't exist
    if 'patient_id' not in df.columns:
        logger.info("Adding patient_id column")
        df['patient_id'] = [f"patient_{i}" for i in range(len(df))]
    
    # Save the fusion test data
    logger.info(f"Saving {len(df)} samples to {output_file}")
    df.to_csv(output_file, index=False)
    logger.info("Fusion test data creation completed!")

if __name__ == "__main__":
    main() 