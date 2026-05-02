#!/usr/bin/env python
"""
Prepare 13-class test and train datasets from existing labeled reports.

This script:
1. Loads the labeled reports file
2. Extracts only the 13 common classes
3. Saves the processed data to test and train CSV files
"""

import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

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
    # Define input and output paths
    input_file = 'data/raw/text/archive (2)/labeled_reports.csv'
    # If labeled_reports.csv is too large, use test.csv instead
    backup_file = 'data/raw/text/archive (2)/test.csv'
    output_test = 'data/processed/text/13class_test.csv'
    output_train = 'data/processed/text/13class_train.csv'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_test), exist_ok=True)
    
    try:
        # Try to load the labeled reports file
        logger.info(f"Attempting to load {input_file}...")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {input_file} with shape {df.shape}")
    except Exception as e:
        logger.warning(f"Failed to load {input_file}: {e}")
        logger.info(f"Attempting to load backup file {backup_file}...")
        try:
            df = pd.read_csv(backup_file)
            logger.info(f"Loaded {backup_file} with shape {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load backup file: {e}")
            raise
    
    # Check if 'report_text' column exists
    text_col = None
    for col in df.columns:
        if 'text' in col.lower() or 'report' in col.lower() or 'impression' in col.lower():
            text_col = col
            logger.info(f"Using column '{text_col}' as report text")
            break
    
    if not text_col:
        logger.warning("No report text column found. Using the first column that's not a class label.")
        # Use the first column that's not in COMMON_CLASSES
        for col in df.columns:
            if col not in COMMON_CLASSES:
                text_col = col
                logger.info(f"Using column '{text_col}' as report text")
                break
    
    if not text_col:
        raise ValueError("Could not identify a text column in the dataset")
    
    # Rename text column to 'report_text' if needed
    if text_col != 'report_text':
        df = df.rename(columns={text_col: 'report_text'})
    
    # Check which common classes exist in the dataset
    existing_classes = [c for c in COMMON_CLASSES if c in df.columns]
    logger.info(f"Found {len(existing_classes)} of {len(COMMON_CLASSES)} common classes in the dataset")
    
    # For any missing classes, add them as all zeros
    for class_name in COMMON_CLASSES:
        if class_name not in df.columns:
            logger.warning(f"Class {class_name} not found, adding as all zeros")
            df[class_name] = 0
    
    # Keep only report_text and the common classes
    columns_to_keep = ['report_text'] + COMMON_CLASSES
    df = df[columns_to_keep]
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the processed data
    logger.info(f"Saving test set with {len(test_df)} samples to {output_test}")
    test_df.to_csv(output_test, index=False)
    
    logger.info(f"Saving train set with {len(train_df)} samples to {output_train}")
    train_df.to_csv(output_train, index=False)
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    main() 