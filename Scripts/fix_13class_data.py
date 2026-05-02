#!/usr/bin/env python
"""
Fix the 13-class dataset by replacing NaN and -1.0 values with 0.0.

This script:
1. Loads the 13-class test and train datasets
2. Replaces NaN and -1.0 values with 0.0
3. Saves the fixed datasets
"""

import os
import pandas as pd
import numpy as np
import logging

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

def fix_dataset(df):
    """
    Fix the dataset by replacing NaN and -1.0 values with 0.0.
    
    Args:
        df: DataFrame to fix
        
    Returns:
        DataFrame: Fixed DataFrame
    """
    # Make a copy to avoid modifying the original
    df_fixed = df.copy()
    
    # For each class column
    for col in COMMON_CLASSES:
        # Count NaN values
        nan_count = df_fixed[col].isna().sum()
        if nan_count > 0:
            logger.info(f"Replacing {nan_count} NaN values in {col} with 0.0")
            df_fixed[col] = df_fixed[col].fillna(0.0)
        
        # Count -1.0 values
        neg_count = (df_fixed[col] == -1.0).sum()
        if neg_count > 0:
            logger.info(f"Replacing {neg_count} -1.0 values in {col} with 0.0")
            df_fixed[col] = df_fixed[col].replace(-1.0, 0.0)
    
    return df_fixed

def main():
    # Define input and output paths
    input_test = 'data/processed/text/13class_test.csv'
    input_train = 'data/processed/text/13class_train.csv'
    output_test = 'data/processed/text/13class_test_fixed.csv'
    output_train = 'data/processed/text/13class_train_fixed.csv'
    
    # Load test dataset
    logger.info(f"Loading test dataset from {input_test}")
    test_df = pd.read_csv(input_test)
    logger.info(f"Loaded test dataset with {len(test_df)} samples")
    
    # Fix test dataset
    logger.info("Fixing test dataset")
    test_df_fixed = fix_dataset(test_df)
    
    # Save fixed test dataset
    logger.info(f"Saving fixed test dataset to {output_test}")
    test_df_fixed.to_csv(output_test, index=False)
    
    # Load train dataset
    logger.info(f"Loading train dataset from {input_train}")
    train_df = pd.read_csv(input_train)
    logger.info(f"Loaded train dataset with {len(train_df)} samples")
    
    # Fix train dataset
    logger.info("Fixing train dataset")
    train_df_fixed = fix_dataset(train_df)
    
    # Save fixed train dataset
    logger.info(f"Saving fixed train dataset to {output_train}")
    train_df_fixed.to_csv(output_train, index=False)
    
    logger.info("Dataset fixing completed!")

if __name__ == "__main__":
    main() 