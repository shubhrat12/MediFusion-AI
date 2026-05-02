import os
import sys
import logging
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import data loaders
from data.data_loader import load_clinical_tabular, load_chestxray_images, load_mtsamples_text

# Import preprocessors
from preprocessing.clinical_preprocessor import TabularPreprocessor
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.text_preprocessor import TextPreprocessor

def test_tabular_loader():
    """Test the tabular data loader and preprocessor"""
    logger.info("Testing tabular data loader...")
    try:
        # Load the data
        df = load_clinical_tabular()
        logger.info(f"Tabular data loaded successfully. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"First 5 rows:\n{df.head()}")
        
        # Test preprocessor
        logger.info("Testing tabular preprocessor...")
        preprocessor = TabularPreprocessor()
        preprocessed = preprocessor.fit_transform(df.drop('target', axis=1))
        logger.info(f"Preprocessed data shape: {preprocessed.shape}")
        logger.info(f"Feature names: {preprocessor.get_feature_names()[:5]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error in tabular testing: {str(e)}", exc_info=True)
        return False

def test_image_loader():
    """Test the image data loader and preprocessor"""
    logger.info("Testing image data loader...")
    try:
        # Load the data with a small sample size
        image_data, image_paths = load_chestxray_images(sample_size=5)
        logger.info(f"Image metadata loaded successfully. Shape: {image_data.shape}")
        logger.info(f"Number of image paths: {len(image_paths)}")
        
        if len(image_paths) > 0:
            # Test preprocessor with the first image
            logger.info("Testing image preprocessor...")
            preprocessor = ImagePreprocessor()
            image_tensor = preprocessor.preprocess(image_paths[0])
            logger.info(f"Preprocessed image tensor shape: {image_tensor.shape}")
        else:
            logger.warning("No image paths found to test preprocessing")
        
        return True
    except Exception as e:
        logger.error(f"Error in image testing: {str(e)}", exc_info=True)
        return False

def test_text_loader():
    """Test the text data loader and preprocessor"""
    logger.info("Testing text data loader...")
    try:
        # Load the data
        texts = load_mtsamples_text()
        logger.info(f"Text data loaded successfully. Number of texts: {len(texts)}")
        if len(texts) > 0:
            logger.info(f"First text (truncated): {texts[0][:100]}...")
        
        # Test preprocessor
        logger.info("Testing text preprocessor...")
        config = {'model_name': 'dmis-lab/biobert-base-cased-v1.1'}
        try:
            preprocessor = TextPreprocessor(config)
            preprocessor.load_tokenizer()
            
            if len(texts) > 0:
                # Test tokenization on first text
                tokenized = preprocessor.tokenize(texts[0])
                logger.info(f"Tokenized input_ids shape: {tokenized['input_ids'].shape}")
        except ImportError:
            logger.warning("Could not load tokenizer - likely missing transformers package")
        
        return True
    except Exception as e:
        logger.error(f"Error in text testing: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting validation of all data loaders and preprocessors")
    
    # Test each modality
    tabular_ok = test_tabular_loader()
    image_ok = test_image_loader()
    text_ok = test_text_loader()
    
    # Report results
    if tabular_ok and image_ok and text_ok:
        logger.info("All loaders validated successfully!")
    else:
        logger.warning("Some loaders failed validation:")
        logger.warning(f"  Tabular: {'OK' if tabular_ok else 'FAILED'}")
        logger.warning(f"  Image: {'OK' if image_ok else 'FAILED'}")
        logger.warning(f"  Text: {'OK' if text_ok else 'FAILED'}") 