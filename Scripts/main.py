#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Modal Health Insights Platform - Main application entry point.

This module initializes and runs the platform's data processing and analysis pipeline.
"""

import os
import sys
import logging
from typing import Dict, Any, List
import time

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Local imports
from utils.logger import setup_logger
from data.data_loader import DataLoader
from preprocessing.clinical_preprocessor import ClinicalPreprocessor
from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.text_preprocessor import TextPreprocessor
from models.model_factory import ModelFactory

# Register configuration schema
cs = ConfigStore.instance()
# cs.store(name="config_schema", node=Config)  # Uncomment when Config dataclass is defined

logger = logging.getLogger(__name__)

# ASCII Art Banner
BANNER = """
╔═════════════════════════════════════════════════════════════════╗
║                                                                 ║
║   ███╗   ███╗██╗   ██╗██╗  ████████╗██╗    ███╗   ███╗ ██████╗  ║
║   ████╗ ████║██║   ██║██║  ╚══██╔══╝██║    ████╗ ████║██╔═══██╗ ║
║   ██╔████╔██║██║   ██║██║     ██║   ██║    ██╔████╔██║██║   ██║ ║
║   ██║╚██╔╝██║██║   ██║██║     ██║   ██║    ██║╚██╔╝██║██║   ██║ ║
║   ██║ ╚═╝ ██║╚██████╔╝███████╗██║   ██████╗██║ ╚═╝ ██║╚██████╔╝ ║
║   ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝   ╚═════╝╚═╝     ╚═╝ ╚═════╝  ║
║                                                                 ║
║            Health Insights Platform - Version 0.1.0             ║
║                                                                 ║
╚═════════════════════════════════════════════════════════════════╝
"""


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the health insights platform pipeline.
    
    Args:
        cfg: Configuration parameters loaded by Hydra
    """
    # Print startup banner
    print(BANNER)
    
    # Setup logging
    setup_logger(cfg.logging.level, cfg.logging.file)
    logger.info(f"Starting Health Insights Platform with config: {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Log execution environment
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Running on: {sys.platform}")
    
    start_time = time.time()
    
    try:
        # Initialize data loaders
        logger.info("Initializing data loaders")
        data_loader = DataLoader(
            clinical_data_path=cfg.data.clinical_path,
            image_data_path=cfg.data.image_path,
            text_data_path=cfg.data.text_path
        )
        
        logger.info("----------------------------------------")
        logger.info("PHASE 1: DATA LOADING")
        logger.info("----------------------------------------")
        # Load data
        clinical_data = data_loader.load_clinical_data()
        logger.info(f"Clinical data loaded: {clinical_data.shape[0]} records with {clinical_data.shape[1]} features")
        
        image_data = data_loader.load_image_data()
        logger.info(f"Image data loaded: {len(image_data)} patients")
        
        text_data = data_loader.load_text_data()
        logger.info(f"Text data loaded: {len(text_data)} patients")
        
        logger.info("----------------------------------------")
        logger.info("PHASE 2: PREPROCESSING")
        logger.info("----------------------------------------")
        # Initialize preprocessors
        logger.info("Setting up preprocessing pipelines")
        clinical_preprocessor = ClinicalPreprocessor(cfg.preprocessing.clinical)
        image_preprocessor = ImagePreprocessor(cfg.preprocessing.image)
        text_preprocessor = TextPreprocessor(cfg.preprocessing.text)
        
        # Preprocess data
        logger.info("Starting preprocessing...")
        
        # TODO: Implement multiprocessing for parallel preprocessing
        processed_clinical = clinical_preprocessor.preprocess(clinical_data)
        logger.info(f"Clinical preprocessing complete: {processed_clinical.shape[0]} records with {processed_clinical.shape[1]} features")
        
        processed_images = image_preprocessor.preprocess(image_data)
        logger.info(f"Image preprocessing complete for {len(processed_images)} patients")
        
        processed_text = text_preprocessor.preprocess(text_data)
        logger.info(f"Text preprocessing complete for {len(processed_text)} patients")
        
        logger.info("----------------------------------------")
        logger.info("PHASE 3: MODEL INITIALIZATION")
        logger.info("----------------------------------------")
        
        # TODO: Add model training, evaluation, and inference code
        # Initialize models
        logger.info("Initializing models")
        model_factory = ModelFactory(cfg.models)
        
        # Placeholder for future model operations
        logger.info("Model training and inference to be implemented in future versions")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Health Insights Platform completed in {elapsed_time:.2f} seconds")
        logger.info("Execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error running Health Insights Platform: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 