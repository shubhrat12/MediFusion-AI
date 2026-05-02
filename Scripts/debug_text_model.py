#!/usr/bin/env python
"""
Debug script for text classification model.

This script:
1. Loads the pretrained text classification model
2. Evaluates it on a small batch of data
3. Prints detailed information about the model outputs
"""

import os
import torch
import pandas as pd
import numpy as np
import logging
from transformers import BertTokenizer, BertForSequenceClassification

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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model_path = "data/raw/text/archive (2)/text_classifier_model.pt"
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create model architecture
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(COMMON_CLASSES),
        problem_type="multi_label_classification"
    )
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Filter state dict to only include matching keys
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model.state_dict():
                if model.state_dict()[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    logger.warning(f"Shape mismatch for {key}: {value.shape} vs {model.state_dict()[key].shape}")
        
        # Load filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        logger.info("Loaded partial state dict with matching keys and shapes")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load a small batch of data
    test_data_path = "data/processed/text/13class_test.csv"
    logger.info(f"Loading test data from {test_data_path}")
    
    df = pd.read_csv(test_data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Take just 5 samples for debugging
    df = df.head(5)
    
    # Process each sample individually
    for i, row in df.iterrows():
        text = row['report_text']
        labels = row[COMMON_CLASSES].values
        
        logger.info(f"Processing sample {i+1}")
        logger.info(f"Text: {text[:100]}...")
        logger.info(f"Labels: {labels}")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            padding='max_length',
            truncation=True
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Print logits
                logger.info(f"Logits: {logits.cpu().numpy()}")
                
                # Convert to probabilities
                probs = torch.sigmoid(logits).cpu().numpy()
                logger.info(f"Probabilities: {probs}")
                
                # Get predictions
                preds = (probs >= 0.5).astype(int)
                logger.info(f"Predictions: {preds}")
                
                # Check for NaN values
                if np.isnan(probs).any():
                    logger.warning("NaN values detected in probabilities")
                
                logger.info("-" * 50)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
    
    logger.info("Debug completed")

if __name__ == "__main__":
    main() 