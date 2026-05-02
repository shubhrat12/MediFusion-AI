#!/usr/bin/env python
"""
Extract text embeddings from clinical text data.

This script:
1. Loads text data from CSV or TXT files
2. Cleans and preprocesses the text
3. Uses a pretrained clinical language model to extract embeddings
4. Saves the embeddings for each patient ID
"""

import os
import sys
import argparse
import logging
import csv
import re
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from models.model_factory import ModelFactory


def clean_text(text: str) -> str:
    """
    Clean and preprocess clinical text.
    
    Args:
        text: Raw clinical text
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove headers/footers (simplified)
    # This can be expanded based on specific format of clinical notes
    text = re.sub(r'^\s*[-_]+.*?\n', '', text, flags=re.MULTILINE)  # Headers
    text = re.sub(r'\n.*?[-_]+\s*$', '', text, flags=re.MULTILINE)  # Footers
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def standardize_medical_terms(text: str, term_map: Optional[Dict[str, str]] = None) -> str:
    """
    Standardize medical terms in text.
    
    Args:
        text: Input text
        term_map: Dictionary mapping non-standard terms to standard ones
        
    Returns:
        str: Text with standardized medical terms
    """
    if term_map is None:
        # Default term map - expand this for your specific use case
        term_map = {
            "chf": "congestive heart failure",
            "cabg": "coronary artery bypass graft",
            "mi": "myocardial infarction",
            "dm": "diabetes mellitus",
            "htn": "hypertension",
            "cad": "coronary artery disease",
            "ckd": "chronic kidney disease",
            "afib": "atrial fibrillation"
        }
    
    # Replace terms
    for term, replacement in term_map.items():
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(term) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def load_text_data(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load text data from a file.
    
    Args:
        file_path: Path to the file containing text data
        
    Returns:
        Tuple[List[str], List[str]]: List of patient IDs and corresponding texts
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        # Attempt to load CSV file
        try:
            df = pd.read_csv(file_path)
            
            # Try to identify patient ID and text columns
            id_col = None
            text_col = None
            
            # Look for common column names
            for col in df.columns:
                if col.lower() in ['id', 'patient', 'patient_id', 'patientid', 'subject_id', 'subjectid', 'patient id']:
                    id_col = col
                elif col.lower() in ['text', 'note', 'report', 'discharge_summary', 'clinical_note', 'content', 'transcript']:
                    text_col = col
            
            # If columns weren't found, use first column as ID and second as text
            if id_col is None or text_col is None:
                logger.warning(f"Couldn't identify ID and text columns. Using first column as ID and second as text.")
                id_col = df.columns[0]
                text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            logger.info(f"Using '{id_col}' as ID column and '{text_col}' as text column")
            
            # Extract data
            patient_ids = df[id_col].astype(str).tolist()
            texts = df[text_col].astype(str).tolist()
            
            return patient_ids, texts
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            return [], []
            
    elif file_ext == '.txt':
        # For text files, assume one document per file with filename as ID
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Use filename as patient ID
            patient_id = os.path.splitext(os.path.basename(file_path))[0]
            
            return [patient_id], [text]
            
        except Exception as e:
            logger.error(f"Error loading text file: {str(e)}")
            return [], []
            
    else:
        logger.error(f"Unsupported file extension: {file_ext}")
        return [], []


def extract_embeddings(
    input_dir: str,
    output_dir: str,
    model_type: str = 'clinicalbert',
    pooling_strategy: str = 'cls',
    batch_size: int = 16,
    max_length: int = 512,
    file_pattern: str = '*.csv',
    device: Optional[str] = None
):
    """
    Extract embeddings from text files and save them.
    
    Args:
        input_dir: Directory containing input text files
        output_dir: Directory to save output embeddings
        model_type: Type of model to use ('clinicalbert', 'biobert', 'pubmedbert')
        pooling_strategy: Pooling strategy ('cls' or 'mean')
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        file_pattern: Pattern to match input files
        device: Device to run inference on ('cuda', 'cpu', None)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Initialize model factory with text model configuration
    model_config = {
        'architecture': model_type,
        'pretrained': True,
        'freeze_encoder': True,  # We're just using the model for inference
        'pooling_strategy': pooling_strategy,
        'output_dim': 768  # Default BERT dimension
    }
    
    factory = ModelFactory({'text': model_config})
    model = factory.get_model('text')
    
    if model is None:
        logger.error("Failed to initialize model")
        return
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Get list of input files
    input_files = list(Path(input_dir).glob(file_pattern))
    logger.info(f"Found {len(input_files)} input files matching pattern '{file_pattern}'")
    
    # Dict to track all patient IDs and their embeddings
    all_embeddings = {}
    patient_id_map = {}
    
    # Process each file
    for file_path in tqdm(input_files, desc="Processing files"):
        logger.info(f"Processing file: {file_path}")
        
        # Load text data
        patient_ids, texts = load_text_data(str(file_path))
        
        if not texts:
            logger.warning(f"No texts found in {file_path}")
            continue
        
        logger.info(f"Loaded {len(texts)} texts from {file_path}")
        
        # Clean and standardize each text
        cleaned_texts = []
        for text in tqdm(texts, desc="Cleaning texts", leave=False):
            # Clean text
            cleaned = clean_text(text)
            
            # Standardize medical terms
            cleaned = standardize_medical_terms(cleaned)
            
            cleaned_texts.append(cleaned)
        
        # Process in batches
        for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Extracting embeddings", leave=False):
            batch_texts = cleaned_texts[i:i+batch_size]
            batch_ids = patient_ids[i:i+batch_size]
            
            # Skip empty batches
            if not batch_texts:
                continue
            
            # Extract embeddings
            with torch.no_grad():
                # Preprocess texts for the model
                inputs = model.preprocess_text(batch_texts, max_length=max_length)
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass through model
                embeddings = model(inputs)
            
            # Add to dictionary
            for j, patient_id in enumerate(batch_ids):
                # Store embeddings as a NumPy array
                all_embeddings[patient_id] = embeddings[j].cpu().numpy()
                
                # Save patient ID mapping for verification
                patient_id_map[patient_id] = {"source_file": str(file_path)}
    
    # Save embeddings to output directory
    logger.info(f"Saving embeddings for {len(all_embeddings)} patients to {output_dir}")
    
    for patient_id, embedding in tqdm(all_embeddings.items(), desc="Saving embeddings"):
        # Create output path
        output_path = os.path.join(output_dir, f"{patient_id}.pt")
        
        # Convert to tensor
        embedding_tensor = torch.tensor(embedding)
        
        # Save embedding
        torch.save(embedding_tensor, output_path)
    
    # Save patient ID mapping as JSON
    mapping_path = os.path.join(output_dir, "patient_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(patient_id_map, f, indent=2)
    
    logger.info(f"Extraction complete. Saved {len(all_embeddings)} embeddings to {output_dir}")


def main():
    """
    Main function to parse command line arguments and run the extraction.
    """
    parser = argparse.ArgumentParser(description='Extract embeddings from clinical text')
    
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input text files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output embeddings')
    
    # Optional arguments
    parser.add_argument('--model_type', type=str, default='clinicalbert',
                        choices=['clinicalbert', 'biobert', 'pubmedbert'],
                        help='Type of model to use')
    parser.add_argument('--pooling_strategy', type=str, default='cls',
                        choices=['cls', 'mean'],
                        help='Pooling strategy for embeddings')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--file_pattern', type=str, default='*.csv',
                        help='Pattern to match input files')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cuda, cpu, None)')
    
    args = parser.parse_args()
    
    # Run extraction
    extract_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        pooling_strategy=args.pooling_strategy,
        batch_size=args.batch_size,
        max_length=args.max_length,
        file_pattern=args.file_pattern,
        device=args.device
    )


if __name__ == '__main__':
    main() 