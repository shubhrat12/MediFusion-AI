"""
Secure text processing module for medical reports and clinical text data.
"""

import os
import sys
import logging
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer
import re
from pathlib import Path

from .data_loader_security import (
    SecureDataLoader,
    verify_memory_usage,
    verify_file_size,
    MAX_CSV_SIZE_MB
)

logger = logging.getLogger(__name__)

class SecureTextProcessor:
    """Secure text processor with input validation and memory limits."""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = None
    
    def load_tokenizer(self, model_name: str = 'dmis-lab/biobert-base-cased-v1.1'):
        """Securely load the tokenizer."""
        try:
            verify_memory_usage()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def validate_text(self, text: str) -> str:
        """Validate and sanitize input text."""
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input")
        
        # Remove potentially harmful characters
        text = re.sub(r'[^\w\s.,;:?!()\[\]{}\-\'\"]+', '', text)
        
        # Limit text length
        if len(text) > self.max_length * 10:  # Approximate character limit
            logger.warning(f"Text length ({len(text)}) exceeds recommended limit")
            text = text[:self.max_length * 10]
        
        return text
    
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text with security checks."""
        try:
            # Verify memory
            verify_memory_usage()
            
            # Validate input
            text = self.validate_text(text)
            
            if not self.tokenizer:
                raise ValueError("Tokenizer not initialized")
            
            # Tokenize with length limit
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise

class SecureTextDataset(torch.utils.data.Dataset):
    """Secure dataset for text data with validation."""
    
    def __init__(self, 
                 texts: List[str],
                 labels: Optional[List[int]] = None,
                 max_length: int = 512):
        self.processor = SecureTextProcessor(max_length)
        self.processor.load_tokenizer()
        
        # Validate and process texts
        self.texts = []
        for text in texts:
            try:
                clean_text = self.processor.validate_text(text)
                self.texts.append(clean_text)
            except Exception as e:
                logger.warning(f"Skipping invalid text: {e}")
        
        self.labels = labels
        
        if labels is not None and len(self.texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Verify memory before processing
        verify_memory_usage()
        
        text = self.texts[idx]
        tokens = self.processor.process_text(text)
        
        if self.labels is not None:
            return tokens, self.labels[idx]
        return tokens

def load_text_data(
    file_path: str,
    text_column: str,
    label_columns: Optional[List[str]] = None
) -> Tuple[List[str], Optional[List[int]]]:
    """
    Securely load text data from CSV file.
    
    Args:
        file_path: Path to CSV file containing text data
        text_column: Name of column containing text
        label_columns: Optional list of label column names
    
    Returns:
        Tuple of (texts, labels)
    """
    try:
        # Verify file
        verify_file_size(file_path, MAX_CSV_SIZE_MB)
        
        # Read data in chunks
        chunk_size = 1000
        texts = []
        labels = [] if label_columns else None
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Validate text column
            if text_column not in chunk.columns:
                raise ValueError(f"Text column '{text_column}' not found")
            
            # Add texts
            texts.extend(chunk[text_column].fillna('').tolist())
            
            # Add labels if needed
            if label_columns:
                for col in label_columns:
                    if col not in chunk.columns:
                        raise ValueError(f"Label column '{col}' not found")
                chunk_labels = chunk[label_columns].values.tolist()
                labels.extend(chunk_labels)
            
            # Check memory
            verify_memory_usage()
        
        return texts, labels
        
    except Exception as e:
        logger.error(f"Error loading text data: {e}")
        raise
