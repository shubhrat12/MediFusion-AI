"""
Multi-modal datasets for medical data.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from data.data_loader_security import (
    verify_memory_usage,
    verify_file_size,
    MAX_IMAGE_SIZE_MB,
    MAX_CSV_SIZE_MB
)

logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """Secure dataset for multi-modal medical data."""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 image_transform=None,
                 text_tokenizer=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.image_paths = []
        self.texts = []
        self.tabular_features = []
        self.labels = []
        
        # Process metadata
        self._process_metadata()
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata file."""
        try:
            metadata_path = self.data_dir / f'{self.split}_metadata.csv'
            verify_file_size(metadata_path, MAX_CSV_SIZE_MB)
            
            df = pd.read_csv(metadata_path)
            required_columns = {'image_path', 'text', 'features', 'labels'}
            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def _process_metadata(self):
        """Process and validate all data entries."""
        try:
            verify_memory_usage()
            
            for _, row in self.metadata.iterrows():
                try:
                    # Validate image
                    img_path = self.data_dir / row['image_path']
                    verify_file_size(img_path, MAX_IMAGE_SIZE_MB)
                    if not img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        continue
                    
                    # Validate text
                    if not isinstance(row['text'], str):
                        continue
                    
                    # Validate features
                    features = eval(row['features'])  # Assuming stored as string repr of dict
                    if not isinstance(features, dict):
                        continue
                    
                    # Store valid data
                    self.image_paths.append(img_path)
                    self.texts.append(row['text'])
                    self.tabular_features.append(features)
                    self.labels.append(eval(row['labels']))
                    
                except Exception as e:
                    logger.warning(f"Skipping invalid entry: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error processing metadata: {e}")
            raise
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            verify_memory_usage()
            
            # Load and process image
            with Image.open(self.image_paths[idx]) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if self.image_transform:
                    img = self.image_transform(img)
            
            # Process text
            text = self.texts[idx]
            if self.text_tokenizer:
                text = self.text_tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
            
            # Convert tabular features to tensor
            features = torch.FloatTensor(list(self.tabular_features[idx].values()))
            
            # Convert labels to tensor
            labels = torch.FloatTensor(self.labels[idx])
            
            return {
                'image': img,
                'text': text,
                'tabular': features,
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            raise
