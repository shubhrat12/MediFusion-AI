"""
Secure evaluation module for fusion models with input validation and memory monitoring.
"""

import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image

from data.data_loader_security import (
    SecureDataLoader,
    verify_memory_usage,
    verify_file_size,
    safe_path_join,
    MAX_IMAGE_SIZE_MB
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureFusionEvaluator:
    """Secure evaluator for fusion models with input validation."""
    
    def __init__(self,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.image_model = None
        self.text_model = None
        self.tabular_model = None
        self.image_transforms = None
        self.text_processor = None
    
    def validate_input_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Validate and load input image securely."""
        try:
            image_path = Path(image_path)
            
            # Validate path
            if not image_path.is_file():
                raise ValueError(f"Image file not found: {image_path}")
            
            # Validate file size
            verify_file_size(image_path, MAX_IMAGE_SIZE_MB)
            
            # Validate file extension
            if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")
            
            # Load and validate image
            with Image.open(image_path) as img:
                # Convert grayscale to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Validate dimensions
                if any(dim < 100 for dim in img.size):
                    raise ValueError(f"Image dimensions too small: {img.size}")
                
                # Create a copy to ensure the file handle is closed
                img_copy = img.copy()
            
            return img_copy
            
        except Exception as e:
            logger.error(f"Error validating input image: {e}")
            raise
    
    def validate_input_text(self, text: str, max_length: int = 10000) -> str:
        """Validate and sanitize input text."""
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Invalid text input")
            
            # Remove potentially harmful characters
            text = text.replace('\x00', '')  # Remove null bytes
            
            # Limit text length
            if len(text) > max_length:
                logger.warning(f"Text length ({len(text)}) exceeds limit, truncating")
                text = text[:max_length]
            
            return text
            
        except Exception as e:
            logger.error(f"Error validating input text: {e}")
            raise
    
    def validate_tabular_input(self, features: Dict[str, float]) -> pd.DataFrame:
        """Validate tabular input features."""
        try:
            required_features = {
                'age', 'gender', 'blood_pressure', 'heart_rate', 'temperature',
                'oxygen_saturation', 'glucose'
            }
            
            # Check for required features
            missing = required_features - set(features.keys())
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            
            # Validate value ranges
            validations = {
                'age': (0, 120),
                'blood_pressure': (60, 200),
                'heart_rate': (30, 200),
                'temperature': (35, 42),
                'oxygen_saturation': (60, 100),
                'glucose': (30, 500)
            }
            
            for feature, (min_val, max_val) in validations.items():
                if feature in features:
                    value = features[feature]
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Invalid type for {feature}: {type(value)}")
                    if not min_val <= value <= max_val:
                        raise ValueError(f"Value for {feature} outside valid range: {value}")
            
            # Convert to DataFrame
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error validating tabular input: {e}")
            raise
    
    def load_models(self, model_paths: Dict[str, str]):
        """Load all models securely."""
        try:
            verify_memory_usage()
            
            # Load models securely from validated paths
            for model_type, path in model_paths.items():
                path = Path(path)
                if not path.is_file():
                    raise FileNotFoundError(f"Model file not found: {path}")
                
                verify_file_size(path, 1000)  # 1GB limit for model files
                
                if model_type == 'image':
                    self.image_model = torch.load(path, map_location=self.device)
                elif model_type == 'text':
                    self.text_model = torch.load(path, map_location=self.device)
                elif model_type == 'tabular':
                    self.tabular_model = torch.load(path, map_location=self.device)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                logger.info(f"Successfully loaded {model_type} model from {path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict(self,
               image_path: Optional[str] = None,
               text: Optional[str] = None,
               tabular_features: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Make predictions with input validation and secure processing."""
        try:
            verify_memory_usage()
            predictions = {}
            
            # Process image if provided
            if image_path:
                img = self.validate_input_image(image_path)
                if self.image_transforms:
                    img = self.image_transforms(img)
                img = img.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    predictions['image'] = self.image_model(img).sigmoid().cpu().numpy()
            
            # Process text if provided
            if text:
                text = self.validate_input_text(text)
                if self.text_processor:
                    text_features = self.text_processor.process_text(text)
                    text_features = {k: v.to(self.device) for k, v in text_features.items()}
                    with torch.no_grad():
                        predictions['text'] = self.text_model(**text_features).sigmoid().cpu().numpy()
            
            # Process tabular features if provided
            if tabular_features:
                df = self.validate_tabular_input(tabular_features)
                with torch.no_grad():
                    predictions['tabular'] = self.tabular_model(
                        torch.FloatTensor(df.values).to(self.device)
                    ).sigmoid().cpu().numpy()
            
            return self.fuse_predictions(predictions)
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def fuse_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Fuse predictions from different models securely."""
        try:
            if not predictions:
                raise ValueError("No predictions to fuse")
            
            # Initialize weights for each model type
            weights = {
                'image': 0.4,
                'text': 0.4,
                'tabular': 0.2
            }
            
            # Combine predictions with weights
            final_pred = np.zeros_like(next(iter(predictions.values())))
            weight_sum = 0
            
            for model_type, pred in predictions.items():
                if model_type in weights:
                    final_pred += pred * weights[model_type]
                    weight_sum += weights[model_type]
            
            if weight_sum > 0:
                final_pred /= weight_sum
            
            # Convert to disease probabilities
            diseases = [
                'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
                'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                'Pleural Effusion', 'Pleural Other', 'Fracture',
                'Support Devices', 'No Finding'
            ]
            
            return {disease: float(prob) for disease, prob in zip(diseases, final_pred[0])}
            
        except Exception as e:
            logger.error(f"Error fusing predictions: {e}")
            raise
