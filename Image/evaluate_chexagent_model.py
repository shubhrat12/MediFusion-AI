#!/usr/bin/env python
"""
Evaluation script for fine-tuned CheXagent model on 13 common classes.

This script:
1. Loads the fine-tuned CheXagent model
2. Evaluates it on the test dataset
3. Reports comprehensive metrics (F1 Score, AUC, Accuracy, per-class metrics)
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)
from torch.utils.data import DataLoader
from torchvision import transforms
import json

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

class ChestXrayDataset(torch.utils.data.Dataset):
    """Dataset for chest X-ray images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.FloatTensor(label)


def load_test_data(data_path):
    """Load test data from CSV file."""
    logger.info(f"Loading test data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Extract image paths and labels
    image_paths = df['image_path'].tolist()
    labels = df[COMMON_CLASSES].values
    
    logger.info(f"Loaded {len(image_paths)} test samples")
    return image_paths, labels


def load_model(model_path, device):
    """Load the fine-tuned CheXagent model."""
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load the model
        model = torch.load(model_path, map_location=device)
        model.eval()
        logger.info("Successfully loaded model")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    return model


def evaluate(model, dataloader, device):
    """Evaluate the model on test data."""
    logger.info("Starting evaluation...")
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities >= 0.5).astype(int)
            
            all_labels.append(labels)
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
    
    # Concatenate batches
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    
    # Calculate metrics
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
    metrics['precision_macro'] = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    try:
        metrics['auc_roc_macro'] = roc_auc_score(all_labels, all_probabilities, average='macro')
    except ValueError as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        metrics['auc_roc_macro'] = float('nan')
    
    # Per-class metrics
    metrics['per_class'] = {}
    for i, class_name in enumerate(COMMON_CLASSES):
        metrics['per_class'][class_name] = {
            'precision': precision_score(all_labels[:, i], all_predictions[:, i], zero_division=0),
            'recall': recall_score(all_labels[:, i], all_predictions[:, i], zero_division=0),
            'f1': f1_score(all_labels[:, i], all_predictions[:, i], zero_division=0),
            'auc': roc_auc_score(all_labels[:, i], all_probabilities[:, i]) if len(np.unique(all_labels[:, i])) > 1 else float('nan')
        }
    
    # Get classification report
    report = classification_report(
        all_labels, all_predictions,
        target_names=COMMON_CLASSES,
        output_dict=True,
        zero_division=0
    )
    metrics['classification_report'] = report
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned CheXagent model")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument('--test_data', type=str, required=True,
                        help="Path to test data CSV")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load test data
    image_paths, labels = load_test_data(args.test_data)
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ChestXrayDataset(image_paths, labels, transform=transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    metrics = evaluate(model, test_dataloader, device)
    
    # Save metrics
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print key metrics
    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  AUC-ROC Macro: {metrics['auc_roc_macro']:.4f}")
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main() 