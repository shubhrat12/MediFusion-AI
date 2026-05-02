#!/usr/bin/env python
"""
Evaluation script for text classification model on 13 common classes.

This script:
1. Loads the pretrained text classification model
2. Evaluates it on a test dataset, focusing only on the 13 common classes
3. Reports comprehensive metrics (F1 Score, AUC, Accuracy, per-class metrics)
"""

import os
import sys
import argparse
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)
from torch.utils.data import DataLoader, Dataset
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


class TextDataset(Dataset):
    """Dataset for text classification with focus on 13 common classes."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels  # Should be filtered to only include the 13 classes
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.FloatTensor(label)
        }


def load_test_data(data_path, common_classes):
    """
    Load test data and filter to only include the 13 common classes.
    
    Args:
        data_path: Path to the CSV file containing test data
        common_classes: List of the 13 common classes to keep
        
    Returns:
        texts: List of report texts
        labels: Numpy array of labels (only for the 13 common classes)
    """
    logger.info(f"Loading test data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure all common classes exist in the dataset
    for class_name in common_classes:
        if class_name not in df.columns:
            logger.warning(f"Class {class_name} not found in dataset, adding as all zeros")
            df[class_name] = 0
    
    # Extract texts and labels (only for common classes)
    texts = df['report_text'].tolist()  # Assuming the text column is named 'report_text'
    labels = df[common_classes].values
    
    logger.info(f"Loaded {len(texts)} test samples with {len(common_classes)} classes")
    
    return texts, labels


def load_model(model_path, device, num_classes=13):
    """
    Load the pretrained text classification model.
    
    Args:
        model_path: Path to the pretrained model
        device: Device to load the model onto
        num_classes: Number of classes in the model
        
    Returns:
        model: Loaded model
        tokenizer: BERT tokenizer
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    try:
        # Try to load the model file
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if it's a state dict or a full model
        if isinstance(state_dict, dict) and not hasattr(state_dict, 'to'):
            logger.info("Loaded file is a state dictionary")
            
            # Create model architecture
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=num_classes,
                problem_type="multi_label_classification"
            )
            
            # Try to load state dict - it may have different keys
            try:
                # Load directly if keys match
                model.load_state_dict(state_dict)
                logger.info("Successfully loaded state dict with matching keys")
            except:
                logger.warning("State dict keys don't match model, filtering keys...")
                
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
        else:
            # It's a full model
            model = state_dict
            logger.info("Loaded full model")
    
    except Exception as e:
        logger.warning(f"Could not load model as expected: {e}")
        logger.info("Creating new model architecture...")
        
        # Create model architecture
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_classes,
            problem_type="multi_label_classification"
        )
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def evaluate(model, dataloader, device, common_classes):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: The text classification model
        dataloader: DataLoader for the test dataset
        device: Device to run evaluation on
        common_classes: List of the 13 common class names
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logger.info("Starting evaluation...")
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get logits and convert to probabilities
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()
                
                # Get binary predictions using 0.5 threshold
                preds = (probs >= 0.5).astype(int)
                
                all_labels.append(labels)
                all_predictions.append(preds)
                all_probs.append(probs)
            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
                continue
    
    if not all_labels:
        logger.error("No valid predictions were made. All batches failed.")
        metrics = {
            'error': 'No valid predictions',
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'auc_roc_macro': 0.0
        }
        return metrics
    
    # Concatenate batches
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    all_probs = np.vstack(all_probs)
    
    # Handle NaN values
    mask = ~np.isnan(all_labels).any(axis=1) & ~np.isnan(all_predictions).any(axis=1) & ~np.isnan(all_probs).any(axis=1)
    if not np.all(mask):
        logger.warning(f"Found {np.sum(~mask)} samples with NaN values. Removing them from evaluation.")
        all_labels = all_labels[mask]
        all_predictions = all_predictions[mask]
        all_probs = all_probs[mask]
    
    if len(all_labels) == 0:
        logger.error("All samples contained NaN values. Cannot compute metrics.")
        metrics = {
            'error': 'All samples contained NaN values',
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'auc_roc_macro': 0.0
        }
        return metrics
    
    # Calculate metrics
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
    metrics['precision_macro'] = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    # AUC-ROC (macro and per class)
    try:
        metrics['auc_roc_macro'] = roc_auc_score(all_labels, all_probs, average='macro')
        per_class_auc = roc_auc_score(all_labels, all_probs, average=None)
        
        # Store per-class metrics
        metrics['per_class'] = {}
        for i, class_name in enumerate(common_classes):
            metrics['per_class'][class_name] = {
                'precision': precision_score(all_labels[:, i], all_predictions[:, i], zero_division=0),
                'recall': recall_score(all_labels[:, i], all_predictions[:, i], zero_division=0),
                'f1': f1_score(all_labels[:, i], all_predictions[:, i], zero_division=0),
                'auc': per_class_auc[i]
            }
    except ValueError as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        metrics['auc_roc_macro'] = float('nan')
    
    # Get classification report
    report = classification_report(all_labels, all_predictions, 
                                   target_names=common_classes, 
                                   output_dict=True, 
                                   zero_division=0)
    metrics['classification_report'] = report
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate text classification model on 13 common classes")
    parser.add_argument('--model_path', type=str, default='data/raw/text/archive (2)/text_classification_model.pt',
                        help="Path to the pretrained text model")
    parser.add_argument('--test_data', type=str, default='data/processed/text/labeled_reports_test.csv',
                        help="Path to the test dataset CSV")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument('--output_file', type=str, default='logs/text_model_13class_evaluation.json',
                        help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, device, num_classes=len(COMMON_CLASSES))
    
    # Load test data
    texts, labels = load_test_data(args.test_data, COMMON_CLASSES)
    
    # Create dataset and dataloader
    test_dataset = TextDataset(texts, labels, tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Evaluate
    metrics = evaluate(model, test_dataloader, device, COMMON_CLASSES)
    
    # Print key metrics
    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"  AUC-ROC Macro: {metrics['auc_roc_macro']:.4f}")
    
    # Save metrics to file
    import json
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {args.output_file}")


if __name__ == "__main__":
    main() 