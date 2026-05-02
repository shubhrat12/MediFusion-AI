#!/usr/bin/env python
"""
Fine-tuning script for text classification model on 13 common classes.

This script:
1. Loads the pretrained text classification model
2. Reinitializes the final classification layer for 13 classes
3. Fine-tunes the model on a dataset with only the 13 common classes
4. Saves the fine-tuned model
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup

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


def load_data(data_path, common_classes, val_size=0.1, random_state=42):
    """
    Load training data and filter to only include the 13 common classes.
    Split into train and validation sets.
    
    Args:
        data_path: Path to the CSV file containing training data
        common_classes: List of the 13 common classes to keep
        val_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_texts: List of training report texts
        val_texts: List of validation report texts
        train_labels: Numpy array of training labels (only for the 13 common classes)
        val_labels: Numpy array of validation labels (only for the 13 common classes)
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure all common classes exist in the dataset
    for class_name in common_classes:
        if class_name not in df.columns:
            logger.warning(f"Class {class_name} not found in dataset, adding as all zeros")
            df[class_name] = 0
    
    # Extract texts and labels (only for common classes)
    texts = df['report_text'].tolist()  # Assuming the text column is named 'report_text'
    labels = df[common_classes].values
    
    # Split into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_size, random_state=random_state
    )
    
    logger.info(f"Train samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    return train_texts, val_texts, train_labels, val_labels


def load_pretrained_model(model_path, device, num_classes=13, freeze_base=True):
    """
    Load the pretrained text classification model and reinitialize the final layer.
    
    Args:
        model_path: Path to the pretrained model
        device: Device to load the model onto
        num_classes: Number of classes in the model
        freeze_base: Whether to freeze the BERT base model
        
    Returns:
        model: Modified model with reinitialized classifier
        tokenizer: BERT tokenizer
    """
    logger.info(f"Loading pretrained model from {model_path}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    try:
        # Try to load the full model with architecture
        pretrained_model = torch.load(model_path, map_location=device)
        
        # Check if it's a state dict or a full model
        if isinstance(pretrained_model, dict) and 'state_dict' in pretrained_model:
            # It's a checkpoint with state dict
            state_dict = pretrained_model['state_dict']
        elif isinstance(pretrained_model, dict):
            # It's just a state dict
            state_dict = pretrained_model
        else:
            # It's a full model, extract state dict
            state_dict = pretrained_model.state_dict()
            
    except Exception as e:
        logger.warning(f"Could not load model as expected: {e}")
        logger.info("Creating new model from BERT base...")
        
        # Create new model from scratch
        config = BertConfig.from_pretrained(
            'bert-base-uncased',
            num_labels=num_classes,
            problem_type='multi_label_classification'
        )
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
        model.to(device)
        
        if freeze_base:
            logger.info("Freezing BERT base layers")
            for param in model.bert.parameters():
                param.requires_grad = False
        
        return model, tokenizer
    
    # Create new model with correct output size
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=num_classes,
        problem_type='multi_label_classification'
    )
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    
    # Filter state dict to only include matching keys
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key in model.state_dict() and 'classifier' not in key:
            if model.state_dict()[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                logger.warning(f"Shape mismatch for {key}: {value.shape} vs {model.state_dict()[key].shape}")
    
    # Load filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)
    logger.info("Loaded pretrained weights for all layers except classifier")
    
    # Freeze BERT base if requested
    if freeze_base:
        logger.info("Freezing BERT base layers")
        for param in model.bert.parameters():
            param.requires_grad = False
    
    model.to(device)
    return model, tokenizer


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    device,
    epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=0,
    verbose=True
):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        device: Device to train on
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for learning rate scheduler
        verbose: Whether to print progress
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Set model to training mode
    model.train()
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize loss function (BCEWithLogitsLoss for multi-label)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_dataloader
        
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update running loss
            running_loss += loss.item()
            
            if verbose:
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        # Evaluate on validation set
        val_loss, val_f1 = evaluate_model(model, val_dataloader, device, criterion)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "models/text_classifier_13class_best.pt")
            logger.info(f"Saved new best model with Val F1: {val_f1:.4f}")
    
    return model, history


def evaluate_model(model, dataloader, device, criterion=None):
    """
    Evaluate the model on validation data.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for validation data
        device: Device to evaluate on
        criterion: Loss function (optional)
        
    Returns:
        avg_loss: Average loss
        f1: F1 score (macro)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            # Calculate loss
            if criterion:
                loss = criterion(logits, labels)
                running_loss += loss.item()
            
            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            # Move to CPU for metric calculation
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
    
    # Concatenate all batches
    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    # Calculate average loss
    avg_loss = running_loss / len(dataloader) if criterion else 0.0
    
    return avg_loss, f1


def main():
    parser = argparse.ArgumentParser(description="Fine-tune text classification model for 13 common classes")
    parser.add_argument('--pretrained_model', type=str, default='data/raw/text/archive (2)/text_classifier_model.pt',
                        help="Path to the pretrained text model")
    parser.add_argument('--train_data', type=str, default='data/processed/text/13class_train_fixed.csv',
                        help="Path to the training dataset CSV")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument('--freeze_base', action='store_true',
                        help="Whether to freeze the BERT base model")
    parser.add_argument('--output_model', type=str, default='models/text_classifier_13class.pt',
                        help="Path to save the fine-tuned model")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load pretrained model and tokenizer
    model, tokenizer = load_pretrained_model(
        args.pretrained_model,
        device,
        num_classes=len(COMMON_CLASSES),
        freeze_base=args.freeze_base
    )
    
    # Load and split data
    train_texts, val_texts, train_labels, val_labels = load_data(
        args.train_data,
        COMMON_CLASSES
    )
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Train model
    logger.info("Starting training...")
    model, history = train_model(
        model,
        train_dataloader,
        val_dataloader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Load best model
    try:
        model.load_state_dict(torch.load("models/text_classifier_13class_best.pt"))
        logger.info("Loaded best model from checkpoint")
    except:
        logger.warning("Could not load best model, using last epoch")
    
    # Save final model
    torch.save(model.state_dict(), args.output_model)
    logger.info(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main() 