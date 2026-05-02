#!/usr/bin/env python
"""
Evaluation script for text embeddings.

This script:
1. Loads the precomputed text embeddings
2. Creates a simple MLP classifier to predict labels from embeddings
3. Evaluates the classifier and reports metrics
4. Saves results to JSON and CSV files
"""

import os
import sys
import json
import argparse
import logging
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)


class EmbeddingMLP(nn.Module):
    """Simple MLP for classifying text embeddings."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=14, dropout_rate=0.2):
        """
        Initialize the model.
        
        Args:
            input_dim: Dimension of the input embeddings
            hidden_dim: Dimension of hidden layer
            output_dim: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super(EmbeddingMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def load_embeddings_and_labels(
    embeddings_dir,
    label_path=None,
    max_samples=1000,
    random_seed=42
):
    """
    Load embeddings and corresponding labels.
    
    Args:
        embeddings_dir: Directory containing text embedding files
        label_path: Path to CSV/JSON file with patient_id to label mapping
        max_samples: Maximum number of samples to load
        random_seed: Random seed for reproducibility
        
    Returns:
        embeddings: Tensor of embeddings
        labels: Tensor of labels (or None if no labels)
        patient_ids: List of patient IDs
    """
    # Set random seed
    random.seed(random_seed)
    
    # Get all embedding files
    embedding_files = glob.glob(os.path.join(embeddings_dir, "*.pt"))
    logger.info(f"Found {len(embedding_files)} embedding files")
    
    # Limit the number of files
    if max_samples and max_samples < len(embedding_files):
        embedding_files = random.sample(embedding_files, max_samples)
        logger.info(f"Sampled {len(embedding_files)} embedding files")
    
    # Load embeddings
    embeddings = []
    patient_ids = []
    
    for file_path in tqdm(embedding_files, desc="Loading embeddings"):
        # Extract patient ID from filename
        patient_id = os.path.basename(file_path).replace(".pt", "")
        
        # Load embedding
        embedding = torch.load(file_path)
        
        embeddings.append(embedding)
        patient_ids.append(patient_id)
    
    # Stack embeddings into tensor
    embeddings = torch.stack(embeddings)
    
    # Load labels if provided
    labels = None
    if label_path:
        if label_path.endswith('.csv'):
            df = pd.read_csv(label_path)
            # Assuming the first column is patient_id and the rest are labels
            label_df = df[df.iloc[:, 0].isin(patient_ids)]
            # Sort the label_df to match the order of patient_ids
            label_df = label_df.set_index(label_df.columns[0]).loc[patient_ids].reset_index()
            labels = torch.tensor(label_df.iloc[:, 1:].values, dtype=torch.float32)
        elif label_path.endswith('.json'):
            with open(label_path, 'r') as f:
                label_dict = json.load(f)
            # Extract labels for patient_ids
            label_list = []
            for pid in patient_ids:
                if pid in label_dict:
                    label_list.append(label_dict[pid])
                else:
                    # If no label found, use zeros
                    label_list.append([0] * 14)  # Assuming 14 classes
            labels = torch.tensor(label_list, dtype=torch.float32)
    
    return embeddings, labels, patient_ids


def train_and_evaluate(
    embeddings_dir,
    label_path=None,
    output_dir='logs',
    test_split=0.2,
    batch_size=32,
    learning_rate=0.001,
    num_epochs=10,
    hidden_dim=256,
    random_seed=42,
    max_samples=1000
):
    """
    Train a classifier on text embeddings and evaluate it.
    
    Args:
        embeddings_dir: Directory containing text embedding files
        label_path: Path to file with patient_id to label mapping
        output_dir: Directory to save output files
        test_split: Proportion of data to use for testing
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        hidden_dim: Hidden dimension of the MLP
        random_seed: Random seed for reproducibility
        max_samples: Maximum number of samples to load
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embeddings and labels
    embeddings, labels, patient_ids = load_embeddings_and_labels(
        embeddings_dir=embeddings_dir,
        label_path=label_path,
        max_samples=max_samples,
        random_seed=random_seed
    )
    
    # Log embedding shape
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    # Save sample embedding shape to metrics file
    metrics = {
        'embedding_shape': list(embeddings.shape),
        'embedding_dim': embeddings.shape[1],
        'num_samples': embeddings.shape[0]
    }
    
    # If no labels, just report embedding stats
    if labels is None:
        logger.info("No labels provided. Saving embedding stats only.")
        metrics_path = os.path.join(output_dir, 'text_embedding_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        return metrics
    
    # Create dataset
    dataset = TensorDataset(embeddings, labels)
    
    # Split into train and test sets
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = embeddings.shape[1]
    output_dim = labels.shape[1] if len(labels.shape) > 1 else 1
    
    model = EmbeddingMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    # Define loss function and optimizer
    if output_dim > 1:
        criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    else:
        criterion = nn.BCEWithLogitsLoss()  # For binary classification
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            all_outputs.append(outputs.cpu())
            all_labels.append(batch_y.cpu())
    
    # Concatenate outputs and labels
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    # Convert to numpy
    outputs_np = all_outputs.numpy()
    labels_np = all_labels.numpy()
    
    # Calculate predictions
    if output_dim > 1:
        # Multi-label classification
        probs_np = 1 / (1 + np.exp(-outputs_np))  # Sigmoid
        preds_np = (probs_np >= 0.5).astype(float)
        
        # Calculate metrics
        accuracy = accuracy_score(labels_np, preds_np)
        precision_macro = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        recall_macro = recall_score(labels_np, preds_np, average='macro', zero_division=0)
        f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        
        # Calculate AUC-ROC if possible
        try:
            auc_roc = roc_auc_score(labels_np, probs_np, average='macro')
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
            auc_roc = float('nan')
        
        # Add metrics to dictionary
        metrics.update({
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'auc_roc_macro': float(auc_roc)
        })
    else:
        # Binary classification
        probs_np = 1 / (1 + np.exp(-outputs_np.flatten()))  # Sigmoid
        preds_np = (probs_np >= 0.5).astype(float)
        
        # Calculate metrics
        accuracy = accuracy_score(labels_np, preds_np)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        
        # Calculate AUC-ROC if possible
        try:
            auc_roc = roc_auc_score(labels_np, probs_np)
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
            auc_roc = float('nan')
        
        # Add metrics to dictionary
        metrics.update({
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc_roc)
        })
    
    # Print metrics
    logger.info("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        if key != 'embedding_shape' and key != 'embedding_dim' and key != 'num_samples':
            logger.info(f"{key}: {value:.4f}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'text_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save model
    model_path = os.path.join(output_dir, 'text_embedding_classifier.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate text embeddings")
    
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="data/processed/text_embeddings",
        help="Directory containing text embedding files"
    )
    
    parser.add_argument(
        "--label_path",
        type=str,
        default=None,
        help="Path to file with patient_id to label mapping (CSV or JSON)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimization"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the MLP"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to load"
    )
    
    args = parser.parse_args()
    
    train_and_evaluate(
        embeddings_dir=args.embeddings_dir,
        label_path=args.label_path,
        output_dir=args.output_dir,
        test_split=args.test_split,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        random_seed=args.random_seed,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main() 