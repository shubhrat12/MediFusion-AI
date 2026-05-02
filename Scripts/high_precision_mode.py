#!/usr/bin/env python
"""
High Precision Model for Chest X-ray Classification

This script focuses on achieving high precision for chest X-ray classification by:
1. Focusing only on the most common/identifiable conditions
2. Using higher classification thresholds (0.5-0.9) to improve precision
3. Using precision-weighted loss and metrics
4. Using MONAI's DenseNet121 pretrained on medical images
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
try:
    from models.attention_xray_model import AttentionXrayModel
    from data.data_loader import load_chestxray_images
    logger.info("Successfully imported model dependencies")
except ImportError as e:
    logger.error(f"Error importing dependencies: {str(e)}")
    raise

class HighPrecisionDataset(torch.utils.data.Dataset):
    """
    Dataset focused on high precision for X-ray images.
    """
    def __init__(self, image_paths, labels, transform=None, preprocessor=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocessor = preprocessor  # Optional image preprocessing
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            # Apply preprocessor if available (CLAHE, noise reduction)
            if self.preprocessor:
                image = self.preprocessor(image)
                
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {str(e)}")
            # Return a black image of the right size and the label
            return torch.zeros((1, 224, 224)), label

class PrecisionFocusedLoss(nn.Module):
    """
    Loss function that prioritizes precision over recall.
    """
    def __init__(self, pos_weight=None, gamma=2.0, precision_weight=0.7):
        super().__init__()
        self.base_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.gamma = gamma
        self.precision_weight = precision_weight  # Weight for precision vs recall
        
    def forward(self, logits, targets):
        # Standard BCE loss with logits
        bce_loss = self.base_loss(logits, targets)
        
        # Calculate predicted probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal weights
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Add additional weight to prioritize precision
        # Higher weight for false positives (prioritizing precision)
        precision_factor = torch.where(
            (targets == 0) & (probs > 0.1),  # Potential false positives
            torch.ones_like(targets) * self.precision_weight,
            torch.ones_like(targets) * (1 - self.precision_weight)
        )
        
        # Combine weights
        weight = focal_weight * precision_factor
        
        # Apply weights to BCE loss
        loss = bce_loss * weight
        
        # Return mean loss
        return loss.mean()

def compute_metrics(y_true, y_pred, y_score, class_names):
    """
    Compute classification metrics with emphasis on precision.
    """
    # Calculate macro and micro metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_score, average='macro')
    except:
        auc = 0.5  # Default AUC
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    try:
        auc_per_class = roc_auc_score(y_true, y_score, average=None)
    except:
        auc_per_class = [0.5] * len(class_names)  # Default AUC per class
    
    # Create metrics dictionary
    metrics = {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'auc': auc,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'auc_per_class': auc_per_class
    }
    
    return metrics

def print_metrics_table(metrics, class_names, epoch=None, total_epochs=None):
    """
    Print a nicely formatted table of metrics.
    """
    header = "=" * 70
    if epoch is not None and total_epochs is not None:
        title = f"===== Epoch {epoch}/{total_epochs} Validation Metrics ====="
    else:
        title = "===== Validation Metrics ====="
    
    print(header)
    print(" " * 14 + title)
    print(header)
    
    # Print overall metrics
    print(f"{'Metric':<20} | {'Value':>10}")
    print("-" * 70)
    print(f"{'F1 Macro':<20} | {metrics['f1_macro']:>10.4f}")
    print(f"{'F1 Micro':<20} | {metrics['f1_micro']:>10.4f}")
    print(f"{'Precision Macro':<20} | {metrics['precision_macro']:>10.4f}")
    print(f"{'Recall Macro':<20} | {metrics['recall_macro']:>10.4f}")
    print(f"{'AUC':<20} | {metrics['auc']:>10.4f}")
    print(header)
    
    # Print per-class metrics
    print("Per-Class Metrics:")
    print(f"{'Class':<20} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'AUC':>10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        precision = metrics['precision_per_class'][i]
        recall = metrics['recall_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        auc = metrics['auc_per_class'][i]
        
        print(f"{class_name:<20} | {precision:>10.4f} | {recall:>10.4f} | {f1:>10.4f} | {auc:>10.4f}")
    
    print(header)
    print()

def find_high_precision_thresholds(y_true, y_score, class_names, target_precision=0.7):
    """
    Find thresholds that achieve a target precision.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        class_names: List of class names
        target_precision: Target precision to achieve
        
    Returns:
        np.ndarray: Optimal thresholds for each class
    """
    n_classes = y_true.shape[1]
    # High threshold range to improve precision
    thresholds = np.linspace(0.5, 0.95, 50)
    optimal_thresholds = np.zeros(n_classes)
    
    logger.info(f"Finding thresholds for {n_classes} classes with target precision: {target_precision:.2f}")
    
    for i in range(n_classes):
        best_f1 = 0
        best_threshold = 0.5  # Default threshold
        best_precision = 0
        best_recall = 0
        
        # Try each threshold
        for threshold in thresholds:
            # Apply threshold
            y_pred_i = (y_score[:, i] >= threshold).astype(int)
            
            # Calculate precision and recall
            precision = precision_score(y_true[:, i], y_pred_i, zero_division=0)
            recall = recall_score(y_true[:, i], y_pred_i, zero_division=0)
            
            # Calculate F1 score
            if precision > 0 and recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            # Update best if precision is above target and F1 is improved
            if precision >= target_precision:
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
        
        # If no threshold meets the target, find the one with highest precision
        if best_f1 == 0:
            for threshold in thresholds:
                y_pred_i = (y_score[:, i] >= threshold).astype(int)
                precision = precision_score(y_true[:, i], y_pred_i, zero_division=0)
                recall = recall_score(y_true[:, i], y_pred_i, zero_division=0)
                
                # Calculate F1 score
                if precision > 0 and recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                
                if precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
                    best_f1 = f1
                    best_recall = recall
        
        # Save the best threshold for this class
        optimal_thresholds[i] = best_threshold
        
        logger.info(f"  Class {class_names[i]}: threshold={best_threshold:.2f}, "
                    f"precision={best_precision:.4f}, recall={best_recall:.4f}, f1={best_f1:.4f}")
    
    return optimal_thresholds

def filter_top_classes(df, image_paths, top_n=2):
    """
    Filter dataset to include only the top N most common diseases.
    
    Args:
        df: DataFrame with disease labels
        image_paths: List of image paths
        top_n: Number of top classes to keep
        
    Returns:
        Tuple of (filtered_df, filtered_image_paths, selected_classes)
    """
    # Get disease columns (exclude metadata columns)
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower()]
    disease_cols = [col for col in df.columns if col not in id_cols and df[col].dtype != 'object']
    
    # Calculate prevalence for each disease
    prevalence = {}
    for col in disease_cols:
        pos_count = df[col].sum()
        prevalence[col] = pos_count / len(df)
    
    # Sort diseases by prevalence
    sorted_diseases = sorted(prevalence.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N most common diseases
    selected_classes = [d[0] for d in sorted_diseases[:top_n]]
    
    logger.info(f"Selected top {top_n} classes: {', '.join(selected_classes)}")
    
    # Print prevalence for selected classes
    for cls in selected_classes:
        pos_count = df[cls].sum()
        neg_count = len(df) - pos_count
        logger.info(f"  {cls}: {pos_count} positive, {neg_count} negative, {pos_count/len(df)*100:.2f}%")
    
    return df, image_paths, selected_classes

def train_high_precision_model(
    model_name='densenet121',
    top_n_classes=3,
    epochs=20,
    batch_size=8,
    learning_rate=2e-5,
    sample_size=5000,
    dropout_rate=0.5,
    target_precision=0.7,
    weight_decay=1e-3
):
    """
    Train a model with focus on high precision.
    
    Args:
        model_name: Name of the backbone model
        top_n_classes: Number of top classes to focus on
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        sample_size: Number of samples to use
        dropout_rate: Dropout rate
        target_precision: Target precision to achieve
        weight_decay: Weight decay for optimizer
        
    Returns:
        Dict: Training metrics
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load ChestX-ray14 dataset
    logger.info(f"Loading ChestX-ray14 dataset with sample_size={sample_size}...")
    df, image_paths = load_chestxray_images(
        sample_size=sample_size,
        balanced_sampling=True
    )
    
    # Filter to top N classes
    df, image_paths, selected_classes = filter_top_classes(df, image_paths, top_n=top_n_classes)
    
    # Create transformations - stronger augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Create binary labels for selected classes
    labels = np.zeros((len(df), len(selected_classes)), dtype=np.float32)
    for i, row in df.iterrows():
        for j, disease in enumerate(selected_classes):
            if disease in df.columns and row[disease] == 1:
                labels[i, j] = 1.0
    
    # Convert to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Split data
    indices = np.arange(len(df))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, 
        stratify=np.argmax(labels, axis=1) if np.any(labels.sum(axis=1) > 0) else None
    )
    
    # Create datasets
    train_dataset = HighPrecisionDataset(
        [image_paths[i] for i in train_indices],
        labels_tensor[train_indices],
        transform=train_transform
    )
    
    val_dataset = HighPrecisionDataset(
        [image_paths[i] for i in val_indices],
        labels_tensor[val_indices],
        transform=val_transform
    )
    
    # Calculate weights for balanced sampling
    train_labels = labels[train_indices]
    class_counts = train_labels.sum(axis=0)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    
    # Calculate sample weights
    sample_weights = train_labels @ class_weights
    sample_weights = np.maximum(sample_weights, 0.1)  # Ensure minimum weight
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    
    # Create sampler with replacement for better balance
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Calculate class weights for loss function
    pos_weights = torch.tensor(
        [len(df) / max(df[cls].sum(), 1) for cls in selected_classes], 
        dtype=torch.float32
    ).to(device)
    
    # Create model
    try:
        # Import MONAI's DenseNet121 with attention mechanism
        model = AttentionXrayModel(
            num_classes=len(selected_classes),
            pretrained=True,
            unfreeze_blocks=4,  # Unfreeze all blocks for thorough fine-tuning
            dropout_rate=dropout_rate
        )
        logger.info("Using MONAI's DenseNet121 with medical imaging pretraining")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise
    
    model = model.to(device)
    
    # Create optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler - linear decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Longer warmup
        div_factor=25,
        final_div_factor=1000
    )
    
    # Create precision-focused loss function
    criterion = PrecisionFocusedLoss(pos_weight=pos_weights, precision_weight=0.7)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_f1 = 0.0
    best_model_state = None
    best_metrics = None
    optimal_thresholds = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, (batch_X, batch_y) in enumerate(progress):
            # Move data to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast('cuda'):
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            scheduler.step()
            
            # Update statistics
            train_loss += loss.item()
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate average loss
        train_loss /= len(train_loader)
        logger.info(f"Training Loss: {train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Collect all validation data
        all_y_true = []
        all_y_scores = []
        
        with torch.no_grad():
            progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            
            for batch_X, batch_y in progress:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Forward pass
                with autocast('cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                # Convert sigmoid outputs to probabilities
                probs = torch.sigmoid(outputs)
                
                # Collect predictions and targets
                all_y_true.append(batch_y.cpu().numpy())
                all_y_scores.append(probs.cpu().numpy())
                
                # Update statistics
                val_loss += loss.item()
        
        # Calculate average loss
        val_loss /= len(val_loader)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        # Concatenate all validation results
        y_true = np.vstack(all_y_true)
        y_score = np.vstack(all_y_scores)
        
        # Find high precision thresholds
        optimal_thresholds = find_high_precision_thresholds(
            y_true, y_score, selected_classes, target_precision=target_precision
        )
        
        # Apply optimal thresholds
        y_pred = (y_score >= optimal_thresholds).astype(np.int32)
        
        # Calculate metrics
        val_metrics = compute_metrics(y_true, y_pred, y_score, selected_classes)
        
        # Print metrics
        print_metrics_table(val_metrics, selected_classes, epoch+1, epochs)
        
        # Check if this is the best model so far
        current_f1 = val_metrics['f1_macro']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_state = model.state_dict()
            best_metrics = val_metrics.copy()
            
            # Save checkpoint
            torch.save({
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': best_metrics,
                'thresholds': optimal_thresholds,
                'class_names': selected_classes
            }, f"models/checkpoints/high_precision_{model_name}_checkpoint.pt")
            
            logger.info(f"Saved best model with F1 macro: {best_f1:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    
    # Save final model
    final_model_path = f"models/checkpoints/final_high_precision_{model_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'thresholds': optimal_thresholds,
        'metrics': best_metrics,
        'class_names': selected_classes
    }, final_model_path)
    
    logger.info(f"Training completed. Best F1 macro: {best_f1:.4f}")
    logger.info(f"Final model saved to {final_model_path}")
    
    return {
        'f1_macro': best_f1,
        'model_name': model_name,
        'selected_classes': selected_classes,
        'epochs': epochs,
        'thresholds': optimal_thresholds,
        'precision': best_metrics['precision_macro'],
        'recall': best_metrics['recall_macro']
    }

def main():
    parser = argparse.ArgumentParser(description='Train a high precision chest X-ray model')
    parser.add_argument('--top_n', type=int, default=2, 
                        help='Number of top classes to focus on')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--sample_size', type=int, default=5000, help='Number of samples to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--target_precision', type=float, default=0.7, 
                        help='Target precision to achieve')
    
    args = parser.parse_args()
    
    # Train model
    results = train_high_precision_model(
        top_n_classes=args.top_n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sample_size=args.sample_size,
        dropout_rate=args.dropout,
        target_precision=args.target_precision
    )
    
    # Print final results
    logger.info(f"Final F1 macro: {results['f1_macro']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}")
    logger.info(f"Selected classes: {', '.join(results['selected_classes'])}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 