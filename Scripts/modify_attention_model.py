#!/usr/bin/env python
"""
Modifications to the attention model for improved F1 score.

This script implements several improvements to the base attention model:
1. Increases model capacity with a deeper network
2. Uses advanced data augmentation specific to medical images
3. Implements class balancing techniques for imbalanced classification
4. Uses a more aggressive learning rate schedule
5. Optimizes thresholds for maximizing F1 score
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from models.attention_xray_model import AttentionXrayModel, EfficientNetXrayModel, AdaptiveFocalLoss
from data.data_loader import load_chestxray_images
from data.datasets import create_data_loaders
from preprocessing.xray_enhancer import SafeXrayTransform

# Check for utils.samplers and use a fallback if needed
if os.path.exists(os.path.join(project_root, 'utils', 'samplers.py')):
    from utils.samplers import BalancedSampler
    logger.info("Successfully imported BalancedSampler")
else:
    logger.warning("utils/samplers.py not found, will use default sampler in data loaders")
    # Define a placeholder class if needed
    class BalancedSampler:
        def __init__(self, *args, **kwargs):
            pass

def fine_tune_attention_model(
    model_name: str = 'efficientnet_b2',
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    sample_size: int = 2000,
    dropout_rate: float = 0.6,  # Increased dropout for better generalization
    gamma: float = 2.5,         # Increased gamma for more focus on hard examples
    alpha: float = 0.75,        # Increased alpha to address class imbalance 
    weight_decay: float = 1e-3, # Increased weight decay to reduce overfitting
    label_smoothing: float = 0.1, # Added label smoothing for better generalization
    mixup_alpha: float = 0.4    # Increased mixup alpha for stronger augmentation
):
    """
    Fine-tune the attention model with optimized parameters.
    
    Args:
        model_name: Model architecture to use
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Base learning rate
        sample_size: Maximum number of samples to use
        dropout_rate: Dropout rate for model regularization
        gamma: Focal loss gamma parameter
        alpha: Focal loss alpha parameter
        weight_decay: Weight decay for optimizer
        label_smoothing: Label smoothing parameter
        mixup_alpha: Alpha parameter for mixup augmentation
        
    Returns:
        tuple: (model, metrics, optimal_thresholds)
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load ChestX-ray14 metadata and images
    logger.info(f"Loading ChestX-ray14 dataset with sample_size={sample_size}...")
    df, image_paths = load_chestxray_images(
        sample_size=sample_size,
        balanced_sampling=True
    )
    
    # Create advanced transformations for X-rays
    train_transform = SafeXrayTransform(
        image_size=(224, 224),
        apply_clahe=True,
        apply_noise_reduction=True,
        training=True,
        # Enhanced augmentation parameters
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.0  # Usually no vertical flips for medical images
    )
    
    val_transform = SafeXrayTransform(
        image_size=(224, 224),
        apply_clahe=True,
        apply_noise_reduction=True,
        training=False
    )
    
    # Create data loaders with balanced sampling
    train_loader, val_loader, disease_names = create_data_loaders(
        df=df,
        image_paths=image_paths,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        test_size=0.2,
        num_workers=4,
        pin_memory=True,
        use_balanced_sampler=True
    )
    
    # Calculate class weights for focal loss
    class_counts = np.array([df[disease].sum() for disease in disease_names])
    total_samples = len(df)
    neg_counts = total_samples - class_counts
    # Scaled inverse frequency
    pos_weights = torch.tensor(neg_counts / class_counts, dtype=torch.float32).to(device)
    class_weights = torch.tensor(1 - (class_counts / total_samples), dtype=torch.float32).to(device)
    
    # Create model
    num_classes = len(disease_names)
    if model_name.startswith('efficientnet'):
        model = EfficientNetXrayModel(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=True,
            dropout_rate=dropout_rate
        )
    else:
        model = AttentionXrayModel(
            num_classes=num_classes,
            pretrained=True,
            unfreeze_blocks=4,  # Unfreeze all blocks for fine-tuning
            dropout_rate=dropout_rate
        )
    
    model = model.to(device)
    logger.info(f"Created model with {model.count_trainable_parameters()} trainable parameters")
    
    # Use AdaptiveFocalLoss with class weights
    criterion = AdaptiveFocalLoss(
        gamma=gamma,
        alpha=alpha,
        reduction='mean',
        label_smoothing=label_smoothing,
        pos_weight=pos_weights,
        class_weights=class_weights
    )
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler with cosine annealing and warmup
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # 20% warmup
        div_factor=25,  # Initial LR is max_lr/25
        final_div_factor=1000  # Final LR is max_lr/1000
    )
    
    # Train the model
    from train_attention_xray import train_epoch, validate, find_optimal_thresholds, print_metrics_table
    
    best_f1 = 0.0
    best_model_state = None
    best_metrics = None
    optimal_thresholds = None
    
    logger.info(f"Starting training for {epochs} epochs with {len(train_loader)} batches per epoch")
    
    for epoch in range(epochs):
        # Train for an epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            use_amp=True,
            use_mixup=True,
            mixup_alpha=mixup_alpha,
            epoch=epoch,
            num_epochs=epochs
        )
        
        # Validate and find optimal thresholds
        val_loss, y_true, y_score = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=disease_names,
            optimal_thresholds=None  # Always re-optimize thresholds
        )
        
        # Find optimal thresholds
        optimal_thresholds = find_optimal_thresholds(y_true, y_score, disease_names)
        
        # Compute metrics with optimal thresholds
        # Use the optimal thresholds to make predictions
        y_pred = (y_score >= optimal_thresholds).astype(np.int32)
        
        # Import metrics functions
        from train_attention_xray import compute_metrics
        
        # Compute metrics
        val_metrics = compute_metrics(y_true, y_pred, y_score, disease_names)
        
        # Print metrics
        print_metrics_table(val_metrics, disease_names, epoch+1, epochs)
        
        # Check if this is the best model so far
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            best_metrics = val_metrics.copy()
            
            # Save checkpoint
            model_path = os.path.join("models/checkpoints", f"optimized_{model_name}_checkpoint.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "thresholds": optimal_thresholds,
                "class_names": disease_names
            }, model_path)
            
            logger.info(f"Saved best model with F1 macro: {best_f1:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.info(f"Training completed. Best F1 macro: {best_f1:.4f}")
    
    return model, best_metrics, optimal_thresholds

def main():
    """
    Main function to parse arguments and run the optimized training.
    """
    parser = argparse.ArgumentParser(description='Optimize Attention Model for X-ray Classification')
    parser.add_argument('--model', type=str, default='efficientnet_b2', 
                        choices=['densenet121', 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--sample_size', type=int, default=2000, help='Maximum number of samples to use')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate for model regularization')
    parser.add_argument('--gamma', type=float, default=2.5, help='Focal loss gamma parameter')
    parser.add_argument('--alpha', type=float, default=0.75, help='Focal loss alpha parameter')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing parameter')
    parser.add_argument('--mixup_alpha', type=float, default=0.4, help='Alpha parameter for mixup augmentation')
    
    args = parser.parse_args()
    
    # Run the optimized training
    model, metrics, thresholds = fine_tune_attention_model(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        sample_size=args.sample_size,
        dropout_rate=args.dropout,
        gamma=args.gamma,
        alpha=args.alpha,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha
    )
    
    # Print final metrics
    logger.info(f"Final F1 macro: {metrics['f1_macro']:.4f}")
    logger.info(f"Final F1 micro: {metrics['f1_micro']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 