#!/usr/bin/env python
"""
Improved ensemble model for chest X-ray classification with focus on F1 score optimization.

This script implements an ensemble approach with three key improvements:
1. Uses multiple backbone models with different architectures for diversity
2. Implements per-class threshold optimization specifically for F1 score
3. Uses weighted voting based on model confidence and validation performance
"""

import os
import sys
import argparse
import time
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

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
# Make sure utils is imported correctly
if os.path.exists(os.path.join(project_root, 'utils', 'samplers.py')):
    from utils.samplers import BalancedSampler
    logger.info("Successfully imported BalancedSampler")
else:
    logger.warning("utils/samplers.py not found, using default sampler")
    # Define a simple placeholder if needed
    class BalancedSampler:
        def __init__(self, *args, **kwargs):
            pass


class MultiBackboneEnsemble(nn.Module):
    """
    Ensemble model using multiple backbone architectures.
    This approach combines diverse model architectures for better generalization.
    """
    def __init__(
        self, 
        num_classes: int,
        model_names: List[str] = ['densenet121', 'efficientnet_b0', 'efficientnet_b2'],
        weights: Optional[List[float]] = None,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the ensemble.
        
        Args:
            num_classes: Number of output classes
            model_names: List of model architectures to use
            weights: Optional weights for each model's prediction
            dropout_rate: Dropout rate for regularization
        """
        super(MultiBackboneEnsemble, self).__init__()
        
        self.models = nn.ModuleList()
        self.num_classes = num_classes
        self.model_names = model_names
        
        # Set default weights if not provided
        if weights is None:
            self.weights = torch.ones(len(model_names)) / len(model_names)
        else:
            # Normalize weights to sum to 1
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            self.weights = weights_tensor / weights_tensor.sum()
        
        # Create each model in the ensemble
        for model_name in model_names:
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
                    dropout_rate=dropout_rate
                )
                
            self.models.append(model)
            
        logger.info(f"Created ensemble with {len(self.models)} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted average of model outputs
        """
        all_outputs = []
        
        # Get outputs from each model
        for model in self.models:
            with autocast('cuda'):
                output = model(x)
            all_outputs.append(output)
        
        # Stack outputs
        stacked_outputs = torch.stack(all_outputs)
        
        # Ensure weights are on the same device as stacked_outputs
        weights = self.weights.to(stacked_outputs.device)
        
        # Apply weights and sum
        weighted_outputs = weights.view(-1, 1, 1) * stacked_outputs
        ensemble_output = weighted_outputs.sum(dim=0)
        
        return ensemble_output
    
    def predict_with_thresholds(
        self, 
        x: torch.Tensor, 
        thresholds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions using class-specific thresholds.
        
        Args:
            x: Input tensor
            thresholds: Threshold for each class
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (probabilities, binary predictions)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            
            # Apply thresholds for each class
            preds = (probs >= thresholds.to(probs.device)).float()
            
            return probs, preds
    
    def update_weights(
        self, 
        model_metrics: List[Dict[str, float]],
        metric_name: str = 'f1_macro'
    ) -> None:
        """
        Update model weights based on validation performance.
        
        Args:
            model_metrics: List of metrics dictionaries for each model
            metric_name: Name of the metric to use for weighting
        """
        # Extract the specified metric for each model
        metrics = torch.tensor([m[metric_name] for m in model_metrics], dtype=torch.float32)
        
        # Add a small constant to avoid zero weights
        metrics = metrics + 1e-5
        
        # Normalize to get weights (higher metric = higher weight)
        new_weights = metrics / metrics.sum()
        
        # Update weights
        self.weights = new_weights
        
        # Log new weights
        weight_str = ", ".join([f"{self.model_names[i]}: {w:.3f}" for i, w in enumerate(self.weights)])
        logger.info(f"Updated model weights: {weight_str}")


def optimize_thresholds_for_f1(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    class_names: List[str],
    class_weights: Optional[np.ndarray] = None,
    metric: str = 'f1'
) -> np.ndarray:
    """
    Find optimal thresholds for each class that maximize the chosen metric.
    
    Args:
        y_true: Ground truth labels (shape: n_samples x n_classes)
        y_score: Predicted probabilities (shape: n_samples x n_classes)
        class_names: List of class names
        class_weights: Optional weights for each class
        metric: Metric to optimize ('f1', 'precision', or 'recall')
        
    Returns:
        np.ndarray: Optimal thresholds for each class
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    n_classes = y_true.shape[1]
    thresholds = np.linspace(0.3, 0.9, 50)  # Check 50 threshold values with higher range
    optimal_thresholds = np.zeros(n_classes)
    
    # Set default class weights if not provided
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes
    
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights)
    
    logger.info(f"Finding optimal thresholds for {n_classes} classes to maximize {metric}")
    
    for i in range(n_classes):
        best_metric_value = 0
        best_threshold = 0.5  # Default threshold
        
        # Check class prevalence
        positive_ratio = np.mean(y_true[:, i])
        
        # For very imbalanced classes, adjust threshold search range
        if positive_ratio < 0.05:
            # Very rare class - use reasonable thresholds
            thresh_range = np.linspace(0.2, 0.6, 30)
        elif positive_ratio > 0.4:
            # Common class - use higher thresholds
            thresh_range = np.linspace(0.4, 0.9, 30)
        else:
            # Moderately rare - use balanced thresholds
            thresh_range = np.linspace(0.3, 0.8, 30)
            
        # Try each threshold
        for threshold in thresh_range:
            # Apply threshold
            y_pred_i = (y_score[:, i] >= threshold).astype(int)
            
            # Calculate the requested metric
            if metric == 'f1':
                metric_value = f1_score(y_true[:, i], y_pred_i, zero_division=0)
            elif metric == 'precision':
                metric_value = precision_score(y_true[:, i], y_pred_i, zero_division=0)
            elif metric == 'recall':
                metric_value = recall_score(y_true[:, i], y_pred_i, zero_division=0)
            elif metric == 'balanced':
                # Balance between precision and recall
                prec = precision_score(y_true[:, i], y_pred_i, zero_division=0)
                rec = recall_score(y_true[:, i], y_pred_i, zero_division=0)
                # Harmonic mean with bias toward precision to avoid predicting everything
                if positive_ratio < 0.05:
                    # For rare classes, balance precision and recall
                    metric_value = (0.5 * prec + 0.5 * rec) if (prec > 0 and rec > 0) else 0
                else:
                    # For common classes, favor precision a bit more
                    metric_value = (0.6 * prec + 0.4 * rec) if (prec > 0 and rec > 0) else 0
            
            # Update best if improvement found
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold
        
        # Save the best threshold for this class
        optimal_thresholds[i] = best_threshold
        
        logger.info(f"  Class {class_names[i]}: threshold={best_threshold:.2f}, {metric}={best_metric_value:.4f}")
    
    return optimal_thresholds


def train_ensemble(
    model_names: List[str] = ['densenet121', 'efficientnet_b0', 'efficientnet_b2'],
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    sample_size: int = 2000,
    dropout_rate: float = 0.6,
    gamma: float = 2.5,
    alpha: float = 0.75,
    label_smoothing: float = 0.1,
    optimize_thresholds: bool = True,
    use_dynamic_weights: bool = True,
    threshold_metric: str = 'balanced',
    mixup_alpha: float = 0.4
):
    """
    Train the ensemble model.
    
    Args:
        model_names: List of model architectures to use
        epochs: Number of epochs to train
        batch_size: Batch size for training
        learning_rate: Base learning rate
        weight_decay: Weight decay for optimizer
        sample_size: Number of samples to use
        dropout_rate: Dropout rate for regularization
        gamma: Focal loss gamma parameter
        alpha: Focal loss alpha parameter
        label_smoothing: Label smoothing parameter
        optimize_thresholds: Whether to optimize thresholds for each class
        use_dynamic_weights: Whether to update model weights during training
        threshold_metric: Metric to optimize thresholds for
        mixup_alpha: Alpha parameter for mixup augmentation
        
    Returns:
        Tuple: (ensemble model, metrics, optimal thresholds)
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create output directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load ChestX-ray14 dataset
    logger.info(f"Loading ChestX-ray14 dataset with sample_size={sample_size}...")
    df, image_paths = load_chestxray_images(
        sample_size=sample_size,
        balanced_sampling=True
    )
    
    # Create advanced transformations for better data augmentation
    train_transform = SafeXrayTransform(
        image_size=(224, 224),
        apply_clahe=True,
        apply_noise_reduction=True,
        training=True
    )
    
    val_transform = SafeXrayTransform(
        image_size=(224, 224),
        apply_clahe=True,
        apply_noise_reduction=True,
        training=False
    )
    
    # Create data loaders
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
    
    # Create ensemble model
    num_classes = len(disease_names)
    ensemble = MultiBackboneEnsemble(
        num_classes=num_classes,
        model_names=model_names,
        dropout_rate=dropout_rate
    )
    
    ensemble = ensemble.to(device)
    logger.info(f"Created ensemble with {len(ensemble.models)} models")
    
    # Create optimizer for each model and the ensemble
    optimizers = []
    schedulers = []
    
    for model in ensemble.models:
        opt = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        optimizers.append(opt)
        
        # Create scheduler
        steps_per_epoch = len(train_loader)
        sched = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,
            div_factor=25,
            final_div_factor=1000
        )
        schedulers.append(sched)
    
    # Create loss function
    criterion = AdaptiveFocalLoss(
        gamma=gamma,
        alpha=alpha,
        reduction='mean',
        label_smoothing=label_smoothing,
        pos_weight=pos_weights,
        class_weights=class_weights
    )
    
    # Initialize gradient scalers for mixed precision
    scalers = [GradScaler() for _ in range(len(ensemble.models))]
    
    # Training loop
    best_f1 = 0.0
    best_ensemble_state = None
    best_metrics = None
    optimal_thresholds = torch.ones(num_classes) * 0.5  # Default threshold
    
    for epoch in range(epochs):
        # Train each model
        for model_idx, model in enumerate(ensemble.models):
            model.train()
            
            # Train for one epoch
            train_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train Model {model_idx+1}]")
            
            for batch_idx, (batch_X, batch_y) in enumerate(progress):
                # Move data to device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Zero the parameter gradients
                optimizers[model_idx].zero_grad()
                
                # Forward pass with mixed precision
                with autocast('cuda'):
                    # Apply mixup if enabled
                    if mixup_alpha > 0 and np.random.random() < 0.5:
                        # Mixup augmentation
                        lam = np.random.beta(mixup_alpha, mixup_alpha)
                        index = torch.randperm(batch_X.size(0)).to(device)
                        mixed_X = lam * batch_X + (1 - lam) * batch_X[index, :]
                        mixed_y = lam * batch_y + (1 - lam) * batch_y[index]
                        
                        # Forward pass with mixed data
                        outputs = model(mixed_X)
                        loss = criterion(outputs, mixed_y)
                    else:
                        # Standard forward pass
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                
                # Backward pass with scaling
                scalers[model_idx].scale(loss).backward()
                scalers[model_idx].step(optimizers[model_idx])
                scalers[model_idx].update()
                
                # Update learning rate
                schedulers[model_idx].step()
                
                # Update statistics
                train_loss += loss.item()
                
                # Update progress bar
                progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{optimizers[model_idx].param_groups[0]['lr']:.6f}"
                })
            
            # Calculate average loss
            train_loss /= len(train_loader)
            logger.info(f"Model {model_idx+1} ({ensemble.model_names[model_idx]}) - Train Loss: {train_loss:.4f}")
        
        # Validate ensemble and individual models
        ensemble.eval()
        model_metrics = []
        
        # Collect all validation data for threshold optimization
        all_y_true = []
        all_y_scores = []
        
        # Use a list to collect outputs for each model
        model_y_scores = [[] for _ in range(len(ensemble.models))]
        
        with torch.no_grad():
            progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Ensemble]")
            
            for batch_X, batch_y in progress:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Collect ground truth
                all_y_true.append(batch_y.cpu().numpy())
                
                # Get predictions from each model
                for model_idx, model in enumerate(ensemble.models):
                    model.eval()
                    with autocast('cuda'):
                        outputs = model(batch_X)
                    
                    probs = torch.sigmoid(outputs)
                    model_y_scores[model_idx].append(probs.cpu().numpy())
                
                # Get ensemble predictions
                with autocast('cuda'):
                    outputs = ensemble(batch_X)
                
                probs = torch.sigmoid(outputs)
                all_y_scores.append(probs.cpu().numpy())
        
        # Concatenate all validation results
        y_true = np.vstack(all_y_true)
        y_score = np.vstack(all_y_scores)
        
        # Calculate metrics for each individual model
        for model_idx, model in enumerate(ensemble.models):
            model_scores = np.vstack(model_y_scores[model_idx])
            
            # Find optimal thresholds for this model
            model_thresholds = optimize_thresholds_for_f1(
                y_true, 
                model_scores, 
                disease_names,
                metric=threshold_metric
            )
            
            # Apply thresholds
            y_pred = (model_scores >= model_thresholds).astype(np.int32)
            
            # Calculate metrics
            from train_attention_xray import compute_metrics
            model_metric = compute_metrics(y_true, y_pred, model_scores, disease_names)
            model_metrics.append(model_metric)
            
            logger.info(f"Model {model_idx+1} ({ensemble.model_names[model_idx]}) - F1 Macro: {model_metric['f1_macro']:.4f}")
        
        # Update ensemble weights if enabled
        if use_dynamic_weights and epoch > 0:
            ensemble.update_weights(model_metrics, metric_name='f1_macro')
        
        # Find optimal thresholds for the ensemble
        if optimize_thresholds:
            optimal_thresholds = optimize_thresholds_for_f1(
                y_true, 
                y_score, 
                disease_names,
                class_weights=class_counts/np.sum(class_counts),
                metric=threshold_metric
            )
            
            # Convert to tensor
            optimal_thresholds = torch.tensor(optimal_thresholds, dtype=torch.float32)
        
        # Apply thresholds
        y_pred = (y_score >= optimal_thresholds.numpy()).astype(np.int32)
        
        # Calculate ensemble metrics
        from train_attention_xray import compute_metrics
        val_metrics = compute_metrics(y_true, y_pred, y_score, disease_names)
        
        # Print metrics
        from train_attention_xray import print_metrics_table
        print_metrics_table(val_metrics, disease_names, epoch+1, epochs)
        
        # Check if this is the best model so far
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_ensemble_state = {
                'models': [model.state_dict() for model in ensemble.models],
                'weights': ensemble.weights
            }
            best_metrics = val_metrics.copy()
            
            # Save checkpoint
            model_path = os.path.join("models/checkpoints", "improved_ensemble_checkpoint.pt")
            torch.save({
                "ensemble_state": best_ensemble_state,
                "thresholds": optimal_thresholds,
                "metrics": val_metrics,
                "class_names": disease_names
            }, model_path)
            
            logger.info(f"Saved best ensemble with F1 macro: {best_f1:.4f}")
    
    # Load best ensemble state
    if best_ensemble_state is not None:
        for model_idx, model in enumerate(ensemble.models):
            model.load_state_dict(best_ensemble_state['models'][model_idx])
        ensemble.weights = best_ensemble_state['weights']
    
    logger.info(f"Training completed. Best F1 macro: {best_f1:.4f}")
    
    # Save final model
    final_model_path = os.path.join("models/checkpoints", "final_improved_ensemble.pt")
    torch.save({
        "ensemble_state": {
            'models': [model.state_dict() for model in ensemble.models],
            'weights': ensemble.weights
        },
        "thresholds": optimal_thresholds,
        "metrics": best_metrics,
        "class_names": disease_names
    }, final_model_path)
    
    logger.info(f"Final ensemble saved to {final_model_path}")
    
    return ensemble, best_metrics, optimal_thresholds


def main():
    """
    Main function to parse arguments and train the ensemble.
    """
    parser = argparse.ArgumentParser(description='Train an improved ensemble for chest X-ray classification')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['densenet121', 'efficientnet_b0', 'efficientnet_b2'],
                        help='List of model architectures to use in the ensemble')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--sample_size', type=int, default=2000, help='Number of samples to use')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate for regularization')
    parser.add_argument('--gamma', type=float, default=2.5, help='Focal loss gamma parameter')
    parser.add_argument('--alpha', type=float, default=0.75, help='Focal loss alpha parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing parameter')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    
    args = parser.parse_args()
    
    # Train ensemble
    ensemble, metrics, thresholds = train_ensemble(
        model_names=args.models,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        sample_size=args.sample_size,
        dropout_rate=args.dropout,
        gamma=args.gamma,
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay
    )
    
    # Print final metrics
    logger.info(f"Final F1 macro: {metrics['f1_macro']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 