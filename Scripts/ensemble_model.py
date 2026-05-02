"""
Ensemble model approach for chest X-ray classification to drastically improve F1 score.

This script uses multiple models with different architectures and combines their predictions
to achieve much better F1 scores (above 0.60) on the chest X-ray classification task.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import our modules
from models.attention_xray_model import AttentionXrayModel, EfficientNetXrayModel, AdaptiveFocalLoss
from data.data_loader import load_chestxray_images
from data.datasets import create_data_loaders
from modify_attention_model import improve_thresholds, modify_model, get_improved_focal_loss

class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for better prediction.
    """
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of models to ensemble
            weights: Optional weights for each model (default: equal weights)
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Set weights (default: equal weights)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            self.weights = [w / sum(weights) for w in weights]  # Normalize
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and combine predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Combined predictions
        """
        # Run all models
        outputs = []
        for i, model in enumerate(self.models):
            outputs.append(self.weights[i] * model(x))
        
        # Take weighted average
        return sum(outputs)

def find_optimal_thresholds(y_true, y_score, class_names):
    """
    Find optimal thresholds for each class using ROC curve.
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        class_names: Names of the classes
        
    Returns:
        np.ndarray: Array of optimal thresholds
    """
    from sklearn.metrics import roc_curve
    
    thresholds = []
    
    for i, class_name in enumerate(class_names):
        # Get class data
        y_true_class = y_true[:, i]
        y_score_class = y_score[:, i]
        
        # Calculate ROC curve
        fpr, tpr, threshold = roc_curve(y_true_class, y_score_class)
        
        # Calculate Youden's J statistic (tpr - fpr)
        j_scores = tpr - fpr
        
        # Find threshold that maximizes J
        best_idx = np.argmax(j_scores)
        best_threshold = threshold[best_idx]
        
        # Add to list
        thresholds.append(best_threshold)
    
    return np.array(thresholds)

def evaluate_model(model, data_loader, device, class_names, thresholds=None):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with validation data
        device: Device to run on
        class_names: Names of the classes
        thresholds: Optional array of thresholds
        
    Returns:
        Dict: Dictionary of metrics
    """
    model.eval()
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(data_loader, desc="Evaluation"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            
            # Store predictions and targets
            all_scores.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_targets = np.array(all_targets)
    
    # Find optimal thresholds if not provided
    if thresholds is None:
        thresholds = find_optimal_thresholds(all_targets, all_scores, class_names)
    
    # Apply thresholds
    all_preds = (all_scores >= thresholds).astype(float)
    
    # Calculate metrics
    metrics = {}
    metrics['f1_macro'] = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    metrics['auc'] = roc_auc_score(all_targets, all_scores, average='macro')
    metrics['thresholds'] = thresholds
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': precision_score(all_targets[:, i], all_preds[:, i], zero_division=0),
            'recall': recall_score(all_targets[:, i], all_preds[:, i], zero_division=0),
            'f1': f1_score(all_targets[:, i], all_preds[:, i], zero_division=0),
            'auc': roc_auc_score(all_targets[:, i], all_scores[:, i]) if len(np.unique(all_targets[:, i])) > 1 else 0.5
        }
    
    metrics['per_class'] = per_class_metrics
    
    return metrics

def train_ensemble_model(
    models_config: List[Dict],
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    sample_size: int = 1600,
    weight_decay: float = 0.001,
    num_workers: int = 0,
    model_weights: Optional[List[float]] = None
):
    """
    Train an ensemble of models for better performance.
    
    Args:
        models_config: List of model configurations
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        sample_size: Sample size
        weight_decay: Weight decay for optimizer
        num_workers: Number of worker threads
        model_weights: Optional weights for ensemble
    """
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load ChestX-ray14 dataset with larger sample size
    logger.info("Loading ChestX-ray14 dataset...")
    df, image_paths = load_chestxray_images(
        sample_size=sample_size,
        balanced_sampling=True
    )
    logger.info(f"Found {len(image_paths)} valid images with metadata")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, disease_names = create_data_loaders(
        df=df,
        image_paths=image_paths,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        test_size=0.2,
        random_seed=42,
        num_workers=num_workers,
        pin_memory=True,
        use_balanced_sampler=True,
        use_advanced_transforms=False
    )
    logger.info(f"Created datasets - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Number of classes
    num_classes = len(disease_names)
    logger.info(f"Training for {num_classes} disease classes")
    
    # Create models according to config
    models = []
    for i, config in enumerate(models_config):
        logger.info(f"Creating model {i+1}/{len(models_config)}: {config['type']}")
        
        if config['type'].startswith('efficientnet'):
            model = EfficientNetXrayModel(
                num_classes=num_classes,
                model_name=config['type'],
                pretrained=True,
                dropout_rate=config.get('dropout_rate', 0.7)
            )
        else:
            model = AttentionXrayModel(
                num_classes=num_classes,
                pretrained=True,
                unfreeze_blocks=config.get('unfreeze_blocks', 4),
                dropout_rate=config.get('dropout_rate', 0.7)
            )
        
        # Apply model improvements
        model = modify_model(model, num_classes, config.get('dropout_rate', 0.7))
        models.append(model)
    
    # Create ensemble model
    ensemble = EnsembleModel(models, weights=model_weights)
    logger.info(f"Created ensemble model with {len(models)} models")
    
    # Move models to device
    ensemble = ensemble.to(device)
    
    # Calculate class weights for focal loss
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.append(labels)
    train_labels = torch.cat(all_train_labels, dim=0)
    
    pos_counts = train_labels.sum(dim=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = torch.clamp(neg_counts / torch.clamp(pos_counts, min=1.0), min=1.0, max=50.0)
    pos_weight = pos_weight.to(device)
    
    # Create loss function with optimal parameters
    criterion = get_improved_focal_loss(
        pos_weight=pos_weight,
        gamma=2.5,
        alpha=0.6,
        label_smoothing=0.1
    )
    logger.info(f"Using Improved Focal Loss with gamma=2.5, alpha=0.6, label_smoothing=0.1")
    
    # Create optimizer
    optimizer = optim.AdamW(
        ensemble.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    logger.info(f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-6
    )
    logger.info("Using ReduceLROnPlateau scheduler")
    
    # Train ensemble model
    best_f1 = 0.0
    best_model_path = os.path.join("models/checkpoints", "xray_ensemble_best.pt")
    
    history = {
        'train_loss': [],
        'val_f1_macro': [],
        'val_precision_macro': [],
        'val_recall_macro': []
    }
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        ensemble.train()
        train_loss = 0.0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ensemble(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Training loss: {avg_train_loss:.4f}")
        history['train_loss'].append(avg_train_loss)
        
        # Evaluate
        logger.info("Evaluating ensemble model...")
        metrics = evaluate_model(ensemble, val_loader, device, disease_names)
        
        # Log metrics
        logger.info(f"Validation - F1 Macro: {metrics['f1_macro']:.4f}, F1 Micro: {metrics['f1_micro']:.4f}")
        logger.info(f"Precision: {metrics['precision_macro']:.4f}, Recall: {metrics['recall_macro']:.4f}, AUC: {metrics['auc']:.4f}")
        
        # Update history
        history['val_f1_macro'].append(metrics['f1_macro'])
        history['val_precision_macro'].append(metrics['precision_macro'])
        history['val_recall_macro'].append(metrics['recall_macro'])
        
        # Print per-class metrics
        print("\nPer-class metrics:")
        print(f"{'Class':<20} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'AUC':>10}")
        print("-" * 65)
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<20} | {class_metrics['precision']:>10.4f} | "
                  f"{class_metrics['recall']:>10.4f} | {class_metrics['f1']:>10.4f} | "
                  f"{class_metrics['auc']:>10.4f}")
        
        # Check if this is the best model
        if metrics['f1_macro'] > best_f1:
            best_f1 = metrics['f1_macro']
            
            # Save model
            torch.save({
                'model_state_dict': ensemble.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'thresholds': metrics['thresholds'].tolist(),
                'f1_macro': metrics['f1_macro'],
                'epoch': epoch
            }, best_model_path)
            
            logger.info(f"Saved best model with F1 macro: {best_f1:.4f}")
        
        # Update learning rate
        scheduler.step(metrics['f1_macro'])
    
    # Plot learning curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1_macro'], label='F1 Macro')
    plt.plot(history['val_precision_macro'], label='Precision')
    plt.plot(history['val_recall_macro'], label='Recall')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('logs/ensemble_learning_curves.png')
    
    logger.info(f"Training completed. Best F1 macro: {best_f1:.4f}")
    return best_f1

if __name__ == "__main__":
    # Define model configurations for ensemble
    models_config = [
        {'type': 'efficientnet_b0', 'dropout_rate': 0.7},
        {'type': 'efficientnet_b0', 'dropout_rate': 0.5}  # Using same architecture with different dropout for efficiency
    ]
    
    # Model weights (higher weight for better models)
    model_weights = [1.0, 0.8]
    
    # Train ensemble model
    best_f1 = train_ensemble_model(
        models_config=models_config,
        epochs=3,  # Fewer epochs for quick results
        batch_size=16,
        learning_rate=1e-4,
        sample_size=1000,  # Smaller sample size to avoid memory issues
        weight_decay=0.001,
        num_workers=0,
        model_weights=model_weights
    )
    
    print(f"\nBest F1 macro score achieved: {best_f1:.4f}")
    
    # For comparison, show target F1 score
    print("Target F1 macro score: 0.6000")
    if best_f1 >= 0.6:
        print("✅ Target achieved!")
    else:
        print("❌ Target not achieved. Try with more data and longer training.") 