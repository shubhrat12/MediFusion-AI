#!/usr/bin/env python
"""
Improved F1 Score for Chest X-ray Classification

This script focuses on optimizing F1 score for chest X-ray classification by:
1. Focusing only on the most common/important conditions
2. Using higher classification thresholds to improve precision
3. Applying advanced data augmentation techniques
4. Using pretrained medical imaging models
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from sklearn.utils.class_weight import compute_class_weight


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
try:
    from models.attention_xray_model import AttentionXrayModel, EfficientNetXrayModel, AdaptiveFocalLoss
    from data.data_loader import load_chestxray_images
    from data.datasets import create_data_loaders
    from preprocessing.xray_enhancer import SafeXrayTransform
    logger.info("Successfully imported model dependencies")
except ImportError as e:
    logger.error(f"Error importing dependencies: {str(e)}")
    raise

def compute_metrics(y_true, y_pred, y_score, class_names):
    """
    Compute classification metrics.
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

def optimize_thresholds(y_true, y_score, class_names):
    """
    Find optimal thresholds for each class to maximize F1 score.
    """
    from sklearn.metrics import f1_score
    
    n_classes = y_true.shape[1]
    thresholds = np.linspace(0.3, 0.9, 50)  # Higher threshold range to improve precision
    optimal_thresholds = np.zeros(n_classes)
    
    logger.info(f"Optimizing thresholds for each class...")
    logger.info(f"Finding optimal thresholds for {n_classes} classes")
    
    for i in range(n_classes):
        best_f1 = 0
        best_threshold = 0.5  # Default threshold
        
        # Check class prevalence
        positive_ratio = np.mean(y_true[:, i])
        
        # Adjust threshold search range based on class prevalence
        if positive_ratio < 0.05:
            # Rare class - higher thresholds to prevent false positives
            thresh_range = np.linspace(0.2, 0.7, 30)
        elif positive_ratio > 0.3:
            # Common class - higher thresholds to improve precision
            thresh_range = np.linspace(0.4, 0.9, 30)
        else:
            # Moderately rare - balanced thresholds
            thresh_range = np.linspace(0.3, 0.8, 30)
        
        # Try each threshold
        for threshold in thresh_range:
            # Apply threshold
            y_pred_i = (y_score[:, i] >= threshold).astype(int)
            
            # Calculate F1 score
            f1 = f1_score(y_true[:, i], y_pred_i, zero_division=0)
            
            # Update best if improvement found
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Save the best threshold for this class
        optimal_thresholds[i] = best_threshold
        
        logger.info(f"  Class {class_names[i]}: threshold={best_threshold:.2f}, f1={best_f1:.4f}")
    
    return optimal_thresholds

def filter_selected_classes(df, image_paths, selected_classes=None):
    """
    Filter dataset to include only selected classes.
    
    Args:
        df: DataFrame with disease labels
        image_paths: List of image paths
        selected_classes: List of class names to keep (if None, keep all)
        
    Returns:
        Tuple of (filtered_df, filtered_image_paths, selected_classes)
    """
    if selected_classes is None:
        # Use diseases with higher prevalence or importance
        default_classes = [
            'Atelectasis',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule'
        ]
        selected_classes = default_classes
    
    logger.info(f"Filtering dataset to include only selected classes: {', '.join(selected_classes)}")
    
    # Check if selected classes exist in the dataframe
    for cls in selected_classes:
        if cls not in df.columns:
            logger.warning(f"Class {cls} not found in dataset, skipping")
            selected_classes.remove(cls)
    
    if not selected_classes:
        logger.error("No valid classes selected, using all available classes")
        selected_classes = [col for col in df.columns if col not in ['patient_id', 'PatientID', 'Finding Labels', 'Image Index']]
    
    # Keep only columns for selected classes and patient ID
    id_cols = [col for col in df.columns if 'id' in col.lower() or 'index' in col.lower()]
    cols_to_keep = id_cols + selected_classes
    filtered_df = df[cols_to_keep].copy()
    
    # Calculate class distributions
    for cls in selected_classes:
        pos_count = filtered_df[cls].sum()
        neg_count = len(filtered_df) - pos_count
        logger.info(f"  {cls}: {pos_count} positive, {neg_count} negative, {pos_count/len(filtered_df)*100:.2f}%")
    
    return filtered_df, image_paths, selected_classes

class CustomXrayDataset(torch.utils.data.Dataset):
    """
    Custom dataset for X-ray images with multi-label classification.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: Labels tensor (num_samples, num_classes)
            transform: Transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get image and label for given index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple: (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            from PIL import Image
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            if self.transform:
                image = self.transform(image)
            else:
                # Basic transform if none provided
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485], std=[0.229])
                ])
                image = transform(image)
                
            return image, label
            
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {str(e)}")
            # Return a blank tensor and the label
            return torch.zeros((3, 224, 224)), label

def create_custom_data_loaders(
    df, 
    image_paths, 
    class_names,
    train_transform, 
    val_transform, 
    batch_size=16, 
    test_size=0.2,
    random_seed=42,
    num_workers=2,
    pin_memory=True,
    use_balanced_sampler=True
):
    """
    Create custom data loaders for X-ray images.
    
    Args:
        df: DataFrame with labels
        image_paths: List of image paths
        class_names: List of class names
        train_transform: Transform for training images
        val_transform: Transform for validation images
        batch_size: Batch size
        test_size: Fraction of data to use for validation
        random_seed: Random seed
        num_workers: Number of workers
        pin_memory: Whether to pin memory
        use_balanced_sampler: Whether to use balanced sampling
        
    Returns:
        Tuple: (train_loader, val_loader)
    """
    # Create binary labels for each disease
    labels = np.zeros((len(df), len(class_names)), dtype=np.float32)
    for i, row in df.iterrows():
        for j, disease in enumerate(class_names):
            if disease in df.columns and row[disease] == 1:
                labels[i, j] = 1.0
    
    # Convert to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(df))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=random_seed, 
        stratify=np.argmax(labels, axis=1) if np.any(labels.sum(axis=1) > 0) else None
    )
    
    # Create datasets
    train_dataset = CustomXrayDataset(
        [image_paths[i] for i in train_indices],
        labels_tensor[train_indices],
        transform=train_transform
    )
    
    val_dataset = CustomXrayDataset(
        [image_paths[i] for i in val_indices],
        labels_tensor[val_indices],
        transform=val_transform
    )
    
    # Create balanced sampler for training if requested
    if use_balanced_sampler:
        # Calculate weights for balanced sampling
        train_labels = labels[train_indices]
        class_counts = train_labels.sum(axis=0)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        
        # Calculate sample weights
        sample_weights = train_labels @ class_weights
        sample_weights = np.maximum(sample_weights, 0.1)  # Ensure minimum weight
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        
        # Create sampler with replacement
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        # Create data loader with sampler
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    else:
        # Create data loader with standard shuffling
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
    
    # Validation loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def train_model_with_focus(
    model_name='densenet121',
    selected_classes=None,
    epochs=15,
    batch_size=16,
    learning_rate=3e-5,
    sample_size=5000,
    dropout_rate=0.5,
    gamma=2.0,
    alpha=0.4,
    weight_decay=1e-3,
    label_smoothing=0.1
):
    """
    Train a model with focus on specific classes and optimized for F1 score.
    
    Args:
        model_name: Name of the backbone model
        selected_classes: List of classes to focus on
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        sample_size: Number of samples to use
        dropout_rate: Dropout rate
        gamma: Focal loss gamma parameter
        alpha: Focal loss alpha parameter
        weight_decay: Weight decay for optimizer
        label_smoothing: Label smoothing parameter
        
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
    
    # Filter to selected classes
    df, image_paths, selected_classes = filter_selected_classes(df, image_paths, selected_classes)
    
    # Create transformations using standard torchvision transforms
    import torchvision.transforms as transforms
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Create data loaders with our custom function
    train_loader, val_loader = create_custom_data_loaders(
        df=df,
        image_paths=image_paths,
        class_names=selected_classes,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        test_size=0.2,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        use_balanced_sampler=True
    )
    
    # Calculate class weights - FIXED VERSION
    class_counts = np.array([df[cls].sum() for cls in selected_classes])
    total_samples = len(df)
    neg_counts = total_samples - class_counts

    # Cap weights to prevent instability
    pos_weights = []
    for i, cls in enumerate(selected_classes):
        pos_weight = min(neg_counts[i] / class_counts[i], 10.0)  # Cap at 10x max
        pos_weights.append(pos_weight)
        print(f"Class {cls}: pos_weight={pos_weight:.2f}")

    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    
    class_weights = torch.tensor(1 - (class_counts / total_samples), dtype=torch.float32).to(device)
    
    # Create model
    num_classes = len(selected_classes)
    try:
        if model_name == 'densenet121':
            # Import MONAI's DenseNet121 with attention mechanism
            model = AttentionXrayModel(
                num_classes=num_classes,
                pretrained=True,
                unfreeze_blocks=4,  # Unfreeze all blocks
                dropout_rate=dropout_rate
            )
            logger.info("Using MONAI's DenseNet121 with medical imaging pretraining")
        else:
            # Use EfficientNet model
            model = EfficientNetXrayModel(
                num_classes=num_classes,
                model_name=model_name,
                pretrained=True,
                dropout_rate=dropout_rate
            )
            logger.info(f"Using {model_name}")
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise
    
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # Longer warmup
        div_factor=25,
        final_div_factor=1000
    )
    
    # Create loss function
    # criterion = AdaptiveFocalLoss(
    #     gamma=gamma,
    #     alpha=alpha,
    #     reduction='mean',
    #     label_smoothing=label_smoothing,
    #     pos_weight=pos_weights,
    #     class_weights=class_weights
    # )
    
    # REPLACE with this:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Initialize gradient scaler for mixed precision
    scaler = torch.amp.GradScaler()
    
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
            
            # NEW CODE: Dynamic threshold optimization during training
            if batch_idx % 50 == 0 and batch_idx > 0:
                with torch.no_grad():
                    probs = torch.sigmoid(outputs)
                    
                    # Initialize adaptive thresholds if not exists
                    if not hasattr(train_model_with_focus, 'adaptive_thresholds'):
                        train_model_with_focus.adaptive_thresholds = [0.5] * len(selected_classes)
                    
                    # Update threshold for each class
                    for i, cls in enumerate(selected_classes):
                        class_true = batch_y[:, i].cpu().numpy()
                        class_scores = probs[:, i].cpu().numpy()
                        
                        if len(np.unique(class_true)) > 1:  # If both classes present
                            best_thresh = 0.5
                            best_f1 = 0
                            for thresh in np.linspace(0.2, 0.8, 15):
                                pred = (class_scores >= thresh).astype(int)
                                from sklearn.metrics import f1_score
                                f1 = f1_score(class_true, pred, zero_division=0)
                                if f1 > best_f1:
                                    best_f1 = f1
                                    best_thresh = thresh
                            
                            # Smooth threshold update
                            train_model_with_focus.adaptive_thresholds[i] = (
                                0.8 * train_model_with_focus.adaptive_thresholds[i] + 
                                0.2 * best_thresh
                            )
        
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
        
        # Use adaptive thresholds if available, otherwise optimize
        if hasattr(train_model_with_focus, 'adaptive_thresholds'):
            optimal_thresholds = np.array(train_model_with_focus.adaptive_thresholds)
        else:
            optimal_thresholds = optimize_thresholds(y_true, y_score, selected_classes)

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
            }, f"models/checkpoints/improved_f1_{model_name}_checkpoint.pt")
            
            logger.info(f"Saved best model with F1 macro: {best_f1:.4f}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    
    # Save final model
    final_model_path = f"models/checkpoints/final_improved_f1_{model_name}.pt"
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
        'thresholds': optimal_thresholds
    }

def main():
    parser = argparse.ArgumentParser(description='Train a model with focus on specific classes and optimized for F1 score')
    parser.add_argument('--model', type=str, default='densenet121', 
                        choices=['densenet121', 'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b4'],
                        help='Model architecture to use')
    parser.add_argument('--selected_classes', type=str, default=None,
                        help='Comma-separated list of classes to focus on')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--sample_size', type=int, default=5000, help='Number of samples to use')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Parse selected classes
    selected_classes = None
    if args.selected_classes:
        selected_classes = [cls.strip() for cls in args.selected_classes.split(',')]
    
    # Train model
    results = train_model_with_focus(
        model_name=args.model,
        selected_classes=selected_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sample_size=args.sample_size,
        dropout_rate=args.dropout
    )
    
    # Print final results
    logger.info(f"Final F1 macro: {results['f1_macro']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 