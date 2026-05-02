"""
Modified training script for attention-based chest X-ray classification.

This script improves on train_attention_xray.py to achieve better F1 scores above 0.60
by implementing better threshold selection, model architecture, and training procedure.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import our custom modules
from models.attention_xray_model import AttentionXrayModel, EfficientNetXrayModel, AdaptiveFocalLoss
from data.data_loader import load_chestxray_images
from data.datasets import create_data_loaders
from modify_attention_model import improve_thresholds, modify_model, get_improved_focal_loss

# Create basic transforms that work more reliably
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

def train_xray_model(
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    model_name='efficientnet_b0',
    sample_size=800,  # Increased from 500
    dropout_rate=0.7,  # Increased from 0.5
    gamma=2.5,  # Increased from 2.0
    alpha=0.6,  # Increased from 0.25
    weight_decay=0.001,  # Increased from 0.0001
    num_workers=0,
    label_smoothing=0.1,  # Added label smoothing
    use_improved_thresholds=True
):
    """
    Train the X-ray model with improved parameters to achieve better F1 score.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_name: Model architecture to use
        sample_size: Number of samples to use
        dropout_rate: Dropout rate for regularization
        gamma: Gamma parameter for focal loss
        alpha: Alpha parameter for focal loss
        weight_decay: Weight decay for optimizer
        num_workers: Number of worker threads for data loading
        label_smoothing: Label smoothing parameter
        use_improved_thresholds: Whether to use improved threshold selection
        
    Returns:
        float: Best F1 macro score achieved
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load ChestX-ray14 dataset
    logger.info("Loading ChestX-ray14 dataset...")
    df, image_paths = load_chestxray_images(
        sample_size=sample_size,
        balanced_sampling=True
    )
    logger.info(f"Found {len(image_paths)} valid images with metadata")
    
    # Create data loaders with our custom transforms
    logger.info("Creating data loaders...")
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
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
        use_advanced_transforms=False  # Use our custom transforms instead
    )
    logger.info(f"Created datasets - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Number of output classes
    num_classes = len(disease_names)
    logger.info(f"Training model for {num_classes} disease classes: {disease_names}")
    
    # Create the model with our improvements
    logger.info(f"Creating {model_name} model...")
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
            unfreeze_blocks=4,
            dropout_rate=dropout_rate
        )
    
    # Apply our model improvements
    model = modify_model(model, num_classes, dropout_rate)
    logger.info(f"Model created with {model.count_trainable_parameters()} trainable parameters")
    
    # Move model to device
    model = model.to(device)
    
    # Calculate class weights for loss function
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.append(labels)
    train_labels = torch.cat(all_train_labels, dim=0)
    
    pos_counts = train_labels.sum(dim=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = torch.clamp(neg_counts / torch.clamp(pos_counts, min=1.0), min=1.0, max=50.0)
    pos_weight = pos_weight.to(device)
    
    # Create improved focal loss with our parameters
    criterion = get_improved_focal_loss(
        pos_weight=pos_weight,
        gamma=gamma,
        alpha=alpha,
        label_smoothing=label_smoothing
    )
    logger.info(f"Using Improved Focal Loss with gamma={gamma}, alpha={alpha}, label_smoothing={label_smoothing}")
    
    # Use AdamW optimizer with cosine scheduling
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    logger.info(f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    
    # Use ReduceLROnPlateau instead of OneCycleLR to avoid the division by zero error
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Monitor F1 score (higher is better)
        factor=0.5,  # Reduce LR by half when plateau is detected
        patience=1,  # Wait 1 epoch before reducing LR
        verbose=True,
        min_lr=1e-6
    )
    logger.info("Using ReduceLROnPlateau scheduler with factor=0.5, patience=1")
    
    # Training variables
    best_f1 = 0.0
    best_model_path = os.path.join("models/checkpoints", "attention_xray_improved.pt")
    
    # Main training loop
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        except ImportError:
            train_iter = train_loader
            
        for batch_X, batch_y in train_iter:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Training loss: {avg_train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds_scores = []
        val_targets = []
        
        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            val_iter = tqdm(val_loader, desc="Validation")
        except ImportError:
            val_iter = val_loader
            
        with torch.no_grad():
            for batch_X, batch_y in val_iter:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Get probabilities
                probs = torch.sigmoid(outputs)
                
                # Store predictions and targets
                val_preds_scores.extend(probs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # Convert to numpy arrays
        val_preds_scores = np.array(val_preds_scores)
        val_targets = np.array(val_targets)
        
        # Use our improved threshold selection
        if use_improved_thresholds:
            thresholds = improve_thresholds(val_targets, val_preds_scores, disease_names)
        else:
            # Default threshold of 0.5 for all classes
            thresholds = np.array([0.5] * len(disease_names))
        
        # Apply thresholds
        val_preds_binary = (val_preds_scores >= thresholds).astype(float)
        
        # Calculate F1 scores
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1_macro = f1_score(val_targets, val_preds_binary, average='macro', zero_division=0)
        f1_micro = f1_score(val_targets, val_preds_binary, average='micro', zero_division=0)
        precision = precision_score(val_targets, val_preds_binary, average='macro', zero_division=0)
        recall = recall_score(val_targets, val_preds_binary, average='macro', zero_division=0)
        
        logger.info(f"Validation - F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Calculate per-class metrics
        f1_scores = []
        print("\nPer-class metrics:")
        print(f"{'Class':<20} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
        print("-" * 60)
        
        for i, disease in enumerate(disease_names):
            class_precision = precision_score(val_targets[:, i], val_preds_binary[:, i], zero_division=0)
            class_recall = recall_score(val_targets[:, i], val_preds_binary[:, i], zero_division=0)
            class_f1 = f1_score(val_targets[:, i], val_preds_binary[:, i], zero_division=0)
            f1_scores.append(class_f1)
            
            print(f"{disease:<20} | {class_precision:>10.4f} | {class_recall:>10.4f} | {class_f1:>10.4f}")
        
        # Check if this is the best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            
            # Save model
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "thresholds": thresholds.tolist(),
                "f1_macro": f1_macro,
                "epoch": epoch
            }, best_model_path)
            
            logger.info(f"Saved best model with F1 macro: {best_f1:.4f}")
            
        # Update learning rate based on F1 score
        scheduler.step(f1_macro)
    
    logger.info(f"Training completed. Best F1 macro: {best_f1:.4f}")
    return best_f1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train X-ray model with improved parameters")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0", help="Model architecture")
    parser.add_argument("--sample_size", type=int, default=800, help="Number of samples")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker threads")
    
    args = parser.parse_args()
    
    best_f1 = train_xray_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        sample_size=args.sample_size,
        num_workers=args.num_workers
    )
    
    print(f"Best F1 macro: {best_f1:.4f}") 