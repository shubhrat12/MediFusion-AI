#!/usr/bin/env python
"""
Training script for image classification model on the ChestX-ray14 dataset.

This script:
1. Loads and preprocesses the ChestX-ray14 dataset
2. Uses a pre-trained DenseNet121 model with progressive unfreezing
3. Trains the model with weighted BCEWithLogitsLoss and label smoothing
4. Evaluates performance with per-class dynamic thresholds
5. Saves the trained model
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torchvision
import torchvision.transforms as transforms
from torchvision.models import densenet121, resnet50
from PIL import Image
import logging
import time
import csv
import cv2
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from data.data_loader import load_chestxray_images
from preprocessing.image_preprocessor import ImagePreprocessor
from data.dummy_data import generate_dummy_images
# Import for CheXagent support
from models.model_factory import ModelFactory
# Import FocalLoss from attention_xray_model
from models.attention_xray_model import FocalLoss


def find_chestxray_image(image_id, base_dir="data/raw/images/archive"):
    """
    Find the path to a chest X-ray image based on its ID.
    
    Args:
        image_id: Image ID (e.g. '00000001_000.png')
        base_dir: Base directory where images are stored
        
    Returns:
        str or None: Path to the image if found, None otherwise
    """
    # Try different possible paths (with forward slashes)
    paths = [
        os.path.join(base_dir, image_id),
        os.path.join(base_dir, "images_001", "images", image_id),  # All images seem to be in this folder
    ]
    
    # Debug info
    for path in paths:
        exists = os.path.exists(path)
        print(f"Checking path: {path}, exists: {exists}")
        if exists:
            return path
    
    return None


# Class for handling PIL images for CheXagent preprocessing
class CheXagentDataset(Dataset):
    """
    Custom dataset for CheXagent model with multi-label support.
    """
    def __init__(self, image_paths, labels):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: Multi-hot encoded labels for each image (shape: n_samples x n_classes)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, labels)
        """
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, labels


def get_transforms(is_training=True):
    """
    Get data augmentation transforms.
    """
    if is_training:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.Flip(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2(),
        ])


class ChestXrayDataset(Dataset):
    """
    Custom dataset for ChestX-ray14 images with multi-label support.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: Multi-hot encoded labels for each image (shape: n_samples x n_classes)
            transform: Optional transformations to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load and convert to grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']
            
            return img, torch.FloatTensor(label)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a dummy tensor if image loading fails
            dummy = torch.zeros((1, 224, 224))
            return dummy, torch.FloatTensor(label)


def count_trainable_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(y_true, y_pred, y_score, class_names):
    """
    Compute comprehensive evaluation metrics for multi-label classification.
    
    Args:
        y_true: Ground truth labels (shape: n_samples x n_classes)
        y_pred: Predicted binary labels (shape: n_samples x n_classes)
        y_score: Predicted probabilities/scores (shape: n_samples x n_classes)
        class_names: Names of the classes
        
    Returns:
        Dict: Dictionary containing various metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Try to compute AUC - may fail if only one class is present
    try:
        metrics['auc'] = roc_auc_score(y_true, y_score, average='macro')
    except ValueError:
        metrics['auc'] = float('nan')
    
    # Per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        try:
            class_metrics[class_name] = {
                'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            }
            
            # Try to compute AUC for this class
            if len(np.unique(y_true[:, i])) > 1:
                class_metrics[class_name]['auc'] = roc_auc_score(y_true[:, i], y_score[:, i])
            else:
                class_metrics[class_name]['auc'] = float('nan')
                
        except Exception as e:
            logger.warning(f"Error computing metrics for class {class_name}: {str(e)}")
            class_metrics[class_name] = {
                'precision': float('nan'),
                'recall': float('nan'),
                'f1': float('nan'),
                'auc': float('nan')
            }
    
    metrics['per_class'] = class_metrics
    return metrics


def print_metrics(metrics: Dict, detailed: bool = False):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        detailed: Whether to print detailed per-class metrics
    """
    logger.info(f"Overall Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  AUC:       {metrics['auc']:.4f}")
    
    if detailed and 'per_class' in metrics:
        logger.info("\nPer-class Metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {class_metrics['precision']:.4f}")
            logger.info(f"    Recall:    {class_metrics['recall']:.4f}")
            logger.info(f"    F1 Score:  {class_metrics['f1']:.4f}")
            logger.info(f"    AUC:       {class_metrics['auc']:.4f}")


def find_optimal_thresholds(y_true, y_score, class_names):
    """Find optimal threshold for each class based on F1 score"""
    thresholds = []
    best_f1s = []
    for i in range(y_true.shape[1]):
        best_f1 = 0
        best_threshold = 0.5  # Default
        # Try different thresholds and find the one that maximizes F1
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_score[:, i] > threshold).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds.append(best_threshold)
        best_f1s.append(best_f1)
    
    # Log the results
    for i, (class_name, threshold, f1) in enumerate(zip(class_names, thresholds, best_f1s)):
        logger.info(f"Optimal threshold for {class_name}: {threshold:.2f}, F1: {f1:.4f}")
    
    return np.array(thresholds)


def mixup_data(x, y, alpha=1.0):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class AdaptiveHistogramEqualization:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to tensor images"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy array
            np_img = img.cpu().numpy()
            # If image has 3 channels (C, H, W)
            if np_img.shape[0] == 3:
                # Process as grayscale (average channels)
                gray = (0.299 * np_img[0] + 0.587 * np_img[1] + 0.114 * np_img[2]).astype(np.uint8)
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                enhanced = clahe.apply(gray)
                # Stack back to 3 channels
                result = np.stack([enhanced, enhanced, enhanced])
                return torch.from_numpy(result).float() / 255.0
            return img
        return img


def load_class_names(metadata_dir="data/processed/metadata"):
    """
    Load class names from the metadata file.
    
    Args:
        metadata_dir: Directory containing the metadata files
        
    Returns:
        list: List of class names
    """
    class_names_file = os.path.join(metadata_dir, 'class_names.txt')
    if os.path.exists(class_names_file):
        with open(class_names_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    else:
        # Fallback to default classes if file doesn't exist
        return [
            'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
            'Fracture', 'Support Devices', 'No Finding'
        ]


def load_data_from_metadata(metadata_file):
    """
    Load image paths and labels from metadata CSV file.
    
    Args:
        metadata_file: Path to metadata CSV file containing image paths and labels
        
    Returns:
        Tuple of (image_paths, labels)
    """
    logger.info(f"Loading data from {metadata_file}")
    df = pd.read_csv(metadata_file)
    
    # Extract image paths and labels
    image_paths = df['image_path'].values
    
    # Get label columns (all except image_path)
    label_cols = [col for col in df.columns if col != 'image_path']
    labels = df[label_cols].values
    
    return image_paths, labels, label_cols


def train(
    epochs: int = 100,  # Increased epochs
    batch_size: int = 32,
    learning_rate: float = 3e-4,  # Slightly increased learning rate
    random_seed: int = 42,
    test_size: float = 0.2,
    use_dummy_data: bool = False,
    dummy_samples: int = 200,
    sample_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_amp: bool = True,
    threshold: float = 0.5,
    use_mixup: bool = True,
    mixup_alpha: float = 0.4,  # Increased mixup alpha
    unfreeze_layers: int = 8,
    model_type: str = 'chexagent',
    finetune_chexagent: bool = True,  # Changed default to True
    early_stopping_patience: int = 15,  # Increased patience
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    use_cosine_schedule: bool = True,  # New parameter
    train_data: str = None,
    test_data: str = None
):
    """
    Main training function with improved configuration.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    if use_dummy_data:
        logger.info(f"Generating {dummy_samples} dummy samples")
        train_samples = int(dummy_samples * 0.8)
        test_samples = dummy_samples - train_samples
        logger.info("Generating dummy data...")
        train_images, train_labels = generate_dummy_images(train_samples)
        test_images, test_labels = generate_dummy_images(test_samples)
        class_names = [f'class_{i}' for i in range(train_labels.shape[1])]
        logger.info(f"Generated {train_samples} training and {test_samples} test samples")
    else:
        if not (train_data and test_data):
            logger.error("No data files provided and not using dummy data")
            return
            
        # Load data from metadata files
        train_images, train_labels, class_names = load_data_from_metadata(train_data)
        test_images, test_labels, _ = load_data_from_metadata(test_data)
    
    # Create datasets with improved transforms
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    train_dataset = ChestXrayDataset(train_images, train_labels, transform=train_transform)
    test_dataset = ChestXrayDataset(test_images, test_labels, transform=val_transform)
    
    # Create data loaders
    logger.info(f"Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Initialize model
    num_classes = len(class_names)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Create model factory
    model_factory = ModelFactory(num_classes=num_classes)
    
    if model_type == 'chexagent':
        logger.info(f"Creating CheXagent model (finetune={finetune_chexagent})")
        model = model_factory.create_chexagent_model(finetune=finetune_chexagent)
    else:
        logger.info("Creating standard DenseNet121 model")
        model = model_factory.create_densenet_model()
    
    model = model.to(device)
    logger.info(f"Model created with {count_trainable_parameters(model):,} trainable parameters")
    
    # Calculate class weights for weighted loss
    pos_weights = []
    logger.info("Class weights calculated based on data distribution (capped):")
    for i, class_name in enumerate(class_names):
        pos_count = np.sum(train_labels[:, i] == 1)
        neg_count = np.sum(train_labels[:, i] == 0)
        weight = neg_count / pos_count if pos_count > 0 else 10.0  # Cap at 10
        pos_weights.append(min(weight, 10.0))
        logger.info(f"  {class_name}: {pos_weights[-1]:.2f} (original: {weight:.2f}, pos: {pos_count}, neg: {neg_count})")
    
    pos_weights = torch.FloatTensor(pos_weights).to(device)
    
    # Initialize loss function with label smoothing
    if use_focal_loss:
        criterion = FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            label_smoothing=0.1  # Added label smoothing
        )
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weights,
            label_smoothing=0.1
        )
    
    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01  # Added weight decay
    )
    
    # Initialize learning rate scheduler
    if use_cosine_schedule:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Initial cycle length
            T_mult=2,  # Cycle length multiplier
            eta_min=learning_rate * 0.01  # Minimum learning rate
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )
    
    # Initialize gradient scaler for AMP
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(train_bar):
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            
            # Forward pass with AMP
            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type=device.type):
                    if use_mixup:
                        mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
                        outputs = model(mixed_inputs)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if use_mixup:
                    mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
                    outputs = model(mixed_inputs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f"{train_loss/(batch_idx+1):.4f}"})
        
        # Evaluation
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs = inputs.to(device)
                targets = targets.float().to(device)
                
                if use_amp:
                    with autocast(device_type=device.type):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
        
        # Compute metrics
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)
        all_preds = (all_outputs >= threshold).astype(int)
        
        metrics = compute_metrics(all_targets, all_preds, all_outputs, class_names)
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(metrics['f1'])
        
        # Print metrics
        logger.info(f"\nEpoch {epoch+1}/{epochs}:")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
        logger.info(f"Val Loss: {val_loss/len(test_loader):.4f}")
        print_metrics(metrics)
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            logger.info(f"New best F1 score: {best_f1:.4f}")
            torch.save(model.state_dict(), 'models/chexagent/chexagent_13class.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Training completed!")
    return model, best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CheXagent model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model_type", type=str, default='chexagent')
    parser.add_argument("--finetune_chexagent", action='store_true')
    parser.add_argument("--use_focal_loss", action='store_true')
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--use_mixup", action='store_true')
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--unfreeze_layers", type=int, default=8)
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--use_cosine_schedule", action='store_true', help="Use cosine annealing scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--train_data", type=str, help="Path to training metadata CSV")
    parser.add_argument("--test_data", type=str, help="Path to test metadata CSV")
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_type=args.model_type,
        finetune_chexagent=args.finetune_chexagent,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        unfreeze_layers=args.unfreeze_layers,
        use_amp=args.use_amp,
        use_cosine_schedule=args.use_cosine_schedule,
        early_stopping_patience=args.early_stopping_patience,
        train_data=args.train_data,
        test_data=args.test_data
    ) 