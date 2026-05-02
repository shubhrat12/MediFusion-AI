#!/usr/bin/env python
"""
Evaluation script for multimodal fusion models.

This script evaluates trained fusion models on a held-out test set:
1. Loads the best trained model
2. Evaluates on test patients
3. Calculates and reports comprehensive metrics
4. Saves predictions and results to files

Compatible with all three fusion strategies: early, late, and hybrid fusion.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from secure_fusion_evaluator import SecureFusionEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from models.model_factory import ModelFactory
from models.fusion.multimodal_fusion import EarlyFusionModel, LateFusionModel, HybridFusionModel
from data.datasets import MultiModalDataset
from torch.utils.data import DataLoader

def check_version_compatibility(model_version: str, supported_version: str = "1.0.0") -> bool:
    """
    Check if the model version is compatible with the supported version.
    Args:
        model_version (str): Version string from the checkpoint.
        supported_version (str): Supported version string.
    Returns:
        bool: True if compatible, False otherwise.
    """
    # For now, just check major version compatibility
    model_major = model_version.split('.')[0]
    supported_major = supported_version.split('.')[0]
    return model_major == supported_major


def calculate_metrics(
    targets: np.ndarray, 
    predictions: np.ndarray, 
    probabilities: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for multi-label classification.
    
    Args:
        targets: Ground truth binary labels, shape (n_samples, n_classes)
        predictions: Predicted binary labels, shape (n_samples, n_classes)
        probabilities: Predicted probabilities, shape (n_samples, n_classes)
        class_names: Optional list of class names for reporting
        
    Returns:
        Dict[str, Any]: Dictionary of metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate accuracy (sample-wise)
    metrics['accuracy'] = float(accuracy_score(targets, predictions))
    
    # Calculate precision, recall, and F1 (macro-averaged across classes)
    metrics['precision_macro'] = float(precision_score(targets, predictions, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(targets, predictions, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(targets, predictions, average='macro', zero_division=0))
    
    # Calculate micro-averaged metrics (useful for imbalanced datasets)
    metrics['precision_micro'] = float(precision_score(targets, predictions, average='micro', zero_division=0))
    metrics['recall_micro'] = float(recall_score(targets, predictions, average='micro', zero_division=0))
    metrics['f1_micro'] = float(f1_score(targets, predictions, average='micro', zero_division=0))
    
    # Calculate weighted F1 (accounts for class imbalance)
    metrics['f1_weighted'] = float(f1_score(targets, predictions, average='weighted', zero_division=0))
    
    # Calculate AUC-ROC (macro and per class)
    try:
        # Macro-averaged AUC-ROC
        metrics['auc_roc_macro'] = float(roc_auc_score(targets, probabilities, average='macro'))
        
        # Per-class AUC-ROC
        per_class_auc = roc_auc_score(targets, probabilities, average=None)
        metrics['auc_roc_per_class'] = per_class_auc.tolist()
        
        # Store each class's AUC with name if available
        n_classes = targets.shape[1]
        for i in range(n_classes):
            class_name = class_names[i] if class_names and i < len(class_names) else f'class_{i}'
            metrics[f'auc_{class_name}'] = float(per_class_auc[i])
    
    except ValueError as e:
        # This can happen if a class has all negative or all positive samples
        logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
        metrics['auc_roc_macro'] = float('nan')
    
    # Calculate average precision (PR-AUC) per class and macro
    try:
        # Macro-averaged PR-AUC
        metrics['pr_auc_macro'] = float(average_precision_score(targets, probabilities, average='macro'))
        
        # Per-class PR-AUC
        per_class_pr_auc = average_precision_score(targets, probabilities, average=None)
        metrics['pr_auc_per_class'] = per_class_pr_auc.tolist()
    
    except ValueError as e:
        logger.warning(f"Could not calculate PR-AUC: {str(e)}")
        metrics['pr_auc_macro'] = float('nan')
    
    # Get classification report as dictionary (precision, recall, f1 per class)
    try:
        class_report = classification_report(targets, predictions, output_dict=True, zero_division=0)
        metrics['classification_report'] = class_report
    except Exception as e:
        logger.warning(f"Could not generate classification report: {str(e)}")
    
    return metrics


def load_model(
    config_path: str,
    model_path: str,
    device: torch.device
) -> Tuple[Dict[str, Any], torch.nn.Module]:
    """
    Load the fusion model using configuration and checkpoint with security checks.
    
    Args:
        config_path: Path to model configuration JSON
        model_path: Path to model checkpoint (.pt file)
        device: Device to load the model on
        
    Returns:
        Tuple[Dict[str, Any], torch.nn.Module]: Config and loaded model
    """
    # Validate paths
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Validate file extensions
    if not config_path.endswith('.json'):
        raise ValueError("Config file must be a .json file")
    if not model_path.endswith('.pt'):
        raise ValueError("Model file must be a .pt file")
        
    # Check file sizes
    config_size = os.path.getsize(config_path)
    model_size = os.path.getsize(model_path)
    
    if config_size > 10 * 1024 * 1024:  # 10MB limit for config
        raise ValueError(f"Config file too large: {config_size} bytes")
    if model_size > 2 * 1024 * 1024 * 1024:  # 2GB limit for model
        raise ValueError(f"Model file too large: {model_size} bytes")
    
    try:
        # Load configuration with size limit
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate config schema
        required_fields = {'fusion_strategy', 'input_dims'}
        if not all(field in config for field in required_fields):
            raise ValueError(f"Missing required fields in config: {required_fields}")
            
        # Extract model parameters
        fusion_strategy = config.get('fusion_strategy', 'hybrid_fusion')
        input_dims = config.get('input_dims', {})
        
        # Validate fusion strategy
        valid_strategies = {'early_fusion', 'late_fusion', 'hybrid_fusion'}
        if fusion_strategy not in valid_strategies:
            raise ValueError(f"Invalid fusion strategy: {fusion_strategy}")
        
        # Create appropriate model based on fusion strategy
        if fusion_strategy == 'early_fusion':
            model = EarlyFusionModel(
                input_dims=input_dims,
                hidden_dims=config.get('hidden_dims', [512, 256, 128]),
                output_dim=config.get('output_dim', 14),
                dropout_rate=config.get('dropout_rate', 0.5),
                use_batch_norm=config.get('use_batch_norm', True)
            )
        elif fusion_strategy == 'late_fusion':
            model = LateFusionModel(
                input_dims=input_dims,
                hidden_dims=config.get('hidden_dims', {}),
                output_dim=config.get('output_dim', 14),
                dropout_rate=config.get('dropout_rate', 0.5),
                use_batch_norm=config.get('use_batch_norm', True)
            )
        else:  # hybrid_fusion
            model = HybridFusionModel(
                input_dims=input_dims,
                intermediate_dims=config.get('intermediate_dims', {}),
                joint_hidden_dims=config.get('joint_hidden_dims', [512, 256]),
                output_dim=config.get('output_dim', 14),
                dropout_rate=config.get('dropout_rate', 0.5),
                use_batch_norm=config.get('use_batch_norm', True),
                use_attention=config.get('use_attention', True)
            )
        
        # Load model checkpoint with safety checks
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")
            
        # Validate checkpoint contents
        if not isinstance(checkpoint, (dict, torch.nn.Module)):
            raise ValueError("Invalid checkpoint format")
            
        # Load model state with version check
        if 'model_state_dict' in checkpoint:
            # Check for version compatibility if available
            model_version = checkpoint.get('version', '1.0.0')
            if not check_version_compatibility(model_version):
                raise ValueError(f"Incompatible model version: {model_version}")
                
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                raise RuntimeError(f"Failed to load model state: {str(e)}")
        else:
            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                raise RuntimeError(f"Failed to load model state: {str(e)}")
        
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        return config, model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate the model on the test set.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        class_names: Optional list of class names
        
    Returns:
        Tuple[Dict[str, Any], pd.DataFrame]: Metrics and predictions dataframe
    """
    model.eval()
    
    # Lists to store predictions and patient IDs
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_patient_ids = []
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate over batches
        progress_bar = tqdm(test_loader, desc='Evaluating')
        for batch in progress_bar:
            # Get batch data
            tabular_features = batch.get('tabular')
            image_features = batch.get('image')
            text_features = batch.get('text')
            labels = batch.get('labels')
            patient_ids = batch.get('patient_id', [None] * len(labels))
            
            # Move to device
            if tabular_features is not None:
                tabular_features = tabular_features.to(device)
            if image_features is not None:
                image_features = image_features.to(device)
            if text_features is not None:
                text_features = text_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(
                tabular_features=tabular_features,
                image_features=image_features,
                text_features=text_features
            )
            
            # Convert outputs to probabilities and binary predictions
            probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
            predictions = (probabilities > 0.5).astype(np.float32)
            
            # Store for metrics calculation
            all_targets.append(labels.detach().cpu().numpy())
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
            all_patient_ids.extend(patient_ids)
    
    # Combine all batches
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    all_probabilities = np.vstack(all_probabilities)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions, all_probabilities, class_names)
    
    # Create DataFrame with predictions
    columns = [f"class_{i}" if class_names is None else class_names[i] 
              for i in range(all_targets.shape[1])]
    
    # Prepare data for DataFrame
    data = {
        'patient_id': all_patient_ids
    }
    
    # Add true labels
    for i, col in enumerate(columns):
        data[f"true_{col}"] = all_targets[:, i]
    
    # Add predicted labels
    for i, col in enumerate(columns):
        data[f"pred_{col}"] = all_predictions[:, i]
    
    # Add predicted probabilities
    for i, col in enumerate(columns):
        data[f"prob_{col}"] = all_probabilities[:, i]
    
    # Create DataFrame
    predictions_df = pd.DataFrame(data)
    
    return metrics, predictions_df


def run_evaluation(
    config_path: str,
    model_path: str,
    test_dataset_path: str,
    output_dir: str,
    device_str: Optional[str] = None,
    batch_size: int = 32
):
    """
    Run the full evaluation process.
    
    Args:
        config_path: Path to model configuration
        model_path: Path to model checkpoint
        test_dataset_path: Path to test dataset
        output_dir: Directory to save results
        device_str: Device to run evaluation on
        batch_size: Batch size for evaluation
    """
    # Setup device
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and config
    logger.info(f"Loading model from {model_path}")
    config, model = load_model(config_path, model_path, device)
    
    # Extract fusion strategy for logging
    fusion_strategy = config.get('fusion_strategy', 'hybrid_fusion')
    logger.info(f"Loaded {fusion_strategy} model")
    
    # Load test dataset
    logger.info(f"Loading test dataset from {test_dataset_path}")
    test_dataset = MultiModalDataset(test_dataset_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get class names if available
    class_names = config.get('class_names', None)
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    metrics, predictions_df = evaluate_model(model, test_loader, device, class_names)
    
    # Print summary metrics
    logger.info("=== Evaluation Results ===")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"Precision Macro: {metrics['precision_macro']:.4f}")
    logger.info(f"Recall Macro: {metrics['recall_macro']:.4f}")
    logger.info(f"AUC-ROC Macro: {metrics['auc_roc_macro']:.4f}")
    
    # Save predictions to CSV
    predictions_path = os.path.join(output_dir, f"{fusion_strategy}_predictions.csv")
    logger.info(f"Saving predictions to {predictions_path}")
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, f"{fusion_strategy}_metrics.json")
    logger.info(f"Saving metrics to {metrics_path}")
    
    # Filter metrics to save (some may not be JSON serializable)
    serializable_metrics = {}
    for key, value in metrics.items():
        if key != 'classification_report':  # Skip complex nested dictionary
            try:
                # Test if serializable
                json.dumps(value)
                serializable_metrics[key] = value
            except (TypeError, OverflowError):
                logger.warning(f"Metric '{key}' is not JSON serializable, skipping")
    
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info("Evaluation complete!")
    
    return metrics


def evaluate_case(
    evaluator: SecureFusionEvaluator,
    image_path: Optional[str] = None,
    text_path: Optional[str] = None,
    tabular_features: Optional[Dict[str, float]] = None
):
    """
    Evaluate a single case with secure input handling.
    
    Args:
        evaluator: Secure fusion model evaluator
        image_path: Optional path to chest X-ray image
        text_path: Optional path to radiologist report
        tabular_features: Optional dictionary of tabular features
    
    Returns:
        Dictionary of disease probabilities
    """
    try:
        # Load text if path provided
        text = None
        if text_path:
            text_path = Path(text_path)
            if text_path.is_file():
                with open(text_path, 'r') as f:
                    text = f.read()
        
        # Make predictions
        predictions = evaluator.predict(
            image_path=image_path,
            text=text,
            tabular_features=tabular_features
        )
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error evaluating case: {e}")
        raise

def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate multimodal fusion models")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration JSON"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    run_evaluation(
        config_path=args.config,
        model_path=args.model,
        test_dataset_path=args.test_data,
        output_dir=args.output_dir,
        device_str=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()