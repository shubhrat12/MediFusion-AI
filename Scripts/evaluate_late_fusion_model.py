#!/usr/bin/env python
"""
Evaluate Late Fusion Model for Multi-Modal Health Insights Platform

This script loads a trained late fusion model and evaluates it on a new test set.

Input:
- Prediction files from each modality (CSV format)
- Ground truth labels
- Trained late fusion model

Output:
- Evaluation metrics
- Confusion matrix
- Predictions on test set
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union

# ML libraries
import sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def load_model(model_path: str) -> Dict:
    """
    Load trained late fusion model.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Dict: Loaded model dictionary
    """
    logger.info(f"Loading model from {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded {model['classifier_type']} model with {len(model['class_names'])} classes")
    
    return model


def load_test_data(
    tabular_preds_path: str,
    image_preds_path: str,
    text_preds_path: str,
    labels_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load test data from prediction files.
    
    Args:
        tabular_preds_path: Path to tabular predictions CSV
        image_preds_path: Path to image predictions CSV
        text_preds_path: Path to text predictions CSV
        labels_path: Path to ground truth labels CSV
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            Combined features dataframe and labels dataframe
    """
    logger.info("Loading test data...")
    
    # Load prediction files
    tabular_preds = pd.read_csv(tabular_preds_path)
    image_preds = pd.read_csv(image_preds_path)
    text_preds = pd.read_csv(text_preds_path)
    true_labels = pd.read_csv(labels_path)
    
    logger.info(f"Loaded tabular predictions: {tabular_preds.shape}")
    logger.info(f"Loaded image predictions: {image_preds.shape}")
    logger.info(f"Loaded text predictions: {text_preds.shape}")
    logger.info(f"Loaded true labels: {true_labels.shape}")
    
    # Verify all files have patient_id column
    for df, name in zip([tabular_preds, image_preds, text_preds, true_labels], 
                        ["tabular", "image", "text", "labels"]):
        if 'patient_id' not in df.columns:
            raise ValueError(f"{name} predictions file does not have a patient_id column")
    
    # Use patient_id as index
    tabular_preds = tabular_preds.set_index('patient_id')
    image_preds = image_preds.set_index('patient_id')
    text_preds = text_preds.set_index('patient_id')
    true_labels = true_labels.set_index('patient_id')
    
    # Check if all files have the same patients
    common_patients = set(tabular_preds.index) & set(image_preds.index) & set(text_preds.index) & set(true_labels.index)
    logger.info(f"Found {len(common_patients)} common patients across all modalities")
    
    # Filter to common patients
    common_patients_list = sorted(list(common_patients))
    tabular_preds = tabular_preds.loc[common_patients_list]
    image_preds = image_preds.loc[common_patients_list]
    text_preds = text_preds.loc[common_patients_list]
    true_labels = true_labels.loc[common_patients_list]
    
    # Rename columns to avoid conflicts when merging
    for prefix, df in zip(['tab_', 'img_', 'txt_'], [tabular_preds, image_preds, text_preds]):
        df.columns = [prefix + col for col in df.columns]
    
    # Create combined features dataframe
    features_df = pd.concat([tabular_preds, image_preds, text_preds], axis=1)
    
    # Sort both dataframes by index to ensure alignment
    features_df = features_df.sort_index()
    true_labels = true_labels.sort_index()
    
    return features_df, true_labels


def calculate_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        true_labels: Ground truth labels
        predictions: Binary predictions
        probabilities: Prediction probabilities
        class_names: List of class names
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    metrics = {}
    
    # Calculate accuracy (sample-wise)
    metrics['accuracy'] = accuracy_score(true_labels, predictions)
    
    # Calculate precision, recall, and F1 (macro-averaged across classes)
    metrics['precision_macro'] = precision_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(true_labels, predictions, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    # Calculate micro-averaged metrics too (useful for imbalanced datasets)
    metrics['precision_micro'] = precision_score(true_labels, predictions, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(true_labels, predictions, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(true_labels, predictions, average='micro', zero_division=0)
    
    # Try to calculate AUC-ROC (macro and per class)
    try:
        # Macro-averaged AUC-ROC
        metrics['auc_roc_macro'] = roc_auc_score(true_labels, probabilities, average='macro')
        
        # Per-class AUC-ROC
        per_class_auc = roc_auc_score(true_labels, probabilities, average=None)
        
        # Store each class's AUC
        for i, class_name in enumerate(class_names):
            metrics[f'auc_class_{class_name}'] = float(per_class_auc[i])
    
    except ValueError as e:
        # This can happen if a class has all negative or all positive samples
        logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
        metrics['auc_roc_macro'] = float('nan')
    
    return metrics


def plot_confusion_matrices(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    output_dir: str
) -> None:
    """
    Plot confusion matrices for each class.
    
    Args:
        true_labels: Ground truth labels
        predictions: Binary predictions
        class_names: List of class names
        output_dir: Directory to save plots
    """
    logger.info("Plotting confusion matrices...")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix for each class
    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(true_labels[:, i], predictions[:, i])
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=cm,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        
        plt.title(f'Confusion Matrix: {class_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure
        plot_path = os.path.join(output_dir, f'confusion_matrix_{class_name}.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved confusion matrix for {class_name} to {plot_path}")
    
    # Create a composite figure with all classes
    n_classes = len(class_names)
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, class_name in enumerate(class_names):
        plt.subplot(n_rows, n_cols, i + 1)
        cm = confusion_matrix(true_labels[:, i], predictions[:, i])
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=cm,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        
        plt.title(f'{class_name}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
    
    plt.tight_layout()
    composite_path = os.path.join(output_dir, 'confusion_matrices_all.png')
    plt.savefig(composite_path, dpi=300)
    plt.close()
    logger.info(f"Saved composite confusion matrix to {composite_path}")


def evaluate_model(
    meta_model: Dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    output_dir: str
) -> Dict[str, float]:
    """
    Evaluate late fusion model on test data.
    
    Args:
        meta_model: Trained meta-model dictionary
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save outputs
        
    Returns:
        Dict[str, float]: Test metrics
    """
    logger.info("Evaluating late fusion model...")
    
    # Extract components from meta_model
    scaler = meta_model['scaler']
    models = meta_model['models']
    class_names = meta_model['class_names']
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    test_preds = np.zeros((X_test.shape[0], len(class_names)))
    test_probs = np.zeros((X_test.shape[0], len(class_names)))
    
    for i, class_name in enumerate(class_names):
        model = models[class_name]
        if hasattr(model, 'predict_proba'):
            test_probs[:, i] = model.predict_proba(X_test_scaled)[:, 1]
        else:
            test_probs[:, i] = model.predict(X_test_scaled)
        test_preds[:, i] = (test_probs[:, i] > 0.5).astype(int)
    
    # Calculate metrics
    test_metrics = calculate_metrics(
        y_test.values,
        test_preds,
        test_probs,
        class_names
    )
    
    # Log metrics
    logger.info("Test metrics:")
    for metric_name, metric_value in test_metrics.items():
        if not metric_name.startswith('auc_class_'):  # Skip per-class AUC in logs
            logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrices(
        y_test.values,
        test_preds,
        class_names,
        os.path.join(output_dir, 'plots')
    )
    
    # Save predictions
    predictions_df = pd.DataFrame(
        test_probs,
        columns=class_names,
        index=X_test.index
    )
    
    # Add binary predictions
    for i, class_name in enumerate(class_names):
        predictions_df[f'{class_name}_binary'] = test_preds[:, i]
    
    # Reset index to get patient_id as a column
    predictions_df = predictions_df.reset_index()
    
    # Save to CSV
    predictions_file = os.path.join(output_dir, 'late_fusion_eval_preds.csv')
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Saved test predictions to {predictions_file}")
    
    # Save metrics
    metrics_file = os.path.join(output_dir, 'late_fusion_eval_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")
    
    return test_metrics


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate late fusion model")
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/saved/late_fusion_model.pkl",
        help="Path to trained late fusion model"
    )
    
    parser.add_argument(
        "--tabular_preds",
        type=str,
        default="outputs/test/tabular_preds.csv",
        help="Path to tabular predictions CSV"
    )
    
    parser.add_argument(
        "--image_preds",
        type=str,
        default="outputs/test/image_preds.csv",
        help="Path to image predictions CSV"
    )
    
    parser.add_argument(
        "--text_preds",
        type=str,
        default="outputs/test/text_preds.csv",
        help="Path to text predictions CSV"
    )
    
    parser.add_argument(
        "--labels",
        type=str,
        default="outputs/test/true_labels.csv",
        help="Path to ground truth labels CSV"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    meta_model = load_model(args.model)
    
    # Load test data
    X_test, y_test = load_test_data(
        args.tabular_preds,
        args.image_preds,
        args.text_preds,
        args.labels
    )
    
    # Evaluate model
    test_metrics = evaluate_model(
        meta_model,
        X_test,
        y_test,
        args.output_dir
    )
    
    logger.info("Evaluation complete!")
    logger.info(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"AUC-ROC Macro: {test_metrics.get('auc_roc_macro', float('nan')):.4f}")


if __name__ == "__main__":
    main() 