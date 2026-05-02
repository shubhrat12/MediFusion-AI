#!/usr/bin/env python
"""
Weighted model fusion using performance-based weights.

This script implements a weighted averaging approach for multimodal fusion:
1. Loads prediction files from tabular, image, and text models
2. Calculates AUC-ROC scores for each modality on each class
3. Uses these scores as weights to compute weighted average probabilities
4. Applies a threshold of 0.5 to generate binary predictions
5. Evaluates the performance against true labels
6. Saves metrics and predictions to files
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
TABULAR_PREDS_PATH = "outputs/tabular_preds.csv"
IMAGE_PREDS_PATH = "outputs/image_preds.csv"
TEXT_PREDS_PATH = "outputs/text_preds.csv"
TRUE_LABELS_PATH = "outputs/true_labels.csv"
OUTPUT_METRICS_PATH = "logs/fusion_weighted_average_metrics.json"
OUTPUT_PREDS_PATH = "outputs/fusion_weighted_average_preds.csv"

# Create output directories if they don't exist
os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PREDS_PATH), exist_ok=True)


def calculate_metrics(targets, predictions, probabilities):
    """
    Calculate evaluation metrics for multi-class classification.
    
    Args:
        targets: Ground truth binary labels, shape (n_samples, n_classes)
        predictions: Predicted binary labels, shape (n_samples, n_classes)
        probabilities: Predicted probabilities, shape (n_samples, n_classes)
        
    Returns:
        Dict: Dictionary of metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate accuracy
    metrics['accuracy'] = float(accuracy_score(targets, predictions))
    
    # Calculate precision, recall, and F1 (macro-averaged across classes)
    metrics['precision_macro'] = float(precision_score(targets, predictions, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(targets, predictions, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(targets, predictions, average='macro', zero_division=0))
    
    # Calculate micro-averaged metrics (useful for imbalanced datasets)
    metrics['precision_micro'] = float(precision_score(targets, predictions, average='micro', zero_division=0))
    metrics['recall_micro'] = float(recall_score(targets, predictions, average='micro', zero_division=0))
    metrics['f1_micro'] = float(f1_score(targets, predictions, average='micro', zero_division=0))
    
    # Calculate AUC-ROC (macro and per class)
    try:
        # Macro-averaged AUC-ROC
        metrics['auc_roc_macro'] = float(roc_auc_score(targets, probabilities, average='macro'))
        
        # Per-class AUC-ROC
        per_class_auc = roc_auc_score(targets, probabilities, average=None)
        
        # Store each class's AUC
        n_classes = targets.shape[1]
        for i in range(n_classes):
            class_name = f'class_{i}'
            metrics[f'auc_class_{class_name}'] = float(per_class_auc[i])
    
    except ValueError as e:
        # This can happen if a class has all negative or all positive samples
        logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
        metrics['auc_roc_macro'] = float('nan')
    
    return metrics


def calculate_auc_weights(data_frame, true_labels, class_names, model_prefix):
    """
    Calculate weights based on AUC-ROC scores for each class for a specific model.
    
    Args:
        data_frame: DataFrame with predictions
        true_labels: Ground truth labels
        class_names: List of class names
        model_prefix: Prefix used for the model's columns
        
    Returns:
        Dict: Dictionary of AUC-ROC scores per class
    """
    weights = {}
    
    for class_name in class_names:
        try:
            pred_col = f"{model_prefix}_{class_name}"
            true_col = class_name
            
            # Calculate AUC-ROC for this class and model
            auc = roc_auc_score(true_labels[true_col].values, data_frame[pred_col].values)
            weights[class_name] = max(0.5, auc)  # Use at least 0.5 as weight to avoid poor models getting too little weight
        except ValueError as e:
            logger.warning(f"Could not calculate AUC for {model_prefix} on {class_name}: {e}")
            weights[class_name] = 0.5  # Default weight if calculation fails
    
    return weights


def main():
    """Main function to perform weighted average fusion and evaluation."""
    # Load prediction files
    logger.info("Loading prediction files...")
    tabular_preds_df = pd.read_csv(TABULAR_PREDS_PATH)
    image_preds_df = pd.read_csv(IMAGE_PREDS_PATH)
    text_preds_df = pd.read_csv(TEXT_PREDS_PATH)
    true_labels_df = pd.read_csv(TRUE_LABELS_PATH)
    
    # Merge all dataframes on patient_id
    logger.info("Merging prediction dataframes...")
    
    # Rename columns to distinguish between models
    tabular_cols = {col: f"tabular_{col}" for col in tabular_preds_df.columns if col != "patient_id"}
    image_cols = {col: f"image_{col}" for col in image_preds_df.columns if col != "patient_id"}
    text_cols = {col: f"text_{col}" for col in text_preds_df.columns if col != "patient_id"}
    
    tabular_preds_df = tabular_preds_df.rename(columns=tabular_cols)
    image_preds_df = image_preds_df.rename(columns=image_cols)
    text_preds_df = text_preds_df.rename(columns=text_cols)
    
    # Merge dataframes
    merged_df = tabular_preds_df.merge(
        image_preds_df, on="patient_id", how="inner"
    ).merge(
        text_preds_df, on="patient_id", how="inner"
    ).merge(
        true_labels_df, on="patient_id", how="inner"
    )
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    
    # Extract class names
    class_names = [col for col in true_labels_df.columns if col != "patient_id"]
    num_classes = len(class_names)
    
    # Calculate weights for each modality based on AUC-ROC performance
    logger.info("Calculating performance-based weights...")
    
    tabular_weights = calculate_auc_weights(merged_df, true_labels_df, class_names, "tabular")
    image_weights = calculate_auc_weights(merged_df, true_labels_df, class_names, "image")
    text_weights = calculate_auc_weights(merged_df, true_labels_df, class_names, "text")
    
    # Log the weights
    logger.info("Model weights per class:")
    for class_name in class_names:
        logger.info(f"{class_name} - Tabular: {tabular_weights[class_name]:.4f}, "
                    f"Image: {image_weights[class_name]:.4f}, "
                    f"Text: {text_weights[class_name]:.4f}")
    
    # Initialize arrays for fused predictions
    fused_probs = np.zeros((len(merged_df), num_classes))
    
    # Compute weighted average probabilities across models
    logger.info("Computing weighted average probabilities...")
    for i, class_name in enumerate(class_names):
        tabular_col = f"tabular_{class_name}"
        image_col = f"image_{class_name}"
        text_col = f"text_{class_name}"
        
        # Get weights for this class
        tab_weight = tabular_weights[class_name]
        img_weight = image_weights[class_name]
        txt_weight = text_weights[class_name]
        
        # Normalize weights
        total_weight = tab_weight + img_weight + txt_weight
        tab_weight_norm = tab_weight / total_weight
        img_weight_norm = img_weight / total_weight
        txt_weight_norm = txt_weight / total_weight
        
        # Apply weighted average
        fused_probs[:, i] = (
            tab_weight_norm * merged_df[tabular_col].values +
            img_weight_norm * merged_df[image_col].values +
            txt_weight_norm * merged_df[text_col].values
        )
    
    # Convert to binary predictions using threshold 0.5
    logger.info("Applying threshold for binary predictions...")
    fused_preds = (fused_probs >= 0.5).astype(int)
    
    # Extract true labels
    true_labels = merged_df[class_names].values
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    metrics = calculate_metrics(true_labels, fused_preds, fused_probs)
    
    # Save weights used
    metrics['fusion_weights'] = {
        'tabular': {cls: float(tabular_weights[cls]) for cls in class_names},
        'image': {cls: float(image_weights[cls]) for cls in class_names},
        'text': {cls: float(text_weights[cls]) for cls in class_names}
    }
    
    # Save metrics
    logger.info(f"Saving metrics to {OUTPUT_METRICS_PATH}")
    with open(OUTPUT_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create output dataframe with patient_id and fused probabilities
    output_df = pd.DataFrame({"patient_id": merged_df["patient_id"]})
    for i, class_name in enumerate(class_names):
        output_df[class_name] = fused_probs[:, i]
    
    # Save predictions
    logger.info(f"Saving predictions to {OUTPUT_PREDS_PATH}")
    output_df.to_csv(OUTPUT_PREDS_PATH, index=False)
    
    # Log key metrics
    logger.info(f"Fusion Weighted Average - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Fusion Weighted Average - F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"Fusion Weighted Average - AUC-ROC Macro: {metrics['auc_roc_macro']:.4f}")


if __name__ == "__main__":
    main() 