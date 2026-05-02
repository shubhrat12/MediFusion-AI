#!/usr/bin/env python
"""
Create dummy data for testing the late fusion pipeline.

This script generates:
- Tabular model predictions
- Image model predictions
- Text model predictions
- Ground truth labels

All with the same structure and patient IDs for testing.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Set random seed for reproducibility
np.random.seed(42)

def generate_dummy_predictions(n_samples=500, n_classes=5, bias=0.0, noise=0.3):
    """
    Generate dummy predictions with specific bias and noise levels.
    
    Args:
        n_samples: Number of patients
        n_classes: Number of disease classes
        bias: Bias to add (positive values make predictions higher)
        noise: Amount of noise to add
        
    Returns:
        DataFrame with predictions
    """
    # Generate patient IDs
    patient_ids = [f"PAT_{i:05d}" for i in range(n_samples)]
    
    # Generate predictions (probabilities)
    base_preds = np.random.random((n_samples, n_classes)) * 0.5  # Base predictions in [0, 0.5]
    
    # Add bias and noise
    predictions = base_preds + bias + np.random.normal(0, noise, (n_samples, n_classes))
    
    # Clip to valid probability range
    predictions = np.clip(predictions, 0, 1)
    
    # Create column names
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Create DataFrame
    df = pd.DataFrame(predictions, columns=class_names)
    df.insert(0, 'patient_id', patient_ids)
    
    return df

def generate_ground_truth(n_samples=500, n_classes=5):
    """
    Generate ground truth labels with realistic class distribution.
    
    Args:
        n_samples: Number of patients
        n_classes: Number of disease classes
        
    Returns:
        DataFrame with binary labels
    """
    # Generate patient IDs (same as predictions)
    patient_ids = [f"PAT_{i:05d}" for i in range(n_samples)]
    
    # Generate ground truth with class imbalance
    # Most diseases have around 10-15% prevalence
    prevalences = np.random.uniform(0.1, 0.15, n_classes)
    
    # Generate binary labels based on prevalence
    labels = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        labels[:, i] = np.random.random(n_samples) < prevalences[i]
    
    # Create column names
    class_names = [f"class_{i}" for i in range(n_classes)]
    
    # Create DataFrame
    df = pd.DataFrame(labels, columns=class_names)
    df.insert(0, 'patient_id', patient_ids)
    
    return df

def calculate_f1_macro(preds_df, labels_df):
    """Calculate F1 Macro score between predictions and labels."""
    # Convert probabilities to binary predictions
    binary_preds = (preds_df.iloc[:, 1:].values > 0.5).astype(int)
    labels = labels_df.iloc[:, 1:].values
    
    # Calculate F1 macro
    f1 = f1_score(labels, binary_preds, average='macro')
    return f1

def main():
    """Generate and save dummy data."""
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Parameters
    n_samples = 500
    n_classes = 5
    
    # Generate ground truth
    print("Generating ground truth labels...")
    labels_df = generate_ground_truth(n_samples, n_classes)
    
    # Generate predictions with different accuracy levels
    print("Generating model predictions...")
    
    # Tabular predictions - moderate performance, lower noise
    tabular_df = generate_dummy_predictions(n_samples, n_classes, bias=0.2, noise=0.2)
    
    # Image predictions - best performance, lowest noise
    image_df = generate_dummy_predictions(n_samples, n_classes, bias=0.3, noise=0.15)
    
    # Text predictions - lowest performance, highest noise
    text_df = generate_dummy_predictions(n_samples, n_classes, bias=0.1, noise=0.25)
    
    # Calculate F1 scores
    tabular_f1 = calculate_f1_macro(tabular_df, labels_df)
    image_f1 = calculate_f1_macro(image_df, labels_df)
    text_f1 = calculate_f1_macro(text_df, labels_df)
    
    print(f"Tabular model F1 Macro: {tabular_f1:.4f}")
    print(f"Image model F1 Macro: {image_f1:.4f}")
    print(f"Text model F1 Macro: {text_f1:.4f}")
    
    # Save files
    labels_df.to_csv("outputs/true_labels.csv", index=False)
    tabular_df.to_csv("outputs/tabular_preds.csv", index=False)
    image_df.to_csv("outputs/image_preds.csv", index=False)
    text_df.to_csv("outputs/text_preds.csv", index=False)
    
    # Create a test set (20% of samples)
    test_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
    test_patient_ids = [f"PAT_{i:05d}" for i in test_indices]
    
    # Create test directory
    os.makedirs("outputs/test", exist_ok=True)
    
    # Filter test samples
    labels_test_df = labels_df[labels_df['patient_id'].isin(test_patient_ids)]
    tabular_test_df = tabular_df[tabular_df['patient_id'].isin(test_patient_ids)]
    image_test_df = image_df[image_df['patient_id'].isin(test_patient_ids)]
    text_test_df = text_df[text_df['patient_id'].isin(test_patient_ids)]
    
    # Save test files
    labels_test_df.to_csv("outputs/test/true_labels.csv", index=False)
    tabular_test_df.to_csv("outputs/test/tabular_preds.csv", index=False)
    image_test_df.to_csv("outputs/test/image_preds.csv", index=False)
    text_test_df.to_csv("outputs/test/text_preds.csv", index=False)
    
    print("\nDummy data created successfully:")
    print(f"- {n_samples} patients")
    print(f"- {n_classes} disease classes")
    print(f"- Files saved in 'outputs/' and 'outputs/test/' directories")

if __name__ == "__main__":
    main() 