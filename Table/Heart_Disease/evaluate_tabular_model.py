#!/usr/bin/env python
"""
Evaluation script for tabular model.

This script:
1. Loads the trained tabular model
2. Evaluates it on test data
3. Calculates comprehensive metrics (F1, Accuracy, Precision, Recall, AUC-ROC)
4. Saves predictions as CSV and metrics to JSON
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import project modules
from data.data_loader import load_clinical_tabular
from preprocessing.clinical_preprocessor import TabularPreprocessor
from models.tabular_model import MLPClassifier

def load_model(model_path, device):
    """
    Load the tabular model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        input_size: Model input size
        feature_names: Names of input features
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters
    input_size = checkpoint['input_size']
    hidden_sizes = checkpoint['hidden_sizes']
    feature_names = checkpoint['feature_names']
    dropout_rate = checkpoint.get('dropout_rate', 0.2)
    use_batch_norm = checkpoint.get('use_batch_norm', True)
    
    # Initialize model
    model = MLPClassifier(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=1,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, input_size, feature_names

def calculate_metrics(targets, predictions, probabilities):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted binary labels
        probabilities: Predicted probabilities
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Binary metrics
    metrics['accuracy'] = float(accuracy_score(targets, predictions))
    metrics['precision'] = float(precision_score(targets, predictions, zero_division=0))
    metrics['recall'] = float(recall_score(targets, predictions, zero_division=0))
    metrics['f1_score'] = float(f1_score(targets, predictions, zero_division=0))
    
    # AUC-ROC
    metrics['auc_roc'] = float(roc_auc_score(targets, probabilities))
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity and NPV
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    
    return metrics

def evaluate(
    model_path='models/checkpoints/tabular_model.pt',
    test_size=0.2,
    batch_size=32,
    random_seed=42,
    output_dir='logs'
):
    """
    Evaluate the tabular model.
    
    Args:
        model_path: Path to the model checkpoint
        test_size: Proportion of data to use for test set
        batch_size: Batch size for evaluation
        random_seed: Random seed for reproducibility
        output_dir: Directory to save output files
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model, input_size, feature_names = load_model(model_path, device)
    print("Model loaded successfully")
    
    # Load data
    print("Loading UCI Heart Disease dataset...")
    df = load_clinical_tabular()
    print(f"Dataset loaded, shape: {df.shape}")
    
    # Convert target to binary classification (0 = no disease, 1 = disease)
    df["target"] = (df["target"] > 0).astype(int)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Preprocess features
    print("Preprocessing features...")
    preprocessor = TabularPreprocessor(config={
        'impute_strategy': 'mean',
        'categorical_encoding': 'one-hot',
        'normalization': 'standard'
    })
    
    X_processed = preprocessor.fit_transform(X)
    print(f"Preprocessed data shape: {X_processed.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    print(f"Test set size: {X_test.shape[0]}")
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # Create dataset and dataloader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Evaluate the model
    print("Evaluating model...")
    all_probabilities = []
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_probabilities.append(probs.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate predictions and targets
    all_probabilities = np.concatenate(all_probabilities).flatten()
    all_predictions = np.concatenate(all_predictions).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions, all_probabilities)
    
    # Print metrics
    print("\n=== Evaluation Results ===")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, 'tabular_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Create DataFrame with original features, predictions and ground truth
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    results_df = pd.DataFrame({
        'target': all_targets,
        'prediction': all_predictions,
        'probability': all_probabilities
    })
    
    # Combine with original features
    results_df = pd.concat([X_test_df, results_df], axis=1)
    
    # Save predictions to CSV
    predictions_path = os.path.join(output_dir, 'tabular_predictions.csv')
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate the tabular model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/checkpoints/tabular_model.pt",
        help="Path to the model checkpoint"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for test set"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs",
        help="Directory to save output files"
    )
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        test_size=args.test_size,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 