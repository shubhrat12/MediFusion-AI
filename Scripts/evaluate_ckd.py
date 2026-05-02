#!/usr/bin/env python
"""
Chronic Kidney Disease Model Evaluation Script
This script:
1. Loads the trained CKD model and its weights
2. Evaluates performance on test data
3. Saves detailed evaluation results in a JSON file
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import os

def evaluate_ckd_model():
    """Evaluate the chronic kidney disease prediction model on test data"""
    try:
        # Load model and preprocessing objects
        model = joblib.load('models/ckd_model.joblib')
        scaler = joblib.load('models/ckd_scaler.joblib')
        label_encoders = joblib.load('models/ckd_encoders.joblib')
        
        # Load data
        from ckd_model import load_data
        df = load_data()
        
        # Split data first (using same random_state as training)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
        
        # Preprocess test data
        from ckd_model import preprocess_data
        X_test_scaled, y_test, _, _ = preprocess_data(test_df)
        
        # Get predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate metrics
        metrics = {
            'model_name': 'Chronic Kidney Disease Prediction Model (Random Forest)',
            'model_parameters': model.get_params(),
            'performance_metrics': {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
            },
            'confusion_matrix': {
                'true_negative': int(conf_matrix[0][0]),
                'false_positive': int(conf_matrix[0][1]),
                'false_negative': int(conf_matrix[1][0]),
                'true_positive': int(conf_matrix[1][1])
            },
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': {
                name: float(importance) 
                for name, importance in zip(X_test_scaled.columns, model.feature_importances_)
            }
        }
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_test_scaled, y_test, cv=5)
        metrics['cross_validation'] = {
            'scores': [float(score) for score in cv_scores],
            'mean_score': float(cv_scores.mean()),
            'std_score': float(cv_scores.std())
        }
        
        # Add dataset statistics
        metrics['dataset_info'] = {
            'total_samples': int(len(df)),
            'test_samples': int(len(test_df)),
            'feature_count': int(X_test_scaled.shape[1]),
            'class_distribution': {
                'train': {str(k): int(v) for k, v in train_df['class'].value_counts().items()},
                'test': {str(k): int(v) for k, v in test_df['class'].value_counts().items()}
            }
        }
        
        # Save metrics to JSON
        os.makedirs('models', exist_ok=True)
        with open('models/ckd_model_detailed_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nCKD Model Evaluation Results:")
        print(f"Accuracy: {metrics['performance_metrics']['accuracy']:.4f}")
        print(f"Precision: {metrics['performance_metrics']['precision']:.4f}")
        print(f"Recall: {metrics['performance_metrics']['recall']:.4f}")
        print(f"F1 Score: {metrics['performance_metrics']['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['performance_metrics']['auc_roc']:.4f}")
        print(f"\nCross-validation mean accuracy: {metrics['cross_validation']['mean_score']:.4f}")
        print(f"Cross-validation std: {metrics['cross_validation']['std_score']:.4f}")
        
        print("\nTop 5 Most Important Features:")
        sorted_features = sorted(
            metrics['feature_importance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")
        
        print("\nDetailed evaluation results saved to: models/ckd_model_detailed_evaluation.json")
        return metrics
        
    except Exception as e:
        print(f"Error evaluating CKD model: {str(e)}")
        return None

if __name__ == "__main__":
    evaluate_ckd_model() 