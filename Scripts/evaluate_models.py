#!/usr/bin/env python
"""
Model Evaluation Script
This script:
1. Loads both trained models (Diabetes and CKD)
2. Evaluates their performance on proper test data
3. Saves evaluation results in separate JSON files
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import os

def evaluate_diabetes_model():
    """Evaluate the diabetes prediction model on a proper test set"""
    try:
        # Load model and preprocessing objects
        model = joblib.load('models/diabetes_model_xgb.joblib')
        scaler = joblib.load('models/diabetes_scaler.joblib')
        pt = joblib.load('models/diabetes_power_transformer.joblib')
        
        # Load data
        columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        df = pd.read_csv('data/diabetes.csv', names=columns, skiprows=1)
        
        # Split data first (using same random_state as training)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Outcome'])
        
        # Preprocess test data using the same steps as in training
        from diabetes_model import preprocess_data
        X_test_scaled, y_test, _, _ = preprocess_data(test_df)
        
        # Get predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': 'Diabetes Prediction Model (XGBoost)',
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': {
                name: float(importance) 
                for name, importance in zip(X_test_scaled.columns, model.feature_importances_)
            }
        }
        
        # Save metrics to JSON
        with open('models/diabetes_model_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("Diabetes model evaluation completed and saved!")
        return metrics
        
    except Exception as e:
        print(f"Error evaluating diabetes model: {str(e)}")
        return None

def evaluate_ckd_model():
    """Evaluate the chronic kidney disease prediction model on a proper test set"""
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
        
        # Preprocess test data using the same steps as in training
        from ckd_model import preprocess_data
        X_test_scaled, y_test, _, _ = preprocess_data(test_df)
        
        # Get predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': 'Chronic Kidney Disease Prediction Model (Random Forest)',
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'auc_roc': float(roc_auc_score(y_test, y_pred_proba)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': {
                name: float(importance) 
                for name, importance in zip(X_test_scaled.columns, model.feature_importances_)
            }
        }
        
        # Save metrics to JSON
        with open('models/ckd_model_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("CKD model evaluation completed and saved!")
        return metrics
        
    except Exception as e:
        print(f"Error evaluating CKD model: {str(e)}")
        return None

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("Starting model evaluation...")
    
    # Evaluate Diabetes model
    print("\nEvaluating Diabetes Prediction Model...")
    diabetes_metrics = evaluate_diabetes_model()
    if diabetes_metrics:
        print("\nDiabetes Model Metrics:")
        print(f"Accuracy: {diabetes_metrics['accuracy']:.4f}")
        print(f"F1 Score: {diabetes_metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {diabetes_metrics['auc_roc']:.4f}")
    
    # Evaluate CKD model
    print("\nEvaluating Chronic Kidney Disease Model...")
    ckd_metrics = evaluate_ckd_model()
    if ckd_metrics:
        print("\nCKD Model Metrics:")
        print(f"Accuracy: {ckd_metrics['accuracy']:.4f}")
        print(f"F1 Score: {ckd_metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {ckd_metrics['auc_roc']:.4f}")
    
    print("\nEvaluation complete! Results have been saved to:")
    print("- models/diabetes_model_evaluation.json")
    print("- models/ckd_model_evaluation.json")

if __name__ == "__main__":
    main() 