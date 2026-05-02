#!/usr/bin/env python
"""
New Diabetes Prediction Model
This script:
1. Loads and preprocesses the new BRFSS2015 Diabetes Dataset
2. Trains an optimized model using XGBoost
3. Evaluates the model performance
4. Saves the model and evaluation results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import xgboost as xgb
from scipy.stats import uniform, randint
import joblib
import json
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

def load_data():
    """Load the BRFSS2015 Diabetes Dataset"""
    df = pd.read_csv('data/raw/tabular/diabetes/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    return df

def preprocess_data(df):
    """
    Preprocess the diabetes dataset:
    1. Handle any missing values
    2. Scale numerical features
    3. Encode categorical features if any
    """
    # Separate features and target
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler

def train_model(X_train, y_train):
    """Train an optimized XGBoost model with hyperparameter tuning"""
    # Define the parameter space for random search
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 7),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(1, 4),
    }
    
    # Create base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    # Set up RandomizedSearchCV
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit RandomizedSearchCV
    search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")
    
    return search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the model and save detailed results"""
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate metrics
    metrics = {
        'model_name': 'New Diabetes Prediction Model (XGBoost)',
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
            for name, importance in zip(feature_names, model.feature_importances_)
        }
    }
    
    # Save metrics to JSON
    with open('models/new_diabetes_model_evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {metrics['performance_metrics']['accuracy']:.4f}")
    print(f"Precision: {metrics['performance_metrics']['precision']:.4f}")
    print(f"Recall: {metrics['performance_metrics']['recall']:.4f}")
    print(f"F1 Score: {metrics['performance_metrics']['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['performance_metrics']['auc_roc']:.4f}")
    
    print("\nTop 10 Most Important Features:")
    sorted_features = sorted(
        metrics['feature_importance'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")
    
    return metrics

def main():
    # Load data
    print("Loading BRFSS2015 Diabetes dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Print dataset statistics
    print("\nClass distribution:")
    print(df['Diabetes_binary'].value_counts())
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_scaled, y, scaler = preprocess_data(df)
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\nTraining XGBoost model with hyperparameter tuning...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test, X_scaled.columns)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'models/new_diabetes_model.joblib')
    joblib.dump(scaler, 'models/new_diabetes_scaler.joblib')
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    main() 