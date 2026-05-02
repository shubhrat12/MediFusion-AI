#!/usr/bin/env python
"""
Chronic Kidney Disease Prediction Model
This script:
1. Loads and preprocesses the UCI Chronic Kidney Disease Dataset
2. Trains a binary classification model
3. Evaluates the model performance
4. Saves the trained model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib
import os

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_data():
    """
    Load the Chronic Kidney Disease Dataset.
    Handle the specific format of the .arff file.
    """
    # Column names for the dataset
    columns = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc',
        'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
        'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet',
        'pe', 'ane', 'class'
    ]
    
    # Read the ARFF file manually
    data = []
    with open('data/raw/tabular/chronic+kidney+disease/Chronic_Kidney_Disease/Chronic_Kidney_Disease/chronic_kidney_disease.arff', 'r') as f:
        # Skip header until data section
        for line in f:
            if line.strip().lower() == '@data':
                break
        
        # Read data section
        for line in f:
            if line.strip() and not line.startswith('%'):
                values = line.strip().split(',')
                if len(values) == len(columns):
                    data.append(values)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df

def preprocess_data(df):
    """
    Preprocess the CKD dataset:
    1. Handle missing values
    2. Encode categorical variables
    3. Scale numerical features
    """
    # Separate numerical and categorical columns
    numerical_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    categorical_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    
    # Handle missing values in numerical columns
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # Handle missing values and encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        # Fill missing values with mode
        df[col] = df[col].fillna(df[col].mode()[0])
        
        # Encode categorical variables
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    df['class'] = le_target.fit_transform(df['class'].astype(str))
    label_encoders['class'] = le_target
    
    # Split features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame to keep column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler, label_encoders

def train_model(X_train, y_train):
    """
    Train a Random Forest model for CKD prediction
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print various metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Print metrics
    print("\n=== Model Evaluation Metrics ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
    
    # Print detailed classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    return metrics

def main():
    # Load data
    print("Loading Chronic Kidney Disease dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_scaled, y, scaler, label_encoders = preprocess_data(df)
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\nTraining Random Forest model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model, scaler, and encoders
    print("\nSaving model, scaler, and encoders...")
    joblib.dump(model, 'models/ckd_model.joblib')
    joblib.dump(scaler, 'models/ckd_scaler.joblib')
    joblib.dump(label_encoders, 'models/ckd_encoders.joblib')
    print("Model, scaler, and encoders saved successfully!")

if __name__ == "__main__":
    main() 