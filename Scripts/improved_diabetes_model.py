#!/usr/bin/env python
"""
Memory-Efficient Diabetes Prediction Model
Optimized version that reduces memory usage while maintaining good performance
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
import gc

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

def optimize_dtypes(df):
    """Optimize data types to reduce memory usage"""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def create_basic_features(df):
    """Create essential features while minimizing memory usage"""
    # BMI Categories (simplified)
    df['BMI_Category'] = pd.cut(df['BMI'], 
        bins=[0, 18.5, 25, 30, float('inf')],
        labels=[0, 1, 2, 3]).astype('int8')
    
    # Age Categories (simplified)
    df['Age_Category'] = pd.cut(df['Age'], 
        bins=[0, 40, 60, float('inf')],
        labels=[0, 1, 2]).astype('int8')
    
    # Basic Health Risk Score
    df['HealthRiskScore'] = (
        df['HighBP'].astype('float32') * 2 + 
        df['HighChol'].astype('float32') * 2 + 
        df['HeartDiseaseorAttack'].astype('float32') * 3 + 
        df['GenHlth'].astype('float32')
    ).astype('float32')
    
    # Basic Lifestyle Score
    df['LifestyleScore'] = (
        df['PhysActivity'].astype('float32') * 2 -
        df['Smoker'].astype('float32') * 2 +
        (df['Fruits'] + df['Veggies']).astype('float32')
    ).astype('float32')
    
    # Essential interactions
    df['BMI_Age'] = (df['BMI'] * df['Age']).astype('float32')
    df['BP_Cholesterol'] = (df['HighBP'] * df['HighChol']).astype('int8')
    
    # Clean up memory
    gc.collect()
    
    return df

def preprocess_data(df):
    """Memory-efficient preprocessing pipeline"""
    # Drop unnecessary columns
    df = df.drop(['AgeCat', 'BMICat'], axis=1, errors='ignore')
    
    # Optimize data types
    df = optimize_dtypes(df)
    
    # Create basic features
    df = create_basic_features(df)
    
    # Separate features and target
    y = df['Diabetes_binary']
    X = df.drop('Diabetes_binary', axis=1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to float32 to save memory
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, dtype='float32')
    
    # Clean up memory
    del df
    gc.collect()
    
    return X_scaled, y, scaler

def train_model(X_train, y_train):
    """Train an optimized XGBoost model with memory-efficient settings"""
    # Create a smaller validation set
    X_train_main, X_valid, y_train_main, y_valid = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Memory-efficient parameter space
    param_dist = {
        'n_estimators': randint(100, 500),  # Reduced number of trees
        'max_depth': randint(3, 7),  # Shallower trees
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 5),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(1, 2)
    }
    
    # Create base model with memory-efficient settings
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',  # Memory efficient algorithm
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        eval_metric=['auc', 'error'],
        max_bin=256,  # Reduce number of bins
        grow_policy='lossguide'  # More memory-efficient tree growing
    )
    
    # RandomizedSearchCV with fewer iterations
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,  # Reduced iterations
        scoring=['accuracy', 'f1', 'roc_auc'],
        refit='f1',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit RandomizedSearchCV
    search.fit(
        X_train_main, 
        y_train_main,
        eval_set=[(X_valid, y_valid)],
        verbose=True
    )
    
    # Clean up memory
    gc.collect()
    
    return search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the model with basic metrics"""
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'performance_metrics': {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
        },
        'feature_importance': {
            name: float(importance) 
            for name, importance in zip(feature_names, model.feature_importances_)
        }
    }
    
    # Save metrics
    with open('models/diabetes_model_evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print results
    print("\nModel Evaluation Results:")
    for metric, value in metrics['performance_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Print top 10 features only
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
    """Main function with memory-efficient processing"""
    print("Loading dataset...")
    # Read data in chunks to reduce memory usage
    df = pd.read_csv(
        'data/raw/tabular/diabetes/diabetes_binary_5050split_health_indicators_BRFSS2015.csv',
        dtype={
            'Diabetes_binary': 'int8',
            'HighBP': 'int8',
            'HighChol': 'int8',
            'BMI': 'float32',
            'Smoker': 'int8',
            'Stroke': 'int8',
            'HeartDiseaseorAttack': 'int8',
            'PhysActivity': 'int8',
            'Fruits': 'int8',
            'Veggies': 'int8',
            'HvyAlcoholConsump': 'int8',
            'AnyHealthcare': 'int8',
            'NoDocbcCost': 'int8',
            'GenHlth': 'int8',
            'MentHlth': 'int8',
            'PhysHlth': 'int8',
            'DiffWalk': 'int8',
            'Sex': 'int8',
            'Age': 'int8',
            'Education': 'int8',
            'Income': 'int8'
        }
    )
    
    print(f"Dataset shape: {df.shape}")
    print("\nPreprocessing data...")
    X, y, scaler = preprocess_data(df)
    
    # Free memory
    del df
    gc.collect()
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test, X.columns)
    
    print("\nSaving model...")
    joblib.dump(model, 'models/diabetes_model.joblib')
    joblib.dump(scaler, 'models/diabetes_scaler.joblib')
    
    print("Done!")

if __name__ == '__main__':
    main() 