#!/usr/bin/env python
"""
Diabetes Prediction Model using Pima Indians Diabetes Dataset
This script:
1. Loads and preprocesses the Pima Indians Diabetes Dataset
2. Performs feature selection and engineering
3. Trains an optimized XGBoost model with hyperparameter tuning
4. Evaluates the model performance
5. Saves the trained model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from scipy.stats import uniform, randint
import joblib
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

def load_data():
    """Load the Pima Indians Diabetes Dataset."""
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df = pd.read_csv('data/diabetes.csv', names=columns, skiprows=1)
    return df

def create_features(df):
    """Create new features from existing ones"""
    df_new = df.copy()
    
    # Create polynomial features for important numeric columns
    df_new['Glucose_squared'] = df_new['Glucose'] ** 2
    df_new['BMI_squared'] = df_new['BMI'] ** 2
    
    # Create interaction features
    df_new['Glucose_BMI'] = df_new['Glucose'] * df_new['BMI']
    df_new['Age_BMI'] = df_new['Age'] * df_new['BMI']
    df_new['Glucose_Age'] = df_new['Glucose'] * df_new['Age']
    
    # Create ratio features
    df_new['BMI_BloodPressure'] = df_new['BMI'] / df_new['BloodPressure'].replace(0, np.nan)
    df_new['Glucose_BloodPressure'] = df_new['Glucose'] / df_new['BloodPressure'].replace(0, np.nan)
    df_new['Glucose_Insulin'] = df_new['Glucose'] / df_new['Insulin'].replace(0, np.nan)
    
    # Create age-related features
    df_new['Age_Risk'] = np.where(df_new['Age'] > 50, 1, 0)
    
    # Create BMI categories
    df_new['BMI_Category'] = pd.cut(
        df_new['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    ).astype(str)
    
    # One-hot encode categorical features
    df_new = pd.get_dummies(df_new, columns=['BMI_Category'])
    
    # Handle missing values in new features
    df_new = df_new.fillna(df_new.median())
    
    return df_new

def preprocess_data(df):
    """
    Preprocess the diabetes dataset:
    1. Handle missing values
    2. Create new features
    3. Scale features
    4. Select best features
    """
    # Handle missing values (0s) in specific columns
    zero_value_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_value_cols:
        df.loc[df[col] == 0, col] = np.nan
        df[col] = df[col].fillna(df[col].median())
    
    # Create new features
    df = create_features(df)
    
    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Apply Yeo-Johnson transformation to handle skewness
    pt = PowerTransformer(method='yeo-johnson')
    X_scaled = pt.fit_transform(X_scaled)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler, pt

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
        'scale_pos_weight': [len(y_train[y_train==0]) / len(y_train[y_train==1])],
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
        scoring='roc_auc',
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
    
    # Get feature importances
    feature_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': search.best_estimator_.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_imp.head(15))
    
    return search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print various metrics"""
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
    
    # Perform cross-validation
    print("\n=== 5-Fold Cross-validation Scores ===")
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return metrics

def main():
    # Load data
    print("Loading Pima Indians Diabetes dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_scaled, y, scaler, pt = preprocess_data(df)
    
    # Split data with stratification
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\nTraining XGBoost model with hyperparameter tuning...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model and preprocessing objects
    print("\nSaving model and preprocessing objects...")
    joblib.dump(model, 'models/diabetes_model_xgb.joblib')
    joblib.dump(scaler, 'models/diabetes_scaler.joblib')
    joblib.dump(pt, 'models/diabetes_power_transformer.joblib')
    print("Model and preprocessing objects saved successfully!")

if __name__ == "__main__":
    main() 