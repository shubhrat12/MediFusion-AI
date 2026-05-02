import pandas as pd
import numpy as np
from pathlib import Path

def create_ckd_dataset(n_samples=400):
    """Create a sample CKD dataset with realistic values"""
    np.random.seed(42)
    
    # Generate data
    data = {
        'age': np.random.normal(60, 15, n_samples).clip(20, 90),
        'bp': np.random.normal(130, 20, n_samples).clip(80, 180),
        'sg': np.random.choice(['1.005', '1.010', '1.015', '1.020', '1.025'], n_samples),
        'al': np.random.choice(['0', '1', '2', '3', '4', '5'], n_samples),
        'su': np.random.choice(['0', '1', '2', '3', '4', '5'], n_samples),
        'rbc': np.random.choice(['normal', 'abnormal'], n_samples),
        'pc': np.random.choice(['normal', 'abnormal'], n_samples),
        'pcc': np.random.choice(['present', 'notpresent'], n_samples),
        'ba': np.random.choice(['present', 'notpresent'], n_samples),
        'bgr': np.random.normal(140, 40, n_samples).clip(70, 300),
        'bu': np.random.normal(50, 20, n_samples).clip(10, 150),
        'sc': np.random.normal(2, 1, n_samples).clip(0.4, 8),
        'sod': np.random.normal(135, 5, n_samples).clip(120, 150),
        'pot': np.random.normal(4, 0.5, n_samples).clip(2.5, 6.5),
        'hemo': np.random.normal(12, 2, n_samples).clip(6, 18),
        'pcv': np.random.normal(40, 5, n_samples).clip(20, 60),
        'wc': np.random.normal(8000, 2000, n_samples).clip(3000, 15000),
        'rc': np.random.normal(4.5, 0.8, n_samples).clip(2.5, 7),
        'htn': np.random.choice(['yes', 'no'], n_samples),
        'dm': np.random.choice(['yes', 'no'], n_samples),
        'cad': np.random.choice(['yes', 'no'], n_samples),
        'appet': np.random.choice(['good', 'poor'], n_samples),
        'pe': np.random.choice(['yes', 'no'], n_samples),
        'ane': np.random.choice(['yes', 'no'], n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable based on features
    conditions = (
        (df['sc'] > 3) |
        ((df['hemo'] < 9) & (df['rc'] < 3)) |
        ((df['bp'] > 160) & (df['al'] > '2'))
    )
    df['class'] = np.where(conditions, 'ckd', 'notckd')
    
    # Add some missing values
    for col in df.columns:
        mask = np.random.random(n_samples) < 0.05  # 5% missing values
        df.loc[mask, col] = '?'
    
    return df

def main():
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create and save CKD dataset
    df = create_ckd_dataset()
    df.to_csv(data_dir / 'chronic_kidney_disease.data', index=False)
    print("Created CKD dataset with realistic values")

if __name__ == "__main__":
    main() 