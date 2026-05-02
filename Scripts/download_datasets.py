import os
import pandas as pd
import requests
from pathlib import Path

def download_file(url, filename):
    """Download a file from a URL to the specified filename"""
    response = requests.get(url)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

def main():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Download Pima Indians Diabetes Dataset
    diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    diabetes_file = data_dir / "diabetes.csv"
    download_file(diabetes_url, diabetes_file)
    
    # Download Chronic Kidney Disease Dataset
    ckd_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/chronic_kidney_disease.data"
    ckd_file = data_dir / "chronic_kidney_disease.data"
    download_file(ckd_url, ckd_file)

if __name__ == "__main__":
    main() 