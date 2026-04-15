"""
Script to download all required datasets for disease prediction.
"""

import os
import pandas as pd
import numpy as np


def create_data_directory():
    """Create data directory if it doesn't exist."""
    os.makedirs('data', exist_ok=True)
    print("✅ Data directory ready")


def download_heart_disease():
    """Download Heart Disease dataset."""
    print("\n1️⃣ Downloading Heart Disease dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    
    try:
        df = pd.read_csv(url, names=columns, header=None)
        df.to_csv('data/heart_disease.csv', index=False)
        print(f"   ✅ Heart Disease dataset saved ({df.shape[0]} records)")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_diabetes():
    """Download Diabetes dataset."""
    print("\n2️⃣ Downloading Diabetes dataset...")
    
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    try:
        df = pd.read_csv(url, names=columns, header=0)
        df.to_csv('data/diabetes.csv', index=False)
        print(f"   ✅ Diabetes dataset saved ({df.shape[0]} records)")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_breast_cancer():
    """Download Breast Cancer dataset."""
    print("\n3️⃣ Downloading Breast Cancer dataset...")
    
    try:
        from sklearn.datasets import load_breast_cancer
        
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df.to_csv('data/breast_cancer.csv', index=False)
        print(f"   ✅ Breast Cancer dataset saved ({df.shape[0]} records)")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def verify_datasets():
    """Verify downloaded datasets."""
    print("\n🔍 Verifying datasets...\n")
    
    datasets = {
        'data/heart_disease.csv': 'Heart Disease',
        'data/diabetes.csv': 'Diabetes',
        'data/breast_cancer.csv': 'Breast Cancer'
    }
    
    for filepath, name in datasets.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"✅ {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"⚠️  {name}: File not found")


def main():
    """Main function."""
    print("="*80)
    print("📥 DISEASE PREDICTION DATASETS DOWNLOADER")
    print("="*80)
    
    create_data_directory()
    
    download_heart_disease()
    download_diabetes()
    download_breast_cancer()
    
    verify_datasets()
    
    print("\n✨ Download complete!")
    print("="*80)


if __name__ == "__main__":
    main()