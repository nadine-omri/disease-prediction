"""
Utility functions for data loading and preprocessing.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


class DataLoader:
    """Class for loading and validating medical datasets."""
    
    @staticmethod
    def load_heart_disease(filepath='data/heart_disease.csv'):
        """
        Load heart disease dataset.
        
        Parameters:
        -----------
        filepath : str
            Path to the heart disease CSV file
            
        Returns:
        --------
        df : pd.DataFrame
            Loaded dataset
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Remplacer les valeurs manquantes "?" par NaN
        df = df.replace('?', np.nan)
        
        return df
    
    @staticmethod
    def load_diabetes(filepath='data/diabetes.csv'):
        """
        Load diabetes dataset.
        
        Parameters:
        -----------
        filepath : str
            Path to the diabetes CSV file
            
        Returns:
        --------
        df : pd.DataFrame
            Loaded dataset
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        df = pd.read_csv(filepath)
        df = df.replace('?', np.nan)
        
        return df
    
    @staticmethod
    def load_breast_cancer(filepath='data/breast_cancer.csv'):
        """
        Load breast cancer dataset.
        
        Parameters:
        -----------
        filepath : str
            Path to the breast cancer CSV file
            
        Returns:
        --------
        df : pd.DataFrame
            Loaded dataset
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        df = pd.read_csv(filepath)
        df = df.replace('?', np.nan)
        
        return df
    
    @staticmethod
    def load_any_dataset(filepath):
        """
        Load any CSV dataset.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        df : pd.DataFrame
            Loaded dataset
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        df = pd.read_csv(filepath)
        df = df.replace('?', np.nan)
        
        return df


class DataValidator:
    """Class for validating dataset quality."""
    
    @staticmethod
    def check_missing_values(df):
        """
        Check for missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to check
            
        Returns:
        --------
        dict : Missing values information
        """
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        return {
            'missing_counts': missing,
            'missing_percent': missing_percent,
            'total_missing': missing.sum()
        }
    
    @staticmethod
    def check_data_types(df):
        """
        Check data types of the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to check
            
        Returns:
        --------
        dict : Data type information
        """
        return {
            'data_types': df.dtypes,
            'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
    
    @staticmethod
    def get_dataset_info(df):
        """
        Get comprehensive information about the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
            
        Returns:
        --------
        dict : Comprehensive dataset information
        """
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': DataValidator.check_missing_values(df),
            'data_types': DataValidator.check_data_types(df),
            'duplicates': df.duplicated().sum(),
            'numeric_stats': df.describe().to_dict()
        }


class ModelPersistence:
    """Class for saving and loading trained models."""
    
    @staticmethod
    def save_model(model, filepath):
        """
        Save a trained model to disk.
        
        Parameters:
        -----------
        model : object
            Trained model
        filepath : str
            Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model saved at {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the model file
            
        Returns:
        --------
        model : object
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found at {filepath}")
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {filepath}")
        return model
    
    @staticmethod
    def save_models_dict(models_dict, filepath):
        """
        Save multiple models in a dictionary.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of models
        filepath : str
            Path to save the models
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(models_dict, f)
        print(f"✅ Models saved at {filepath}")
    
    @staticmethod
    def load_models_dict(filepath):
        """
        Load multiple models from a dictionary.
        
        Parameters:
        -----------
        filepath : str
            Path to the models file
            
        Returns:
        --------
        dict : Dictionary of loaded models
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Models file not found at {filepath}")
        
        with open(filepath, 'rb') as f:
            models_dict = pickle.load(f)
        print(f"✅ Models loaded from {filepath}")
        return models_dict


class FeatureEngineer:
    """Class for feature engineering and selection."""
    
    @staticmethod
    def create_directories():
        """Create necessary project directories."""
        directories = ['data', 'models', 'results', 'results/visualizations']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory '{directory}' ready")
    
    @staticmethod
    def remove_outliers(df, columns, threshold=3):
        """
        Remove outliers using z-score method.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        columns : list
            Columns to check for outliers
        threshold : float
            Z-score threshold
            
        Returns:
        --------
        pd.DataFrame : Dataset without outliers
        """
        from scipy import stats
        
        df_clean = df.copy()
        for col in columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[(z_scores < threshold).all(axis=1)]
        
        return df_clean
    
    @staticmethod
    def get_feature_correlations(df, target_col):
        """
        Get feature correlations with target.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        target_col : str
            Target column name
            
        Returns:
        --------
        pd.Series : Correlations sorted by absolute value
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        correlations = df.corr()[target_col].sort_values(ascending=False)
        return correlations


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer les répertoires
    FeatureEngineer.create_directories()
    
    # Charger les données
    loader = DataLoader()
    try:
        df_heart = loader.load_heart_disease()
        print("✅ Heart disease dataset loaded")
        print(f"Shape: {df_heart.shape}")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")