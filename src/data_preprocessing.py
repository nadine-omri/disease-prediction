"""
Data Preprocessing Module
Handles data loading, cleaning, normalization, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Class for preprocessing medical data."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        strategy : str
            Strategy for handling missing values: 'mean', 'median', 'forward_fill', 'drop'
            
        Returns:
        --------
        pd.DataFrame : Dataset with missing values handled
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy)
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        elif strategy == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        return df_clean
    
    def encode_categorical(self, df, categorical_cols):
        """
        Encode categorical variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        categorical_cols : list
            List of categorical column names
            
        Returns:
        --------
        pd.DataFrame : Dataset with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def normalize_features(self, df_train, df_test=None, method='standard'):
        """
        Normalize/Scale features.
        
        Parameters:
        -----------
        df_train : pd.DataFrame
            Training dataset
        df_test : pd.DataFrame, optional
            Test dataset
        method : str
            Scaling method: 'standard' or 'minmax'
            
        Returns:
        --------
        tuple : Scaled training data and optionally scaled test data
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        df_train_scaled = scaler.fit_transform(df_train)
        df_train_scaled = pd.DataFrame(df_train_scaled, columns=df_train.columns)
        
        if df_test is not None:
            df_test_scaled = scaler.transform(df_test)
            df_test_scaled = pd.DataFrame(df_test_scaled, columns=df_test.columns)
            return df_train_scaled, df_test_scaled
        
        return df_train_scaled
    
    def split_data(self, X, y, test_size=0.2, stratify=True):
        """
        Split data into train and test sets.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        test_size : float
            Proportion of test set
        stratify : bool
            Whether to stratify the split
            
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test
        """
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test
    
    def full_preprocessing(self, df, target_col, test_size=0.2, categorical_cols=None):
        """
        Perform complete preprocessing pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        target_col : str
            Name of target column
        test_size : float
            Proportion of test set
        categorical_cols : list
            List of categorical column names
            
        Returns:
        --------
        tuple : X_train, X_test, y_train, y_test (preprocessed)
        """
        # Handle missing values
        df_clean = self.handle_missing_values(df, strategy='mean')
        
        # Separate features and target
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Encode categorical variables
        if categorical_cols:
            X = self.encode_categorical(X, categorical_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        # Normalize features
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test, method='standard')
        
        return X_train_scaled, X_test_scaled, y_train, y_test


# Convenience functions
def load_and_preprocess_heart_disease(filepath='data/heart_disease.csv', test_size=0.2):
    """Load and preprocess heart disease dataset."""
    df = pd.read_csv(filepath)
    df = df.replace('?', np.nan)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.full_preprocessing(
        df, target_col='num', test_size=test_size
    )
    
    return X_train, X_test, y_train, y_test

def load_and_preprocess_diabetes(filepath='data/diabetes.csv', test_size=0.2):
    """Load and preprocess diabetes dataset."""
    df = pd.read_csv(filepath)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.full_preprocessing(
        df, target_col='Outcome', test_size=test_size
    )
    
    return X_train, X_test, y_train, y_test

def load_and_preprocess_breast_cancer(filepath='data/breast_cancer.csv', test_size=0.2):
    """Load and preprocess breast cancer dataset."""
    df = pd.read_csv(filepath)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.full_preprocessing(
        df, target_col='target', test_size=test_size, categorical_cols=['diagnosis']
    )
    
    return X_train, X_test, y_train, y_test