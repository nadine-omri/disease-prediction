"""
Machine Learning Models Module
Implements SVM, Logistic Regression, Random Forest, and XGBoost classifiers.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class ModelFactory:
    """Factory class for creating and configuring ML models."""
    
    @staticmethod
    def create_svm(kernel='rbf', C=1.0, gamma='scale', probability=True):
        """
        Create SVM classifier.
        
        Parameters:
        -----------
        kernel : str
            Kernel type: 'linear', 'rbf', 'poly'
        C : float
            Regularization parameter
        gamma : str or float
            Kernel coefficient
        probability : bool
            Enable probability estimates
            
        Returns:
        --------
        SVC : Configured SVM model
        """
        return SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=42,
            verbose=0
        )
    
    @staticmethod
    def create_logistic_regression(C=1.0, max_iter=1000, solver='lbfgs'):
        """
        Create Logistic Regression classifier.
        
        Parameters:
        -----------
        C : float
            Inverse of regularization strength
        max_iter : int
            Maximum number of iterations
        solver : str
            Algorithm to use: 'lbfgs', 'liblinear', 'saga'
            
        Returns:
        --------
        LogisticRegression : Configured Logistic Regression model
        """
        return LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    @staticmethod
    def create_random_forest(n_estimators=100, max_depth=None, min_samples_split=2, 
                            min_samples_leaf=1):
        """
        Create Random Forest classifier.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int or None
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split
        min_samples_leaf : int
            Minimum samples in leaf
            
        Returns:
        --------
        RandomForestClassifier : Configured Random Forest model
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    @staticmethod
    def create_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1, 
                       subsample=0.8, colsample_bytree=0.8):
        """
        Create XGBoost classifier.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate (eta)
        subsample : float
            Subsample ratio
        colsample_bytree : float
            Column sample ratio
            
        Returns:
        --------
        XGBClassifier : Configured XGBoost model
        """
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss'
        )


class HyperparameterTuner:
    """Class for hyperparameter tuning.""" 
    
    @staticmethod
    def tune_svm(X_train, y_train, cv=5):
        """
        Tune SVM hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        SVC : Best SVM model
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best SVM parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    @staticmethod
    def tune_random_forest(X_train, y_train, cv=5):
        """
        Tune Random Forest hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        RandomForestClassifier : Best Random Forest model
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best Random Forest parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    @staticmethod
    def tune_xgboost(X_train, y_train, cv=5):
        """
        Tune XGBoost hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        XGBClassifier : Best XGBoost model
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.3]
        }
        
        xgb = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        grid_search = GridSearchCV(xgb, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best XGBoost parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_


class ModelTrainer:
    """Class for training models."""
    
    def __init__(self):
        self.models = {}
        self.factory = ModelFactory()
    
    def train_all_models(self, X_train, y_train, tune=False):
        """
        Train all models.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        dict : Dictionary of trained models
        """
        print("🤖 Training machine learning models...")
        
        # SVM
        print("\n1️⃣ Training SVM...")
        if tune:
            self.models['SVM'] = HyperparameterTuner.tune_svm(X_train, y_train)
        else:
            self.models['SVM'] = self.factory.create_svm()
            self.models['SVM'].fit(X_train, y_train)
        print("   ✅ SVM trained")
        
        # Logistic Regression
        print("\n2️⃣ Training Logistic Regression...")
        self.models['Logistic Regression'] = self.factory.create_logistic_regression()
        self.models['Logistic Regression'].fit(X_train, y_train)
        print("   ✅ Logistic Regression trained")
        
        # Random Forest
        print("\n3️⃣ Training Random Forest...")
        if tune:
            self.models['Random Forest'] = HyperparameterTuner.tune_random_forest(X_train, y_train)
        else:
            self.models['Random Forest'] = self.factory.create_random_forest()
            self.models['Random Forest'].fit(X_train, y_train)
        print("   ✅ Random Forest trained")
        
        # XGBoost
        print("\n4️⃣ Training XGBoost...")
        if tune:
            self.models['XGBoost'] = HyperparameterTuner.tune_xgboost(X_train, y_train)
        else:
            self.models['XGBoost'] = self.factory.create_xgboost()
            self.models['XGBoost'].fit(X_train, y_train)
        print("   ✅ XGBoost trained")
        
        print("\n✨ All models trained successfully!")
        return self.models
    
    def predict_all(self, X_test):
        """
        Make predictions with all models.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
            
        Returns:
        --------
        dict : Predictions from all models
        """
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
        
        return predictions
    
    def predict_proba_all(self, X_test):
        """
        Get probability predictions from all models.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
            
        Returns:
        --------
        dict : Probability predictions from all models
        """
        proba_predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba_predictions[name] = model.predict_proba(X_test)
            else:
                proba_predictions[name] = model.decision_function(X_test)
        
        return proba_predictions