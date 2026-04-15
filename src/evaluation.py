"""
Model Evaluation Module
Implements evaluation metrics and model comparison.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Class for evaluating model performance."""
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_pred_proba=None, average='binary'):
        """
        Compute evaluation metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities
        average : str
            Averaging method for multi-class: 'binary', 'micro', 'macro', 'weighted'
            
        Returns:
        --------
        dict : Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # Compute ROC-AUC if probabilities are provided and it's binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name="Model"):
        """
        Evaluate a single model.
        
        Parameters:
        -----------
        model : object
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict : Evaluation results
        """
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        metrics = ModelEvaluator.compute_metrics(y_test, y_pred, y_pred_proba)
        
        return {
            'model_name': model_name,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
    
    @staticmethod
    def evaluate_all_models(models_dict, X_test, y_test):
        """
        Evaluate all models.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models
        X_test : array-like
            Test features
        y_test : array-like
            Test target
            
        Returns:
        --------
        dict : Evaluation results for all models
        """
        results = {}
        
        for name, model in models_dict.items():
            print(f"📊 Evaluating {name}...")
            results[name] = ModelEvaluator.evaluate_model(model, X_test, y_test, name)
        
        return results
    
    @staticmethod
    def cross_validate(model, X_train, y_train, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        model : object
            Model to validate
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        cv : int
            Number of folds
            
        Returns:
        --------
        dict : Cross-validation results
        """
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }


class ResultsComparator:
    """Class for comparing model results."""
    
    @staticmethod
    def compare_models(results_dict):
        """
        Compare model performance.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary of evaluation results
            
        Returns:
        --------
        pd.DataFrame : Comparison dataframe
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1', 0),
                'ROC-AUC': metrics.get('roc_auc', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')
        
        return comparison_df
    
    @staticmethod
    def rank_models(results_dict, metric='f1'):
        """
        Rank models by a specific metric.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary of evaluation results
        metric : str
            Metric to rank by
            
        Returns:
        --------
        pd.Series : Ranked models
        """
        ranking = []
        
        for model_name, results in results_dict.items():
            metric_value = results['metrics'].get(metric, 0)
            ranking.append({'Model': model_name, metric: metric_value})
        
        ranking_df = pd.DataFrame(ranking)
        ranking_df = ranking_df.sort_values(metric, ascending=False).reset_index(drop=True)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df
    
    @staticmethod
    def get_best_model(results_dict, metric='f1'):
        """
        Get the best model by a specific metric.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary of evaluation results
        metric : str
            Metric to evaluate by
            
        Returns:
        --------
        str : Name of the best model
        """
        best_model = None
        best_score = -1
        
        for model_name, results in results_dict.items():
            score = results['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score


class ReportGenerator:
    """Class for generating evaluation reports."""
    
    @staticmethod
    def generate_text_report(results_dict, save_path='results/evaluation_reports.txt'):
        """
        Generate text report of all evaluations.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary of evaluation results
        save_path : str
            Path to save the report
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DISEASE PREDICTION MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name, results in results_dict.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"MODEL: {model_name}\n")
                f.write(f"{'='*80}\n\n")
                
                # Metrics
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 40 + "\n")
                for metric, value in results['metrics'].items():
                    f.write(f"{metric.upper():20s}: {value:.4f}\n")
                
                # Classification Report
                f.write("\n\nCLASSIFICATION REPORT:\n")
                f.write("-" * 40 + "\n")
                f.write(results['classification_report'])
                
                # Confusion Matrix
                f.write("\n\nCONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                f.write(str(results['confusion_matrix']))
                f.write("\n\n")
        
        print(f"✅ Report saved at {save_path}")
    
    @staticmethod
    def print_summary(results_dict):
        """
        Print summary of all models.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary of evaluation results
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80 + "\n")
        
        comparison_df = ResultsComparator.compare_models(results_dict)
        print(comparison_df.to_string())
        
        print("\n" + "-"*80)
        best_model, best_score = ResultsComparator.get_best_model(results_dict, 'f1')
        print(f"🏆 Best Model: {best_model} (F1-Score: {best_score:.4f})")
        print("="*80 + "\n")
