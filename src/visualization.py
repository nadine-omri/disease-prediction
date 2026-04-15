"""
Visualization Module
Creates plots and visualizations for model analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """Class for creating visualizations."""
    
    @staticmethod
    def plot_confusion_matrix(cm, model_name, save_path='results/visualizations/'):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        cm : array-like
            Confusion matrix
        model_name : str
            Name of the model
        save_path : str
            Path to save the figure
        """
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = f"{save_path}confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, model_name, save_path='results/visualizations/'):
        """
        Plot ROC-AUC curve.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Name of the model
        save_path : str
            Path to save the figure
        """
        os.makedirs(save_path, exist_ok=True)
        
        if len(np.unique(y_true)) == 2:
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC-AUC Curve - {model_name}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            filename = f"{save_path}roc_curve_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {filename}")
    
    @staticmethod
    def plot_model_comparison(comparison_df, save_path='results/visualizations/'):
        """
        Plot model comparison.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            Comparison dataframe
        save_path : str
            Path to save the figure
        """
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_title(metric, fontsize=12, fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = f"{save_path}model_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")
    
    @staticmethod
    def plot_feature_importance(model, feature_names, model_name, save_path='results/visualizations/'):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        model : object
            Trained model
        feature_names : list
            List of feature names
        model_name : str
            Name of the model
        save_path : str
            Path to save the figure
        """
        os.makedirs(save_path, exist_ok=True)
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
            plt.title(f'Top 20 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            
            filename = f"{save_path}feature_importance_{model_name.replace(' ', '_').lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {filename}")
    
    @staticmethod
    def plot_correlation_heatmap(X, save_path='results/visualizations/'):
        """
        Plot correlation heatmap.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
        save_path : str
            Path to save the figure
        """
        os.makedirs(save_path, exist_ok=True)
        
        corr_matrix = X.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"{save_path}correlation_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")
    
    @staticmethod
    def plot_distribution(df, column, save_path='results/visualizations/'):
        """
        Plot feature distribution.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data dataframe
        column : str
            Column name
        save_path : str
            Path to save the figure
        """
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=column, kde=True, color='steelblue', edgecolor='black')
        plt.title(f'Distribution of {column}', fontsize=14, fontweight='bold')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        filename = f"{save_path}distribution_{column.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")


class DashboardGenerator:
    """Class for generating comprehensive dashboards.""" 
    
    @staticmethod
    def generate_full_report(results_dict, X_test, comparison_df, save_path='results/visualizations/'):
        """
        Generate full visualization report.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary of evaluation results
        X_test : pd.DataFrame
            Test features
        comparison_df : pd.DataFrame
            Comparison dataframe
        save_path : str
            Path to save visualizations
        """
        print("\n📊 Generating visualizations...")
        
        # Model comparison
        Visualizer.plot_model_comparison(comparison_df, save_path)
        
        # Correlation heatmap
        Visualizer.plot_correlation_heatmap(X_test, save_path)
        
        # Confusion matrices and ROC curves
        for model_name, results in results_dict.items():
            # Confusion matrix
            Visualizer.plot_confusion_matrix(results['confusion_matrix'], model_name, save_path)
            
            # ROC curve
            if results['y_pred_proba'] is not None:
                Visualizer.plot_roc_curve(results['y_pred'], results['y_pred_proba'], 
                                         model_name, save_path)
            
            # Feature importance
            # Visualizer.plot_feature_importance(model, X_test.columns, model_name, save_path)
        
        print("✅ All visualizations generated!")
