"""
Main training script for disease prediction models.
Trains all models and generates evaluation reports.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from src.utils import FeatureEngineer, DataLoader, ModelPersistence
from src.data_preprocessing import DataPreprocessor
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator, ResultsComparator, ReportGenerator
from src.visualization import DashboardGenerator


def main():
    """Main training pipeline."""
    
    print("\n" + "="*80)
    print("🏥 DISEASE PREDICTION MODEL TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Create directories
    print("📁 Creating project directories...")
    FeatureEngineer.create_directories()
    
    # Load dataset
    print("\n📊 Loading dataset...")
    try:
        # Try to load from downloaded CSV files first
        if os.path.exists('data/breast_cancer.csv'):
            print("   Loading from local file...")
            df = pd.read_csv('data/breast_cancer.csv')
            print(f"✅ Local Breast Cancer dataset loaded: {df.shape}")
        else:
            # Fallback to sklearn dataset
            print("   Loading from sklearn...")
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            print(f"✅ Breast Cancer dataset loaded: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    preprocessor = DataPreprocessor(random_state=42)
    
    X_train, X_test, y_train, y_test = preprocessor.full_preprocessing(
        df, target_col='target', test_size=0.2
    )
    
    print(f"✅ Preprocessing complete!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Train models
    print("\n🤖 Training models...")
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train, y_train, tune=False)
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    results = ModelEvaluator.evaluate_all_models(models, X_test, y_test)
    
    # Compare models
    print("\n🏆 Comparing models...")
    comparison_df = ResultsComparator.compare_models(results)
    print("\n" + comparison_df.to_string())
    
    # Generate reports
    print("\n📝 Generating reports...")
    ReportGenerator.generate_text_report(results)
    ReportGenerator.print_summary(results)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    try:
        DashboardGenerator.generate_full_report(results, X_test, comparison_df)
    except Exception as e:
        print(f"⚠️  Visualization warning: {e}")
    
    # Save models
    print("\n💾 Saving models...")
    ModelPersistence.save_models_dict(models, 'models/trained_models.pkl')
    
    print("\n✨ Training pipeline completed successfully!")
    print("="*80 + "\n")
    
    # Print best model
    best_model_name, best_score = ResultsComparator.get_best_model(results, 'f1')
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   F1-Score: {best_score:.4f}\n")


if __name__ == "__main__":
    main()