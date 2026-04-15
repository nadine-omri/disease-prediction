"""
Prediction script for making disease predictions on new data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from src.utils import ModelPersistence
from src.data_preprocessing import DataPreprocessor


def make_predictions(input_data, model_path='models/trained_models.pkl'):
    """
    Make predictions using trained models.
    
    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        New patient data for prediction
    model_path : str
        Path to trained models
        
    Returns:
    --------
    dict : Predictions from all models
    """
    # Vérifier si les modèles existent
    if not os.path.exists(model_path):
        print("❌ Modèles non trouvés!")
        print(f"   Chemin attendu: {model_path}")
        print("\n💡 Exécutez d'abord: python train.py")
        return None
    
    # Charger les modèles
    print("📂 Chargement des modèles entraînés...")
    try:
        models = ModelPersistence.load_models_dict(model_path)
        print(f"✅ {len(models)} modèles chargés avec succès!\n")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None
    
    # Charger les features du dataset d'entraînement
    print("📊 Chargement des features du dataset...")
    try:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        feature_names = data.feature_names
        print(f"✅ {len(feature_names)} features trouvées\n")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None
    
    # Convertir l'entrée en DataFrame si nécessaire
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
    
    # S'assurer que les colonnes correspondent aux features d'entraînement
    if list(input_df.columns) != list(feature_names):
        print("⚠️  Les colonnes d'entrée ne correspondent pas aux features d'entraînement!")
        print(f"   Colonnes attendues ({len(feature_names)}): {list(feature_names)[:5]}...")
        
        # Si l'entrée a le bon nombre de features, renommer les colonnes
        if len(input_df.columns) == len(feature_names):
            print("   Renommage automatique des colonnes...")
            input_df.columns = feature_names
        else:
            print(f"❌ Nombre de features incorrect: {len(input_df.columns)} au lieu de {len(feature_names)}")
            return None
    
    # Prétraiter les données d'entrée
    print("🔧 Prétraitement des données d'entrée...")
    preprocessor = DataPreprocessor()
    input_scaled = preprocessor.normalize_features(input_df)
    
    # Faire les prédictions
    print("\n🤖 Prédictions en cours...\n")
    predictions = {}
    
    for model_name, model in models.items():
        try:
            pred = model.predict(input_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                confidence = max(proba) * 100
                predictions[model_name] = {
                    'prediction': pred,
                    'confidence': confidence
                }
                result_text = "POSITIF 🔴" if pred == 1 else "NÉGATIF 🟢"
                print(f"✅ {model_name:20s}: {result_text} (Confiance: {confidence:.2f}%)")
            else:
                predictions[model_name] = {
                    'prediction': pred,
                    'confidence': None
                }
                result_text = "POSITIF 🔴" if pred == 1 else "NÉGATIF 🟢"
                print(f"✅ {model_name:20s}: {result_text}")
        except Exception as e:
            print(f"❌ {model_name}: Erreur - {e}")
    
    return predictions


def print_predictions(predictions):
    """Print predictions in a readable format."""
    if not predictions:
        return
    
    print("\n" + "="*80)
    print("RÉSULTATS DES PRÉDICTIONS")
    print("="*80 + "\n")
    
    for model_name, result in predictions.items():
        result_text = "CANCER DÉTECTÉ 🔴" if result['prediction'] == 1 else "NORMAL 🟢"
        print(f"Modèle: {model_name}")
        print(f"  Diagnostic: {result_text}")
        if result['confidence'] is not None:
            print(f"  Confiance: {result['confidence']:.2f}%")
        print()


def load_sample_data():
    """Load a real sample from the breast cancer dataset."""
    print("📥 Chargement d'un échantillon du dataset...\n")
    
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    
    # Prendre le premier cas
    sample = data.data[0]
    feature_names = data.feature_names
    
    sample_dict = dict(zip(feature_names, sample))
    
    print(f"✅ Échantillon du patient chargé (Cas réel du dataset)")
    print(f"   Features: {len(feature_names)}\n")
    
    return sample_dict


if __name__ == "__main__":
    print("="*80)
    print("🏥 DISEASE PREDICTION - PRÉDICTIONS")
    print("="*80 + "\n")
    
    # Option 1: Utiliser un échantillon réel du dataset
    print("💡 Option 1: Utiliser un échantillon réel du dataset\n")
    sample_data = load_sample_data()
    
    print("🤔 Faire une prédiction sur cet échantillon...")
    predictions = make_predictions(sample_data)
    
    if predictions:
        print_predictions(predictions)
    else:
        print("\n⚠️  Les modèles n'ont pas pu être chargés.")
        print("👉 Veuillez d'abord entraîner les modèles avec: python train.py")
    
    # Option 2: Ajouter vos propres données
    print("\n" + "="*80)
    print("💡 Option 2: Pour utiliser vos propres données")
    print("="*80)
    print("""
Modifiez le fichier predict.py et remplacez:
    sample_data = load_sample_data()

Par:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    feature_names = data.feature_names
    
    # Vos valeurs personnalisées
    your_values = [12.0, 15.0, 80.0, ...]  # 30 valeurs
    sample_data = dict(zip(feature_names, your_values))

Ou chargez depuis un CSV:
    df = pd.read_csv('your_data.csv')
    sample_data = df.iloc[0].to_dict()
""")