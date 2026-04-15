import joblib
import numpy as np
import pandas as pd

# Load saved models
model1 = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')

# Function to make predictions on new patient data

def make_predictions(patient_data):
    # Ensure the input is in the correct format
    data = pd.DataFrame(patient_data)
    
    # Make predictions
    predictions_model1 = model1.predict(data)
    confidence_model1 = model1.predict_proba(data).max(axis=1)
    predictions_model2 = model2.predict(data)
    confidence_model2 = model2.predict_proba(data).max(axis=1)
    
    # Combine results
    results = {
        'model1_predictions': predictions_model1,
        'model1_confidence': confidence_model1,
        'model2_predictions': predictions_model2,
        'model2_confidence': confidence_model2,
    }
    return results

# Example usage
if __name__ == '__main__':
    new_patient_data = [{ 'feature1': value1, 'feature2': value2 }]  # Replace with actual new patient data
    predictions = make_predictions(new_patient_data)
    print(predictions)