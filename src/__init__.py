"""
Disease Prediction from Medical Data

A machine learning system for predicting diseases using classification techniques.
Supports: Heart Disease, Diabetes, Breast Cancer
Algorithms: SVM, Logistic Regression, Random Forest, XGBoost
"""

__version__ = "1.0.0"
__author__ = "Nadine Omri"

from . import data_preprocessing
from . import models
from . import evaluation
from . import visualization
from . import utils

__all__ = [
    'data_preprocessing',
    'models',
    'evaluation',
    'visualization',
    'utils',
]