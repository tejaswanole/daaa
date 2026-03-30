"""
pipeline/trainer.py
===================
Orchestrates model training by wrapping core module functions.
Provides a clean interface for the UI to call.
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path so core imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_generator import generate_workload_data, prepare_data
from core.models import kmeans_profiler, train_svm, train_lstm, train_vanilla_rf, train_ga_rf


def run_data_generation(n_samples=20000, random_state=42):
    """
    Generate synthetic workload data and prepare train/test splits.
    
    Returns:
        dict with keys: X, y, feature_names, X_train, X_test,
              X_train_scaled, X_test_scaled, y_train, y_test, scaler,
              X_train_aug, X_test_aug, kmeans
    """
    # Generate raw data
    X, y, feature_names = generate_workload_data(
        n_samples=n_samples, random_state=random_state
    )
    
    # Split and scale
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y)
    
    # K-Means augmentation
    X_train_aug, X_test_aug, kmeans = kmeans_profiler(X_train_scaled, X_test_scaled, k=5)
    
    return {
        'X': X, 'y': y, 'feature_names': feature_names,
        'X_train': X_train, 'X_test': X_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'y_train': y_train, 'y_test': y_test, 'scaler': scaler,
        'X_train_aug': X_train_aug, 'X_test_aug': X_test_aug,
        'kmeans': kmeans,
    }


def train_single_model(model_name, X_train_aug, y_train, X_test_aug, y_test,
                        pop_size=20, generations=15):
    """
    Train a single model by name.
    
    Args:
        model_name: One of 'SVM', 'LSTM', 'Random Forest', 'GA-RF'
        X_train_aug, y_train, X_test_aug, y_test: Training/test data
        pop_size, generations: GA-RF specific parameters
    
    Returns:
        (model_object, metrics_dict, predictions, ga_fitness_or_None)
    """
    ga_fitness = None
    
    if model_name == 'SVM':
        model, metrics, preds = train_svm(X_train_aug, y_train, X_test_aug, y_test)
    elif model_name == 'LSTM':
        model, metrics, preds = train_lstm(
            X_train_aug, y_train, X_test_aug, y_test,
            epochs=30, batch_size=64
        )
    elif model_name == 'Random Forest':
        model, metrics, preds = train_vanilla_rf(X_train_aug, y_train, X_test_aug, y_test)
    elif model_name == 'GA-RF':
        model, metrics, preds, ga_fitness = train_ga_rf(
            X_train_aug, y_train, X_test_aug, y_test,
            pop_size=pop_size, generations=generations
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, metrics, preds, ga_fitness


def train_all_models(X_train_aug, y_train, X_test_aug, y_test,
                     pop_size=20, generations=15, progress_callback=None):
    """
    Train all 4 models sequentially.
    
    Args:
        progress_callback: Optional callable(model_name, step, total) for progress updates
    
    Returns:
        dict with keys: models, metrics, predictions, ga_fitness
    """
    model_names = ['SVM', 'LSTM', 'Random Forest', 'GA-RF']
    results = {
        'models': {},
        'metrics': [],
        'predictions': {},
        'ga_fitness': None,
    }
    
    for i, name in enumerate(model_names):
        if progress_callback:
            progress_callback(name, i, len(model_names))
        
        model, metrics, preds, ga_fitness = train_single_model(
            name, X_train_aug, y_train, X_test_aug, y_test,
            pop_size=pop_size, generations=generations
        )
        
        results['models'][name] = model
        results['metrics'].append(metrics)
        results['predictions'][name] = preds
        
        if ga_fitness is not None:
            results['ga_fitness'] = ga_fitness
    
    if progress_callback:
        progress_callback('Done', len(model_names), len(model_names))
    
    return results
