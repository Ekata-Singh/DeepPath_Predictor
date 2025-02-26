#!/usr/bin/env python3
"""
Script to train the combinational depth prediction model.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import DepthPredictor, train_and_evaluate_models

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a combinational depth prediction model.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data CSV file')
    parser.add_argument('--test_data_path', type=str, default=None,
                        help='Path to the test data CSV file (optional)')
    parser.add_argument('--model_output', type=str, required=True,
                        help='Path to save the trained model')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'linear', 'ridge', 'lasso', 'svr', 'mlp'],
                        help='Type of model to train')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing (if no test data provided)')
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Whether to tune hyperparameters')
    parser.add_argument('--compare_models', action='store_true',
                        help='Whether to compare different model types')
    parser.add_argument('--plot_results', action='store_true',
                        help='Whether to plot the results')
    
    return parser.parse_args()

def load_data(data_path, test_data_path=None, test_size=0.2):
    """
    Load the training and test data.
    
    Args:
        data_path (str): Path to the training data CSV file
        test_data_path (str): Path to the test data CSV file (optional)
        test_size (float): Fraction of data to use for testing (if no test data provided)
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    # Load training data
    data = pd.read_csv(data_path)
    
    # If test data is provided, load it
    if test_data_path:
        test_data = pd.read_csv(test_data_path)
        X_train = data.drop('actual_depth', axis=1)
        y_train = data['actual_depth']
        X_test = test_data.drop('actual_depth', axis=1)
        y_test = test_data['actual_depth']
    else:
        # Split the data into training and test sets
        X = data.drop('actual_depth', axis=1)
        y = data['actual_depth']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, y_train, X_test, y_test

def plot_results(y_test, y_pred, model_type):
    """
    Plot the results of the model.
    
    Args:
        y_test (pd.Series): Actual depths
        y_pred (np.ndarray): Predicted depths
        model_type (str): Type of model used
    """
    plt.figure(figsize=(10, 6))
    
    # Plot actual vs predicted
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    
    plt.title(f'Actual vs Predicted Combinational Depth ({model_type})')
    plt.xlabel('Actual Depth')
    plt.ylabel('Predicted Depth')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/actual_vs_predicted_{model_type}.png')
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    errors = y_test - y_pred
    plt.hist(errors, bins=20, alpha=0.7)
    plt.title(f'Error Distribution ({model_type})')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'plots/error_distribution_{model_type}.png')

def main():
    """Main function."""
    args = parse_args()
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(
        args.data_path, args.test_data_path, args.test_size
    )
    
    if args.compare_models:
        # Compare different model types
        print("Comparing different model types...")
        results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
        
        # Find the best model
        best_model_type = min(results.keys(), key=lambda k: results[k]['eval_metrics']['mse'])
        print(f"\nBest model: {best_model_type}")
        
        # Use the best model type
        model_type = best_model_type
    else:
        # Use the specified model type
        model_type = args.model_type
    
    # Create and train the model
    print(f"Training {model_type} model...")
    model = DepthPredictor(model_type=model_type)
    
    if args.tune_hyperparams:
        # Tune hyperparameters
        print("Tuning hyperparameters...")
        tuning_results = model.tune_hyperparameters(X_train, y_train)
        print(f"Best parameters: {tuning_results['best_params']}")
        print(f"Best MSE: {tuning_results['best_score']:.4f}")
    
    # Train the model
    train_metrics = model.train(X_train, y_train)
    print(f"Training MSE: {train_metrics['mse']:.4f}")
    print(f"Training MAE: {train_metrics['mae']:.4f}")
    print(f"Training R²: {train_metrics['r2']:.4f}")
    
    # Evaluate the model
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"Test MSE: {eval_metrics['mse']:.4f}")
    print(f"Test MAE: {eval_metrics['mae']:.4f}")
    print(f"Test R²: {eval_metrics['r2']:.4f}")
    print(f"Within ±1: {eval_metrics['within_one']:.2%}")
    
    if args.plot_results:
        # Plot the results
        plot_results(y_test, eval_metrics['predictions'], model_type)
    
    # Save the model
    model.save(args.model_output)
    print(f"Model saved to {args.model_output}")

if __name__ == "__main__":
    main() 