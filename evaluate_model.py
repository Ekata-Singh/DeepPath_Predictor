#!/usr/bin/env python3
"""
Script to evaluate the trained model on test data and show predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import DepthPredictor

def main():
    """Main function to evaluate the model on test data."""
    # Check if model exists
    model_path = 'models/depth_predictor.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        sys.exit(1)
    
    # Load test data
    test_data_path = 'data/test_data.csv'
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file '{test_data_path}' not found.")
        sys.exit(1)
    
    test_data = pd.read_csv(test_data_path)
    
    # Load the model
    print(f"Loading model from '{model_path}'...")
    model = DepthPredictor.load(model_path)
    
    # Prepare features and target
    X_test = test_data.drop('actual_depth', axis=1)
    y_test = test_data['actual_depth']
    
    # Make predictions
    print("Making predictions on test data...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    within_one = np.mean(np.abs(y_test - y_pred) <= 1.0)
    
    print(f"\nTest Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Within ±1: {within_one:.2%}")
    
    # Show predictions for each signal
    print("\nPredictions for each signal:")
    print("=" * 80)
    print(f"{'Module':<20} {'Signal':<20} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 80)
    
    for i in range(len(test_data)):
        module = test_data.iloc[i]['module_name']
        signal = test_data.iloc[i]['signal_name']
        actual = test_data.iloc[i]['actual_depth']
        predicted = y_pred[i]
        error = actual - predicted
        
        print(f"{module:<20} {signal:<20} {actual:<10.0f} {predicted:<10.2f} {error:<10.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Actual vs Predicted Combinational Depth')
    plt.xlabel('Actual Depth')
    plt.ylabel('Predicted Depth')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/actual_vs_predicted_test_data.png')
    print("\nPlot saved to 'plots/actual_vs_predicted_test_data.png'")

if __name__ == "__main__":
    main() 