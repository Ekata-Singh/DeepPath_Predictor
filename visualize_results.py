#!/usr/bin/env python3
"""
Script to generate detailed visualizations of model performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import DepthPredictor

def main():
    """Main function to visualize model results."""
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
    
    # Load training data for comparison
    train_data_path = 'data/training_data.csv'
    if not os.path.exists(train_data_path):
        print(f"Error: Training data file '{train_data_path}' not found.")
        sys.exit(1)
    
    test_data = pd.read_csv(test_data_path)
    train_data = pd.read_csv(train_data_path)
    
    # Load the model
    print(f"Loading model from '{model_path}'...")
    model = DepthPredictor.load(model_path)
    
    # Prepare features and target
    X_test = test_data.drop('actual_depth', axis=1)
    y_test = test_data['actual_depth']
    X_train = train_data.drop('actual_depth', axis=1)
    y_train = train_data['actual_depth']
    
    # Make predictions
    print("Making predictions on test and training data...")
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # 1. Actual vs Predicted scatter plot with error bars
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.7, label='Test Data', color='blue')
    plt.scatter(y_train, y_pred_train, alpha=0.3, label='Training Data', color='green')
    plt.plot([min(min(y_test), min(y_train)), max(max(y_test), max(y_train))], 
             [min(min(y_test), min(y_train)), max(max(y_test), max(y_train))], 
             'r--', label='Perfect Prediction')
    
    # Add error bars for test data
    for i in range(len(y_test)):
        plt.plot([y_test.iloc[i], y_test.iloc[i]], [y_test.iloc[i], y_pred_test[i]], 'b-', alpha=0.3)
    
    plt.title('Actual vs Predicted Combinational Depth')
    plt.xlabel('Actual Depth')
    plt.ylabel('Predicted Depth')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/actual_vs_predicted_with_errors.png')
    print("Plot saved to 'plots/actual_vs_predicted_with_errors.png'")
    
    # 2. Error distribution histogram
    plt.figure(figsize=(12, 8))
    errors_test = y_test - y_pred_test
    errors_train = y_train - y_pred_train
    
    plt.hist(errors_test, bins=10, alpha=0.7, label='Test Data', color='blue')
    plt.hist(errors_train, bins=10, alpha=0.3, label='Training Data', color='green')
    
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/error_distribution_detailed.png')
    print("Plot saved to 'plots/error_distribution_detailed.png'")
    
    # 3. Module-wise performance
    plt.figure(figsize=(14, 10))
    
    # Group by module and calculate mean absolute error
    module_errors = pd.DataFrame({
        'module': test_data['module_name'],
        'error': np.abs(y_test - y_pred_test)
    })
    module_mae = module_errors.groupby('module')['error'].mean().sort_values()
    
    # Plot module-wise MAE
    ax = module_mae.plot(kind='bar', color='skyblue')
    plt.title('Module-wise Mean Absolute Error')
    plt.xlabel('Module')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(module_mae):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/module_wise_mae.png')
    print("Plot saved to 'plots/module_wise_mae.png'")
    
    # 4. Feature importance analysis (if model supports it)
    try:
        # Get feature importances if available
        if hasattr(model.model.named_steps['regressor'], 'coef_'):
            # For linear models
            importances = np.abs(model.model.named_steps['regressor'].coef_)
            feature_names = model.feature_columns
            
            # Create DataFrame for plotting
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
            plt.title('Feature Importance')
            plt.xlabel('Absolute Coefficient Value')
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png')
            print("Plot saved to 'plots/feature_importance.png'")
        
        elif hasattr(model.model.named_steps['regressor'], 'feature_importances_'):
            # For tree-based models
            importances = model.model.named_steps['regressor'].feature_importances_
            feature_names = model.feature_columns
            
            # Create DataFrame for plotting
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png')
            print("Plot saved to 'plots/feature_importance.png'")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")
    
    # 5. Depth-wise accuracy
    plt.figure(figsize=(12, 8))
    
    # Create a DataFrame with actual and predicted depths
    depth_accuracy = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_test,
        'Error': np.abs(y_test - y_pred_test),
        'Within_One': np.abs(y_test - y_pred_test) <= 1
    })
    
    # Group by actual depth and calculate percentage within ±1
    depth_accuracy_grouped = depth_accuracy.groupby('Actual')['Within_One'].mean() * 100
    
    # Plot depth-wise accuracy
    ax = depth_accuracy_grouped.plot(kind='bar', color='lightgreen')
    plt.title('Accuracy by Actual Depth (% Within ±1)')
    plt.xlabel('Actual Depth')
    plt.ylabel('Percentage Within ±1')
    plt.grid(True, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(depth_accuracy_grouped):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/depth_wise_accuracy.png')
    print("Plot saved to 'plots/depth_wise_accuracy.png'")
    
    # 6. Confusion matrix-like visualization for rounded predictions
    plt.figure(figsize=(10, 8))
    
    # Round predictions to nearest integer
    y_pred_rounded = np.round(y_pred_test).astype(int)
    
    # Get unique depth values
    all_depths = sorted(set(list(y_test.unique()) + list(y_pred_rounded)))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred_rounded, labels=all_depths)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_depths, yticklabels=all_depths)
    plt.title('Confusion Matrix (Rounded Predictions)')
    plt.xlabel('Predicted Depth')
    plt.ylabel('Actual Depth')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    print("Plot saved to 'plots/confusion_matrix.png'")
    
    print("\nAll visualizations have been saved to the 'plots' directory.")

if __name__ == "__main__":
    main() 