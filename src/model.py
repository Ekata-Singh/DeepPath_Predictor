#!/usr/bin/env python3
"""
Model module for RTL combinational depth prediction.
This module defines the ML models used for predicting combinational depth.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class DepthPredictor:
    """
    Class for predicting combinational depth of RTL signals.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the depth predictor with a specific model type.
        
        Args:
            model_type (str): Type of model to use. Options:
                - 'random_forest'
                - 'gradient_boosting'
                - 'linear'
                - 'ridge'
                - 'lasso'
                - 'svr'
                - 'mlp'
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.feature_columns = [
            'fan_in', 'operators_count', 'conditional_count',
            'arithmetic_ops', 'logical_ops', 'comparison_ops', 'mux_count'
        ]
        
    def _create_model(self):
        """Create the ML model based on the specified type."""
        if self.model_type == 'random_forest':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                ))
            ])
        elif self.model_type == 'gradient_boosting':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                ))
            ])
        elif self.model_type == 'linear':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        elif self.model_type == 'ridge':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=1.0, random_state=42))
            ])
        elif self.model_type == 'lasso':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Lasso(alpha=1.0, random_state=42))
            ])
        elif self.model_type == 'svr':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))
            ])
        elif self.model_type == 'mlp':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    max_iter=500,
                    random_state=42
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y):
        """
        Train the model on the given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values (combinational depths)
        """
        # Extract only the feature columns we need
        X_features = X[self.feature_columns]
        
        # Train the model
        self.model.fit(X_features, y)
        
        # Return training metrics
        y_pred = self.model.predict(X_features)
        return {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
    
    def predict(self, X):
        """
        Predict combinational depths for the given features.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.ndarray: Predicted combinational depths
        """
        # Extract only the feature columns we need
        X_features = X[self.feature_columns]
        
        # Make predictions
        return self.model.predict(X_features)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values (combinational depths)
            
        Returns:
            dict: Evaluation metrics
        """
        # Extract only the feature columns we need
        X_features = X[self.feature_columns]
        
        # Make predictions
        y_pred = self.model.predict(X_features)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate accuracy within +/- 1 level
        within_one = np.mean(np.abs(y - y_pred) <= 1.0)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'within_one': within_one,
            'predictions': y_pred
        }
    
    def tune_hyperparameters(self, X, y, param_grid=None):
        """
        Tune hyperparameters using grid search.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values
            param_grid (dict): Parameter grid for grid search
            
        Returns:
            dict: Best parameters and score
        """
        # Extract only the feature columns we need
        X_features = X[self.feature_columns]
        
        # Default parameter grids for different model types
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [None, 10, 20, 30],
                    'regressor__min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__learning_rate': [0.01, 0.1, 0.2],
                    'regressor__max_depth': [3, 5, 7]
                }
            elif self.model_type in ['ridge', 'lasso']:
                param_grid = {
                    'regressor__alpha': [0.1, 1.0, 10.0]
                }
            elif self.model_type == 'svr':
                param_grid = {
                    'regressor__C': [0.1, 1.0, 10.0],
                    'regressor__epsilon': [0.01, 0.1, 0.2]
                }
            elif self.model_type == 'mlp':
                param_grid = {
                    'regressor__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'regressor__alpha': [0.0001, 0.001, 0.01]
                }
            else:
                param_grid = {}
        
        # Create grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_features, y)
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_  # Convert back to MSE
        }
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load(cls, filepath, model_type='random_forest'):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to load the model from
            model_type (str): Type of the model
            
        Returns:
            DepthPredictor: Loaded model
        """
        # Create a new instance
        instance = cls(model_type=model_type)
        
        # Load the model
        instance.model = joblib.load(filepath)
        
        return instance

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple models to find the best one.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        
    Returns:
        dict: Results for each model
    """
    model_types = [
        'random_forest',
        'gradient_boosting',
        'linear',
        'ridge',
        'lasso',
        'svr',
        'mlp'
    ]
    
    results = {}
    
    for model_type in model_types:
        print(f"Training {model_type} model...")
        
        # Create and train model
        model = DepthPredictor(model_type=model_type)
        train_metrics = model.train(X_train, y_train)
        
        # Evaluate model
        eval_metrics = model.evaluate(X_test, y_test)
        
        # Store results
        results[model_type] = {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }
        
        print(f"  MSE: {eval_metrics['mse']:.4f}")
        print(f"  MAE: {eval_metrics['mae']:.4f}")
        print(f"  R²: {eval_metrics['r2']:.4f}")
        print(f"  Within ±1: {eval_metrics['within_one']:.2%}")
    
    return results

if __name__ == "__main__":
    import sys
    import pandas as pd
    
    if len(sys.argv) < 4:
        print("Usage: python model.py <train_data.csv> <test_data.csv> <output_model.joblib>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Split into features and target
    X_train = train_data.drop('actual_depth', axis=1)
    y_train = train_data['actual_depth']
    X_test = test_data.drop('actual_depth', axis=1)
    y_test = test_data['actual_depth']
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Find best model
    best_model_type = min(results.keys(), key=lambda k: results[k]['eval_metrics']['mse'])
    print(f"\nBest model: {best_model_type}")
    
    # Train final model
    final_model = DepthPredictor(model_type=best_model_type)
    final_model.train(X_train, y_train)
    
    # Save model
    final_model.save(output_file)
    print(f"Model saved to {output_file}") 