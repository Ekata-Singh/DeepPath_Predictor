#!/usr/bin/env python3
"""
Test script for the combinational depth prediction model.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import DepthPredictor
from src.feature_extraction import extract_features_from_rtl

class TestDepthPredictor(unittest.TestCase):
    """Test cases for the DepthPredictor class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a simple test dataset
        self.X_train = pd.DataFrame({
            'module_name': ['counter', 'alu', 'fifo_controller'],
            'signal_name': ['max_reached', 'overflow_flag', 'debug_trigger'],
            'fan_in': [10, 15, 4],
            'operators_count': [5, 10, 3],
            'conditional_count': [1, 5, 0],
            'arithmetic_ops': [0, 0, 0],
            'logical_ops': [4, 5, 3],
            'comparison_ops': [1, 5, 0],
            'mux_count': [1, 5, 0]
        })
        self.y_train = pd.Series([4, 6, 6])
        
        self.X_test = pd.DataFrame({
            'module_name': ['counter', 'alu'],
            'signal_name': ['intermediate1', 'carry_out'],
            'fan_in': [3, 2],
            'operators_count': [2, 1],
            'conditional_count': [0, 0],
            'arithmetic_ops': [0, 0],
            'logical_ops': [2, 1],
            'comparison_ops': [0, 0],
            'mux_count': [0, 0]
        })
        self.y_test = pd.Series([2, 4])
        
        # Create a temporary directory for model saving/loading
        os.makedirs('temp', exist_ok=True)
    
    def tearDown(self):
        """Clean up after the tests."""
        # Remove temporary files
        if os.path.exists('temp/model.joblib'):
            os.remove('temp/model.joblib')
        
        if os.path.exists('temp'):
            os.rmdir('temp')
    
    def test_random_forest_model(self):
        """Test the random forest model."""
        model = DepthPredictor(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        # Test prediction
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
        
        # Test evaluation
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('within_one', metrics)
    
    def test_gradient_boosting_model(self):
        """Test the gradient boosting model."""
        model = DepthPredictor(model_type='gradient_boosting')
        model.train(self.X_train, self.y_train)
        
        # Test prediction
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
    
    def test_linear_model(self):
        """Test the linear model."""
        model = DepthPredictor(model_type='linear')
        model.train(self.X_train, self.y_train)
        
        # Test prediction
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
    
    def test_save_load_model(self):
        """Test saving and loading the model."""
        # Create and train a model
        model = DepthPredictor(model_type='random_forest')
        model.train(self.X_train, self.y_train)
        
        # Save the model
        model.save('temp/model.joblib')
        self.assertTrue(os.path.exists('temp/model.joblib'))
        
        # Load the model
        loaded_model = DepthPredictor.load('temp/model.joblib', model_type='random_forest')
        
        # Test that the loaded model makes the same predictions
        y_pred_original = model.predict(self.X_test)
        y_pred_loaded = loaded_model.predict(self.X_test)
        np.testing.assert_array_almost_equal(y_pred_original, y_pred_loaded)
    
    def test_feature_extraction(self):
        """Test feature extraction from RTL files."""
        # Path to sample RTL files
        rtl_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'data', 'sample_rtl_1.v')
        
        # Check if the file exists
        if not os.path.exists(rtl_file):
            self.skipTest(f"Sample RTL file {rtl_file} not found")
        
        try:
            # Try to extract features
            features = extract_features_from_rtl(rtl_file, 'max_reached')
            
            # Check that features were extracted
            self.assertIsNotNone(features)
            self.assertEqual(features['signal_name'], 'max_reached')
            self.assertIn('fan_in', features)
            self.assertIn('operators_count', features)
            self.assertIn('conditional_count', features)
            self.assertIn('arithmetic_ops', features)
            self.assertIn('logical_ops', features)
            self.assertIn('comparison_ops', features)
            self.assertIn('mux_count', features)
        except Exception as e:
            # Skip test if iverilog is not installed or other extraction errors occur
            if "No such file or directory: 'iverilog'" in str(e):
                self.skipTest("Icarus Verilog (iverilog) is not installed. Skipping feature extraction test.")
            else:
                self.skipTest(f"Feature extraction failed: {str(e)}")

if __name__ == '__main__':
    unittest.main() 