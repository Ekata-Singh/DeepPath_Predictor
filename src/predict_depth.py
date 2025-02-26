#!/usr/bin/env python3
"""
Script to predict the combinational depth of a signal in an RTL file.
"""

import os
import sys
import argparse
import pandas as pd
import joblib

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import DepthPredictor
from src.feature_extraction import extract_features_from_rtl

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict the combinational depth of a signal in an RTL file.')
    parser.add_argument('--rtl_file', type=str, required=True,
                        help='Path to the RTL file')
    parser.add_argument('--signal', type=str, required=True,
                        help='Name of the signal to predict depth for')
    parser.add_argument('--model_path', type=str, default='models/depth_predictor.joblib',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'linear', 'ridge', 'lasso', 'svr', 'mlp'],
                        help='Type of model to use')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Check if the RTL file exists
    if not os.path.isfile(args.rtl_file):
        print(f"Error: RTL file '{args.rtl_file}' does not exist.")
        sys.exit(1)
    
    # Check if the model file exists
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file '{args.model_path}' does not exist.")
        sys.exit(1)
    
    # Extract features from the RTL file for the specified signal
    print(f"Extracting features from '{args.rtl_file}' for signal '{args.signal}'...")
    features = extract_features_from_rtl(args.rtl_file, args.signal)
    
    if features is None:
        print(f"Error: Signal '{args.signal}' not found in '{args.rtl_file}'.")
        sys.exit(1)
    
    # Convert features to a DataFrame
    features_df = pd.DataFrame([features])
    
    # Load the model
    print(f"Loading model from '{args.model_path}'...")
    model = DepthPredictor.load(args.model_path, model_type=args.model_type)
    
    # Predict the depth
    print("Predicting combinational depth...")
    depth = model.predict(features_df)[0]
    
    # Print the result
    print(f"\nPredicted combinational depth for signal '{args.signal}': {depth:.2f}")
    print(f"Rounded depth: {round(depth)}")
    
    # Print feature importance if the model supports it
    if args.model_type in ['random_forest', 'gradient_boosting']:
        try:
            # Get feature importances
            importances = model.model.named_steps['regressor'].feature_importances_
            feature_names = model.feature_columns
            
            # Sort features by importance
            indices = importances.argsort()[::-1]
            
            print("\nFeature importances:")
            for i in indices:
                print(f"  {feature_names[i]}: {importances[i]:.4f}")
        except:
            pass

if __name__ == "__main__":
    main() 