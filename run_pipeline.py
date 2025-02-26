#!/usr/bin/env python3
"""
Script to run the entire RTL combinational depth prediction pipeline.
"""

import os
import sys
import argparse
import subprocess

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the RTL combinational depth prediction pipeline.')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--predict', action='store_true',
                        help='Predict combinational depth for a signal')
    parser.add_argument('--test', action='store_true',
                        help='Run tests')
    parser.add_argument('--rtl_file', type=str,
                        help='Path to the RTL file for prediction')
    parser.add_argument('--signal', type=str,
                        help='Name of the signal to predict depth for')
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare different model types')
    
    return parser.parse_args()

def run_train():
    """Run the training script."""
    print("=== Training the Model ===")
    cmd = [
        "python", "src/train_model.py",
        "--data_path", "data/training_data.csv",
        "--test_data_path", "data/test_data.csv",
        "--model_output", "models/depth_predictor.joblib"
    ]
    
    # Add compare_models flag if specified
    if args.compare_models:
        cmd.append("--compare_models")
        cmd.append("--plot_results")
    
    # Run the command
    subprocess.run(cmd)

def run_predict(rtl_file, signal):
    """Run the prediction script."""
    print(f"=== Predicting Depth for Signal '{signal}' in '{rtl_file}' ===")
    
    # Check if the RTL file exists
    if not os.path.exists(rtl_file):
        print(f"Error: RTL file '{rtl_file}' does not exist.")
        return
    
    # Check if the model exists
    if not os.path.exists("models/depth_predictor.joblib"):
        print("Error: Model file 'models/depth_predictor.joblib' does not exist. Please train the model first.")
        return
    
    # Check if iverilog is installed
    import shutil
    if shutil.which('iverilog') is None:
        print("Warning: Icarus Verilog (iverilog) is not installed.")
        print("For full RTL parsing functionality, please install Icarus Verilog.")
        print("Attempting to proceed with limited functionality...")
    
    cmd = [
        "python", "src/predict_depth.py",
        "--rtl_file", rtl_file,
        "--signal", signal,
        "--model_path", "models/depth_predictor.joblib"
    ]
    
    # Run the command
    result = subprocess.run(cmd)
    
    # Check if the command was successful
    if result.returncode != 0:
        print("Error: Prediction failed. This could be due to missing dependencies or issues with the RTL file.")
        print("If you need full RTL parsing functionality, please install Icarus Verilog (iverilog).")

def run_tests():
    """Run the tests."""
    print("=== Running Tests ===")
    cmd = ["python", "-m", "unittest", "discover", "tests"]
    
    # Run the command
    subprocess.run(cmd)

def main():
    """Main function."""
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    if args.train:
        run_train()
    
    if args.predict:
        if not args.rtl_file or not args.signal:
            print("Error: --rtl_file and --signal are required for prediction.")
            sys.exit(1)
        
        run_predict(args.rtl_file, args.signal)
    
    if args.test:
        run_tests()
    
    # If no action specified, print help
    if not (args.train or args.predict or args.test):
        parser.print_help()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the RTL combinational depth prediction pipeline.')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--predict', action='store_true',
                        help='Predict combinational depth for a signal')
    parser.add_argument('--test', action='store_true',
                        help='Run tests')
    parser.add_argument('--rtl_file', type=str,
                        help='Path to the RTL file for prediction')
    parser.add_argument('--signal', type=str,
                        help='Name of the signal to predict depth for')
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare different model types')
    
    args = parser.parse_args()
    
    main() 