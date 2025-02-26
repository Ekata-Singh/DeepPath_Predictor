#!/bin/bash
# Script to run the complete RTL Combinational Depth Predictor pipeline
# Note: On Windows, use 'python' instead of 'python3'

# Don't exit on error, we'll handle errors gracefully
set +e

echo "===== RTL Combinational Depth Predictor Pipeline ====="
echo "Starting pipeline execution..."

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models plots

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Train the model with model comparison
echo -e "\n===== Step 1: Training the model ====="
echo "Training the model and comparing different model types..."
python run_pipeline.py --train --compare_models
if [ $? -ne 0 ]; then
    echo "Warning: Model training encountered issues. Check the output above for details."
fi

# Evaluate the model on test data
echo -e "\n===== Step 2: Evaluating the model ====="
echo "Evaluating the model on test data..."
python evaluate_model.py
if [ $? -ne 0 ]; then
    echo "Warning: Model evaluation encountered issues. Check the output above for details."
fi

# Generate detailed visualizations
echo -e "\n===== Step 3: Generating visualizations ====="
echo "Generating detailed visualizations..."
python visualize_results.py
if [ $? -ne 0 ]; then
    echo "Warning: Visualization generation encountered issues. Check the output above for details."
fi

# Run tests
echo -e "\n===== Step 4: Running tests ====="
echo "Running tests..."
python -m unittest discover tests
if [ $? -ne 0 ]; then
    echo "Note: Some tests may have failed. This could be due to missing optional dependencies like Icarus Verilog."
    echo "If you need full RTL parsing functionality, please install Icarus Verilog (iverilog)."
fi

echo -e "\n===== Pipeline completed! ====="
echo "Results:"
echo "- Trained model saved to: models/depth_predictor.joblib"
echo "- Evaluation metrics in terminal output above"
echo "- Visualizations saved to: plots/ directory"
echo "- Test results in terminal output above"
echo ""
echo "Optional Dependencies:"
echo "- For full RTL parsing functionality, install Icarus Verilog: http://iverilog.icarus.com/"
echo ""
echo "To make predictions on new RTL files, use:"
echo "python run_pipeline.py --predict --rtl_file path/to/your/rtl_file.v --signal signal_name" 