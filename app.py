import streamlit as st
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from src.model import DepthPredictor
import tempfile

def load_model():
    """Load the trained model."""
    model_path = 'models/depth_predictor.joblib'
    if os.path.exists(model_path):
        return DepthPredictor.load(model_path)
    return None

def train_model():
    """Run model training."""
    cmd = [
        "python", "src/train_model.py",
        "--data_path", "data/training_data.csv",
        "--test_data_path", "data/test_data.csv",
        "--model_output", "models/depth_predictor.joblib",
        "--plot_results"
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    return process.stdout, process.stderr

def predict_depth(rtl_file_path, signal_name):
    """Run depth prediction for a signal."""
    cmd = [
        "python", "src/predict_depth.py",
        "--rtl_file", rtl_file_path,
        "--signal", signal_name,
        "--model_path", "models/depth_predictor.joblib"
    ]
    process = subprocess.run(cmd, capture_output=True, text=True)
    return process.stdout, process.stderr

def main():
    st.title("RTL Combinational Depth Predictor")
    st.write("Predict combinational depth for RTL signals using machine learning")

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Train Model", "Predict Depth", "View Results"])

    # Tab 1: Train Model
    with tab1:
        st.header("Train Model")
        st.write("Train a new model using your dataset")
        
        if st.button("Train New Model"):
            with st.spinner("Training model..."):
                stdout, stderr = train_model()
                if stderr:
                    st.error(f"Error during training:\n{stderr}")
                else:
                    st.success("Model trained successfully!")
                    st.text(stdout)
                
                # Display training plots if they exist
                if os.path.exists('plots/model_comparison.png'):
                    st.image('plots/model_comparison.png', 
                            caption='Model Comparison Results')

    # Tab 2: Predict Depth
    with tab2:
        st.header("Predict Signal Depth")
        
        # Check if model exists
        model = load_model()
        if model is None:
            st.warning("No trained model found. Please train a model first.")
        else:
            # File uploader for RTL file
            uploaded_file = st.file_uploader("Upload RTL File", type=['v'])
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.v') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    rtl_file_path = tmp_file.name

                # Signal name input
                signal_name = st.text_input("Enter Signal Name")
                
                if signal_name and st.button("Predict Depth"):
                    with st.spinner("Predicting depth..."):
                        stdout, stderr = predict_depth(rtl_file_path, signal_name)
                        if stderr:
                            st.error(f"Error during prediction:\n{stderr}")
                        else:
                            st.success("Prediction complete!")
                            st.text(stdout)
                
                # Cleanup temporary file
                os.unlink(rtl_file_path)

    # Tab 3: View Results
    with tab3:
        st.header("View Results")
        
        if os.path.exists('plots/actual_vs_predicted_test_data.png'):
            st.image('plots/actual_vs_predicted_test_data.png', 
                    caption='Actual vs Predicted Depths')
        
        if os.path.exists('data/test_data.csv'):
            test_data = pd.read_csv('data/test_data.csv')
            st.write("### Recent Predictions")
            st.dataframe(test_data)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    main() 