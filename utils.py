import pickle
import streamlit as st
import os
import subprocess

def write_requirements_file():
    requirements = """
streamlit
google-generativeai
pandas
matplotlib
seaborn
scikit-learn
kaggle
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())

def create_venv():
    venv_folder = "venv"
    if not os.path.exists(venv_folder):
        subprocess.run(["python", "-m", "venv", venv_folder])
        
        # Install required packages in the virtual environment
        subprocess.run([os.path.join(venv_folder, 'Scripts', 'pip'), 'install', '-r', 'requirements.txt'])

    return venv_folder

def generate_loading_code(filename, feature_names):
    # Generate dynamic input fields for Streamlit
    input_statements = "\n    ".join([f"{name} = st.number_input('Enter value for {name}')" for name in feature_names])
    
    # Create the prediction statement using user inputs from Streamlit
    prediction_statement = f"prediction = model.predict([[{', '.join([name for name in feature_names])}]])"
    
    code = f"""
import pickle
import streamlit as st

# Load the model from file
def load_model():
    with open('{filename}', 'rb') as f:
        model = pickle.load(f)
    return model

# Streamlit UI for predictions
def predict():
    st.title("Model Prediction App")
    
    model = load_model()
    
    # Create input fields for each feature
    {input_statements}
    
    # Predict the output
    if st.button("Predict"):
        {prediction_statement}
        st.write("Predicted output:", prediction[0])
    
if __name__ == "__main__":
    predict()
"""

    # Write the code to a Python file
    with open("load_model.py", "w") as f:
        f.write(code.strip())  # Use .strip() to remove leading/trailing whitespace
