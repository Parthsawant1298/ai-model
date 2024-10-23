import streamlit as st
import pandas as pd
import io
from data_handling import download_kaggle_dataset, generate_dataset_from_text
from preprocessing import preprocess_dataset
from model_training import train_models, save_best_model
from visualization import visualize_results
from utils import write_requirements_file, generate_loading_code

# Configure API keys securely
import google.generativeai as genai
from kaggle.api.kaggle_api_extended import KaggleApi

# Set up the Generative AI API
genai.configure(api_key="AIzaSyDoR10wPWSnCCLXHZWWrlrAg7XCXFzzpx8")  # Replace with your API key
kaggle_api = KaggleApi()
kaggle_api.authenticate()

# Set page config and custom CSS
st.set_page_config(page_title="AI-Generated ML System", layout="wide")
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #4e54c8, #8f94fb);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #4e54c8, #8f94fb);
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stButton>button {
        color: #4e54c8;
        background-color: #ffffff;
        border-radius: 20px;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        color: #4e54c8;
        background-color: #ffffff;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎉 AI-Generated Machine Learning System")
st.markdown("""
Welcome to the AI-Generated Machine Learning System! Here, you can generate datasets, train machine learning models, 
and get AI-powered explanations of the results. Please follow the instructions below to get started.
""")

# Sidebar for inputs
st.sidebar.header("Project Configuration")
text_prompt = st.sidebar.text_area("Describe your project requirements:", 
                                  placeholder="",
                                  height=150)
task_type = st.sidebar.selectbox("Select Task Type", options=["regression", "classification"])

# Button to generate project
if st.sidebar.button("Generate Project"):
    if text_prompt:
        # First, try to download dataset from Kaggle
        downloaded_file_name = download_kaggle_dataset(text_prompt)
        
        if downloaded_file_name:
            st.success("Dataset found on Kaggle.")
            df = pd.read_csv(downloaded_file_name)
            st.subheader("Dataset:")
            st.dataframe(df)
        else:
            st.warning("No Kaggle dataset found, generating a new dataset...")
            generated_data = generate_dataset_from_text(text_prompt)
            df = pd.read_csv(io.StringIO(generated_data))
            st.subheader("Generated Dataset:")
            st.dataframe(df)

        try:
            st.subheader("Preprocessing Data...")
            X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_dataset(df, task_type)

            if X_train is not None and X_test is not None:
                st.subheader("Training Model...")
                best_model, best_model_name, best_score, y_pred = train_models(X_train, y_train, X_test, y_test, task_type)
                
                # Save model and generate necessary files
                save_best_model(best_model)
                generate_loading_code("best_model.pkl", feature_names)
                write_requirements_file()

                st.success(f"Successfully trained {best_model_name} with a score of {best_score:.4f}. Model saved as 'best_model.pkl'. Loading code saved as 'load_model.py'. Requirements file created as 'requirements.txt'.")

                # Visualization and AI-powered explanation
                st.subheader("📊 Model Analysis with AI Explanations")
                visualize_results(task_type, y_test, y_pred, best_model, X_test, feature_names, text_prompt)

                # Download buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                          with open("best_model.pkl", "rb") as f:
                            st.download_button(label="Download Best Model (best_model.pkl)", data=f, file_name="best_model.pkl")
                with col2:
                      with open("load_model.py", "rb") as f:
                       st.download_button(label="Download Loading Code (load_model.py)", data=f, file_name="load_model.py")
                with col3:
                      with open("requirements.txt", "rb") as f:
                       st.download_button(label="Download Requirements (requirements.txt)", data=f, file_name="requirements.txt")

        except Exception as e:
            st.error(f"Error processing the generated dataset: {e}")

# Add a footer
st.markdown("---")
st.markdown("Created with ❤️ by AI-Generated ML System")