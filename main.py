import re  # Add this import for regular expressions
import streamlit as st
import pandas as pd
import io
import zipfile  # For creating ZIP files
import os  # For running commands
from data_handling import download_kaggle_dataset, generate_dataset_from_text
from preprocessing import preprocess_dataset
from model_training import train_models, save_best_model
from visualization import visualize_results
from utils import write_requirements_file, generate_loading_code
from visualization_nlp import visualize_nlp_results
import google.generativeai as genai
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import json

# # Load environment variables
# load_dotenv()

# # Configure the Google Gemini API key
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("Google API key not found! Please add it to the .env file.")

# Configure the Google Gemini API key through streamlit secrets
api_key = st.secrets("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found! Please add it to the .env file.")
    st.stop()

genai.configure(api_key=api_key)

# def load_kaggle_credentials():
#     try:
#         with open('kaggle.json', 'r') as f:
#             kaggle_api = json.load(f)
#             os.environ['KAGGLE_USERNAME'] = kaggle_api['username']
#             os.environ['KAGGLE_KEY'] = kaggle_api['key']
#     except FileNotFoundError:
#         st.error("Kaggle API key file not found! Please add [kaggle.json](http://_vscodecontentref_/7) to the project directory.")
#         st.stop()
#     except KeyError:
#         st.error("Invalid Kaggle API key file format! Please ensure it contains 'username' and 'key'.")
#         st.stop()

# # Load Kaggle credentials
# load_kaggle_credentials()

# Load Kaggle credentials from Streamlit secrets
def load_kaggle_credentials():
    try:
        kaggle_username = st.secrets["kaggle"]["username"]
        kaggle_key = st.secrets["kaggle"]["key"]
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
    except KeyError:
        st.error("Kaggle API key not found in Streamlit secrets! Please add it to the secrets.")
        st.stop()

# Load Kaggle credentials
load_kaggle_credentials()

# Create datasets directory if it doesn't exist
datasets_dir = 'datasets'
os.makedirs(datasets_dir, exist_ok=True)

# Set page config and custom CSS
st.set_page_config(page_title="AI-Generated Machine Learning System", layout="wide")
st.markdown("""<style>
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
        width: 100%;  /* Make buttons full width */
        height: 50px; /* Set height for uniformity */
        font-size: 16px; /* Adjust font size */
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        color: #4e54c8;
        background-color: #ffffff;
        border-radius: 10px;
    }
</style>""", unsafe_allow_html=True)

st.title("🎉 AI-Generated Machine Learning System")
st.markdown("""Welcome to the AI-Generated Machine Learning System! Here, you can generate datasets, train machine learning models, 
and get AI-powered explanations of the results. Please follow the instructions below to get started.""")

# Sidebar for inputs
st.sidebar.header("Project Configuration")

# File uploader for CSV files
uploaded_file = st.sidebar.file_uploader("Upload Your Own CSV File (optional)", type=["csv"])
text_prompt = st.sidebar.text_area("Describe your project requirements:", 
                                    placeholder="Type project description here...", height=150)
task_type = st.sidebar.selectbox("Select Task Type", options=["regression", "classification", "nlp"])

# Define words to ignore in prompt
def clean_prompt(text):
    ignore_words = ["give", "me", "project", "for"]
    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in ignore_words) + r')\b'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    return cleaned_text

# Use cleaned text in prompt
cleaned_text_prompt = clean_prompt(text_prompt)

# Initialize dataframe
df = None

# Create a row for the buttons
button_col1, button_col2 = st.sidebar.columns(2)

# Button to process project
if button_col1.button("Build Project"):
    if uploaded_file:
        # Remove old files in the datasets directory
        for file in os.listdir(datasets_dir):
            file_path = os.path.join(datasets_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Read the uploaded CSV file and save it to the datasets directory
        new_file_path = os.path.join(datasets_dir, uploaded_file.name)
        with open(new_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(new_file_path)
        st.success("CSV file uploaded and saved successfully.")
        st.subheader("Uploaded Dataset:")
        st.dataframe(df)
    elif cleaned_text_prompt:
        # Try to download dataset from Kaggle
        downloaded_file_name = download_kaggle_dataset(cleaned_text_prompt)
        if downloaded_file_name:
            st.success("Dataset found.")
            df = pd.read_csv(downloaded_file_name)
            st.subheader("Dataset:")
            st.dataframe(df)
        else:
            st.warning("No Kaggle dataset found, generating a new dataset...")
            generated_data = generate_dataset_from_text(cleaned_text_prompt)
            df = pd.read_csv(io.StringIO(generated_data))
            st.subheader("Generated Dataset:")
            st.dataframe(df)

    if df is not None:
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

                # Create a ZIP file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                    zipf.write("best_model.pkl")
                    zipf.write("load_model.py")
                    zipf.write("requirements.txt")

                zip_buffer.seek(0)  # Seek to the beginning of the BytesIO buffer

                st.success(f"Successfully trained {best_model_name} with a score of {best_score:.2f}. Model files created.")

                # Download button for the ZIP file
                st.download_button(label="Download Project", data=zip_buffer, file_name="project.zip", mime="application/zip")

                # Visualization and AI-powered explanation
                st.subheader("📊 Model Analysis with AI Explanations")
                if task_type == 'nlp':
                    visualize_nlp_results(y_test, y_pred, text_prompt)
                else:
                    visualize_results(task_type, y_test, y_pred, best_model, X_test, feature_names, text_prompt)

        except Exception as e:
            st.error(f"Error processing the dataset: {e}")

if button_col2.button("Customize"):
    # Run the command to execute the generate.py script
    os.system("streamlit run generate.py")
    st.success("Customize Dataset command executed.")

# Add a new button for data analysis
if button_col2.button("Data Analysis"):
    # Run the command to execute the app3.py script
    os.system("streamlit run app3.py")
    st.success("Data Analysis command executed.")

# Add a button for the chatbot in the previous column
if button_col1.button("Chatbot"):
    # Run the command to execute the chatbot.py script
    os.system("streamlit run chatbot.py")
    st.success("Chatbot command executed.")

# Add a footer
st.markdown("---")
st.markdown("Created with ❤️ by AI-Generated ML System")

# /workspaces/ai-model/app3.py