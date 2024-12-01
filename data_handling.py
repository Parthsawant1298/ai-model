import os
import streamlit as st
import pandas as pd
import io
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import google.generativeai as genai
import json

# def load_kaggle_credentials():
#     try:
#         with open('kaggle.json', 'r') as f:
#             kaggle_api = json.load(f)
#             os.environ['KAGGLE_USERNAME'] = kaggle_api['username']
#             os.environ['KAGGLE_KEY'] = kaggle_api['key']
#     except FileNotFoundError:
#         st.error("Kaggle API key file not found! Please add kaggle.json to the project directory.")
#         st.stop()
#     except KeyError:
#         st.error("Invalid Kaggle API key file format! Please ensure it contains 'username' and 'key'.")
#         st.stop()


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


def download_kaggle_dataset(query):
    dataset_folder = "datasets"
    load_kaggle_credentials()
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    
    try:
        # Clear the dataset folder before downloading the new dataset
        if os.path.exists(dataset_folder):
            for file_name in os.listdir(dataset_folder):
                file_path = os.path.join(dataset_folder, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove directory
                except Exception as e:
                    st.error(f"Error deleting file {file_name}: {e}")

        # Search for datasets matching the query
        datasets = kaggle_api.dataset_list(search=query)
        if datasets:
            dataset = datasets[0]  # Get the first result
            dataset_name = dataset.ref  # Dataset reference
            
            # Download the dataset files to the dataset folder
            kaggle_api.dataset_download_files(dataset_name, path=dataset_folder, unzip=True)
            
            # List the files in the dataset folder to find the actual downloaded file
            downloaded_files = os.listdir(dataset_folder)
            
            # Return the first CSV file found
            for file in downloaded_files:
                if file.endswith('.csv'):
                    return os.path.join(dataset_folder, file)  # Return the full path to the CSV file
        
        return None
    except Exception as e:
        st.error(f"Error searching for Kaggle datasets: {e}")
        return None

def generate_dataset_from_text(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([ 
        f"Generate a dataset in CSV format based on the following text without explanation, "
        f"just data and I want 200 rows and 5 columns, avoid repeating data both numeric as well as "
        f"categorical also strictly don't give ''' csv ''' or '''  ''' with dataset : {text}."
    ])
    return response.text