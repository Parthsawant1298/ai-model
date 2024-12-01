import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import io
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     st.error("Google API key not found! Please add it to the .env file.")
#     st.stop()

# Load Google API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]
if not api_key:
    st.error("Google API key not found! Please add it to the Streamlit secrets.")
    st.stop()

genai.configure(api_key=api_key)

# Function to generate dataset based on a text prompt using Google Generative AI
def generate_dataset_from_text(text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([
        "Generate a dataset in CSV format based on the following text with no explanation, just data. Each row should have the same number of values as the header and also ''' csv ''' or '''  ''' should strictly not be there in data.:", text
    ])
    return response.text

# Function to clean CSV data for parsing
def clean_csv_data(csv_data):
    lines = csv_data.splitlines()
    cleaned_lines = []
    
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:  # Ensure the line is not empty
            cleaned_lines.append(cleaned_line)
    
    return "\n".join(cleaned_lines)

# Enhanced validation to give better feedback
def validate_csv_format(csv_data):
    lines = csv_data.splitlines()
    if not lines:
        return False, "The CSV data is empty."
    
    header_columns = len(lines[0].split(','))
    errors = []
    
    for i, line in enumerate(lines[1:], start=2):  # Starting from line 2 (after header)
        if len(line.split(',')) != header_columns:
            errors.append(f"Line {i} has {len(line.split(','))} columns (expected {header_columns}).")
    
    if errors:
        return False, "\n".join(errors)
    
    return True, None

# Function to save the generated dataset in the "datasets" folder
def save_dataset_to_folder(csv_data, file_name="generated_dataset.csv"):
    dataset_folder = "datasets"
    
    # Create the dataset folder if it doesn't exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Remove any existing files in the dataset folder
    for file in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Define the full path to save the CSV file
    file_path = os.path.join(dataset_folder, file_name)
    
    # Save the new CSV data to the file
    with open(file_path, 'w') as file:
        file.write(csv_data)
    
    return file_path

# Streamlit App
st.title("🗂️ Dataset Generator from Text")

# Input field for text prompt
text_prompt = st.text_area("Enter a prompt to generate a dataset:", height=150)

if st.button("Generate Dataset"):
    if text_prompt:
        try:
            # Generate dataset from the text prompt
            csv_data = generate_dataset_from_text(text_prompt)
           
            
            # Clean the CSV data
            cleaned_csv_data = clean_csv_data(csv_data)
            
            # Validate the cleaned CSV format
            is_valid, error_message = validate_csv_format(cleaned_csv_data)
            
            if not is_valid:
                st.error(f"The cleaned CSV data is incorrectly formatted. Error:\n{error_message}")
            else:
                # Try to convert CSV data into a DataFrame
                try:
                    df = pd.read_csv(io.StringIO(cleaned_csv_data))
                    st.subheader("Preview of Generated Dataset:")
                    st.dataframe(df)

                    # Save the dataset to the "datasets" folder
                    saved_file_path = save_dataset_to_folder(cleaned_csv_data)

                    # Provide download options for the generated CSV
                    st.download_button(
                        label="Download CSV",
                        data=cleaned_csv_data,
                        file_name='generated_dataset.csv',
                        mime='text/csv'
                    )
                    st.success(f"CSV file saved to {saved_file_path} and ready for download!")
                except pd.errors.ParserError as e:
                    st.error(f"Error parsing CSV data: {e}. The cleaned CSV might still be incorrectly formatted.")
                except Exception as e:
                    st.error(f"Unexpected error while processing CSV: {e}")

        except Exception as e:
            st.error(f"Error generating dataset: {e}")
    else:
        st.warning("Please provide a text prompt.") 