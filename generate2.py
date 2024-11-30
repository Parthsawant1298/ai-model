import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import io

# Configure API key directly
genai.configure(api_key="AIzaSyDoR10wPWSnCCLXHZWWrlrAg7XCXFzzpx8")

# Function to generate JSON dataset based on user prompt
def generate_json_dataset_from_prompt(prompt_text):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    prompt = f"Generate a dataset in JSON format based on the following prompt:\n\n{prompt_text}\n\nReturn JSON data only, no explanation."
    response = model.generate_content(prompt)
    return response.text

# Function to convert JSON data to CSV
def json_to_csv(json_data):
    data = json.loads(json_data)  # Convert JSON text to Python list of dicts
    df = pd.DataFrame(data)  # Create DataFrame from list of dicts
    csv_data = df.to_csv(index=False)  # Convert DataFrame to CSV
    return csv_data

# Function to save the generated dataset in the "datasets" folder
def save_dataset_to_folder(csv_data, file_name="generated_dataset.csv"):
    dataset_folder = "datasets"
    
    # Create the dataset folder if it doesn't exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Define the full path to save the CSV file
    file_path = os.path.join(dataset_folder, file_name)
    
    # Save the CSV data to the file
    with open(file_path, 'w') as file:
        file.write(csv_data)
    
    return file_path

# Streamlit App
st.title("🗂️ Custom Dataset Generator from Prompt")

# Input field for user-defined prompt
user_prompt = st.text_area("Enter a prompt to generate a dataset:", height=150)

if st.button("Generate Dataset"):
    if user_prompt:
        try:
            # Generate JSON dataset from the user prompt
            json_data = generate_json_dataset_from_prompt(user_prompt)
            
            # Convert JSON data to CSV format
            csv_data = json_to_csv(json_data)
            
            # Try to convert CSV data into a DataFrame for display
            try:
                df = pd.read_csv(io.StringIO(csv_data))
                st.subheader("Preview of Generated Dataset:")
                st.dataframe(df)

                # Save the dataset to the "datasets" folder
                saved_file_path = save_dataset_to_folder(csv_data)

                # Provide download options for the generated CSV
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name='generated_dataset.csv',
                    mime='text/csv'
                )
                st.success(f"CSV file saved to {saved_file_path} and ready for download!")
            except pd.errors.ParserError as e:
                st.error(f"Error parsing CSV data: {e}. The JSON might still be incorrectly formatted.")
            except Exception as e:
                st.error(f"Unexpected error while processing CSV: {e}")

        except Exception as e:
            st.error(f"Error generating dataset: {e}")
    else:
        st.warning("Please provide a text prompt.")
