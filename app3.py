import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

from pandasai import SmartDataframe
from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Configure the Google Gemini API key
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     st.error("Google API key not found! Please add it to the .env file.")
#     st.stop()

# Load Google API key from Streamlit secrets
api_key = st.secrets["GOOGLE_API_KEY"]
if not api_key:
    st.error("Google API key not found! Please add it to the Streamlit secrets.")
    st.stop()

# Configure PandasAI API key
pandas_ai_key = st.secrets["PANDASAI_API_KEY"]
if not pandas_ai_key:
    st.error("""PandasAI API key not found! Please:
    1. Go to https://www.pandabi.ai and sign up
    2. From settings go to API keys and copy
    3. Create a .env file and add: PANDASAI_API_KEY=your_api_key_here""")
    st.stop()

genai.configure(api_key=api_key)
os.environ['PANDASAI_API_KEY'] = pandas_ai_key

def chat_with_csv(df, query):
    """Use PandasAI  to chat with CSV data."""
    try:
        os.environ["GOOGLE_API_KEY"] = api_key
        smart_df = SmartDataframe(df, config={"api_key": pandas_ai_key})
        result = smart_df.chat(query)

        if isinstance(result, (int, float)):
            return {"type": "response", "value": str(result)}
        return {"type": "response", "value": result}
        
    except Exception as e:
        return {"type": "error", "value": str(e)}

def check_image_generated(image_path='exports/charts/temp_chart.png'):
    """Check if an image has been generated."""
    return os.path.isfile(image_path)

def load_datasets_from_folder(folder_path='datasets'):
    """Load all CSV files from the specified folder."""
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

def main():
    # Set the title and layout of the Streamlit app
    st.set_page_config(
        page_title="Interactive Data Chat ",
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    st.title("💬 Interactive Data Chat")
    
    # Add description and styling
    st.markdown("""
    <style>
        .description {
            font-size: 20px;
            color: gray;
        }
        .header {
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subheader {
            font-size: 24px;
            color: #555;
        }
        .button {
            background-color: #4CAF50; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='header'>Unlock Insights from Your Data</div>", unsafe_allow_html=True)
    st.markdown("<div class='description'>This app allows you to interact with your CSV data in natural language. Ask questions and get answers powered by llama 3 and PandasAI!</div>", unsafe_allow_html=True)

    # Load CSV files from the datasets folder
    folder_path = 'datasets'
    if os.path.exists(folder_path):
        csv_files = load_datasets_from_folder(folder_path)

        if csv_files:
            selected_file = st.selectbox("Select a Dataset to Chat With", csv_files)
            selected_file_path = os.path.join(folder_path, selected_file)

            try:
                # Load and display the selected CSV file
                data = pd.read_csv(selected_file_path)

                st.subheader("📊 Data Preview")
                st.markdown("Here are the first 8 rows of the selected CSV file:")
                st.dataframe(data.head(8), use_container_width=True)

               
                # Query interface
                st.subheader("🤖 Chat with Your Data")
                input_text = st.text_area("Enter your query about the data:")
                st.caption("Ask questions in natural language, e.g., 'What is the average of column X?'")

                # Perform analysis
                if st.button("🔍 Analyze", type="primary"):
                    if input_text.strip():
                        with st.spinner("Analyzing your data..."):
                            if check_image_generated():
                                os.remove('exports/charts/temp_chart.png')
                            
                            result = chat_with_csv(data, input_text)
                            if result["type"] == "error":
                                st.error(f"Error processing your query: {result['value']}")
                            else:
                                st.success("Analysis Complete!")
                                st.write(result["value"])
                                
                                if check_image_generated():
                                    st.success("An image has been generated successfully!")
                                    st.image('exports/charts/temp_chart.png')
                    else:
                        st.warning("Please enter a valid query.")
                
            except Exception as e:
                st.error(f"Error reading the CSV file: {str(e)}")
        else:
            st.info("No CSV files found in the datasets folder.")
    else:
        st.error("The 'datasets' folder does not exist. Please create it and add CSV files.")

if __name__ == "__main__":
    main()
