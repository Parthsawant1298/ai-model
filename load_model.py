import pickle
import streamlit as st

# Load the model from file
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Streamlit UI for predictions
def predict():
    st.title("Model Prediction App")
    
    model = load_model()
    
    # Create input fields for each feature
    Brand = st.number_input('Enter value for Brand')
    Processor_Speed = st.number_input('Enter value for Processor_Speed')
    RAM_Size = st.number_input('Enter value for RAM_Size')
    Storage_Capacity = st.number_input('Enter value for Storage_Capacity')
    Screen_Size = st.number_input('Enter value for Screen_Size')
    Weight = st.number_input('Enter value for Weight')
    
    # Predict the output
    if st.button("Predict"):
        prediction = model.predict([[Brand, Processor_Speed, RAM_Size, Storage_Capacity, Screen_Size, Weight]])
        st.write("Predicted output:", prediction[0])
    
if __name__ == "__main__":
    predict()