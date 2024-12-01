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
    Unnamed: 0.1 = st.number_input('Enter value for Unnamed: 0.1')
    Unnamed: 0 = st.number_input('Enter value for Unnamed: 0')
    brand = st.number_input('Enter value for brand')
    name = st.number_input('Enter value for name')
    price = st.number_input('Enter value for price')
    spec_rating = st.number_input('Enter value for spec_rating')
    processor = st.number_input('Enter value for processor')
    CPU = st.number_input('Enter value for CPU')
    Ram = st.number_input('Enter value for Ram')
    Ram_type = st.number_input('Enter value for Ram_type')
    ROM = st.number_input('Enter value for ROM')
    ROM_type = st.number_input('Enter value for ROM_type')
    GPU = st.number_input('Enter value for GPU')
    display_size = st.number_input('Enter value for display_size')
    resolution_width = st.number_input('Enter value for resolution_width')
    resolution_height = st.number_input('Enter value for resolution_height')
    OS = st.number_input('Enter value for OS')
    
    # Predict the output
    if st.button("Predict"):
        prediction = model.predict([[Unnamed: 0.1, Unnamed: 0, brand, name, price, spec_rating, processor, CPU, Ram, Ram_type, ROM, ROM_type, GPU, display_size, resolution_width, resolution_height, OS]])
        st.write("Predicted output:", prediction[0])
    
if __name__ == "__main__":
    predict()