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
    input_statements = "\n".join([f"{name} = float(input('Enter value for {name}: '))" for name in feature_names])
    prediction_statement = f"prediction = model.predict([[{', '.join(['str(' + name + ')' for name in feature_names])}]])"
    code = f"""
import pickle

def load_model():
    with open('{filename}', 'rb') as f:
        model = pickle.load(f)
    return model

def predict():
    model = load_model()
    
    {input_statements}
    {prediction_statement}
    
    print("Predicted output:", prediction)

if __name__ == "__main__":
    predict()
"""
    
    with open("load_model.py", "w") as f:
        f.write(code)

def generate_graph_explanation(graph_description):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([ 
        f"Explain the following graph visualization in detail: {graph_description}. "
        f"Include information about what insights can be drawn from the graph, "
        f"how the model's performance or results are reflected in the graph, and "
        f"what specific features or patterns the graph highlights."
    ])
    return response.text
