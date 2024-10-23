import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import io
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import subprocess
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats

# Configure API key securely
genai.configure(api_key="AIzaSyDoR10wPWSnCCLXHZWWrlrAg7XCXFzzpx8")
kaggle_api = KaggleApi()
kaggle_api.authenticate()  # Authenticate with Kaggle API using your credentials

# Set page config for a wider layout
st.set_page_config(page_title="AI-Generated ML System", layout="wide")

# Custom CSS to make the app more attractive
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

def download_kaggle_dataset(query):
    dataset_folder = "datasets"
    
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
        
        st.warning("No suitable datasets found on Kaggle. Falling back to generated data.")
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

def write_requirements_file():
    requirements = """
pickle
streamlit
scikit-learn
pandas
numpy
plotly
scipy
google-generativeai
kaggle
matplotlib
seaborn
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())

def create_venv():
    venv_folder = "venv"
    if not os.path.exists(venv_folder):
        st.text("Creating a virtual environment...")
        subprocess.run(["python", "-m", "venv", venv_folder])
        
        # Install required packages in the virtual environment
        subprocess.run([os.path.join(venv_folder, 'Scripts', 'pip'), 'install', '-r', 'requirements.txt'])

    return venv_folder

def preprocess_dataset(df, task_type):
    if df.empty:
        st.error("The DataFrame is empty after reading the CSV.")
        return None, None, None, None, None, None

    df.replace("None", pd.NA, inplace=True)

    # Separate features and target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)

    if task_type == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor, X.columns.tolist()

def train_models(X_train, y_train, X_test, y_test, task_type):
    models = {
        "Decision Tree": DecisionTreeClassifier() if task_type == 'classification' else DecisionTreeRegressor(),
        "Support Vector Machine": SVC(probability=True) if task_type == 'classification' else SVR(),
        "K-Nearest Neighbors": KNeighborsClassifier() if task_type == 'classification' else KNeighborsRegressor(),
        "Logistic Regression": LogisticRegression(max_iter=1000) if task_type == 'classification' else None,
    }

    best_model = None
    best_model_name = ""
    best_score = -float('inf')

    for model_name, model in models.items():
        if model:
            param_grid = {}
            if model_name == "Decision Tree":
                param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
            elif model_name == "Support Vector Machine":
                param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            elif model_name == "K-Nearest Neighbors":
                param_grid = {'n_neighbors': [3, 5, 7]}
            elif model_name == "Logistic Regression":
                param_grid = {'C': [0.1, 1, 10]}
            grid_search = GridSearchCV(model, param_grid, scoring='accuracy' if task_type == 'classification' else 'r2', cv=5)
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            score = accuracy_score(y_test, y_pred) if task_type == 'classification' else r2_score(y_test, y_pred)

            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = model_name

    return best_model, best_model_name, best_score, y_pred

def save_best_model(model):
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)

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

st.title("🎉 AI-Generated Machine Learning System")
st.markdown(""" 
Welcome to the AI-Generated Machine Learning System! Here, you can generate datasets, train machine learning models, and save the best model for future use. Please follow the instructions below to get started. 
""")

# Sidebar for inputs
st.sidebar.header("Project Configuration")
text_prompt = st.sidebar.text_area("Describe your project requirements:", height=150)
task_type = st.sidebar.selectbox("Select Task Type", options=["regression", "classification"])

# Button to generate project
if st.sidebar.button("Generate Project"):
    if text_prompt:
        # First, try to download dataset from Kaggle
        downloaded_file_name = download_kaggle_dataset(text_prompt)
        
        if downloaded_file_name:
            st.success("Dataset found on Kaggle.")
            # Read the downloaded dataset directly into a variable
            df = pd.read_csv(downloaded_file_name)
            st.subheader("Dataset:")
            st.dataframe(df)  # Display the DataFrame in a nice format

            try:
                # Proceed with the Kaggle dataset
                st.subheader("Preprocessing Data...")
                csv_data = df.to_csv(index=False)
                X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_dataset(df, task_type)

                if X_train is not None and X_test is not None:
                    st.subheader("Training Model...")
                    best_model, best_model_name, best_score, y_pred = train_models(X_train, y_train, X_test, y_test, task_type)
                    st.success(f"Best Model: {best_model_name} with Score: {best_score:.2f}")

                    # Save the model
                    save_best_model(best_model)

                    # Generate loading code
                    generate_loading_code("best_model.pkl", feature_names)

                    # Write requirements file
                    write_requirements_file()

                    # Display success message
                    st.success(f"Successfully trained {best_model_name} with a score of {best_score:.4f}. Model saved as 'best_model.pkl'. Loading code saved as 'load_model.py'. Requirements file created as 'requirements.txt'.")
                    
                    # Visualizations
                    st.subheader("📊 Model Performance Visualization")
                    
                    if task_type == 'classification':
                        col1, col2 = st.columns(2)
                        with col1:
                            # Confusion Matrix
                            cm = confusion_matrix(y_test, y_pred)
                            fig_cm = px.imshow(cm, text_auto=True, 
                                               labels=dict(x="Predicted", y="Actual"),
                                               x=['Class 0', 'Class 1'],
                                               y=['Class 0', 'Class 1'])
                            fig_cm.update_layout(title="Confusion Matrix")
                            st.plotly_chart(fig_cm)
                            if st.button("Explain Confusion Matrix"):
                                explanation = generate_graph_explanation("Confusion Matrix for classification model")
                                st.write(explanation)
                            
                            # ROC Curve
                            fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
                            fig_roc = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                              title=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})')
                            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                            st.plotly_chart(fig_roc)
                            if st.button("Explain ROC Curve"):
                                explanation = generate_graph_explanation("ROC Curve for classification model")
                                st.write(explanation)

                        with col2:
                            # Precision-Recall Curve
                            precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
                            fig_pr = px.line(x=recall, y=precision, labels={'x': 'Recall', 'y': 'Precision'},
                                             title=f'Precision-Recall Curve (AP = {average_precision_score(y_test, best_model.predict_proba(X_test)[:, 1]):.2f})')
                            st.plotly_chart(fig_pr)
                            if st.button("Explain Precision-Recall Curve"):
                                explanation = generate_graph_explanation("Precision-Recall Curve for classification model")
                                st.write(explanation)
                            
                            # Feature Importance
                            if hasattr(best_model, 'feature_importances_'):
                                importances = best_model.feature_importances_
                                indices = np.argsort(importances)[::-1]
                                fig_importance = px.bar(x=importances[indices],
                                                        y=[feature_names[i] for i in indices],
                                                        labels={'x': 'Importance', 'y': 'Feature'},
                                                        title='Feature Importance',
                                                        orientation='h')
                                st.plotly_chart(fig_importance)
                                if st.button("Explain Feature Importance"):
                                    explanation = generate_graph_explanation("Feature Importance graph for classification model")
                                    st.write(explanation)
                            else:
                                try:
                                    result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=0)
                                    sorted_idx = result.importances_mean.argsort()[::-1]
                                    limited_idx = sorted_idx[sorted_idx < len(feature_names)]
                                    if limited_idx.size > 0:
                                        fig_importance = px.bar(x=result.importances_mean[limited_idx],
                                                                y=[feature_names[i] for i in limited_idx],
                                                                labels={'x': 'Importance', 'y': 'Feature'},
                                                                title='Feature Importance (Permutation)',
                                                                orientation='h')
                                        st.plotly_chart(fig_importance)
                                        if st.button("Explain Feature Importance"):
                                            explanation = generate_graph_explanation("Feature Importance (Permutation) graph for classification model")
                                            st.write(explanation)
                                    else:
                                        st.write("No valid features for importance calculation.")
                                except Exception as e:
                                    st.write("Error calculating feature importance:", e)

                        # Classification Report
                        report = classification_report(y_test, y_pred, output_dict=True)
                        df_report = pd.DataFrame(report).transpose()
                        st.table(df_report)
                        if st.button("Explain Classification Report"):
                            explanation = generate_graph_explanation("Classification Report table for the model")
                            st.write(explanation)

                    else:  # Regression
                        col1, col2 = st.columns(2)
                        with col1:
                            # Actual vs Predicted
                            fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
                            fig_scatter.update_layout(title="Actual vs Predicted Values")
                            st.plotly_chart(fig_scatter)
                            if st.button("Explain Actual vs Predicted"):
                                explanation = generate_graph_explanation("Actual vs Predicted Values scatter plot for regression model")
                                st.write(explanation)
                            
                            # Residual plot
                            residuals = y_test - y_pred
                            fig_residuals = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'})
                            fig_residuals.update_layout(title="Residual Plot")
                            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_residuals)
                            if st.button("Explain Residual Plot"):
                                explanation = generate_graph_explanation("Residual Plot for regression model")
                                st.write(explanation)

                        with col2:
                            # Q-Q Plot
                            fig_qq = px.scatter(x=stats.probplot(residuals, dist="norm")[0][0],
                                                y=stats.probplot(residuals, dist="norm")[0][1],
                                                labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                                                title='Q-Q Plot')
                            fig_qq.add_shape(type='line', line=dict(dash='dash'), x0=fig_qq.data[0].x.min(), x1=fig_qq.data[0].x.max(),
                                             y0=fig_qq.data[0].y.min(), y1=fig_qq.data[0].y.max())
                            st.plotly_chart(fig_qq)
                            if st.button("Explain Q-Q Plot"):
                                explanation = generate_graph_explanation("Q-Q Plot for regression model")
                                st.write(explanation)
                            
                            # Prediction Error Distribution
                            errors = y_test - y_pred
                            fig_error_dist = px.histogram(errors, nbins=30, labels={'value': 'Prediction Error'},
                                                          title='Prediction Error Distribution')
                            fig_error_dist.add_vline(x=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_error_dist)
                            if st.button("Explain Prediction Error Distribution"):
                                explanation = generate_graph_explanation("Prediction Error Distribution histogram for regression model")
                                st.write(explanation)

                        # Regression Metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"Root Mean Squared Error: {rmse:.4f}")
                        st.write(f"R-squared Score: {r2:.4f}")
                        if st.button("Explain Regression Metrics"):
                            explanation = generate_graph_explanation(f"Regression metrics: MSE={mse:.4f}, RMSE={rmse:.4f}, R-squared={r2:.4f}")
                            st.write(explanation)

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
                st.error(f"Error processing the Kaggle dataset: {e}")
                st.info("Falling back to generated data...")
                # Generate a new dataset
                generated_data = generate_dataset_from_text(text_prompt)
                generated_df = pd.read_csv(io.StringIO(generated_data))
                try:
                    # Preprocess the generated dataset
                    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_dataset(generated_df, task_type)

                    if X_train is not None and X_test is not None:
                        st.subheader("Training Models...")
                        best_model, best_model_name, best_score, y_pred = train_models(X_train, y_train, X_test, y_test, task_type)
                        st.success(f"Best Model: {best_model_name} with Score: {best_score:.2f}")

                        save_best_model(best_model)
                        st.success("Model saved successfully!")

                        # Generate loading code
                        generate_loading_code("best_model.pkl", feature_names)
                        st.success("Loading code generated successfully!")

                       # Visualizations
                        st.subheader("📊 Model Performance Visualization")
            
                        if task_type == 'classification':
                         col1, col2 = st.columns(2)
                        with col1:
                    # Confusion Matrix
                         cm = confusion_matrix(y_test, y_pred)
                         fig_cm = px.imshow(cm, text_auto=True, 
                                       labels=dict(x="Predicted", y="Actual"),
                                       x=['Class 0', 'Class 1'],
                                       y=['Class 0', 'Class 1'])
                    fig_cm.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig_cm)
                    if st.button("Explain Confusion Matrix"):
                        generate_and_cache_explanation("Confusion Matrix")
                    if "explanation_Confusion Matrix" in st.session_state:
                        st.write(st.session_state["explanation_Confusion Matrix"])
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
                    fig_roc = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                      title=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})')
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc)
                    if st.button("Explain ROC Curve"):
                        generate_and_cache_explanation("ROC Curve")
                    if "explanation_ROC Curve" in st.session_state:
                        st.write(st.session_state["explanation_ROC Curve"])

                    with col2:
                    # Precision-Recall Curve
                     precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
                     fig_pr = px.line(x=recall, y=precision, labels={'x': 'Recall', 'y': 'Precision'},
                                     title=f'Precision-Recall Curve (AP = {average_precision_score(y_test, best_model.predict_proba(X_test)[:, 1]):.2f})')
                     st.plotly_chart(fig_pr)
                    if st.button("Explain Precision-Recall Curve"):
                        generate_and_cache_explanation("Precision-Recall Curve")
                    if "explanation_Precision-Recall Curve" in st.session_state:
                        st.write(st.session_state["explanation_Precision-Recall Curve"])
                    
                    # Feature Importance
                    if hasattr(best_model, 'feature_importances_'):
                        importances = best_model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        fig_importance = px.bar(x=importances[indices],
                                                y=[feature_names[i] for i in indices],
                                                labels={'x': 'Importance', 'y': 'Feature'},
                                                title='Feature Importance',
                                                orientation='h')
                        st.plotly_chart(fig_importance)
                        if st.button("Explain Feature Importance"):
                            generate_and_cache_explanation("Feature Importance")
                        if "explanation_Feature Importance" in st.session_state:
                            st.write(st.session_state["explanation_Feature Importance"])
                    else:
                        try:
                            result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=0)
                            sorted_idx = result.importances_mean.argsort()[::-1]
                            limited_idx = sorted_idx[sorted_idx < len(feature_names)]
                            if limited_idx.size > 0:
                                fig_importance = px.bar(x=result.importances_mean[limited_idx],
                                                        y=[feature_names[i] for i in limited_idx],
                                                        labels={'x': 'Importance', 'y': 'Feature'},
                                                        title='Feature Importance (Permutation)',
                                                        orientation='h')
                                st.plotly_chart(fig_importance)
                                if st.button("Explain Feature Importance (Permutation)"):
                                    generate_and_cache_explanation("Feature Importance (Permutation)")
                                if "explanation_Feature Importance (Permutation)" in st.session_state:
                                    st.write(st.session_state["explanation_Feature Importance (Permutation)"])
                            else:
                                st.write("No valid features for importance calculation.")
                        except Exception as e:
                            st.write("Error calculating feature importance:", e)

                # Classification Report
                            report = classification_report(y_test, y_pred, output_dict=True)
                            df_report = pd.DataFrame(report).transpose()
                            st.table(df_report)
                            if st.button("Explain Classification Report"):
                              generate_and_cache_explanation("Classification Report")
                            if "explanation_Classification Report" in st.session_state:
                               st.write(st.session_state["explanation_Classification Report"])

                        else:  # Regression
                          col1, col2 = st.columns(2)
                        with col1:
                    # Actual vs Predicted
                         fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
                         fig_scatter.update_layout(title="Actual vs Predicted Values")
                         st.plotly_chart(fig_scatter)
                        if st.button("Explain Actual vs Predicted"):
                          generate_and_cache_explanation("Actual vs Predicted")
                        if "explanation_Actual vs Predicted" in st.session_state:
                          st.write(st.session_state["explanation_Actual vs Predicted"])
                    
                    # Residual plot
                    residuals = y_test - y_pred
                    fig_residuals = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'})
                    fig_residuals.update_layout(title="Residual Plot")
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_residuals)
                    if st.button("Explain Residual Plot"):
                        generate_and_cache_explanation("Residual Plot")
                    if "explanation_Residual Plot" in st.session_state:
                        st.write(st.session_state["explanation_Residual Plot"])

                        with col2:
                    # Q-Q Plot
                         fig_qq = px.scatter(x=stats.probplot(residuals, dist="norm")[0][0],
                                        y=stats.probplot(residuals, dist="norm")[0][1],
                                        labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                                        title='Q-Q Plot')
                         fig_qq.add_shape(type='line', line=dict(dash='dash'), x0=fig_qq.data[0].x.min(), x1=fig_qq.data[0].x.max(),
                                     y0=fig_qq.data[0].y.min(), y1=fig_qq.data[0].y.max())
                    st.plotly_chart(fig_qq)
                    if st.button("Explain Q-Q Plot"):
                        generate_and_cache_explanation("Q-Q Plot")
                    if "explanation_Q-Q Plot" in st.session_state:
                        st.write(st.session_state["explanation_Q-Q Plot"])
                    
                    # Prediction Error Distribution
                    errors = y_test - y_pred
                    fig_error_dist = px.histogram(errors, nbins=30, labels={'value': 'Prediction Error'},
                                                  title='Prediction Error Distribution')
                    fig_error_dist.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_error_dist)
                    if st.button("Explain Prediction Error Distribution"):
                        generate_and_cache_explanation("Prediction Error Distribution")
                    if "explanation_Prediction Error Distribution" in st.session_state:
                        st.write(st.session_state["explanation_Prediction Error Distribution"])

                # Regression Metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                
                    st.write(f"Mean Squared Error: {mse:.4f}")
                    st.write(f"Root Mean Squared Error: {rmse:.4f}")
                    st.write(f"R-squared Score: {r2:.4f}")
                    if st.button("Explain Regression Metrics"):
                     generate_and_cache_explanation("Regression Metrics")
                    if "explanation_Regression Metrics" in st.session_state:
                     st.write(st.session_state["explanation_Regression Metrics"])

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

        else:
            # If no dataset was found, generate a new dataset
            st.warning("No Kaggle dataset found, generating a new dataset...")
            generated_data = generate_dataset_from_text(text_prompt)
            generated_df = pd.read_csv(io.StringIO(generated_data))
            try:
                # Preprocess the generated dataset
                X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_dataset(generated_df, task_type)

                if X_train is not None and X_test is not None:
                    st.subheader("Training Models...")
                    best_model, best_model_name, best_score, y_pred = train_models(X_train, y_train, X_test, y_test, task_type)
                    st.success(f"Best Model: {best_model_name} with Score: {best_score:.2f}")

                    save_best_model(best_model)
                    st.success("Model saved successfully!")

                    # Generate loading code
                    generate_loading_code("best_model.pkl", feature_names)
                    st.success("Loading code generated successfully!")
                     
                    # Visualizations
                    st.subheader("📊 Model Performance Visualization")
            
                if task_type == 'classification':
                 col1, col2 = st.columns(2)
                with col1:
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, 
                                       labels=dict(x="Predicted", y="Actual"),
                                       x=['Class 0', 'Class 1'],
                                       y=['Class 0', 'Class 1'])
                    fig_cm.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig_cm)
                    if st.button("Explain Confusion Matrix"):
                        generate_and_cache_explanation("Confusion Matrix")
                    if "explanation_Confusion Matrix" in st.session_state:
                        st.write(st.session_state["explanation_Confusion Matrix"])
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
                    fig_roc = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                                      title=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})')
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc)
                    if st.button("Explain ROC Curve"):
                        generate_and_cache_explanation("ROC Curve")
                    if "explanation_ROC Curve" in st.session_state:
                        st.write(st.session_state["explanation_ROC Curve"])

                with col2:
                    # Precision-Recall Curve
                    precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
                    fig_pr = px.line(x=recall, y=precision, labels={'x': 'Recall', 'y': 'Precision'},
                                     title=f'Precision-Recall Curve (AP = {average_precision_score(y_test, best_model.predict_proba(X_test)[:, 1]):.2f})')
                    st.plotly_chart(fig_pr)
                    if st.button("Explain Precision-Recall Curve"):
                        generate_and_cache_explanation("Precision-Recall Curve")
                    if "explanation_Precision-Recall Curve" in st.session_state:
                        st.write(st.session_state["explanation_Precision-Recall Curve"])
                    
                    # Feature Importance
                    if hasattr(best_model, 'feature_importances_'):
                        importances = best_model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        fig_importance = px.bar(x=importances[indices],
                                                y=[feature_names[i] for i in indices],
                                                labels={'x': 'Importance', 'y': 'Feature'},
                                                title='Feature Importance',
                                                orientation='h')
                        st.plotly_chart(fig_importance)
                       
                    else:
                        try:
                            result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=0)
                            sorted_idx = result.importances_mean.argsort()[::-1]
                            limited_idx = sorted_idx[sorted_idx < len(feature_names)]
                            if limited_idx.size > 0:
                                fig_importance = px.bar(x=result.importances_mean[limited_idx],
                                                        y=[feature_names[i] for i in limited_idx],
                                                        labels={'x': 'Importance', 'y': 'Feature'},
                                                        title='Feature Importance (Permutation)',
                                                        orientation='h')
                                st.plotly_chart(fig_importance)
                                
                            else:
                                st.write("No valid features for importance calculation.")
                        except Exception as e:
                            st.write("Error calculating feature importance:", e)

                # Classification Report
                report = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.table(df_report)
                if st.button("Explain Classification Report"):
                    generate_and_cache_explanation("Classification Report")
                if "explanation_Classification Report" in st.session_state:
                    st.write(st.session_state["explanation_Classification Report"])

                else:  # Regression
                 col1, col2 = st.columns(2)
                with col1:
                    # Actual vs Predicted
                    fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
                    fig_scatter.update_layout(title="Actual vs Predicted Values")
                    st.plotly_chart(fig_scatter)
                    if st.button("Explain Actual vs Predicted"):
                        generate_and_cache_explanation("Actual vs Predicted")
                    if "explanation_Actual vs Predicted" in st.session_state:
                        st.write(st.session_state["explanation_Actual vs Predicted"])
                    
                    # Residual plot
                    residuals = y_test - y_pred
                    fig_residuals = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'})
                    fig_residuals.update_layout(title="Residual Plot")
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_residuals)
                    if st.button("Explain Residual Plot"):
                        generate_and_cache_explanation("Residual Plot")
                    if "explanation_Residual Plot" in st.session_state:
                        st.write(st.session_state["explanation_Residual Plot"])

                with col2:
                    # Q-Q Plot
                    fig_qq = px.scatter(x=stats.probplot(residuals, dist="norm")[0][0],
                                        y=stats.probplot(residuals, dist="norm")[0][1],
                                        labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                                        title='Q-Q Plot')
                    fig_qq.add_shape(type='line', line=dict(dash='dash'), x0=fig_qq.data[0].x.min(), x1=fig_qq.data[0].x.max(),
                                     y0=fig_qq.data[0].y.min(), y1=fig_qq.data[0].y.max())
                    st.plotly_chart(fig_qq)
                    if st.button("Explain Q-Q Plot"):
                        generate_and_cache_explanation("Q-Q Plot")
                    if "explanation_Q-Q Plot" in st.session_state:
                        st.write(st.session_state["explanation_Q-Q Plot"])
                    
                    # Prediction Error Distribution
                    errors = y_test - y_pred
                    fig_error_dist = px.histogram(errors, nbins=30, labels={'value': 'Prediction Error'},
                                                  title='Prediction Error Distribution')
                    fig_error_dist.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_error_dist)
                    if st.button("Explain Prediction Error Distribution"):
                        generate_and_cache_explanation("Prediction Error Distribution")
                    if "explanation_Prediction Error Distribution" in st.session_state:
                        st.write(st.session_state["explanation_Prediction Error Distribution"])

                # Regression Metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"Root Mean Squared Error: {rmse:.4f}")
                st.write(f"R-squared Score: {r2:.4f}")
                if st.button("Explain Regression Metrics"):
                    generate_and_cache_explanation("Regression Metrics")
                if "explanation_Regression Metrics" in st.session_state:
                    st.write(st.session_state["explanation_Regression Metrics"])

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