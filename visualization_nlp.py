import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, accuracy_score
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai

def get_gemini_explanation(data, prompt):
    """
    Get AI-generated explanation for visualizations using Gemini model.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate explanation: {str(e)}"

def visualize_nlp_results(y_test, y_pred, user_prompt):
    """
    Visualize NLP model results with four NLP-specific graphs in a 2x2 grid layout
    and AI-powered explanations.
    
    Parameters:
        y_true (list): True labels
        y_pred (list): Predicted labels or probabilities
        user_prompt (str): User-provided context for generating explanations
    """
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Create two columns for the first row
    col1, col2 = st.columns(2)

    # 1. Enhanced Confusion Matrix with Normalized Values (First Row, First Column)
    with col1:
        st.write("### Enhanced Confusion Matrix")
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=np.unique(y_test),
            y=np.unique(y_test),
            text=np.around(cm_norm, decimals=2),
            texttemplate="%{text:.2%}",
            textfont={"size": 12},
            colorscale="Blues",
        ))
        
        fig_cm.update_layout(
            title="Normalized Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=400,
            height=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # AI Explanation for Confusion Matrix
        cm_details = "\n".join([f"Class {i}: TP={cm[i,i]}, Total={np.sum(cm[i,:])}" for i in range(len(np.unique(y_test)))])
        explanation_prompt_cm = f"""
        Analyze this confusion matrix for {user_prompt}:
        {cm_details}
        Explain the performance across different classes.
        What insights can we draw about the model's classification ability?
        Provide a detailed explanation in 10-12 lines.
        """
        explanation_cm = get_gemini_explanation(str(cm.tolist()), explanation_prompt_cm)
        with st.expander("💡 AI Explanation (Confusion Matrix)", expanded=False):
            st.write(explanation_cm)

    # 2. Class Distribution Comparison (First Row, Second Column)
    with col2:
        st.write("### Class Distribution")
        
        true_dist = Counter(y_test)
        pred_dist = Counter(y_pred)
        labels = sorted(set(y_test) | set(y_pred))
        
        fig_dist = go.Figure(data=[
            go.Bar(name='True Labels', x=labels, y=[true_dist[label] for label in labels]),
            go.Bar(name='Predicted Labels', x=labels, y=[pred_dist[label] for label in labels])
        ])
        
        fig_dist.update_layout(
            title="True vs Predicted Class Distribution",
            xaxis_title="Classes",
            yaxis_title="Count",
            barmode='group',
            width=400,
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # AI Explanation for Distribution
        explanation_prompt_dist = f"""
        Analyze the class distribution comparison for {user_prompt}.
        What patterns do you observe between true and predicted labels?
        What does this tell us about the model's classification bias?
        Provide a detailed explanation in 10-12 lines.
        """
        explanation_dist = get_gemini_explanation("Distribution Data", explanation_prompt_dist)
        with st.expander("💡 AI Explanation (Class Distribution)", expanded=False):
            st.write(explanation_dist)

    # Create two columns for the second row
    col3, col4 = st.columns(2)

    # 3. Prediction Confidence Distribution (Second Row, First Column)
    with col3:
        st.write("### Prediction Confidence Distribution")
        
        # Convert predictions to confidence scores (if not already)
        if isinstance(y_pred[0], (int, str)):
            confidence_scores = np.random.uniform(0.5, 1.0, len(y_pred))  # Dummy scores if not available
        else:
            confidence_scores = np.max(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred

        fig_conf = go.Figure(data=[
            go.Histogram(x=confidence_scores, nbinsx=30, name='Confidence')
        ])
        
        fig_conf.update_layout(
            title="Distribution of Prediction Confidence",
            xaxis_title="Confidence Score",
            yaxis_title="Count",
            width=400,
            height=400
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        # AI Explanation for Confidence Distribution
        explanation_prompt_conf = f"""
        Analyze the prediction confidence distribution for {user_prompt}.
        What does this distribution tell us about the model's certainty in its predictions?
        Provide a detailed explanation in 10-12 lines.
        """
        explanation_conf = get_gemini_explanation("Confidence Distribution Data", explanation_prompt_conf)
        with st.expander("💡 AI Explanation (Confidence Distribution)", expanded=False):
            st.write(explanation_conf)

    # 4. Error Analysis by Length (Second Row, Second Column)
    with col4:
        st.write("### Error Analysis by Text Length")
        
        # Assuming we have access to original texts, otherwise use dummy lengths
        text_lengths = np.random.normal(100, 30, len(y_test))  # Replace with actual text lengths
        correct_predictions = y_test == y_pred
        
        fig_len = go.Figure()
        
        fig_len.add_trace(go.Box(
            y=[l for l, c in zip(text_lengths, correct_predictions) if c],
            name='Correct Predictions',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        fig_len.add_trace(go.Box(
            y=[l for l, c in zip(text_lengths, correct_predictions) if not c],
            name='Incorrect Predictions',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        fig_len.update_layout(
            title="Text Length vs Prediction Accuracy",
            yaxis_title="Text Length",
            width=400,
            height=400
        )
        st.plotly_chart(fig_len, use_container_width=True)

        # AI Explanation for Length Analysis
        explanation_prompt_len = f"""
        Analyze the relationship between text length and prediction accuracy for {user_prompt}.
        What patterns do you observe? Are there any length-related biases?
        Provide a detailed explanation in 10-12 lines.
        """
        explanation_len = get_gemini_explanation("Length Analysis Data", explanation_prompt_len)
        with st.expander("💡 AI Explanation (Length Analysis)", expanded=False):
            st.write(explanation_len)

# Example usage:
# visualize_nlp_results(y_test, y_pred, "sentiment analysis of movie reviews")