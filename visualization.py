import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                           precision_recall_curve, average_precision_score, mean_squared_error, r2_score)
from sklearn.inspection import permutation_importance
from scipy import stats
import google.generativeai as genai

def get_gemini_explanation(data, prompt):
    """
    Get AI-generated explanation for visualizations using Gemini model
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate explanation: {str(e)}"

def visualize_results(task_type, y_test, y_pred, best_model, X_test, feature_names, user_prompt):
    """
    Visualize model results with AI-powered explanations
    """
   
    # Classification Task Visualization
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

            # AI Explanation for Confusion Matrix
            total = np.sum(cm)
            explanation_prompt = f"""
            Analyze this confusion matrix for {user_prompt}:
            - True Positives: {cm[1,1]}
            - False Positives: {cm[0,1]}
            - False Negatives: {cm[1,0]}
            - True Negatives: {cm[0,0]}
            Explain what these numbers mean in the context of {user_prompt} and their implications.
            Al in 10 lines paragraph.
            """
            explanation = get_gemini_explanation(str(cm.tolist()), explanation_prompt)
            with st.expander("💡 AI Explanation", expanded=False):
             st.write("", explanation)
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
            fig_roc = px.line(x=fpr, y=tpr, 
                            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                            title=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})')
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc)

            # AI Explanation for ROC Curve
            explanation_prompt = f"""
            Analyze this ROC curve for {user_prompt}:
            - AUC Score: {auc(fpr, tpr):.2f}
            Explain what this curve and AUC score mean in the context of {user_prompt}.
            How good is the model at distinguishing between classes?
            Al in 10 lines paragraph.
            """
            explanation = get_gemini_explanation(f"AUC: {auc(fpr, tpr)}", explanation_prompt)
            with st.expander("💡 AI Explanation", expanded=False):
             st.write("", explanation)
           

        with col2:
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
            ap_score = average_precision_score(y_test, best_model.predict_proba(X_test)[:, 1])
            fig_pr = px.line(x=recall, y=precision,
                           labels={'x': 'Recall', 'y': 'Precision'},
                           title=f'Precision-Recall Curve (AP = {ap_score:.2f})')
            st.plotly_chart(fig_pr)

            # AI Explanation for Precision-Recall
            explanation_prompt = f"""
            Analyze this Precision-Recall curve for {user_prompt}:
            - Average Precision: {ap_score:.2f}
            Explain what these metrics mean in the context of {user_prompt}.
            What does this tell us about the model's performance?
            All in 10 lines paragraph.
            """
            explanation = get_gemini_explanation(f"AP: {ap_score}", explanation_prompt)
            with st.expander("💡 AI Explanation", expanded=False):
             st.write("", explanation)
          
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

                # AI Explanation for direct feature importance
                top_features = [feature_names[i] for i in indices[:3]]
                explanation_prompt = f"""
                Analyze the feature importance for {user_prompt}:
                Top 3 most important features are: {', '.join(top_features)}
                Explain why these features might be important for {user_prompt} and how they influence the predictions in 10 lines paragraph.
                """
                explanation = get_gemini_explanation(str(dict(zip(feature_names, importances))), explanation_prompt)
                with st.expander("💡 AI Explanation", expanded=False):
                 st.write("", explanation)
            else:
                try:
                    result = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=0)
                    sorted_idx = result.importances_mean.argsort()[::-1]
                    fig_importance = px.bar(x=result.importances_mean[sorted_idx],
                                          y=[feature_names[i] for i in sorted_idx],
                                          labels={'x': 'Importance', 'y': 'Feature'},
                                          title='Feature Importance (Permutation)',
                                          orientation='h')
                    st.plotly_chart(fig_importance)

                    # AI Explanation for permutation importance
                    top_features = [feature_names[i] for i in sorted_idx[:3]]
                    explanation_prompt = f"""
                    Analyze the permutation feature importance for {user_prompt}:
                    Top 3 most important features are: {', '.join(top_features)}
                    Explain why these features might be important for {user_prompt} and how they influence the predictions in 10 lines paragraph.
                    """
                    explanation = get_gemini_explanation(str(dict(zip(feature_names, result.importances_mean))), explanation_prompt)
                    with st.expander("💡 AI Explanation", expanded=False):
                     st.write("", explanation)
                except Exception as e:
                    st.write("Error calculating feature importance:", e)
              
    # Regression Task Visualization
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Actual vs Predicted
            fig_scatter = px.scatter(x=y_test, y=y_pred,
                                   labels={'x': 'Actual', 'y': 'Predicted'},
                                   title="Actual vs Predicted Values")
            st.plotly_chart(fig_scatter)

            # AI Explanation for Actual vs Predicted
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            explanation_prompt = f"""
            Analyze this Actual vs Predicted plot for {user_prompt}:
            - R² Score: {r2:.2f}
            - Mean Squared Error: {mse:.2f}
            Explain what these results mean in the context of {user_prompt}.
            How well is the model performing ? 
            All content in 10 lines paragraph.
            """
            explanation = get_gemini_explanation(f"R2: {r2}, MSE: {mse}", explanation_prompt)
            with st.expander("💡 AI Explanation", expanded=False):
             st.write("", explanation)
           
            # Residual Plot
            residuals = y_test - y_pred
            fig_residuals = px.scatter(x=y_pred, y=residuals,
                                     labels={'x': 'Predicted', 'y': 'Residuals'},
                                     title="Residual Plot")
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals)

            # AI Explanation for Residuals
            explanation_prompt = f"""
            Analyze this Residual plot for {user_prompt}:
            - Mean Residual: {np.mean(residuals):.2f}
            - Std Residual: {np.std(residuals):.2f}
            Explain what these residuals tell us about the model's predictions for {user_prompt} in 10 lines paragraph.
            Are there any patterns or concerns?
            """
            explanation = get_gemini_explanation(f"Residuals stats: {{'mean': {np.mean(residuals)}, 'std': {np.std(residuals)}}}", explanation_prompt)
            with st.expander("💡 AI Explanation", expanded=False):
             st.write("", explanation)
            

        with col2:
            # Q-Q Plot
            fig_qq = px.scatter(x=stats.probplot(residuals, dist="norm")[0][0],
                              y=stats.probplot(residuals, dist="norm")[0][1],
                              labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'},
                              title='Q-Q Plot')
            fig_qq.add_shape(type='line', line=dict(dash='dash'), 
                           x0=fig_qq.data[0].x.min(), x1=fig_qq.data[0].x.max(),
                           y0=fig_qq.data[0].y.min(), y1=fig_qq.data[0].y.max())
            st.plotly_chart(fig_qq)

            # AI Explanation for Q-Q Plot
            explanation_prompt = f"""
            Analyze this Q-Q plot for {user_prompt}:
            Explain what this plot tells us about the normality of residuals and its implications for {user_prompt} in 10 lines paragraph.
            """
            explanation = get_gemini_explanation("Q-Q Plot Analysis", explanation_prompt)
            with st.expander("💡 AI Explanation", expanded=False):
             st.write("", explanation)
            

            # Error Distribution
            fig_error_dist = px.histogram(residuals, nbins=30,
                                        labels={'value': 'Prediction Error'},
                                        title='Prediction Error Distribution')
            fig_error_dist.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_error_dist)

            # AI Explanation for Error Distribution
            explanation_prompt = f"""
            Analyze this Error Distribution for {user_prompt}:
            - Mean Error: {np.mean(residuals):.2f}
            - Std Error: {np.std(residuals):.2f}
            Explain what these stats imply about the predictions in the context of {user_prompt} in 10 lines paragraph.
            """
            explanation = get_gemini_explanation(f"Error Distribution stats: {{'mean': {np.mean(residuals)}, 'std': {np.std(residuals)}}}", explanation_prompt)
            with st.expander("💡 AI Explanation", expanded=False):
             st.write("", explanation)