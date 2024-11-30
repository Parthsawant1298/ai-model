import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier

def train_models(X_train, y_train, X_test, y_test, task_type):
    # Dictionary of models
    models = {
        "Decision Tree": DecisionTreeClassifier() if task_type in ['classification', 'nlp'] else DecisionTreeRegressor(),
        "Support Vector Machine": SVC(probability=True) if task_type in ['classification', 'nlp'] else SVR(),
        "K-Nearest Neighbors": KNeighborsClassifier() if task_type in ['classification', 'nlp'] else KNeighborsRegressor(),
        "Logistic Regression": LogisticRegression(max_iter=1000) if task_type in ['classification', 'nlp'] else None,
        "Random Forest": RandomForestClassifier() if task_type in ['classification', 'nlp'] else RandomForestRegressor(),
        # "Gradient Boosting": GradientBoostingClassifier() if task_type in ['classification', 'nlp'] else GradientBoostingRegressor(),
        # "XGBoost": XGBClassifier(use_label_encoder=False) if task_type in ['classification', 'nlp'] else XGBRegressor(),
        # "LightGBM": LGBMClassifier() if task_type in ['classification', 'nlp'] else LGBMRegressor(),
        # "CatBoost": CatBoostClassifier(silent=True) if task_type in ['classification', 'nlp'] else CatBoostRegressor(silent=True),
        # "Naive Bayes": GaussianNB() if task_type in ['classification', 'nlp'] else None,
        # "Neural Network": MLPClassifier(max_iter=1000) if task_type in ['classification', 'nlp'] else None,
    }

    best_model = None
    best_model_name = ""
    best_score = -float('inf')

    for model_name, model in models.items():
        if model is None:
            continue  # Skip models that are not applicable
        
        param_grid = {}
        if model_name == "Decision Tree":
            param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == "Support Vector Machine":
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        elif model_name == "K-Nearest Neighbors":
            param_grid = {'n_neighbors': [3, 5, 7]}
        elif model_name == "Logistic Regression":
            param_grid = {'C': [0.1, 1, 10]}
        elif model_name == "Random Forest":
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20, 30]}
        # elif model_name == "Gradient Boosting":
        #     param_grid = {'n_estimators': [100], 'learning_rate': [0.01, 0.1, 0.2]}
        # elif model_name == "XGBoost":
        #     param_grid = {'n_estimators': [100], 'learning_rate': [0.01, 0.1, 0.2]}
        # elif model_name == "LightGBM":
        #     param_grid = {'n_estimators': [100], 'learning_rate': [0.01, 0.1, 0.2]}
        # elif model_name == "CatBoost":
        #     param_grid = {'iterations': [100], 'learning_rate': [0.01, 0.1, 0.2]}
        # elif model_name == "Neural Network":
        #     param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh']}

        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy' if task_type in ['classification', 'nlp'] else 'r2', cv=5)

        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            print(f"Error during training {model_name}: {e}")
            continue

        # Evaluate model
        y_pred = grid_search.predict(X_test)
        score = accuracy_score(y_test, y_pred) if task_type in ['classification', 'nlp'] else r2_score(y_test, y_pred)

        print(f"{model_name} - Best Score: {grid_search.best_score_}, Test Score: {score}")

        # Update best model information if applicable
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    if best_model is not None:
        save_best_model(best_model)
    else:
        print("No suitable model was found.")

    return best_model, best_model_name, best_score, best_model.predict(X_test) if best_model else None

def save_best_model(model):
    try:
        with open("best_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Best model saved successfully.")
    except Exception as e:
        print(f"Error saving the model: {e}")
