import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score

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

    return best_model, best_model_name, best_score, best_model.predict(X_test)

def save_best_model(model):
    with open("best_model.pkl", "wb") as f:
        pickle.dump(model, f)