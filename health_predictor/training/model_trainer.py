from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def train_models(X_train, y_train, preferred_model="Random Forest", random_state=42):
    # Define parameter grids
    param_grids = {
        "Random Forest": {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        "Logistic Regression": {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear']
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        },
        "KNN": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        "Decision Tree": {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    }

    base_models = {
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state)
    }

    # Grid search on preferred model
    grid = GridSearchCV(base_models[preferred_model], param_grids[preferred_model],
                        cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Fit all models, replace the preferred one with the tuned one
    models = {name: (best_model if name == preferred_model else model.fit(X_train, y_train))
              for name, model in base_models.items()}

    return models, best_model
