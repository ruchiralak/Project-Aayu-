import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def evaluate_models(models, X_test, y_test, X_full, y_full):
    results = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_full, y_full, cv=5)
        cv_mean = cv_scores.mean()

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "CV Mean Accuracy": cv_mean
        })

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results)
    return results_df, predictions
