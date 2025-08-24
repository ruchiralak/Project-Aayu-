# main.py
import yaml
import pandas as pd
from health_predictor.preprocessing.cleaner import preprocess_data
from health_predictor.features.selector import select_top_features
from health_predictor.training.model_trainer import train_models
from health_predictor.training.model_evaluator import evaluate_models
from health_predictor.visualization.plot_results import visualize_results
from health_predictor.models.model_saver import save_best_model

# === Load Config ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Load Data ===
df = pd.read_csv(config["input_csv"])

# === Step 1: Preprocess Data ===
df_encoded, label_encoders = preprocess_data(df)

# === Step 2: Feature/Target Split ===
X = df_encoded.drop(config["target_column"], axis=1)
y = df_encoded[config["target_column"]]

# === Step 3: Select Top Features ===
X_train, X_test, y_train, y_test, top_features = select_top_features(X, y, config)

# === Step 4: Train Models ===
models = train_models(X_train, y_train, config)

# === Step 5: Evaluate Models ===
results_df, predictions = evaluate_models(models, X_test, y_test, X[top_features], y)

# === Step 6: Visualize Results ===
visualize_results(results_df, y_test, predictions)

# === Step 7: Save Best Model ===
save_best_model(models, results_df, top_features, config)
