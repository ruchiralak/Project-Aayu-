import joblib
import json
import os

def save_best_model(models, results_df, top_features, model_to_save="Random Forest", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    if model_to_save == "auto":
        best_row = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]
        model_to_save = best_row["Model"]

    best_model = models[model_to_save]

    model_path = os.path.join(output_dir, f"{model_to_save.replace(' ', '_').lower()}_model.pkl")
    joblib.dump(best_model, model_path)

    features_path = os.path.join(output_dir, "top_features.json")
    with open(features_path, "w") as f:
        json.dump(top_features, f)

    print(f"\n✅ Saved model: {model_path}")
    print(f"📁 Saved top features: {features_path}")
