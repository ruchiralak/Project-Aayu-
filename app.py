import streamlit as st

import pandas as pd
import joblib
import io
import json

# first streamlit command
st.set_page_config(page_title="Project Aayu Framework", layout="wide")


from health_predictor.preprocessing.cleaner import preprocess_data
from health_predictor.features.selector import select_top_features
from health_predictor.training.model_trainer import train_models
from health_predictor.training.model_evaluator import evaluate_models
from health_predictor.visualization.plot_results import visualize_results
from predictPage import show_predict




def show_dashboard():
    st.title("🧠 Project Aayu : No-Code Framework for Health Prediction Using Machine Learning")

    # Initialize session state variables if not present
    if "df" not in st.session_state:
        st.session_state.df = None
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "top_features" not in st.session_state:
        st.session_state.top_features = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "models" not in st.session_state:
        st.session_state.models = None
    if "target_col" not in st.session_state:
        st.session_state.target_col = None

    uploaded_file = st.file_uploader("📂 Upload your health dataset (CSV)", type=["csv"])

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)

    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        with st.form("options_form"):
            st.subheader("⚙️ Configuration")
            target_col = st.selectbox("🎯 Select target column", df.columns, index=0)
            top_n = st.slider("🔍 Number of top features to select", 1, min(20, len(df.columns) - 1), 8)
            test_size = st.slider("📏 Test size (fraction)", 0.1, 0.5, 0.2)
            random_state = st.number_input("🔹 Random state", min_value=0, max_value=9999, value=42)
            preferred_model = st.selectbox(
                "🔢 Select your preferred model for hyperparameter tuning",
                ["Random Forest", "Logistic Regression", "SVM", "KNN", "Decision Tree"]
            )
            submit_button = st.form_submit_button("🚀 Train and Evaluate Models")

        if submit_button:
            st.info("🔄 Preprocessing and training in progress...")

            df_encoded, _ = preprocess_data(df)
            X = df_encoded.drop(target_col, axis=1)
            y = df_encoded[target_col]

            X_train_sel, X_test_sel, y_train, y_test, top_features = select_top_features(
                X, y,
                top_n=top_n,
                test_size=test_size,
                random_state=random_state
            )

            models, _ = train_models(
                X_train_sel, y_train,
                random_state=random_state,
                preferred_model=preferred_model
            )

            results_df, predictions = evaluate_models(models, X_test_sel, y_test, X[top_features], y)

            # Save results in session state
            st.session_state.results_df = results_df
            st.session_state.top_features = top_features
            st.session_state.predictions = predictions
            st.session_state.models = models
            st.session_state.target_col = target_col
            st.session_state.y_test = y_test

            st.success("✅ Training complete!")

        # If results exist in session, show them
        if st.session_state.results_df is not None:
            st.subheader("📈 Model Performance Summary")
            st.dataframe(st.session_state.results_df.sort_values(by="Accuracy", ascending=False).round(3))

            st.subheader("📌 Top Selected Features")
            st.write(st.session_state.top_features)

            # convert list top features to JSON string
            top_features_json = json.dumps(st.session_state.top_features, indent=2)

            # downloadble JSON file
            st.download_button(
                label="📥 Download Top Features (.json)",
                data=top_features_json,
                file_name="top_selected_features.json",
                mime="application/json"
            )

            st.subheader("📊 Visualizations")
            visualize_results(st.session_state.results_df, st.session_state.y_test, st.session_state.predictions)

            st.subheader("📅 Download Trained Model")
            model_to_save = st.selectbox("Select a model to download", list(st.session_state.models.keys()))

            model_bytes = io.BytesIO()
            joblib.dump(st.session_state.models[model_to_save], model_bytes)
            model_bytes.seek(0)

            st.download_button(
            label=f"Download {model_to_save} Model (.pkl)",
            data=model_bytes,
            file_name=f"{model_to_save.replace(' ', '_').lower()}_model.pkl",
            mime="application/octet-stream"
            )


           

        else:
         st.warning("👆 Please upload a CSV file to get started.")


def main():
    # navigation tab
    page = st.sidebar.radio("Go To", ["Dashboard", "Predict"])
    if page == "Dashboard":
        show_dashboard()
    elif page == "Predict":
        show_predict()


if __name__ == "__main__":
    main()
