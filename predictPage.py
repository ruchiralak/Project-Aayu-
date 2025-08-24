import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from health_predictor.preprocessing.cleaner import preprocess_data


def show_predict():
    st.title("📟 Predict Risk Outcome")

    # Step 1: Check for trained models
    if "models" not in st.session_state or not st.session_state.models:
        st.warning("⚠️ No trained models found. Please train a model from the Dashboard tab first.")
        return

    # Step 2: Select model
    model_name = st.selectbox("🔢 Select a model for prediction", list(st.session_state.models.keys()))
    model = st.session_state.models[model_name]

    # Step 3: Upload JSON for top features
    st.subheader("📤 Upload Top Features JSON")
    features_file = st.file_uploader("Upload a JSON file containing top features", type=["json"])

    if features_file:
        try:
            top_features = json.load(features_file)
            if not isinstance(top_features, list):
                st.error("❌ The uploaded JSON must be a list of feature names.")
                return
        except Exception as e:
            st.error(f"❌ Failed to parse JSON: {e}")
            return
    else:
        st.info("📄 Please upload the JSON file with top features.")
        return

    # Step 4: Upload new test dataset
    st.subheader("📤 Upload New Test Data (CSV)")
    test_file = st.file_uploader("Upload a CSV file with test data", type=["csv"])

    if test_file:
        try:
            test_df = pd.read_csv(test_file)
        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            return
    else:
        st.info("📄 Please upload a CSV file with new test data.")
        return

    # Step 5: Prediction
    if st.button("🔮 Predict"):
        # ✅ NEW: Preprocess uploaded test data
        test_df_processed, _ = preprocess_data(test_df)

        # ✅ Validate required features are in the processed data
        missing = [f for f in top_features if f not in test_df_processed.columns]
        if missing:
            st.error(f"❌ These required features are missing in the test data after preprocessing: {missing}")
            return

        # Use only top features
        input_df = test_df_processed[top_features]

        # Perform prediction
        predictions = model.predict(input_df)
        st.success("✅ Predictions completed!")

        # Add predictions to original (unprocessed) data
        result_df = test_df.copy()
        result_df["Prediction"] = predictions

        # Results and Visualizations (unchanged)
        st.subheader("🧾 Prediction Results (Detailed)")
        positive_count = sum(result_df["Prediction"] == 1)
        negative_count = sum(result_df["Prediction"] == 0)
        total = len(result_df)

        st.markdown(f"""
        🔎 **Summary**:
        - ✅ **Healthy/Low Risk** predictions: `{negative_count}`
        - ⚠️ **At-Risk/Positive** predictions: `{positive_count}`
        - 📊 **Total Samples**: `{total}`
        """)

        st.subheader("📊 Prediction Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Prediction", data=result_df, palette=["#1f77b4", "#d62728"])
        plt.xticks([0, 1], ['Healthy / Low Risk', 'At Risk'])
        plt.xlabel("Prediction Category")
        plt.ylabel("Number of Patients")
        st.pyplot(fig)

        st.subheader("📋 Full Prediction Table")
        def highlight_risk(val):
            color = 'background-color: #ffcccc' if val == 1 else 'background-color: #ccffcc'
            return color
        st.dataframe(result_df.style.applymap(highlight_risk, subset=["Prediction"]))

        st.download_button(
            label="📥 Download Predictions (.csv)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.info("""
        💡 **How to read this?**

        - If you see **"At Risk"** under predictions, that patient may have a higher likelihood of the health issue your model is predicting (like heart disease, diabetes, etc.).
        - A **"Healthy / Low Risk"** result suggests the model did not find strong signs of risk in their profile.
        - This tool is a **support aid**, not a replacement for professional medical advice.
        """)
