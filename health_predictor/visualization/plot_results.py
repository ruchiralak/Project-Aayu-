import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st  # Add this import


def visualize_results(results_df, y_test, predictions):
    st.write("\n📊 Model Performance Summary:")
    st.dataframe(results_df.sort_values(by="Accuracy", ascending=False).round(3))

    # === Bar plots for each metric ===
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "CV Mean Accuracy"]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        sns.barplot(x="Model", y=metric, data=results_df, ax=axs[i])
        axs[i].set_title(metric)
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)
        axs[i].set_ylim(0, 1)
    plt.suptitle("Model Performance Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)  # Use Streamlit to display plot



    # === Heatmap Table ===
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
    sns.heatmap(results_df.set_index("Model"), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax_heatmap)
    ax_heatmap.set_title("Model Performance Heatmap")
    st.pyplot(fig_heatmap)  # Show heatmap plot
