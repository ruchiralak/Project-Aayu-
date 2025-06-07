# Project Aayu

**Project Aayu** is a modular and user-friendly machine learning framework for health risk prediction. It supports both a GUI-based interface (via Streamlit) and CLI usage, allowing developers, data scientists, and health professionals to easily train, evaluate, and export ML models for classification tasks on health datasets.

---

## 🚀 Features

* 📂 Upload and preview health-related datasets (CSV)
* 🧹 Automatic data preprocessing and encoding
* 📊 Feature selection with top N feature filtering
* 🧠 Train multiple models: Random Forest, SVM, Logistic Regression, KNN, Decision Tree
* 📈 Performance evaluation with metrics and visualizations
* 💾 Save and download best models
* 🖥️ GUI (Streamlit) and CLI versions

---

## 🧱 Installation

### 📦 From Source

```bash
# Clone the repository
$ git clone https://github.com/ruchiralak/health_predictor_framework.git
$ cd health_predictor_framework

# Install with dependencies
$ pip install -e .
```

---

## 🧑‍💻 Usage

### 🎛️ GUI (Streamlit)

```bash
$ aayu-gui
```

This launches a Streamlit dashboard where non-technical users can:

* Upload their dataset
* Select configuration options
* Train models and see visual feedback
* Download the best performing model

---

### 💻 CLI Version

You can also run the CLI using:

```bash
$ aayu-cli
```

Customize behavior using `config.yaml`:

```yaml
input_csv: "path/to/dataset.csv"
target_column: "target"
top_n: 8
test_size: 0.2
random_state: 42
preferred_model: "Random Forest"
output_model_path: "saved_model.pkl"
```

---

## 📁 Project Structure

```
health_predictor/
├── app.py                  # Streamlit GUI app
├── main.py                 # CLI entry point
├── preprocessing/
│   └── cleaner.py          # Data cleaning
├── features/
│   └── selector.py         # Feature selector
├── training/
│   ├── model_trainer.py    # Model training
│   └── model_evaluator.py  # Evaluation
├── visualization/
│   └── plot_results.py     # Matplotlib & seaborn visuals
├── models/
│   └── model_saver.py      # Save the best model
```

---

## 📄 License

MIT License

---

## 🙋‍♂️ Author

**Ruchira Lakshan**
📧 [ruchiralakshanm@gmail.com](mailto:ruchiralakshanm@gmail.com)
🔗 [GitHub](https://github.com/ruchiralak)

---

> "Aayu" means "Life" — this tool aims to empower healthcare prediction with the power of open, accessible machine learning.
