# --- 🧮 BMI Predictor ---

## Summary 📋
BMI Predictor is a Machine learning pipeline developed to predict the BMI on a human being by combining basic statistics such as weight, height and genre, surprisingly, this model achieves an accuracy of 100%

## 🚀 Features
- **Machine Learning Pipeline**: End-to-end processing from ETL to Model Training.
- **Accurate Predictions**: Uses Random Forest Classifier to achieve high accuracy.
- **Interactive Dashboard**: A Streamlit-based UI to visualize predicted BMI categories.

## 📂 Project Structure
- `notebooks/`: Jupyter notebooks for ETL, EDA, Training, and Testing.
- `models/`: Trained model binaries (`.joblib` and `.pkl`).
- `data/`: Datasets used for training and testing.
- `dashboard/`: Streamlit web application.

## 💻 How to Run the Dashboard
1. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```
2. **Install dependencies** (if not already installed):
   ```bash
   pip install streamlit plotly pandas scikit-learn joblib
   ```
3. **Launch the application**:
   ```bash
   streamlit run dashboard/app.py
   ```

