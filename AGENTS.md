# AGENTS.md - AI Agent Guidelines for BMI Predictor

## Project Overview
BMI Predictor is a Python ML project using scikit-learn, Streamlit, and Jupyter notebooks for predicting BMI categories from height, weight, and gender data.

## Build/Lint/Test Commands

### Virtual Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run Streamlit dashboard
streamlit run dashboard/app.py

# Run Jupyter notebooks
jupyter notebook notebooks/
```

### MLflow Experiment Tracking
```bash
# Run training with MLflow tracking (CI/CD)
python scripts/train_with_mlflow.py

# View MLflow UI to compare experiments
mlflow ui --backend-store-uri file://$(pwd)/mlruns
```

### Testing
**No formal test framework configured.**
- Manual testing via `notebooks/04-Testing.ipynb`
- Model validation via scikit-learn metrics (accuracy, classification_report)
- Dashboard testing via Streamlit interface

### Code Quality
**No linting/formatting configured.**
Recommended tools:
```bash
# Install and run (optional)
pip install black flake8 pylint
black .
flake8 .
pylint dashboard/
```

## Code Style Guidelines

### Imports
```python
# Standard library first
import os
import sys
import logging

# Third-party libraries second
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
```

### Formatting
- Indentation: 4 spaces
- Line length: ~100 characters (no strict limit observed)
- Docstrings: None used; use inline comments instead
- Comments: Use descriptive inline comments with emojis for visual clarity:
  ```python
  # --- Load Model ---
  # --- Configuration ---
  ```

### Naming Conventions
- Variables: `snake_case` (e.g., `df_bmi`, `bmi_value`, `gender_encoded`)
- Constants: `UPPER_CASE` (e.g., `GENDER_MAP`, `BMI_CATEGORIES`, `RANDOM_STATE`)
- Functions: `snake_case` with descriptive names (e.g., `load_model()`)
- Files: Use descriptive names with dashes for notebooks (e.g., `01-ETL.ipynb`, `03-Training.ipynb`)

### Error Handling
- Use try/except blocks for file operations and model operations
- Use logging for errors in notebooks (not print statements):
  ```python
  logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
  logger = logging.getLogger(__name__)
  logger.error(f"❌ Error message: {e}")
  ```
- Use Streamlit's error display for dashboard errors:
  ```python
  st.error(f"Error loading model: {e}")
  ```

### Docker Containerization
```bash
# Build Docker image
docker build -t bmi-predictor .

# Run container
docker run -p 8501:8501 bmi-predictor

# Using Docker Compose (recommended)
docker-compose up --build

# Access dashboard at: http://localhost:8501
```

### Project Structure
```
bmi-predictor/
├── dashboard/           # Streamlit application
│   └── app.py
├── data/                # CSV datasets
│   ├── bmi.csv
│   └── bmi_cleaned.csv
├── models/              # Trained model binaries
│   ├── *.joblib
│   └── models_exported/
├── notebooks/           # Jupyter notebooks (numbered by workflow)
│   ├── 01-ETL.ipynb
│   ├── 02-EDA.ipynb
│   ├── 03-Training.ipynb
│   └── 04-Testing.ipynb
├── scripts/             # CI/CD and automation scripts
│   ├── train_with_mlflow.py
│   └── utils/
│       └── mlflow_utils.py
├── mlruns/              # MLflow tracking data (local storage)
├── mlruns_artifacts/    # MLflow artifacts (confusion matrices, models)
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose configuration
├── .dockerignore        # Docker build exclusions
├── venv/                # Virtual environment
├── requirements.txt     # Dependencies
└── README.md
```

### Model Development Workflow
1. **ETL** (`01-ETL.ipynb`): Load data, clean duplicates/outliers, encode features
2. **EDA** (`02-EDA.ipynb`): Exploratory data analysis and visualization
3. **Training** (`03-Training.ipynb`): Model training with scikit-learn pipelines
4. **Testing** (`04-Testing.ipynb`): Model validation and testing
5. **MLflow Tracking** (optional via script or notebook): Track hyperparameters, metrics, and artifacts

### MLflow Conventions
- Use local file-based tracking: `mlruns/` directory
- Experiment name: `bmi-predictor-random-forest`
- Track hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `random_state`
- Track metrics: `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, per-class metrics
- Log artifacts: `best_model.pkl`, `confusion_matrix.png`
- Use nested runs for GridSearchCV to track all parameter combinations
- Import utilities from `scripts.utils.mlflow_utils`: `setup_local_mlflow()`, `get_dataset_version()`, `log_classification_metrics()`

### Key Conventions
- Use `RANDOM_STATE = 42` for reproducibility in ML operations
- Use relative paths with `os.path.join()` and `os.path.abspath()` for file operations
- Use dictionaries for categorical mappings (e.g., `GENDER_MAP = {"Male": 0, "Female": 1}`)
- Use snake_case for DataFrame column names with suffixes: `_Value` for calculations, `_Encoded` for encoded values
- Export cleaned data to `data/` folder with `_cleaned.csv` suffix
- Save models using `joblib` or `pickle` in `models/` folder

### Streamlit Dashboard Conventions
- Use `st.set_page_config()` for page configuration
- Use `@st.cache_resource` for expensive operations (model loading)
- Use `st.sidebar` for user inputs
- Use visual feedback (emojis, colored markdown blocks) for results
- Use `st.plotly_chart()` for interactive visualizations
