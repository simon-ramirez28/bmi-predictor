# 🧮 BMI Predictor

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Summary

BMI Predictor is a production-ready Machine Learning pipeline that predicts BMI categories from basic statistics (weight, height, and gender) using a Random Forest Classifier. The model achieves **100% accuracy** on the test set and includes comprehensive experiment tracking with MLflow and Docker containerization for easy deployment.

### 🎯 Key Capabilities
- **6 BMI Categories**: From "Extremely Weak" to "Extreme Obesity"
- **Interactive Dashboard**: Real-time predictions with visual gauges
- **Experiment Tracking**: Full MLflow integration for hyperparameter tuning
- **Production Ready**: Docker containerization with health checks
- **CI/CD Automation**: Automated training scripts for pipelines

---

## 🚀 Features

### 🤖 Machine Learning Pipeline
- End-to-end processing from ETL to Model Training
- Random Forest Classifier with hyperparameter optimization
- Data cleaning, outlier detection, and feature engineering
- 100% accuracy on 98-sample test set

### 📊 Interactive Dashboard
- Streamlit-based web UI for real-time predictions
- Visual gauges and color-coded health categories
- Responsive design with sidebar controls
- **Screenshot**: *Dashboard showing BMI prediction with health gauge*

```
┌─────────────────────────────────────────┐
│  ⚖️ BMI Health Dashboard                │
│                                         │
│  [User Input Panel]    [Results Card]   │
│  - Gender: [Male]      ┌──────────────┐ │
│  - Weight: [70 kg]     │   Normal       │ │
│  - Height: [170 cm]    │  BMI: 24.2    │ │
│                        └──────────────┘ │
│  [Calculate Button]    [Health Gauge]   │
└─────────────────────────────────────────┘
```

### 🔬 MLflow Experiment Tracking
- **Hyperparameter Tracking**: GridSearchCV with 12 parameter combinations
- **Metrics Logging**: Accuracy, precision, recall, F1 per class
- **Artifact Management**: Model files (.pkl) and confusion matrices
- **Model Registry**: Versioned models with MLflow model registry
- **Screenshot**: *MLflow UI showing experiment comparison*

### 🐳 Docker Containerization
- Multi-stage build for optimized image size (~150MB)
- Health checks and automatic restart policies
- Environment variable support for flexible deployment
- Docker Compose for local development

---

## 📂 Project Structure

```
bmi-predictor/
├── 📁 dashboard/              # Streamlit web application
│   └── app.py                # Main dashboard entry point
├── 📁 data/                   # Datasets
│   ├── bmi.csv               # Raw data
│   └── bmi_cleaned.csv       # Preprocessed data (486 samples)
├── 📁 models/                 # Trained models
│   ├── *.joblib              # scikit-learn pipelines
│   └── models_exported/
│       └── bmi_model.pkl     # Production model
├── 📁 notebooks/              # Jupyter notebooks (workflow)
│   ├── 01-ETL.ipynb          # Extract, Transform, Load
│   ├── 02-EDA.ipynb          # Exploratory Data Analysis
│   ├── 03-Training.ipynb     # Model training with MLflow
│   └── 04-Testing.ipynb      # Model validation
├── 📁 scripts/                # CI/CD automation (NEW)
│   ├── train_with_mlflow.py  # Automated training script
│   └── utils/
│       └── mlflow_utils.py   # MLflow helper functions
├── 📁 mlruns/                 # MLflow tracking data (auto-generated)
├── 📁 mlruns_artifacts/       # MLflow artifacts (auto-generated)
├── 🐳 Dockerfile              # Docker image definition
├── 🐳 docker-compose.yml      # Docker Compose configuration
├── 📄 .dockerignore           # Docker build exclusions
├── 📄 requirements.txt        # Python dependencies
├── 📄 AGENTS.md              # AI agent guidelines
└── 📄 README.md              # This file
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│              (Streamlit Dashboard - Port 8501)              │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP Requests
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Docker Container                          │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │  Streamlit App  │───▶│  Model Inference              │  │
│  │  (dashboard/)   │    │  - Random Forest Classifier   │  │
│  └─────────────────┘    │  - Input: Height, Weight,     │  │
│                         │           Gender               │  │
│                         │  - Output: BMI Category (0-5)   │  │
│                         └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼ Training/Experimentation
┌─────────────────────────────────────────────────────────────┐
│                 MLflow Tracking Server                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌────────────────┐  │
│  │  Hyperparameters│ │  Metrics        │ │  Artifacts     │  │
│  │  - n_estimators │ │  - Accuracy     │ │  - .pkl models│  │
│  │  - max_depth    │ │  - Precision    │ │  - Confusion  │  │
│  │  - GridSearchCV │ │  - Recall/F1    │ │    Matrix     │  │
│  └─────────────────┘ └─────────────────┘ └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Screenshot Placeholder**: *Architecture diagram showing data flow from user → Docker → Model → Prediction → Dashboard*

---

## ⚡ Quick Start

Choose your preferred deployment method:

### Option A: 🖥️ Local Development

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run dashboard/app.py

# Access at: http://localhost:8501
```

### Option B: 🐳 Docker (Recommended for Production)

```bash
# Using Docker Compose (recommended)
docker-compose up --build

# Or using Docker directly
docker build -t bmi-predictor .
docker run -p 8501:8501 bmi-predictor

# Access at: http://localhost:8501
```

### Option C: 🔬 MLflow Experiment Tracking

```bash
# Run automated training with full tracking
python scripts/train_with_mlflow.py

# View results in MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/mlruns

# Access at: http://localhost:5000
```

---

## 🔬 MLflow Experiment Tracking

### What Gets Tracked?

| Component | Details |
|-----------|---------|
| **Hyperparameters** | `n_estimators`, `max_depth`, `min_samples_split`, `random_state` |
| **Cross-Validation** | 5-fold GridSearchCV with 12 parameter combinations |
| **Metrics** | Accuracy, Precision, Recall, F1 (macro & weighted) |
| **Artifacts** | `best_model.pkl`, `confusion_matrix.png` |
| **Model Registry** | Versioned models: `bmi-predictor-rf` |

### Example Run Output

```bash
$ python scripts/train_with_mlflow.py

🚀 Starting BMI Predictor Training with MLflow
✅ MLflow configured: /path/to/mlruns
✅ Data loaded: 486 rows, 5 columns
🔍 Starting GridSearchCV...
  Run 1/12: accuracy=0.9511, params={'n_estimators': 50, 'max_depth': 10, ...}
  ...
  Run 12/12: accuracy=0.9588, params={'n_estimators': 100, 'max_depth': None, ...}
🎯 Test Accuracy: 1.0000
✅ Training complete! Run ID: 7f84b4bb50a64f37a7b3eac54f75251d
🌐 View results: mlflow ui --backend-store-uri file:///path/to/mlruns
```

**Screenshot Placeholder**: *MLflow UI showing parallel coordinate plot comparing hyperparameter runs*

---

## 🐳 Docker Containerization

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+ (optional)

### Build & Run

```bash
# Quick start with Docker Compose
docker-compose up --build

# Manual Docker build
docker build -t bmi-predictor .
docker run -d \
  --name bmi-predictor \
  -p 8501:8501 \
  --restart unless-stopped \
  bmi-predictor
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/models_exported/bmi_model.pkl` | Path to model file |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Server bind address |
| `STREAMLIT_SERVER_PORT` | `8501` | Server port |

### Docker Compose Features
- **Health Checks**: Automatic container health monitoring
- **Volume Mounting**: Easy model updates without rebuild
- **Restart Policy**: `unless-stopped` for production stability
- **Port Mapping**: Host 8501 → Container 8501

---

## 📊 Model Development Workflow

1. **📥 ETL** (`01-ETL.ipynb`)
   - Load raw data from `data/bmi.csv`
   - Clean duplicates (11 removed) and outliers (3 removed)
   - Calculate BMI values and encode gender
   - Export: `data/bmi_cleaned.csv` (486 samples)

2. **📈 EDA** (`02-EDA.ipynb`)
   - Statistical analysis and visualizations
   - Distribution analysis by BMI category
   - Correlation matrices and pair plots

3. **🎯 Training** (`03-Training.ipynb`)
   - Train/test split (80/20) with stratification
   - Feature scaling with StandardScaler
   - GridSearchCV hyperparameter tuning
   - MLflow experiment tracking integration

4. **✅ Testing** (`04-Testing.ipynb`)
   - Model validation on holdout set
   - Confusion matrix analysis
   - Classification report generation

5. **🔬 MLflow Tracking** (`scripts/train_with_mlflow.py`)
   - Automated training pipeline
   - Hyperparameter logging
   - Model artifact management
   - Version control with MLflow registry

---

## 📈 Performance Metrics

### Model Performance (Test Set - 98 samples)
- **Accuracy**: 100%
- **Precision**: 1.00 (macro avg)
- **Recall**: 1.00 (macro avg)
- **F1-Score**: 1.00 (macro avg)

### Per-Class Performance

| BMI Category | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 - Extremely Weak | 1.00 | 1.00 | 1.00 | 2 |
| 1 - Weak | 1.00 | 1.00 | 1.00 | 4 |
| 2 - Normal | 1.00 | 1.00 | 1.00 | 14 |
| 3 - Overweight | 1.00 | 1.00 | 1.00 | 13 |
| 4 - Obesity | 1.00 | 1.00 | 1.00 | 26 |
| 5 - Extreme Obesity | 1.00 | 1.00 | 1.00 | 39 |

**Screenshot Placeholder**: *Confusion matrix heatmap showing perfect classification*

---

## 🤝 Contributing

This project follows AI agent guidelines defined in [`AGENTS.md`](AGENTS.md). Key conventions:

- **Code Style**: 4-space indentation, snake_case naming
- **Logging**: Use `logging` module with emoji indicators
- **Paths**: Use `os.path.join()` for cross-platform compatibility
- **Reproducibility**: Always set `RANDOM_STATE = 42`

See [`AGENTS.md`](AGENTS.md) for complete guidelines.

---

## 📝 Additional Information

### Dataset Statistics
- **Total Samples**: 500 (raw) → 486 (cleaned)
- **Features**: Height (cm), Weight (kg), Gender, BMI_Value
- **Target**: 6 BMI categories (Index 0-5)
- **Train/Test Split**: 388 / 98 samples (80/20)

### Tech Stack
- **Python**: 3.11+
- **ML**: scikit-learn, pandas, numpy
- **Dashboard**: Streamlit, Plotly
- **Experiment Tracking**: MLflow 2.10+
- **Containerization**: Docker, Docker Compose
- **Visualization**: Matplotlib, Seaborn

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- BMI dataset from [Kaggle](https://www.kaggle.com/)
- Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/)
- Experiment tracking powered by [MLflow](https://mlflow.org/)

---

## 📞 Support

For issues or questions:
- Check [`AGENTS.md`](AGENTS.md) for development guidelines
- Review the [GitHub Issues](https://github.com/yourusername/bmi-predictor/issues) page

---
