"""
MLflow Utilities for BMI Predictor

Helper functions for consistent MLflow tracking across notebooks and scripts.
"""

import os
import hashlib
import logging

import mlflow

logger = logging.getLogger(__name__)


def get_dataset_version(filepath):
    """Generate a hash-based version identifier for the dataset."""
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:8]


def log_classification_metrics(metrics_dict):
    """Log all classification metrics to MLflow."""
    for metric_name, metric_value in metrics_dict.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name, metric_value)
            logger.info(f'📊 Logged {metric_name}: {metric_value:.4f}')


def setup_local_mlflow(experiment_name='bmi-predictor-random-forest'):
    """Configure MLflow for local file-based tracking."""
    mlflow_dir = os.path.abspath('mlruns')
    os.makedirs(mlflow_dir, exist_ok=True)
    
    mlflow.set_tracking_uri(f'file://{mlflow_dir}')
    mlflow.set_experiment(experiment_name)
    
    return mlflow_dir


def get_mlflow_ui_command():
    """Return the command to launch MLflow UI."""
    mlflow_dir = os.path.abspath('mlruns')
    return f'mlflow ui --backend-store-uri file://{mlflow_dir}'
