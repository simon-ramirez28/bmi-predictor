"""
BMI Predictor MLflow Training Script

This script trains a Random Forest model for BMI category prediction
with full MLflow tracking including hyperparameters, metrics, and artifacts.

Usage:
    source venv/bin/activate
    python scripts/train_with_mlflow.py

View results:
    mlflow ui --backend-store-uri file://$(pwd)/mlruns
"""

import os
import sys
import logging
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

RANDOM_STATE = 42


def setup_mlflow():
    """Configure MLflow tracking for local storage."""
    mlflow_dir = os.path.abspath('mlruns')
    os.makedirs(mlflow_dir, exist_ok=True)
    
    mlflow.set_tracking_uri(f'file://{mlflow_dir}')
    mlflow.set_experiment('bmi-predictor-random-forest')
    
    logger.info(f'✅ MLflow configured: {mlflow_dir}')
    return mlflow_dir


def load_data():
    """Load and validate cleaned BMI dataset."""
    project_root = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(project_root, '..', 'data', 'bmi_cleaned.csv')
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f'✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns')
        return df
    except Exception as e:
        logger.error(f'❌ Error loading data: {e}')
        raise


def prepare_train_test_split(df):
    """Split data into training and testing sets."""
    X = df.drop(columns=['Index'])
    y = df['Index']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
    return X_train, X_test, y_train, y_test


def create_pipeline():
    """Create preprocessing + model pipeline."""
    numeric_features = ['Height', 'Weight', 'BMI_Value']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    
    return pipeline


def train_with_gridsearch(pipeline, X_train, y_train):
    """Train model with GridSearchCV and log all runs."""
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, 
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    logger.info('🔍 Starting GridSearchCV...')
    grid_search.fit(X_train, y_train)
    
    return grid_search


def log_all_cv_runs(grid_search, X_test, y_test):
    """Log all GridSearchCV runs as separate MLflow runs."""
    cv_results = grid_search.cv_results_
    
    for i in range(len(cv_results['params'])):
        with mlflow.start_run(nested=True):
            params = cv_results['params'][i]
            mlflow.log_params(params)
            
            metrics = {
                'cv_mean_accuracy': cv_results['mean_test_score'][i],
                'cv_std_accuracy': cv_results['std_test_score'][i]
            }
            mlflow.log_metrics(metrics)
            
            logger.info(f'  Run {i+1}/{len(cv_results["params"])}: '
                       f'accuracy={metrics["cv_mean_accuracy"]:.4f}, '
                       f'params={params}')


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics dictionary."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    classes = sorted(y_test.unique())
    for i, cls in enumerate(classes):
        metrics[f'precision_class_{cls}'] = precision_score(
            y_test, y_pred, labels=[cls], average='micro', zero_division=0
        )
    
    logger.info(f'🎯 Test Accuracy: {metrics["accuracy"]:.4f}')
    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, save_path):
    """Create and save confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y_test.unique()),
                yticklabels=sorted(y_test.unique()))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - BMI Predictor')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f'📊 Confusion matrix saved: {save_path}')


def save_model(model, save_path):
    """Save model as pickle artifact."""
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'💾 Model saved: {save_path}')


def main():
    """Main training workflow with MLflow tracking."""
    logger.info('🚀 Starting BMI Predictor Training with MLflow')
    
    setup_mlflow()
    
    with mlflow.start_run(run_name='random-forest-gridsearch'):
        mlflow.log_param('random_state', RANDOM_STATE)
        mlflow.log_param('test_size', 0.2)
        
        df = load_data()
        X_train, X_test, y_train, y_test = prepare_train_test_split(df)
        
        mlflow.log_param('train_size', X_train.shape[0])
        mlflow.log_param('test_size_actual', X_test.shape[0])
        mlflow.log_param('n_features', X_train.shape[1])
        
        pipeline = create_pipeline()
        
        grid_search = train_with_gridsearch(pipeline, X_train, y_train)
        
        log_all_cv_runs(grid_search, X_test, y_test)
        
        best_model = grid_search.best_estimator_
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param('best_cv_score', grid_search.best_score_)
        
        metrics, y_pred = evaluate_model(best_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        artifact_dir = os.path.abspath('mlruns_artifacts')
        os.makedirs(artifact_dir, exist_ok=True)
        
        cm_path = os.path.join(artifact_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)
        
        model_path = os.path.join(artifact_dir, 'best_model.pkl')
        save_model(best_model, model_path)
        mlflow.log_artifact(model_path)
        
        mlflow.sklearn.log_model(best_model, 'model', 
                                 registered_model_name='bmi-predictor-rf')
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f'✅ Training complete! Run ID: {run_id}')
        logger.info(f'🌐 View results: mlflow ui --backend-store-uri file://{os.path.abspath("mlruns")}')
        
        print('\n' + '='*60)
        print('CLASSIFICATION REPORT')
        print('='*60)
        print(classification_report(y_test, y_pred))
        print('='*60)


if __name__ == '__main__':
    main()
