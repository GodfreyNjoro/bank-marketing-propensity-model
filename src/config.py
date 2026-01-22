"""
Bank Marketing Propensity Model - Configuration

Centralized configuration for paths, parameters, and model settings.
"""

from typing import Dict, List, Any
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG: Dict[str, Any] = {
    "train_file": str(DATA_DIR / "bank.csv"),
    "test_file": str(DATA_DIR / "test.csv"),
    "delimiter": ";",  # Bank dataset uses semicolon delimiter
    "target_column": "y",
    "positive_class": "yes",
}

# Feature configuration
FEATURE_CONFIG: Dict[str, Any] = {
    # Categorical features that are nominal (no order) - use OneHotEncoder
    "nominal_features": [
        "job", "marital", "education", "default", "housing", 
        "loan", "contact", "month", "poutcome"
    ],
    # Numerical features - apply StandardScaler
    "numerical_features": [
        "age", "balance", "day", "duration", "campaign", "pdays", "previous"
    ],
    # Features to create during feature engineering
    "create_features": True,
}

# Model configuration
MODEL_CONFIG: Dict[str, Any] = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,  # Number of cross-validation folds
    "scoring_metric": "roc_auc",
}

# Random Forest configuration
RF_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",  # Address class imbalance
}

# Random Forest hyperparameter grid for tuning
RF_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# XGBoost configuration
XGBOOST_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "scale_pos_weight": 1,  # Will be adjusted for class imbalance
    "eval_metric": "auc",
}

# XGBoost hyperparameter grid for tuning
XGBOOST_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9],
}

# LightGBM configuration
LIGHTGBM_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": -1,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "is_unbalance": True,  # Address class imbalance
    "verbosity": -1,
}

# LightGBM hyperparameter grid for tuning
LIGHTGBM_PARAM_GRID: Dict[str, List[Any]] = {
    "n_estimators": [50, 100, 200],
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [-1, 5, 10],
}

# SMOTE configuration for handling class imbalance
SMOTE_CONFIG: Dict[str, Any] = {
    "random_state": 42,
    "sampling_strategy": "auto",
}

# Output paths
OUTPUT_CONFIG: Dict[str, str] = {
    "model_path": str(MODELS_DIR / "propensity_model.pkl"),
    "predictions_path": str(OUTPUT_DIR / "predictions.csv"),
    "confusion_matrix_path": str(OUTPUT_DIR / "confusion_matrix.png"),
    "roc_curve_path": str(OUTPUT_DIR / "roc_curve.png"),
    "feature_importance_path": str(OUTPUT_DIR / "feature_importance.png"),
    "model_comparison_path": str(OUTPUT_DIR / "model_comparison.png"),
    "log_file": str(LOGS_DIR / "model_training.log"),
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}

# Database configuration (optional - use environment variables in production)
DB_CONFIG: Dict[str, Any] = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "database": os.environ.get("DB_NAME", "bank_marketing"),
    "user": os.environ.get("DB_USER", "your_username"),
    "password": os.environ.get("DB_PASSWORD", "your_password"),
    "port": int(os.environ.get("DB_PORT", 5432)),
}
