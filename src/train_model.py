"""
Bank Marketing Propensity Model - Training Script

This script handles the complete model training pipeline including:
- Data loading and preprocessing with proper encoding
- Advanced feature engineering
- Model training with multiple algorithms (RandomForest, XGBoost, LightGBM)
- Cross-validation for robust evaluation
- Hyperparameter tuning
- Class imbalance handling with SMOTE
- Model comparison and selection
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    DATA_CONFIG,
    FEATURE_CONFIG,
    LIGHTGBM_CONFIG,
    LIGHTGBM_PARAM_GRID,
    LOGGING_CONFIG,
    MODEL_CONFIG,
    OUTPUT_CONFIG,
    RF_CONFIG,
    RF_PARAM_GRID,
    SMOTE_CONFIG,
    XGBOOST_CONFIG,
    XGBOOST_PARAM_GRID,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    datefmt=LOGGING_CONFIG["datefmt"],
    handlers=[
        logging.FileHandler(OUTPUT_CONFIG["log_file"]),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the training data with correct delimiter.
    
    Args:
        filepath: Path to the data file. Defaults to config setting.
        
    Returns:
        Loaded DataFrame
    """
    if filepath is None:
        filepath = DATA_CONFIG["train_file"]
    
    # Use semicolon delimiter as specified in config
    delimiter = DATA_CONFIG.get("delimiter", ";")
    
    logger.info(f"Loading data from {filepath} with delimiter '{delimiter}'")
    df = pd.read_csv(filepath, delimiter=delimiter)
    logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform advanced feature engineering.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    data = df.copy()
    
    if not FEATURE_CONFIG.get("create_features", True):
        return data
    
    logger.info("Performing feature engineering...")
    
    # Age groups
    data['age_group'] = pd.cut(
        data['age'], 
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )
    
    # Balance categories
    data['balance_category'] = pd.cut(
        data['balance'],
        bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
        labels=['negative', 'low', 'medium', 'high', 'very_high']
    )
    
    # Duration categories (contact duration)
    data['duration_category'] = pd.cut(
        data['duration'],
        bins=[0, 60, 180, 300, 600, np.inf],
        labels=['very_short', 'short', 'medium', 'long', 'very_long']
    )
    
    # Campaign intensity
    data['campaign_intensity'] = pd.cut(
        data['campaign'],
        bins=[0, 1, 3, 5, np.inf],
        labels=['single', 'low', 'medium', 'high']
    )
    
    # Contact recency (based on pdays)
    data['contact_recency'] = data['pdays'].apply(
        lambda x: 'never_contacted' if x == -1 else (
            'recent' if x <= 30 else (
                'moderate' if x <= 180 else 'long_ago'
            )
        )
    )
    
    # Previous contact success indicator
    data['had_previous_contact'] = (data['previous'] > 0).astype(int)
    
    # Interaction features
    data['balance_per_age'] = data['balance'] / (data['age'] + 1)
    data['duration_per_campaign'] = data['duration'] / (data['campaign'] + 1)
    
    # Binary features for certain conditions
    data['is_employed'] = (~data['job'].isin(['unemployed', 'student', 'retired'])).astype(int)
    data['has_loan'] = ((data['housing'] == 'yes') | (data['loan'] == 'yes')).astype(int)
    data['is_default'] = (data['default'] == 'yes').astype(int)
    
    logger.info(f"Feature engineering complete. New shape: {data.shape}")
    
    return data


def preprocess_data(
    df: pd.DataFrame,
    is_training: bool = True,
    preprocessor: Optional[ColumnTransformer] = None
) -> Tuple[pd.DataFrame, np.ndarray, ColumnTransformer, List[str]]:
    """
    Preprocess the data with proper encoding and scaling.
    
    Args:
        df: Input DataFrame
        is_training: Whether this is training data
        preprocessor: Pre-fitted preprocessor for test data
        
    Returns:
        Tuple of (features DataFrame, target array, fitted preprocessor, feature names)
    """
    data = df.copy()
    
    # Extract target variable
    target = None
    target_col = DATA_CONFIG.get("target_column", "y")
    positive_class = DATA_CONFIG.get("positive_class", "yes")
    
    if target_col in data.columns:
        target = (data[target_col] == positive_class).astype(int).values
        data = data.drop(target_col, axis=1)
    
    # Engineer features
    data = engineer_features(data)
    
    # Identify feature types
    nominal_features = FEATURE_CONFIG.get("nominal_features", [])
    numerical_features = FEATURE_CONFIG.get("numerical_features", [])
    
    # Add engineered categorical features
    engineered_categorical = ['age_group', 'balance_category', 'duration_category', 
                             'campaign_intensity', 'contact_recency']
    nominal_features = list(set(nominal_features + engineered_categorical))
    
    # Add engineered numerical features
    engineered_numerical = ['balance_per_age', 'duration_per_campaign', 
                           'had_previous_contact', 'is_employed', 'has_loan', 'is_default']
    numerical_features = list(set(numerical_features + engineered_numerical))
    
    # Filter to existing columns
    nominal_features = [f for f in nominal_features if f in data.columns]
    numerical_features = [f for f in numerical_features if f in data.columns]
    
    logger.info(f"Nominal features: {nominal_features}")
    logger.info(f"Numerical features: {numerical_features}")
    
    # Convert categorical columns to string
    for col in nominal_features:
        data[col] = data[col].astype(str)
    
    # Create or use preprocessor
    if is_training or preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
            ],
            remainder='drop'
        )
        X = preprocessor.fit_transform(data)
    else:
        X = preprocessor.transform(data)
    
    # Get feature names
    feature_names = numerical_features.copy()
    if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
        try:
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(cat_features.tolist())
        except Exception:
            # Fallback if feature names don't match
            cat_features = preprocessor.get_feature_names_out()
            feature_names = list(cat_features)
    
    logger.info(f"Preprocessing complete. Feature matrix shape: {X.shape}")
    
    return X, target, preprocessor, feature_names


def apply_smote(
    X_train: np.ndarray, 
    y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Tuple of resampled (X, y)
    """
    if not SMOTE_AVAILABLE:
        logger.warning("SMOTE not available. Install imbalanced-learn: pip install imbalanced-learn")
        return X_train, y_train
    
    logger.info(f"Class distribution before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    smote = SMOTE(**SMOTE_CONFIG)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Class distribution after SMOTE: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    
    return X_resampled, y_resampled


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of available models with their configurations.
    
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        'RandomForest': RandomForestClassifier(**RF_CONFIG)
    }
    
    if XGBOOST_AVAILABLE:
        # Calculate scale_pos_weight for imbalanced data (will be set during training)
        models['XGBoost'] = xgb.XGBClassifier(**XGBOOST_CONFIG)
    else:
        logger.warning("XGBoost not available. Install with: pip install xgboost")
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(**LIGHTGBM_CONFIG)
    else:
        logger.warning("LightGBM not available. Install with: pip install lightgbm")
    
    return models


def cross_validate_model(
    model: Any, 
    X: np.ndarray, 
    y: np.ndarray, 
    cv_folds: int = 5
) -> Dict[str, float]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary with CV scores
    """
    logger.info(f"Performing {cv_folds}-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=MODEL_CONFIG["random_state"])
    
    # Multiple scoring metrics
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        cv_results[f'{metric}_mean'] = scores.mean()
        cv_results[f'{metric}_std'] = scores.std()
        logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return cv_results


def tune_hyperparameters(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    model_name: str,
    use_random_search: bool = True,
    n_iter: int = 20
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    
    Args:
        model: Base model
        X: Features
        y: Labels
        param_grid: Parameter grid
        model_name: Name of the model
        use_random_search: Whether to use RandomizedSearchCV
        n_iter: Number of iterations for random search
        
    Returns:
        Tuple of (best model, best parameters)
    """
    logger.info(f"Tuning hyperparameters for {model_name}...")
    
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, 
                         random_state=MODEL_CONFIG["random_state"])
    
    if use_random_search:
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=MODEL_CONFIG["scoring_metric"],
            n_jobs=-1,
            random_state=MODEL_CONFIG["random_state"],
            verbose=1
        )
    else:
        search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv,
            scoring=MODEL_CONFIG["scoring_metric"],
            n_jobs=-1,
            verbose=1
        )
    
    search.fit(X, y)
    
    logger.info(f"Best {model_name} parameters: {search.best_params_}")
    logger.info(f"Best {model_name} CV score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_


def train_single_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str
) -> Any:
    """
    Train a single model.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        
    Returns:
        Trained model
    """
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"{model_name} training complete")
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"{model_name} Test Performance:")
    logger.info(f"{'='*50}")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return metrics


def compare_models(
    models_results: Dict[str, Dict[str, float]]
) -> str:
    """
    Compare multiple models and select the best one.
    
    Args:
        models_results: Dictionary mapping model names to their metrics
        
    Returns:
        Name of the best model
    """
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    comparison_df = pd.DataFrame(models_results).T
    logger.info(f"\n{comparison_df.to_string()}")
    
    # Select best model based on ROC-AUC
    best_model = max(models_results, key=lambda x: models_results[x]['roc_auc'])
    logger.info(f"\nBest model (by ROC-AUC): {best_model}")
    
    return best_model


def save_model(
    model: Any,
    preprocessor: ColumnTransformer,
    feature_names: List[str],
    model_name: str,
    metrics: Dict[str, float],
    filepath: Optional[str] = None
) -> None:
    """
    Save trained model and preprocessing artifacts.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        model_name: Name of the model
        metrics: Model performance metrics
        filepath: Output file path
    """
    if filepath is None:
        filepath = OUTPUT_CONFIG["model_path"]
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    model_artifacts = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'model_name': model_name,
        'metrics': metrics,
        'config': {
            'data': DATA_CONFIG,
            'features': FEATURE_CONFIG,
            'model': MODEL_CONFIG
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    logger.info(f"Model artifacts saved to {filepath}")


def main(
    tune_models: bool = False,
    use_smote: bool = True,
    compare_all: bool = True
) -> None:
    """
    Main training pipeline.
    
    Args:
        tune_models: Whether to perform hyperparameter tuning
        use_smote: Whether to apply SMOTE for class imbalance
        compare_all: Whether to compare all available models
    """
    logger.info("="*60)
    logger.info("BANK MARKETING PROPENSITY MODEL TRAINING")
    logger.info("="*60 + "\n")
    
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y, preprocessor, feature_names = preprocess_data(df, is_training=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Apply SMOTE if enabled
    if use_smote:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Get models
    models = get_models()
    param_grids = {
        'RandomForest': RF_PARAM_GRID,
        'XGBoost': XGBOOST_PARAM_GRID if XGBOOST_AVAILABLE else {},
        'LightGBM': LIGHTGBM_PARAM_GRID if LIGHTGBM_AVAILABLE else {}
    }
    
    # Train and evaluate models
    models_results = {}
    trained_models = {}
    
    for model_name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {model_name}")
        logger.info(f"{'='*50}")
        
        # Hyperparameter tuning
        if tune_models and model_name in param_grids and param_grids[model_name]:
            model, best_params = tune_hyperparameters(
                model, X_train_resampled, y_train_resampled,
                param_grids[model_name], model_name,
                use_random_search=True, n_iter=10
            )
        else:
            # Train without tuning
            model = train_single_model(model, X_train_resampled, y_train_resampled, model_name)
        
        # Cross-validation
        cv_results = cross_validate_model(model, X_train_resampled, y_train_resampled, 
                                          MODEL_CONFIG["cv_folds"])
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test, model_name)
        
        models_results[model_name] = test_metrics
        trained_models[model_name] = model
        
        if not compare_all:
            break  # Only train the first model
    
    # Compare models and select the best
    if len(models_results) > 1:
        best_model_name = compare_models(models_results)
    else:
        best_model_name = list(models_results.keys())[0]
    
    best_model = trained_models[best_model_name]
    best_metrics = models_results[best_model_name]
    
    # Save the best model
    save_model(
        best_model, preprocessor, feature_names,
        best_model_name, best_metrics
    )
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Bank Marketing Propensity Model')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE')
    parser.add_argument('--single-model', action='store_true', help='Train only RandomForest')
    
    args = parser.parse_args()
    
    main(
        tune_models=args.tune,
        use_smote=not args.no_smote,
        compare_all=not args.single_model
    )
