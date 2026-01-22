"""
Training script for bank marketing propensity model.

Handles data loading, preprocessing, model training with multiple algos,
cross-validation, hyperparameter tuning, and SMOTE for class imbalance.
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# optional deps - might not be installed
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

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

from config import (DATA_CONFIG, FEATURE_CONFIG, LIGHTGBM_CONFIG, LIGHTGBM_PARAM_GRID,
                    LOGGING_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG, RF_CONFIG, RF_PARAM_GRID,
                    SMOTE_CONFIG, XGBOOST_CONFIG, XGBOOST_PARAM_GRID)

# set up logging
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


def load_data(filepath=None):
    """Load training data. Uses semicolon delimiter by default."""
    if filepath is None:
        filepath = DATA_CONFIG["train_file"]
    
    delimiter = DATA_CONFIG.get("delimiter", ";")
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, delimiter=delimiter)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols")
    
    return df


def engineer_features(df):
    """
    Create derived features from raw data.
    Adds age groups, balance categories, etc.
    """
    data = df.copy()
    
    if not FEATURE_CONFIG.get("create_features", True):
        return data
    
    logger.info("Engineering features...")
    
    # age buckets
    data['age_group'] = pd.cut(
        data['age'], 
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )
    
    # balance buckets
    data['balance_category'] = pd.cut(
        data['balance'],
        bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
        labels=['negative', 'low', 'medium', 'high', 'very_high']
    )
    
    # call duration buckets
    data['duration_category'] = pd.cut(
        data['duration'],
        bins=[0, 60, 180, 300, 600, np.inf],
        labels=['very_short', 'short', 'medium', 'long', 'very_long']
    )
    
    # how many times contacted
    data['campaign_intensity'] = pd.cut(
        data['campaign'],
        bins=[0, 1, 3, 5, np.inf],
        labels=['single', 'low', 'medium', 'high']
    )
    
    # when was last contact
    data['contact_recency'] = data['pdays'].apply(
        lambda x: 'never_contacted' if x == -1 else (
            'recent' if x <= 30 else ('moderate' if x <= 180 else 'long_ago')
        )
    )
    
    # binary indicators
    data['had_previous_contact'] = (data['previous'] > 0).astype(int)
    data['balance_per_age'] = data['balance'] / (data['age'] + 1)
    data['duration_per_campaign'] = data['duration'] / (data['campaign'] + 1)
    data['is_employed'] = (~data['job'].isin(['unemployed', 'student', 'retired'])).astype(int)
    data['has_loan'] = ((data['housing'] == 'yes') | (data['loan'] == 'yes')).astype(int)
    data['is_default'] = (data['default'] == 'yes').astype(int)
    
    logger.info(f"Feature engineering done. Shape: {data.shape}")
    return data


def preprocess_data(df, is_training=True, preprocessor=None):
    """
    Preprocess data with proper encoding and scaling.
    Returns X, y, preprocessor, and feature names.
    """
    data = df.copy()
    
    # extract target
    target = None
    target_col = DATA_CONFIG.get("target_column", "y")
    pos_class = DATA_CONFIG.get("positive_class", "yes")
    
    if target_col in data.columns:
        target = (data[target_col] == pos_class).astype(int).values
        data = data.drop(target_col, axis=1)
    
    # engineer features
    data = engineer_features(data)
    
    # get feature lists
    nominal = FEATURE_CONFIG.get("nominal_features", [])
    numerical = FEATURE_CONFIG.get("numerical_features", [])
    
    # add engineered categoricals
    eng_cat = ['age_group', 'balance_category', 'duration_category', 
               'campaign_intensity', 'contact_recency']
    nominal = list(set(nominal + eng_cat))
    
    # add engineered numericals
    eng_num = ['balance_per_age', 'duration_per_campaign', 
               'had_previous_contact', 'is_employed', 'has_loan', 'is_default']
    numerical = list(set(numerical + eng_num))
    
    # filter to existing cols only
    nominal = [f for f in nominal if f in data.columns]
    numerical = [f for f in numerical if f in data.columns]
    
    logger.info(f"Nominal features: {nominal}")
    logger.info(f"Numerical features: {numerical}")
    
    # convert cats to string for encoder
    for col in nominal:
        data[col] = data[col].astype(str)
    
    # build or use preprocessor
    if is_training or preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal)
            ],
            remainder='drop'
        )
        X = preprocessor.fit_transform(data)
    else:
        X = preprocessor.transform(data)
    
    # get feature names
    feature_names = numerical.copy()
    if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
        try:
            cat_feats = preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(cat_feats.tolist())
        except:
            # fallback
            cat_feats = preprocessor.get_feature_names_out()
            feature_names = list(cat_feats)
    
    logger.info(f"Preprocessing done. Shape: {X.shape}")
    return X, target, preprocessor, feature_names


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance classes."""
    if not SMOTE_AVAILABLE:
        logger.warning("SMOTE not available. pip install imbalanced-learn")
        return X_train, y_train
    
    logger.info(f"Class dist before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    smote = SMOTE(**SMOTE_CONFIG)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Class dist after SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def get_models():
    """Get dict of available models."""
    models = {'RandomForest': RandomForestClassifier(**RF_CONFIG)}
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(**XGBOOST_CONFIG)
    else:
        logger.warning("XGBoost not available")
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(**LIGHTGBM_CONFIG)
    else:
        logger.warning("LightGBM not available")
    
    return models


def cross_validate_model(model, X, y, cv_folds=5):
    """Run k-fold CV and return scores."""
    logger.info(f"Running {cv_folds}-fold CV...")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=MODEL_CONFIG["random_state"])
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = {}
    
    for metric in metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        results[f'{metric}_mean'] = scores.mean()
        results[f'{metric}_std'] = scores.std()
        logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return results


def tune_hyperparameters(model, X, y, param_grid, model_name, 
                         use_random_search=True, n_iter=20):
    """Tune hyperparameters with grid or random search."""
    logger.info(f"Tuning {model_name}...")
    
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, 
                         random_state=MODEL_CONFIG["random_state"])
    
    if use_random_search:
        search = RandomizedSearchCV(
            model, param_distributions=param_grid, n_iter=n_iter,
            cv=cv, scoring=MODEL_CONFIG["scoring_metric"],
            n_jobs=-1, random_state=MODEL_CONFIG["random_state"], verbose=1
        )
    else:
        search = GridSearchCV(
            model, param_grid=param_grid, cv=cv,
            scoring=MODEL_CONFIG["scoring_metric"], n_jobs=-1, verbose=1
        )
    
    search.fit(X, y)
    
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_


def train_single_model(model, X_train, y_train, model_name):
    """Train a single model."""
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"{model_name} training complete")
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"{model_name} Test Performance:")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return metrics


def compare_models(models_results):
    """Compare models and pick the best one by ROC-AUC."""
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    df = pd.DataFrame(models_results).T
    logger.info(f"\n{df.to_string()}")
    
    # best by roc_auc
    best = max(models_results, key=lambda x: models_results[x]['roc_auc'])
    logger.info(f"\nBest model (by ROC-AUC): {best}")
    
    return best


def save_model(model, preprocessor, feature_names, model_name, metrics, filepath=None):
    """Save model and artifacts to pickle."""
    if filepath is None:
        filepath = OUTPUT_CONFIG["model_path"]
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
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
        pickle.dump(artifacts, f)
    
    logger.info(f"Model saved to {filepath}")


def main(tune_models=False, use_smote=True, compare_all=True):
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("BANK MARKETING PROPENSITY MODEL - TRAINING")
    logger.info("="*60 + "\n")
    
    # load data
    df = load_data()
    
    # preprocess
    X, y, preprocessor, feature_names = preprocess_data(df, is_training=True)
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Target dist: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # apply SMOTE if enabled
    if use_smote:
        X_train_res, y_train_res = apply_smote(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    
    # get models
    models = get_models()
    param_grids = {
        'RandomForest': RF_PARAM_GRID,
        'XGBoost': XGBOOST_PARAM_GRID if XGBOOST_AVAILABLE else {},
        'LightGBM': LIGHTGBM_PARAM_GRID if LIGHTGBM_AVAILABLE else {}
    }
    
    # train and evaluate
    results = {}
    trained = {}
    
    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {name}")
        logger.info(f"{'='*50}")
        
        # tune if requested
        if tune_models and name in param_grids and param_grids[name]:
            model, _ = tune_hyperparameters(
                model, X_train_res, y_train_res,
                param_grids[name], name,
                use_random_search=True, n_iter=10
            )
        else:
            model = train_single_model(model, X_train_res, y_train_res, name)
        
        # cross-validate
        cross_validate_model(model, X_train_res, y_train_res, MODEL_CONFIG["cv_folds"])
        
        # test set eval
        test_metrics = evaluate_model(model, X_test, y_test, name)
        
        results[name] = test_metrics
        trained[name] = model
        
        if not compare_all:
            break  # just train one model
    
    # pick best
    if len(results) > 1:
        best_name = compare_models(results)
    else:
        best_name = list(results.keys())[0]
    
    best_model = trained[best_name]
    best_metrics = results[best_name]
    
    # save
    save_model(best_model, preprocessor, feature_names, best_name, best_metrics)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best model: {best_name}")
    logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train propensity model')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE')
    parser.add_argument('--single-model', action='store_true', help='Only train RandomForest')
    
    args = parser.parse_args()
    
    main(
        tune_models=args.tune,
        use_smote=not args.no_smote,
        compare_all=not args.single_model
    )
