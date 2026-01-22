"""
Bank Marketing Propensity Model - Scoring Script

This script loads a trained model and generates predictions on new data.
Includes proper preprocessing pipeline usage and comprehensive logging.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    DATA_CONFIG,
    FEATURE_CONFIG,
    LOGGING_CONFIG,
    OUTPUT_CONFIG,
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


def load_model(
    filepath: Optional[str] = None
) -> Tuple[Any, ColumnTransformer, list, str, Dict[str, Any]]:
    """
    Load trained model and preprocessing artifacts.
    
    Args:
        filepath: Path to model file. Defaults to config setting.
        
    Returns:
        Tuple of (model, preprocessor, feature_names, model_name, config)
    """
    if filepath is None:
        filepath = OUTPUT_CONFIG["model_path"]
    
    logger.info(f"Loading model from {filepath}")
    
    with open(filepath, 'rb') as f:
        model_artifacts = pickle.load(f)
    
    model = model_artifacts['model']
    preprocessor = model_artifacts['preprocessor']
    feature_names = model_artifacts.get('feature_names', [])
    model_name = model_artifacts.get('model_name', 'Unknown')
    config = model_artifacts.get('config', {})
    
    logger.info(f"Model loaded: {model_name}")
    logger.info(f"Model metrics: {model_artifacts.get('metrics', {})}")
    
    return model, preprocessor, feature_names, model_name, config


def load_test_data(
    filepath: Optional[str] = None,
    delimiter: Optional[str] = None
) -> pd.DataFrame:
    """
    Load test data for scoring.
    
    Args:
        filepath: Path to test data file. Defaults to config setting.
        delimiter: CSV delimiter. Defaults to config setting.
        
    Returns:
        Loaded DataFrame
    """
    if filepath is None:
        filepath = DATA_CONFIG["test_file"]
    
    if delimiter is None:
        delimiter = DATA_CONFIG.get("delimiter", ";")
    
    logger.info(f"Loading test data from {filepath}")
    
    df = pd.read_csv(filepath, delimiter=delimiter)
    logger.info(f"Test data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the same feature engineering as during training.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    data = df.copy()
    
    if not FEATURE_CONFIG.get("create_features", True):
        return data
    
    logger.info("Performing feature engineering on test data...")
    
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
    
    # Duration categories
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
    
    # Contact recency
    data['contact_recency'] = data['pdays'].apply(
        lambda x: 'never_contacted' if x == -1 else (
            'recent' if x <= 30 else (
                'moderate' if x <= 180 else 'long_ago'
            )
        )
    )
    
    # Binary features
    data['had_previous_contact'] = (data['previous'] > 0).astype(int)
    data['balance_per_age'] = data['balance'] / (data['age'] + 1)
    data['duration_per_campaign'] = data['duration'] / (data['campaign'] + 1)
    data['is_employed'] = (~data['job'].isin(['unemployed', 'student', 'retired'])).astype(int)
    data['has_loan'] = ((data['housing'] == 'yes') | (data['loan'] == 'yes')).astype(int)
    data['is_default'] = (data['default'] == 'yes').astype(int)
    
    return data


def preprocess_test_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer
) -> np.ndarray:
    """
    Preprocess test data using the saved preprocessor.
    
    Args:
        df: Test DataFrame
        preprocessor: Fitted preprocessor from training
        
    Returns:
        Preprocessed feature array
    """
    data = df.copy()
    
    # Remove target column if present
    target_col = DATA_CONFIG.get("target_column", "y")
    if target_col in data.columns:
        data = data.drop(target_col, axis=1)
    
    # Engineer features
    data = engineer_features(data)
    
    # Convert categorical columns to string
    nominal_features = FEATURE_CONFIG.get("nominal_features", [])
    engineered_categorical = ['age_group', 'balance_category', 'duration_category', 
                             'campaign_intensity', 'contact_recency']
    all_categorical = list(set(nominal_features + engineered_categorical))
    
    for col in all_categorical:
        if col in data.columns:
            data[col] = data[col].astype(str)
    
    # Transform using preprocessor
    logger.info("Transforming test data with preprocessor...")
    X = preprocessor.transform(data)
    
    logger.info(f"Preprocessed test data shape: {X.shape}")
    
    return X


def generate_predictions(
    model: Any,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and propensity scores.
    
    Args:
        model: Trained model
        X_test: Preprocessed test features
        
    Returns:
        Tuple of (predictions, propensity_scores)
    """
    logger.info("Generating predictions...")
    
    predictions = model.predict(X_test)
    propensity_scores = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    return predictions, propensity_scores


def save_predictions(
    predictions: np.ndarray,
    propensity_scores: np.ndarray,
    original_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    include_original_data: bool = False
) -> pd.DataFrame:
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Binary predictions
        propensity_scores: Propensity scores
        original_df: Original test DataFrame (optional)
        output_path: Output file path
        include_original_data: Whether to include original data columns
        
    Returns:
        Results DataFrame
    """
    if output_path is None:
        output_path = OUTPUT_CONFIG["predictions_path"]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    results = pd.DataFrame({
        'prediction': predictions,
        'propensity_score': propensity_scores
    })
    
    # Optionally include original data
    if include_original_data and original_df is not None:
        results = pd.concat([original_df.reset_index(drop=True), results], axis=1)
    
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return results


def generate_summary_statistics(
    predictions: np.ndarray,
    propensity_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Generate summary statistics for predictions.
    
    Args:
        predictions: Binary predictions
        propensity_scores: Propensity scores
        
    Returns:
        Dictionary with summary statistics
    """
    stats = {
        'total_records': len(predictions),
        'predicted_positive': int(predictions.sum()),
        'predicted_negative': int(len(predictions) - predictions.sum()),
        'positive_rate': float(predictions.mean()),
        'avg_propensity_score': float(propensity_scores.mean()),
        'median_propensity_score': float(np.median(propensity_scores)),
        'min_propensity_score': float(propensity_scores.min()),
        'max_propensity_score': float(propensity_scores.max()),
        'std_propensity_score': float(propensity_scores.std()),
        'propensity_percentiles': {
            '25th': float(np.percentile(propensity_scores, 25)),
            '50th': float(np.percentile(propensity_scores, 50)),
            '75th': float(np.percentile(propensity_scores, 75)),
            '90th': float(np.percentile(propensity_scores, 90)),
            '95th': float(np.percentile(propensity_scores, 95)),
        }
    }
    
    return stats


def main(
    test_data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    include_original: bool = False
) -> pd.DataFrame:
    """
    Main scoring pipeline.
    
    Args:
        test_data_path: Path to test data
        model_path: Path to trained model
        output_path: Path for output predictions
        include_original: Whether to include original data in output
        
    Returns:
        Predictions DataFrame
    """
    logger.info("="*60)
    logger.info("BANK MARKETING PROPENSITY MODEL SCORING")
    logger.info("="*60 + "\n")
    
    # Load model
    model, preprocessor, feature_names, model_name, config = load_model(model_path)
    
    # Load test data
    test_df = load_test_data(test_data_path)
    
    # Preprocess
    X_test = preprocess_test_data(test_df, preprocessor)
    
    # Generate predictions
    predictions, propensity_scores = generate_predictions(model, X_test)
    
    # Summary statistics
    stats = generate_summary_statistics(predictions, propensity_scores)
    
    logger.info("\n" + "="*50)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*50)
    logger.info(f"Model used: {model_name}")
    logger.info(f"Total records: {stats['total_records']}")
    logger.info(f"Predicted positive: {stats['predicted_positive']} ({stats['positive_rate']*100:.2f}%)")
    logger.info(f"Average propensity: {stats['avg_propensity_score']:.4f}")
    logger.info(f"Propensity range: [{stats['min_propensity_score']:.4f}, {stats['max_propensity_score']:.4f}]")
    logger.info(f"\nPropensity percentiles:")
    for pct, val in stats['propensity_percentiles'].items():
        logger.info(f"  {pct}: {val:.4f}")
    
    # Save results
    results = save_predictions(
        predictions, propensity_scores, 
        test_df, output_path, 
        include_original
    )
    
    logger.info("\n" + "="*60)
    logger.info("SCORING COMPLETE")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Score data using trained model')
    parser.add_argument('--test-data', type=str, help='Path to test data')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--output', type=str, help='Path for output predictions')
    parser.add_argument('--include-original', action='store_true', 
                        help='Include original data in output')
    
    args = parser.parse_args()
    
    main(
        test_data_path=args.test_data,
        model_path=args.model,
        output_path=args.output,
        include_original=args.include_original
    )
