"""
Scoring script - loads trained model and generates predictions.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

from config import DATA_CONFIG, FEATURE_CONFIG, LOGGING_CONFIG, OUTPUT_CONFIG

# logging setup
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


def load_model(filepath=None):
    """Load model and artifacts from pickle."""
    if filepath is None:
        filepath = OUTPUT_CONFIG["model_path"]
    
    logger.info(f"Loading model from {filepath}")
    
    with open(filepath, 'rb') as f:
        artifacts = pickle.load(f)
    
    model = artifacts['model']
    preprocessor = artifacts['preprocessor']
    feature_names = artifacts.get('feature_names', [])
    model_name = artifacts.get('model_name', 'Unknown')
    config = artifacts.get('config', {})
    
    logger.info(f"Loaded: {model_name}")
    logger.info(f"Metrics: {artifacts.get('metrics', {})}")
    
    return model, preprocessor, feature_names, model_name, config


def load_test_data(filepath=None, delimiter=None):
    """Load test data for scoring."""
    if filepath is None:
        filepath = DATA_CONFIG["test_file"]
    if delimiter is None:
        delimiter = DATA_CONFIG.get("delimiter", ";")
    
    logger.info(f"Loading test data from {filepath}")
    df = pd.read_csv(filepath, delimiter=delimiter)
    logger.info(f"Loaded {len(df)} rows")
    
    return df


def engineer_features(df):
    """
    Feature engineering - same as training.
    # TODO: maybe refactor this to share code with train_model.py
    """
    data = df.copy()
    
    if not FEATURE_CONFIG.get("create_features", True):
        return data
    
    logger.info("Engineering features...")
    
    # age groups
    data['age_group'] = pd.cut(
        data['age'], 
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )
    
    # balance categories
    data['balance_category'] = pd.cut(
        data['balance'],
        bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
        labels=['negative', 'low', 'medium', 'high', 'very_high']
    )
    
    # duration
    data['duration_category'] = pd.cut(
        data['duration'],
        bins=[0, 60, 180, 300, 600, np.inf],
        labels=['very_short', 'short', 'medium', 'long', 'very_long']
    )
    
    # campaign intensity
    data['campaign_intensity'] = pd.cut(
        data['campaign'],
        bins=[0, 1, 3, 5, np.inf],
        labels=['single', 'low', 'medium', 'high']
    )
    
    # contact recency (pdays=-1 means never contacted before)
    data['contact_recency'] = data['pdays'].apply(
        lambda x: 'never_contacted' if x == -1 else (
            'recent' if x <= 30 else ('moderate' if x <= 180 else 'long_ago')
        )
    )
    
    # binary features
    data['had_previous_contact'] = (data['previous'] > 0).astype(int)
    data['balance_per_age'] = data['balance'] / (data['age'] + 1)
    data['duration_per_campaign'] = data['duration'] / (data['campaign'] + 1)
    data['is_employed'] = (~data['job'].isin(['unemployed', 'student', 'retired'])).astype(int)
    data['has_loan'] = ((data['housing'] == 'yes') | (data['loan'] == 'yes')).astype(int)
    data['is_default'] = (data['default'] == 'yes').astype(int)
    
    return data


def preprocess_test_data(df, preprocessor):
    """Preprocess test data using saved preprocessor."""
    data = df.copy()
    
    # drop target if present
    target_col = DATA_CONFIG.get("target_column", "y")
    if target_col in data.columns:
        data = data.drop(target_col, axis=1)
    
    # engineer features
    data = engineer_features(data)
    
    # convert categoricals to string
    nominal = FEATURE_CONFIG.get("nominal_features", [])
    eng_cat = ['age_group', 'balance_category', 'duration_category', 
               'campaign_intensity', 'contact_recency']
    all_cat = list(set(nominal + eng_cat))
    
    for col in all_cat:
        if col in data.columns:
            data[col] = data[col].astype(str)
    
    logger.info("Transforming with preprocessor...")
    X = preprocessor.transform(data)
    logger.info(f"Shape: {X.shape}")
    
    return X


def generate_predictions(model, X_test):
    """Generate predictions and scores."""
    logger.info("Generating predictions...")
    
    preds = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"Generated {len(preds)} predictions")
    return preds, scores


def save_predictions(predictions, scores, original_df=None, 
                     output_path=None, include_original=False):
    """Save predictions to CSV."""
    if output_path is None:
        output_path = OUTPUT_CONFIG["predictions_path"]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    results = pd.DataFrame({
        'prediction': predictions,
        'propensity_score': scores
    })
    
    if include_original and original_df is not None:
        results = pd.concat([original_df.reset_index(drop=True), results], axis=1)
    
    results.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    return results


def generate_summary_statistics(predictions, scores):
    """Get summary stats for predictions."""
    stats = {
        'total_records': len(predictions),
        'predicted_positive': int(predictions.sum()),
        'predicted_negative': int(len(predictions) - predictions.sum()),
        'positive_rate': float(predictions.mean()),
        'avg_propensity_score': float(scores.mean()),
        'median_propensity_score': float(np.median(scores)),
        'min_propensity_score': float(scores.min()),
        'max_propensity_score': float(scores.max()),
        'std_propensity_score': float(scores.std()),
        'propensity_percentiles': {
            '25th': float(np.percentile(scores, 25)),
            '50th': float(np.percentile(scores, 50)),
            '75th': float(np.percentile(scores, 75)),
            '90th': float(np.percentile(scores, 90)),
            '95th': float(np.percentile(scores, 95)),
        }
    }
    return stats


def main(test_data_path=None, model_path=None, output_path=None, include_original=False):
    """Main scoring pipeline."""
    logger.info("="*60)
    logger.info("BANK MARKETING PROPENSITY MODEL - SCORING")
    logger.info("="*60 + "\n")
    
    # load model
    model, preprocessor, feature_names, model_name, config = load_model(model_path)
    
    # load test data
    test_df = load_test_data(test_data_path)
    
    # preprocess
    X_test = preprocess_test_data(test_df, preprocessor)
    
    # predict
    predictions, scores = generate_predictions(model, X_test)
    
    # summary
    stats = generate_summary_statistics(predictions, scores)
    
    logger.info("\n" + "="*50)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Total records: {stats['total_records']}")
    logger.info(f"Predicted positive: {stats['predicted_positive']} ({stats['positive_rate']*100:.2f}%)")
    logger.info(f"Avg propensity: {stats['avg_propensity_score']:.4f}")
    logger.info(f"Range: [{stats['min_propensity_score']:.4f}, {stats['max_propensity_score']:.4f}]")
    logger.info("\nPercentiles:")
    for pct, val in stats['propensity_percentiles'].items():
        logger.info(f"  {pct}: {val:.4f}")
    
    # save
    results = save_predictions(predictions, scores, test_df, output_path, include_original)
    
    logger.info("\n" + "="*60)
    logger.info("SCORING COMPLETE")
    logger.info("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Score data with trained model')
    parser.add_argument('--test-data', type=str, help='Path to test data')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--include-original', action='store_true', 
                        help='Include original data in output')
    
    args = parser.parse_args()
    
    main(
        test_data_path=args.test_data,
        model_path=args.model,
        output_path=args.output,
        include_original=args.include_original
    )
