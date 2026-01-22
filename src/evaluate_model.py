"""
Bank Marketing Propensity Model - Evaluation Script

This script provides comprehensive model evaluation including:
- Performance metrics calculation
- Feature importance analysis
- ROC curves and precision-recall curves
- Model comparison visualizations
- Calibration analysis
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    DATA_CONFIG,
    FEATURE_CONFIG,
    LOGGING_CONFIG,
    MODEL_CONFIG,
    OUTPUT_CONFIG,
)

# Import from train_model for preprocessing
from train_model import engineer_features, load_data, preprocess_data

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
) -> Tuple[Any, Any, List[str], str, Dict[str, Any]]:
    """
    Load trained model and artifacts.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Tuple of (model, preprocessor, feature_names, model_name, metrics)
    """
    if filepath is None:
        filepath = OUTPUT_CONFIG["model_path"]
    
    logger.info(f"Loading model from {filepath}")
    
    with open(filepath, 'rb') as f:
        model_artifacts = pickle.load(f)
    
    return (
        model_artifacts['model'],
        model_artifacts['preprocessor'],
        model_artifacts.get('feature_names', []),
        model_artifacts.get('model_name', 'Unknown'),
        model_artifacts.get('metrics', {})
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'avg_precision': average_precision_score(y_true, y_pred_proba)
    }
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    normalize: bool = False
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Output file path
        normalize: Whether to normalize the matrix
    """
    if save_path is None:
        save_path = OUTPUT_CONFIG["confusion_matrix_path"]
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=['No Subscribe', 'Subscribe'],
        yticklabels=['No Subscribe', 'Subscribe']
    )
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None
) -> float:
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        save_path: Output file path
        
    Returns:
        ROC-AUC score
    """
    if save_path is None:
        save_path = OUTPUT_CONFIG["roc_curve_path"]
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to {save_path}")
    
    return roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None
) -> float:
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        save_path: Output file path
        
    Returns:
        Average precision score
    """
    if save_path is None:
        save_path = str(Path(OUTPUT_CONFIG["roc_curve_path"]).parent / "precision_recall_curve.png")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.axhline(y=y_true.mean(), color='navy', lw=2, linestyle='--', 
                label=f'Baseline (Positive Rate = {y_true.mean():.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-Recall curve saved to {save_path}")
    
    return avg_precision


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    save_path: Optional[str] = None,
    top_n: int = 20
) -> Optional[pd.DataFrame]:
    """
    Plot and save feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        save_path: Output file path
        top_n: Number of top features to show
        
    Returns:
        DataFrame with feature importances
    """
    if save_path is None:
        save_path = OUTPUT_CONFIG["feature_importance_path"]
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return None
    
    importances = model.feature_importances_
    
    # Handle case where feature names don't match
    if len(feature_names) != len(importances):
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))
    plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to {save_path}")
    
    return importance_df


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 10
) -> None:
    """
    Plot calibration curve to assess probability reliability.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        save_path: Output file path
        n_bins: Number of bins
    """
    if save_path is None:
        save_path = str(Path(OUTPUT_CONFIG["roc_curve_path"]).parent / "calibration_curve.png")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', color='darkorange', lw=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2, label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Calibration curve saved to {save_path}")


def plot_propensity_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot propensity score distribution by class.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        save_path: Output file path
    """
    if save_path is None:
        save_path = str(Path(OUTPUT_CONFIG["roc_curve_path"]).parent / "propensity_distribution.png")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot distribution for each class
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, 
             label='No Subscribe', color='steelblue', density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, 
             label='Subscribe', color='darkorange', density=True)
    
    plt.xlabel('Propensity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Propensity Score Distribution by Actual Class', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Propensity distribution plot saved to {save_path}")


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    feature_names: List[str],
    model: Any
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        model_name: Name of the model
        feature_names: Feature names
        model: Trained model
        
    Returns:
        Dictionary with all evaluation results
    """
    logger.info("\n" + "="*60)
    logger.info(f"MODEL EVALUATION REPORT - {model_name}")
    logger.info("="*60)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    logger.info("\n--- Performance Metrics ---")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    logger.info(f"\n--- Classification Report ---\n{classification_report(y_true, y_pred)}")
    
    # Generate all plots
    logger.info("\n--- Generating Visualizations ---")
    plot_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, 
                         save_path=str(Path(OUTPUT_CONFIG["confusion_matrix_path"]).parent / "confusion_matrix_normalized.png"),
                         normalize=True)
    plot_roc_curve(y_true, y_pred_proba)
    plot_precision_recall_curve(y_true, y_pred_proba)
    plot_calibration_curve(y_true, y_pred_proba)
    plot_propensity_distribution(y_true, y_pred_proba)
    
    importance_df = plot_feature_importance(model, feature_names)
    
    # Compile report
    report = {
        'model_name': model_name,
        'metrics': metrics,
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'feature_importance': importance_df.to_dict() if importance_df is not None else None
    }
    
    return report


def main(
    model_path: Optional[str] = None,
    data_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main evaluation pipeline.
    
    Args:
        model_path: Path to trained model
        data_path: Path to evaluation data
        
    Returns:
        Evaluation report dictionary
    """
    logger.info("="*60)
    logger.info("BANK MARKETING PROPENSITY MODEL EVALUATION")
    logger.info("="*60 + "\n")
    
    # Load model
    model, preprocessor, feature_names, model_name, saved_metrics = load_model(model_path)
    logger.info(f"Loaded model: {model_name}")
    logger.info(f"Training metrics: {saved_metrics}")
    
    # Load and preprocess data
    logger.info("\nLoading evaluation data...")
    df = load_data(data_path)
    
    # Preprocess data - fit a new preprocessor to ensure consistency
    # This is needed because evaluation should use the same preprocessing as training
    X, y, eval_preprocessor, feature_names = preprocess_data(df, is_training=True)
    
    if y is None:
        logger.error("No target column found in data. Cannot evaluate.")
        return {}
    
    # Train-test split for evaluation (if using training data)
    # This ensures we evaluate on unseen data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y
    )
    
    logger.info(f"Evaluation set size: {len(y_test)}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate comprehensive report
    report = generate_evaluation_report(
        y_test, y_pred, y_pred_proba,
        model_name, feature_names, model
    )
    
    # Save report
    report_path = Path(OUTPUT_CONFIG["roc_curve_path"]).parent / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Model Evaluation Report - {model_name}\n")
        f.write("="*60 + "\n\n")
        f.write("Performance Metrics:\n")
        for metric, value in report['metrics'].items():
            if isinstance(value, float):
                f.write(f"  {metric}: {value:.4f}\n")
            else:
                f.write(f"  {metric}: {value}\n")
        f.write("\n" + classification_report(y_test, y_pred))
    
    logger.info(f"\nEvaluation report saved to {report_path}")
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--data', type=str, help='Path to evaluation data')
    
    args = parser.parse_args()
    
    main(model_path=args.model, data_path=args.data)
