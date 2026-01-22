"""
Model evaluation script - metrics, plots, and analysis.

Generates ROC curves, confusion matrices, feature importance, etc.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, auc,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from config import (DATA_CONFIG, FEATURE_CONFIG, LOGGING_CONFIG, 
                    MODEL_CONFIG, OUTPUT_CONFIG)
from train_model import engineer_features, load_data, preprocess_data

# logging
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
    """Load trained model and artifacts."""
    if filepath is None:
        filepath = OUTPUT_CONFIG["model_path"]
    
    logger.info(f"Loading model from {filepath}")
    
    with open(filepath, 'rb') as f:
        artifacts = pickle.load(f)
    
    return (
        artifacts['model'],
        artifacts['preprocessor'],
        artifacts.get('feature_names', []),
        artifacts.get('model_name', 'Unknown'),
        artifacts.get('metrics', {})
    )


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate all the evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'avg_precision': average_precision_score(y_true, y_proba)
    }
    
    # confusion matrix breakdown
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=False):
    """Plot confusion matrix heatmap."""
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
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=['No Subscribe', 'Subscribe'],
                yticklabels=['No Subscribe', 'Subscribe'])
    plt.title(title, fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(y_true, y_proba, save_path=None):
    """Plot ROC curve."""
    if save_path is None:
        save_path = OUTPUT_CONFIG["roc_curve_path"]
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved: {save_path}")
    return roc_auc


def plot_precision_recall_curve(y_true, y_proba, save_path=None):
    """Plot precision-recall curve."""
    if save_path is None:
        save_path = str(Path(OUTPUT_CONFIG["roc_curve_path"]).parent / "precision_recall_curve.png")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.axhline(y=y_true.mean(), color='navy', lw=2, linestyle='--', 
                label=f'Baseline ({y_true.mean():.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"PR curve saved: {save_path}")
    return avg_precision


def plot_feature_importance(model, feature_names, save_path=None, top_n=20):
    """Plot top feature importances."""
    if save_path is None:
        save_path = OUTPUT_CONFIG["feature_importance_path"]
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model doesn't have feature_importances_")
        return None
    
    importances = model.feature_importances_
    
    # handle mismatch
    if len(feature_names) != len(importances):
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    # create df and sort
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # plot top N
    top = imp_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top)))
    plt.barh(range(len(top)), top['importance'].values, color=colors)
    plt.yticks(range(len(top)), top['feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance saved: {save_path}")
    return imp_df


def plot_calibration_curve(y_true, y_proba, save_path=None, n_bins=10):
    """Plot calibration curve to check probability reliability."""
    if save_path is None:
        save_path = str(Path(OUTPUT_CONFIG["roc_curve_path"]).parent / "calibration_curve.png")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', color='darkorange', lw=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2, label='Perfect')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Calibration curve saved: {save_path}")


def plot_propensity_distribution(y_true, y_proba, save_path=None):
    """Plot propensity score distribution by actual class."""
    if save_path is None:
        save_path = str(Path(OUTPUT_CONFIG["roc_curve_path"]).parent / "propensity_distribution.png")
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(y_proba[y_true == 0], bins=50, alpha=0.7, 
             label='No Subscribe', color='steelblue', density=True)
    plt.hist(y_proba[y_true == 1], bins=50, alpha=0.7, 
             label='Subscribe', color='darkorange', density=True)
    
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.title('Propensity Score Distribution by Class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Propensity distribution saved: {save_path}")


def generate_evaluation_report(y_true, y_pred, y_proba, model_name, feature_names, model):
    """Generate full evaluation report with all plots."""
    logger.info("\n" + "="*60)
    logger.info(f"EVALUATION REPORT - {model_name}")
    logger.info("="*60)
    
    # metrics
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    logger.info("\n--- Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    
    logger.info(f"\n--- Classification Report ---\n{classification_report(y_true, y_pred)}")
    
    # generate all plots
    logger.info("\n--- Generating plots ---")
    plot_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, 
                         save_path=str(Path(OUTPUT_CONFIG["confusion_matrix_path"]).parent / "confusion_matrix_normalized.png"),
                         normalize=True)
    plot_roc_curve(y_true, y_proba)
    plot_precision_recall_curve(y_true, y_proba)
    plot_calibration_curve(y_true, y_proba)
    plot_propensity_distribution(y_true, y_proba)
    
    imp_df = plot_feature_importance(model, feature_names)
    
    report = {
        'model_name': model_name,
        'metrics': metrics,
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'feature_importance': imp_df.to_dict() if imp_df is not None else None
    }
    
    return report


def main(model_path=None, data_path=None):
    """Main evaluation pipeline."""
    logger.info("="*60)
    logger.info("BANK MARKETING PROPENSITY MODEL - EVALUATION")
    logger.info("="*60 + "\n")
    
    # load model
    model, preprocessor, feature_names, model_name, saved_metrics = load_model(model_path)
    logger.info(f"Model: {model_name}")
    logger.info(f"Saved metrics: {saved_metrics}")
    
    # load and preprocess data
    logger.info("\nLoading data...")
    df = load_data(data_path)
    
    # FIXME: should probably use the same preprocessor from training
    # but for now we just refit to ensure consistency
    X, y, _, feature_names = preprocess_data(df, is_training=True)
    
    if y is None:
        logger.error("No target column found!")
        return {}
    
    # split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y
    )
    
    logger.info(f"Eval set size: {len(y_test)}")
    logger.info(f"Class dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # full report
    report = generate_evaluation_report(y_test, y_pred, y_proba, model_name, feature_names, model)
    
    # save text report
    report_path = Path(OUTPUT_CONFIG["roc_curve_path"]).parent / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Model Evaluation Report - {model_name}\n")
        f.write("="*60 + "\n\n")
        f.write("Performance Metrics:\n")
        for k, v in report['metrics'].items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")
        f.write("\n" + classification_report(y_test, y_pred))
    
    logger.info(f"\nReport saved: {report_path}")
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, help='Model path')
    parser.add_argument('--data', type=str, help='Data path')
    
    args = parser.parse_args()
    main(model_path=args.model, data_path=args.data)
