
"""
Bank Marketing Propensity Model - Evaluation Script

This script provides comprehensive model evaluation including:
- Performance metrics
- Feature importance analysis
- ROC curves and calibration plots
- Model diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_model(filepath='../models/propensity_model.pkl'):
    """Load trained model"""
    with open(filepath, 'rb') as f:
        model_artifacts = pickle.load(f)
    return model_artifacts['model']


def plot_confusion_matrix(y_true, y_pred, save_path='../output/confusion_matrix.png'):
    """Plot confusion matrix"""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_pred_proba, save_path='../output/roc_curve.png'):
    """Plot ROC curve"""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {save_path}")


def plot_feature_importance(model, feature_names, save_path='../output/feature_importance.png'):
    """Plot feature importance"""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to {save_path}")


def generate_evaluation_report(y_true, y_pred, y_pred_proba):
    """Generate comprehensive evaluation report"""
    print("\n=== Model Evaluation Report ===\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Additional metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"\nROC-AUC Score: {roc_auc_score(y_true, y_pred_proba):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_true, y_pred_proba):.4f}")


def main():
    """Main evaluation pipeline"""
    print("=== Bank Marketing Propensity Model Evaluation ===\n")
    
    # Note: This is a template. In practice, you would:
    # 1. Load your test data
    # 2. Load the trained model
    # 3. Generate predictions
    # 4. Create visualizations and reports
    
    print("Evaluation script template created.")
    print("Implement with your specific test data and model.")


if __name__ == "__main__":
    main()
