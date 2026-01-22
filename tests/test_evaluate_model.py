"""
Unit tests for evaluate_model.py

Tests the evaluation metrics and visualization functions.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate_model import calculate_metrics


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_worst_predictions(self):
        """Test metrics with completely wrong predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0, 0])
        y_pred_proba = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert metrics['accuracy'] == 0.0
        assert metrics['recall'] == 0.0
    
    def test_confusion_matrix_values(self):
        """Test that confusion matrix values are calculated."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])  # 1 FP
        y_pred_proba = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'true_positives' in metrics
        assert 'true_negatives' in metrics
        assert 'false_positives' in metrics
        assert 'false_negatives' in metrics
        
        assert metrics['true_positives'] == 2  # Correctly predicted positives
        assert metrics['true_negatives'] == 2  # Correctly predicted negatives
        assert metrics['false_positives'] == 1  # Incorrectly predicted as positive
        assert metrics['false_negatives'] == 0  # Incorrectly predicted as negative
    
    def test_roc_auc_calculation(self):
        """Test ROC-AUC score calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.3, 0.7, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_average_precision_calculation(self):
        """Test average precision score calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.3, 0.7, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'avg_precision' in metrics
        assert 0 <= metrics['avg_precision'] <= 1
    
    def test_specificity_calculation(self):
        """Test specificity calculation."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])  # TN=2, FP=1
        y_pred_proba = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Specificity = TN / (TN + FP) = 2 / (2 + 1) = 0.667
        expected_specificity = 2 / 3
        assert abs(metrics['specificity'] - expected_specificity) < 0.01


class TestMetricsEdgeCases:
    """Tests for edge cases in metric calculation."""
    
    def test_all_positive_predictions(self):
        """Test metrics when all predictions are positive."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])  # All predicted as positive
        y_pred_proba = np.array([0.6, 0.7, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Precision should be 0.5 (2 TP out of 4 predicted positive)
        assert metrics['precision'] == 0.5
        # Recall should be 1.0 (all positives correctly identified)
        assert metrics['recall'] == 1.0
    
    def test_all_negative_predictions(self):
        """Test metrics when all predictions are negative."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])  # All predicted as negative
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])
        
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # With zero_division=0, precision should be 0 when no positive predictions
        assert metrics['precision'] == 0
        # Recall should be 0 (no positives identified)
        assert metrics['recall'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
