# test_evaluate_model.py
# Tests for evaluation metrics

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate_model import calculate_metrics


class TestCalculateMetrics:
    """Metric calculation tests"""
    
    def test_perfect_predictions(self):
        """all correct should give 1.0 everywhere"""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.95])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_worst_predictions(self):
        """all wrong should give 0 accuracy"""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.2, 0.1, 0.05])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert metrics['accuracy'] == 0.0
        assert metrics['recall'] == 0.0
    
    def test_confusion_matrix_values(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])  # 1 FP
        y_proba = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        # verify all cm components present
        assert 'true_positives' in metrics
        assert 'true_negatives' in metrics
        assert 'false_positives' in metrics
        assert 'false_negatives' in metrics
        
        assert metrics['true_positives'] == 2
        assert metrics['true_negatives'] == 2
        assert metrics['false_positives'] == 1
        assert metrics['false_negatives'] == 0
    
    def test_roc_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.3, 0.7, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_avg_precision(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.3, 0.7, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'avg_precision' in metrics
        assert 0 <= metrics['avg_precision'] <= 1
    
    def test_specificity(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])  # TN=2, FP=1
        y_proba = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        # specificity = TN / (TN + FP) = 2/3
        assert abs(metrics['specificity'] - 2/3) < 0.01


class TestEdgeCases:
    """Edge case tests"""
    
    def test_all_positive_preds(self):
        """when everything is predicted positive"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        y_proba = np.array([0.6, 0.7, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        # precision = 2/4 = 0.5 (2 actual positives out of 4 predicted)
        assert metrics['precision'] == 0.5
        # recall = 1.0 (got all actual positives)
        assert metrics['recall'] == 1.0
    
    def test_all_negative_preds(self):
        """when everything is predicted negative"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        # with zero_division=0, precision=0 when no positive preds
        assert metrics['precision'] == 0
        assert metrics['recall'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
