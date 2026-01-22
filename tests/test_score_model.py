# test_score_model.py
# Tests for scoring/prediction functionality

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from score_model import engineer_features, generate_predictions, generate_summary_statistics


class TestEngineerFeatures:
    """Feature engineering tests for scoring"""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'age': [30, 45],
            'balance': [1000, 5000],
            'duration': [200, 400],
            'campaign': [2, 4],
            'pdays': [-1, 100],
            'previous': [0, 2],
            'job': ['admin', 'management'],
            'housing': ['yes', 'no'],
            'loan': ['no', 'yes'],
            'default': ['no', 'no']
        })
    
    def test_all_features_created(self, sample_df):
        result = engineer_features(sample_df)
        
        expected = [
            'age_group', 'balance_category', 'duration_category',
            'campaign_intensity', 'contact_recency', 'had_previous_contact',
            'balance_per_age', 'duration_per_campaign', 'is_employed',
            'has_loan', 'is_default'
        ]
        
        for feat in expected:
            assert feat in result.columns, f"Missing: {feat}"
    
    def test_original_cols_preserved(self, sample_df):
        result = engineer_features(sample_df)
        for col in sample_df.columns:
            assert col in result.columns


class TestGeneratePredictions:
    """Prediction generation tests"""
    
    @pytest.fixture
    def mock_model(self):
        # simple trained model for testing
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    def test_returns_preds_and_scores(self, mock_model):
        X_test = np.random.randn(20, 5)
        preds, scores = generate_predictions(mock_model, X_test)
        
        assert len(preds) == 20
        assert len(scores) == 20
    
    def test_preds_are_binary(self, mock_model):
        X_test = np.random.randn(20, 5)
        preds, _ = generate_predictions(mock_model, X_test)
        
        assert set(np.unique(preds)).issubset({0, 1})
    
    def test_scores_are_probabilities(self, mock_model):
        """scores should be between 0 and 1"""
        X_test = np.random.randn(20, 5)
        _, scores = generate_predictions(mock_model, X_test)
        
        assert (scores >= 0).all()
        assert (scores <= 1).all()


class TestGenerateSummaryStatistics:
    """Summary stats tests"""
    
    def test_all_stats_present(self):
        preds = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        
        stats = generate_summary_statistics(preds, scores)
        
        assert 'total_records' in stats
        assert 'predicted_positive' in stats
        assert 'predicted_negative' in stats
        assert 'positive_rate' in stats
        assert 'avg_propensity_score' in stats
        assert 'propensity_percentiles' in stats
    
    def test_counts_correct(self):
        preds = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        
        stats = generate_summary_statistics(preds, scores)
        
        assert stats['total_records'] == 5
        assert stats['predicted_positive'] == 3
        assert stats['predicted_negative'] == 2
    
    def test_positive_rate(self):
        preds = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        
        stats = generate_summary_statistics(preds, scores)
        assert stats['positive_rate'] == 0.6  # 3/5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
