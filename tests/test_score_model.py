"""
Unit tests for score_model.py

Tests the scoring, prediction generation, and output functionality.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from score_model import (
    engineer_features,
    generate_predictions,
    generate_summary_statistics,
)


class TestEngineerFeatures:
    """Tests for engineer_features function in score_model."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
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
    
    def test_features_created(self, sample_df):
        """Test that all engineered features are created."""
        result = engineer_features(sample_df)
        
        expected_features = [
            'age_group', 'balance_category', 'duration_category',
            'campaign_intensity', 'contact_recency', 'had_previous_contact',
            'balance_per_age', 'duration_per_campaign', 'is_employed',
            'has_loan', 'is_default'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
    
    def test_original_columns_preserved(self, sample_df):
        """Test that original columns are preserved."""
        result = engineer_features(sample_df)
        
        for col in sample_df.columns:
            assert col in result.columns, f"Original column {col} not preserved"


class TestGeneratePredictions:
    """Tests for generate_predictions function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a simple mock model."""
        # Create and train a simple model
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    def test_returns_predictions_and_scores(self, mock_model):
        """Test that function returns both predictions and scores."""
        X_test = np.random.randn(20, 5)
        predictions, scores = generate_predictions(mock_model, X_test)
        
        assert len(predictions) == 20
        assert len(scores) == 20
    
    def test_predictions_are_binary(self, mock_model):
        """Test that predictions are binary (0 or 1)."""
        X_test = np.random.randn(20, 5)
        predictions, _ = generate_predictions(mock_model, X_test)
        
        assert set(np.unique(predictions)).issubset({0, 1})
    
    def test_scores_are_probabilities(self, mock_model):
        """Test that scores are valid probabilities (between 0 and 1)."""
        X_test = np.random.randn(20, 5)
        _, scores = generate_predictions(mock_model, X_test)
        
        assert (scores >= 0).all()
        assert (scores <= 1).all()


class TestGenerateSummaryStatistics:
    """Tests for generate_summary_statistics function."""
    
    def test_all_statistics_calculated(self):
        """Test that all statistics are calculated."""
        predictions = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        
        stats = generate_summary_statistics(predictions, scores)
        
        assert 'total_records' in stats
        assert 'predicted_positive' in stats
        assert 'predicted_negative' in stats
        assert 'positive_rate' in stats
        assert 'avg_propensity_score' in stats
        assert 'propensity_percentiles' in stats
    
    def test_total_records_correct(self):
        """Test that total records is correct."""
        predictions = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        
        stats = generate_summary_statistics(predictions, scores)
        
        assert stats['total_records'] == 5
    
    def test_predicted_positive_correct(self):
        """Test that predicted positive count is correct."""
        predictions = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        
        stats = generate_summary_statistics(predictions, scores)
        
        assert stats['predicted_positive'] == 3
        assert stats['predicted_negative'] == 2
    
    def test_positive_rate_correct(self):
        """Test that positive rate is calculated correctly."""
        predictions = np.array([0, 1, 0, 1, 1])
        scores = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
        
        stats = generate_summary_statistics(predictions, scores)
        
        assert stats['positive_rate'] == 0.6  # 3/5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
