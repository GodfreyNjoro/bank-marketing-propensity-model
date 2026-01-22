"""
Unit tests for train_model.py

Tests the data loading, preprocessing, feature engineering,
and model training functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_model import (
    engineer_features,
    get_models,
    load_data,
    preprocess_data,
)


class TestLoadData:
    """Tests for load_data function."""
    
    def test_load_data_with_semicolon_delimiter(self, tmp_path):
        """Test that data loads correctly with semicolon delimiter."""
        # Create test data with semicolon delimiter
        test_data = 'age;job;marital;y\n30;admin;single;yes\n40;blue-collar;married;no'
        test_file = tmp_path / "test_bank.csv"
        test_file.write_text(test_data)
        
        # Temporarily override config
        import config
        original_path = config.DATA_CONFIG["train_file"]
        config.DATA_CONFIG["train_file"] = str(test_file)
        
        try:
            df = load_data()
            assert len(df) == 2
            assert 'age' in df.columns
            assert 'job' in df.columns
            assert df['age'].iloc[0] == 30
        finally:
            config.DATA_CONFIG["train_file"] = original_path
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")


class TestEngineerFeatures:
    """Tests for engineer_features function."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'age': [25, 35, 50, 65],
            'balance': [-100, 500, 3000, 15000],
            'duration': [30, 120, 250, 700],
            'campaign': [1, 2, 4, 8],
            'pdays': [-1, 15, 100, 300],
            'previous': [0, 1, 3, 0],
            'job': ['unemployed', 'admin', 'management', 'retired'],
            'housing': ['yes', 'no', 'yes', 'no'],
            'loan': ['no', 'yes', 'no', 'no'],
            'default': ['no', 'no', 'yes', 'no']
        })
    
    def test_age_group_creation(self, sample_df):
        """Test age group feature creation."""
        result = engineer_features(sample_df)
        
        assert 'age_group' in result.columns
        # Age 25 should be in '18-25' or '26-35' depending on binning
        assert result['age_group'].notna().all()
    
    def test_balance_category_creation(self, sample_df):
        """Test balance category feature creation."""
        result = engineer_features(sample_df)
        
        assert 'balance_category' in result.columns
        assert result['balance_category'].iloc[0] == 'negative'  # -100
        assert result['balance_category'].iloc[1] == 'low'  # 500
        assert result['balance_category'].iloc[2] == 'high'  # 3000
        assert result['balance_category'].iloc[3] == 'very_high'  # 15000
    
    def test_contact_recency_creation(self, sample_df):
        """Test contact recency feature creation."""
        result = engineer_features(sample_df)
        
        assert 'contact_recency' in result.columns
        assert result['contact_recency'].iloc[0] == 'never_contacted'  # pdays = -1
        assert result['contact_recency'].iloc[1] == 'recent'  # pdays = 15
    
    def test_binary_features_creation(self, sample_df):
        """Test binary feature creation."""
        result = engineer_features(sample_df)
        
        assert 'had_previous_contact' in result.columns
        assert 'is_employed' in result.columns
        assert 'has_loan' in result.columns
        assert 'is_default' in result.columns
        
        # Check is_employed: unemployed and retired should be 0
        assert result['is_employed'].iloc[0] == 0  # unemployed
        assert result['is_employed'].iloc[1] == 1  # admin
        assert result['is_employed'].iloc[3] == 0  # retired
    
    def test_interaction_features_creation(self, sample_df):
        """Test interaction feature creation."""
        result = engineer_features(sample_df)
        
        assert 'balance_per_age' in result.columns
        assert 'duration_per_campaign' in result.columns
        
        # Verify calculations
        expected_balance_per_age = sample_df['balance'] / (sample_df['age'] + 1)
        np.testing.assert_array_almost_equal(
            result['balance_per_age'].values,
            expected_balance_per_age.values
        )


class TestPreprocessData:
    """Tests for preprocess_data function."""
    
    @pytest.fixture
    def sample_df_with_target(self):
        """Create sample DataFrame with target variable."""
        return pd.DataFrame({
            'age': [25, 35, 50, 65, 30, 45],
            'balance': [-100, 500, 3000, 15000, 200, 5000],
            'duration': [30, 120, 250, 700, 90, 400],
            'campaign': [1, 2, 4, 8, 1, 3],
            'pdays': [-1, 15, 100, 300, -1, 50],
            'previous': [0, 1, 3, 0, 0, 2],
            'day': [1, 15, 20, 10, 5, 25],
            'job': ['unemployed', 'admin', 'management', 'retired', 'student', 'technician'],
            'marital': ['single', 'married', 'divorced', 'married', 'single', 'married'],
            'education': ['primary', 'secondary', 'tertiary', 'secondary', 'tertiary', 'secondary'],
            'default': ['no', 'no', 'yes', 'no', 'no', 'no'],
            'housing': ['yes', 'no', 'yes', 'no', 'yes', 'yes'],
            'loan': ['no', 'yes', 'no', 'no', 'no', 'yes'],
            'contact': ['cellular', 'telephone', 'cellular', 'unknown', 'cellular', 'telephone'],
            'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun'],
            'poutcome': ['unknown', 'success', 'failure', 'unknown', 'unknown', 'success'],
            'y': ['no', 'yes', 'no', 'yes', 'no', 'yes']
        })
    
    def test_target_extraction(self, sample_df_with_target):
        """Test that target variable is correctly extracted."""
        X, y, preprocessor, feature_names = preprocess_data(sample_df_with_target)
        
        assert y is not None
        assert len(y) == 6
        # 'yes' should be 1, 'no' should be 0
        expected_y = np.array([0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(y, expected_y)
    
    def test_preprocessor_fit(self, sample_df_with_target):
        """Test that preprocessor is fitted correctly."""
        X, y, preprocessor, feature_names = preprocess_data(sample_df_with_target)
        
        assert preprocessor is not None
        assert 'num' in preprocessor.named_transformers_
        assert 'cat' in preprocessor.named_transformers_
    
    def test_feature_names_created(self, sample_df_with_target):
        """Test that feature names are created."""
        X, y, preprocessor, feature_names = preprocess_data(sample_df_with_target)
        
        assert feature_names is not None
        assert len(feature_names) > 0
    
    def test_numerical_scaling(self, sample_df_with_target):
        """Test that numerical features are scaled."""
        X, y, preprocessor, feature_names = preprocess_data(sample_df_with_target)
        
        # Check that numerical features have mean close to 0 and std close to 1
        # (for scaled features)
        num_features = X[:, :7]  # First 7 columns should be numerical
        
        # Mean should be close to 0 (allowing for small sample size variation)
        assert np.abs(num_features.mean(axis=0)).max() < 0.5


class TestGetModels:
    """Tests for get_models function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        models = get_models()
        assert isinstance(models, dict)
    
    def test_contains_random_forest(self):
        """Test that RandomForest is always available."""
        models = get_models()
        assert 'RandomForest' in models
    
    def test_models_have_fit_method(self):
        """Test that all models have fit method."""
        models = get_models()
        for name, model in models.items():
            assert hasattr(model, 'fit'), f"{name} does not have fit method"
            assert hasattr(model, 'predict'), f"{name} does not have predict method"
            assert hasattr(model, 'predict_proba'), f"{name} does not have predict_proba method"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
