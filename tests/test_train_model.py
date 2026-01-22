# test_train_model.py
# Tests for data loading, preprocessing, and model training

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train_model import engineer_features, get_models, load_data, preprocess_data


class TestLoadData:
    """Data loading tests"""
    
    def test_loads_with_semicolon_delimiter(self, tmp_path):
        """make sure it handles semicolon-separated files"""
        test_data = 'age;job;marital;y\n30;admin;single;yes\n40;blue-collar;married;no'
        test_file = tmp_path / "test_bank.csv"
        test_file.write_text(test_data)
        
        # temporarily override config
        import config
        orig_path = config.DATA_CONFIG["train_file"]
        config.DATA_CONFIG["train_file"] = str(test_file)
        
        try:
            df = load_data()
            assert len(df) == 2
            assert 'age' in df.columns
            assert df['age'].iloc[0] == 30
        finally:
            config.DATA_CONFIG["train_file"] = orig_path
    
    def test_missing_file_raises_error(self):
        """should raise FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path/data.csv")


class TestEngineerFeatures:
    """Feature engineering tests"""
    
    @pytest.fixture
    def sample_df(self):
        """sample data for testing"""
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
    
    def test_age_group_created(self, sample_df):
        result = engineer_features(sample_df)
        assert 'age_group' in result.columns
        assert result['age_group'].notna().all()
    
    def test_balance_category_created(self, sample_df):
        result = engineer_features(sample_df)
        
        assert 'balance_category' in result.columns
        assert result['balance_category'].iloc[0] == 'negative'
        assert result['balance_category'].iloc[1] == 'low'
        assert result['balance_category'].iloc[2] == 'high'
        assert result['balance_category'].iloc[3] == 'very_high'
    
    def test_contact_recency_logic(self, sample_df):
        """verify pdays=-1 maps to never_contacted"""
        result = engineer_features(sample_df)
        
        assert 'contact_recency' in result.columns
        assert result['contact_recency'].iloc[0] == 'never_contacted'  # pdays=-1
        assert result['contact_recency'].iloc[1] == 'recent'  # pdays=15
    
    def test_binary_flags(self, sample_df):
        result = engineer_features(sample_df)
        
        # check all binary features exist
        assert 'had_previous_contact' in result.columns
        assert 'is_employed' in result.columns
        assert 'has_loan' in result.columns
        assert 'is_default' in result.columns
        
        # unemployed and retired should be 0 for is_employed
        assert result['is_employed'].iloc[0] == 0
        assert result['is_employed'].iloc[1] == 1
        assert result['is_employed'].iloc[3] == 0
    
    def test_interaction_features(self, sample_df):
        result = engineer_features(sample_df)
        
        assert 'balance_per_age' in result.columns
        assert 'duration_per_campaign' in result.columns
        
        # verify math
        expected = sample_df['balance'] / (sample_df['age'] + 1)
        np.testing.assert_array_almost_equal(
            result['balance_per_age'].values,
            expected.values
        )


class TestPreprocessData:
    """Preprocessing tests"""
    
    @pytest.fixture
    def sample_df_with_target(self):
        """sample data with target column"""
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
        """verify target is extracted correctly"""
        X, y, preprocessor, feature_names = preprocess_data(sample_df_with_target)
        
        assert y is not None
        assert len(y) == 6
        # yes -> 1, no -> 0
        expected_y = np.array([0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(y, expected_y)
    
    def test_preprocessor_fitted(self, sample_df_with_target):
        X, y, preprocessor, feature_names = preprocess_data(sample_df_with_target)
        
        assert preprocessor is not None
        assert 'num' in preprocessor.named_transformers_
        assert 'cat' in preprocessor.named_transformers_
    
    def test_feature_names_created(self, sample_df_with_target):
        X, y, preprocessor, feature_names = preprocess_data(sample_df_with_target)
        
        assert feature_names is not None
        assert len(feature_names) > 0
    
    def test_numerical_scaling(self, sample_df_with_target):
        """check that numericals are roughly standardized"""
        X, y, preprocessor, feature_names = preprocess_data(sample_df_with_target)
        
        # first 7 cols should be numerical - check mean is close to 0
        num_features = X[:, :7]
        assert np.abs(num_features.mean(axis=0)).max() < 0.5


class TestGetModels:
    """Model getter tests"""
    
    def test_returns_dict(self):
        models = get_models()
        assert isinstance(models, dict)
    
    def test_has_random_forest(self):
        """RF should always be available"""
        models = get_models()
        assert 'RandomForest' in models
    
    def test_models_have_required_methods(self):
        """all models need fit, predict, predict_proba"""
        models = get_models()
        for name, model in models.items():
            assert hasattr(model, 'fit'), f"{name} missing fit"
            assert hasattr(model, 'predict'), f"{name} missing predict"
            assert hasattr(model, 'predict_proba'), f"{name} missing predict_proba"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
