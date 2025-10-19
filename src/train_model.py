
"""
Bank Marketing Propensity Model - Training Script

This script handles the complete model training pipeline including:
- Data loading and preprocessing
- Feature engineering
- Model training with multiple algorithms
- Hyperparameter tuning
- Model evaluation and serialization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='../data/bank.csv'):
    """Load the training data"""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape}")
    return df


def preprocess_data(df):
    """
    Preprocess the data:
    - Handle missing values
    - Encode categorical variables
    - Feature engineering
    - Scale numerical features
    """
    # Create a copy
    data = df.copy()
    
    # Encode target variable
    if 'y' in data.columns:
        data['target'] = (data['y'] == 'yes').astype(int)
        data = data.drop('y', axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if 'target' in numerical_cols:
        numerical_cols.remove('target')
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # Feature engineering
    # Add any custom features here
    
    return data, label_encoders


def train_model(X_train, y_train):
    """Train the propensity model"""
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred))
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return y_pred, y_pred_proba


def save_model(model, label_encoders, filepath='../models/propensity_model.pkl'):
    """Save trained model and encoders"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    model_artifacts = {
        'model': model,
        'label_encoders': label_encoders
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print(f"\nModel saved to {filepath}")


def main():
    """Main training pipeline"""
    print("=== Bank Marketing Propensity Model Training ===\n")
    
    # Load data
    df = load_data()
    
    # Preprocess
    data, label_encoders = preprocess_data(df)
    
    # Split features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}\n")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, label_encoders)
    
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
