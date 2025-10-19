
"""
Bank Marketing Propensity Model - Scoring Script

This script loads a trained model and generates predictions on new data.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_model(filepath='../models/propensity_model.pkl'):
    """Load trained model and encoders"""
    with open(filepath, 'rb') as f:
        model_artifacts = pickle.load(f)
    
    return model_artifacts['model'], model_artifacts['label_encoders']


def load_test_data(filepath='../data/test.csv'):
    """Load test data for scoring"""
    df = pd.read_csv(filepath)
    print(f"Test data loaded: {df.shape}")
    return df


def preprocess_test_data(df, label_encoders):
    """Preprocess test data using saved encoders"""
    data = df.copy()
    
    # Encode categorical variables using saved encoders
    for col, le in label_encoders.items():
        if col in data.columns:
            # Handle unseen categories
            data[col] = data[col].astype(str)
            data[col] = data[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            data[col] = le.transform(data[col])
    
    return data


def generate_predictions(model, X_test):
    """Generate predictions and propensity scores"""
    # Predictions
    predictions = model.predict(X_test)
    propensity_scores = model.predict_proba(X_test)[:, 1]
    
    return predictions, propensity_scores


def save_predictions(predictions, propensity_scores, output_path='../output/predictions.csv'):
    """Save predictions to CSV file"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = pd.DataFrame({
        'prediction': predictions,
        'propensity_score': propensity_scores
    })
    
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    return results


def main():
    """Main scoring pipeline"""
    print("=== Bank Marketing Propensity Model Scoring ===\n")
    
    # Load model
    print("Loading trained model...")
    model, label_encoders = load_model()
    
    # Load test data
    test_df = load_test_data()
    
    # Preprocess
    print("Preprocessing test data...")
    X_test = preprocess_test_data(test_df, label_encoders)
    
    # Generate predictions
    print("Generating predictions...")
    predictions, propensity_scores = generate_predictions(model, X_test)
    
    # Summary statistics
    print(f"\nPrediction Summary:")
    print(f"Total records: {len(predictions)}")
    print(f"Predicted positive: {predictions.sum()} ({predictions.mean()*100:.2f}%)")
    print(f"Average propensity score: {propensity_scores.mean():.4f}")
    print(f"Propensity score range: [{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")
    
    # Save results
    results = save_predictions(predictions, propensity_scores)
    
    print("\n=== Scoring Complete ===")
    
    return results


if __name__ == "__main__":
    main()
