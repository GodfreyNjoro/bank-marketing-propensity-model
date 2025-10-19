# Bank Marketing Propensity Model

## Project Overview
This repository contains a complete end-to-end bank marketing propensity modeling project. The goal is to predict customer propensity to subscribe to a term deposit based on marketing campaign data.

## Project Structure

```
bank-marketing-propensity-model/
├── data/                          # Data files
│   ├── bank.csv                   # Training dataset
│   ├── test.csv                   # Test dataset
│   ├── sample_submission.csv      # Sample submission format
│   └── bank-names.txt             # Data dictionary and column descriptions
├── src/                           # Source code
│   ├── train_model.py            # Model training script
│   ├── score_model.py            # Scoring/prediction script
│   ├── evaluate_model.py         # Model evaluation script
│   └── postgres_integration.py   # PostgreSQL database integration
├── notebooks/                     # Jupyter notebooks for exploration
├── docs/                          # Documentation
└── README.md                      # This file
```

## Dataset Description

The dataset contains information about bank marketing campaigns with the following key features:
- Customer demographics (age, job, marital status, education)
- Financial information (balance, housing loan, personal loan)
- Campaign details (contact type, duration, number of contacts)
- Previous campaign outcomes
- Target variable: Whether the customer subscribed to a term deposit (yes/no)

See `data/bank-names.txt` for detailed column descriptions.

## Project Components

### 1. Model Training
- Feature engineering and preprocessing
- Model selection and hyperparameter tuning
- Cross-validation and performance evaluation
- Model serialization for deployment

### 2. Scoring Pipeline
- Load trained model
- Process new data
- Generate predictions
- Output results in required format

### 3. Model Evaluation
- Performance metrics (accuracy, precision, recall, F1-score, AUC-ROC)
- Confusion matrix analysis
- Feature importance analysis
- Model diagnostics

### 4. Database Integration
- PostgreSQL connection setup
- Data ingestion from database
- Prediction results storage
- Query optimization

### 5. Power BI Integration
- Connect to PostgreSQL database
- Create interactive dashboards
- Visualize model performance
- Monitor campaign effectiveness

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm
pip install psycopg2-binary sqlalchemy
pip install matplotlib seaborn plotly
```

### Running the Project

1. **Train the model:**
```bash
python src/train_model.py
```

2. **Generate predictions:**
```bash
python src/score_model.py
```

3. **Evaluate model performance:**
```bash
python src/evaluate_model.py
```

4. **Database operations:**
```bash
python src/postgres_integration.py
```

## Model Performance

The model uses ensemble methods (Random Forest, XGBoost, LightGBM) to predict customer propensity. Key performance metrics will be documented here after training.

## Power BI Dashboard

The Power BI dashboard provides:
- Campaign performance overview
- Customer segmentation analysis
- Propensity score distribution
- Feature importance visualization
- ROI analysis

## Database Schema

PostgreSQL tables:
- `customers`: Customer demographic and financial data
- `campaigns`: Marketing campaign details
- `predictions`: Model predictions and propensity scores
- `model_performance`: Model evaluation metrics

## Future Enhancements

- Real-time scoring API
- A/B testing framework
- Automated model retraining pipeline
- Advanced feature engineering
- Deep learning models exploration

## License

This project is for educational and analytical purposes.

## Contact

For questions or collaboration, please reach out through GitHub issues.
