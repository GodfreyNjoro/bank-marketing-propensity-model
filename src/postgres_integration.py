"""
PostgreSQL integration for the propensity model.
Handles connections, data storage, and queries.

v1.0 - Jan 2026
"""

import logging
import sys
from typing import Dict, Optional, Any

import pandas as pd

# optional deps
try:
    import psycopg2
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from sqlalchemy import create_engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

from config import DB_CONFIG, LOGGING_CONFIG, OUTPUT_CONFIG

# logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    datefmt=LOGGING_CONFIG["datefmt"],
    handlers=[
        logging.FileHandler(OUTPUT_CONFIG["log_file"]),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PostgresConnector:
    """Database connector for propensity model data."""
    
    def __init__(self, host, database, user, password, port=5432):
        """Initialize connection params."""
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.engine = None
        self.connection = None
        
        logger.info(f"PostgresConnector init: {database}@{host}:{port}")
    
    def connect(self):
        """Establish database connection. Returns True on success."""
        if not PSYCOPG2_AVAILABLE or not SQLALCHEMY_AVAILABLE:
            logger.error("Missing DB libs. pip install psycopg2-binary sqlalchemy")
            return False
        
        try:
            # sqlalchemy for pandas
            conn_str = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(conn_str)
            
            # psycopg2 for raw sql
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            
            logger.info("Connected successfully")
            return True
        
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def create_tables(self):
        """Create tables for the model. Returns True on success."""
        if not self.connection:
            logger.error("Not connected. Call connect() first.")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # customers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id SERIAL PRIMARY KEY,
                    age INTEGER,
                    job VARCHAR(50),
                    marital VARCHAR(20),
                    education VARCHAR(50),
                    default_status VARCHAR(10),
                    balance INTEGER,
                    housing VARCHAR(10),
                    loan VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # campaigns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(customer_id),
                    contact VARCHAR(20),
                    day INTEGER,
                    month VARCHAR(10),
                    duration INTEGER,
                    campaign INTEGER,
                    pdays INTEGER,
                    previous INTEGER,
                    poutcome VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(customer_id),
                    campaign_id INTEGER REFERENCES campaigns(campaign_id),
                    propensity_score FLOAT,
                    prediction INTEGER,
                    model_name VARCHAR(100),
                    model_version VARCHAR(50),
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # model performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    performance_id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100),
                    model_version VARCHAR(50),
                    accuracy FLOAT,
                    precision_score FLOAT,
                    recall FLOAT,
                    f1_score FLOAT,
                    roc_auc FLOAT,
                    avg_precision FLOAT,
                    training_samples INTEGER,
                    test_samples INTEGER,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # feature importance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_importance (
                    importance_id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100),
                    model_version VARCHAR(50),
                    feature_name VARCHAR(100),
                    importance_score FLOAT,
                    feature_rank INTEGER,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            logger.info("Tables created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.connection.rollback()
            return False
    
    def load_data_to_db(self, df, table_name, if_exists='append', index=False):
        """Load DataFrame to table."""
        if not self.engine:
            logger.error("No engine. Call connect() first.")
            return False
        
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
            logger.info(f"Loaded {len(df)} rows to {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading to {table_name}: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """Execute query and return DataFrame."""
        if not self.engine:
            logger.error("No engine")
            return None
        
        try:
            df = pd.read_sql_query(query, self.engine, params=params)
            logger.info(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Query error: {e}")
            return None
    
    def store_predictions(self, predictions_df, model_name="Unknown", model_version="1.0"):
        """Store predictions with model metadata."""
        df = predictions_df.copy()
        df['model_name'] = model_name
        df['model_version'] = model_version
        return self.load_data_to_db(df, 'predictions', if_exists='append')
    
    def store_model_performance(self, metrics, model_name, model_version, 
                                 training_samples, test_samples):
        """Store model performance metrics."""
        perf_df = pd.DataFrame([{
            'model_name': model_name,
            'model_version': model_version,
            'accuracy': metrics.get('accuracy'),
            'precision_score': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1'),
            'roc_auc': metrics.get('roc_auc'),
            'avg_precision': metrics.get('avg_precision'),
            'training_samples': training_samples,
            'test_samples': test_samples
        }])
        return self.load_data_to_db(perf_df, 'model_performance', if_exists='append')
    
    def store_feature_importance(self, importance_df, model_name, model_version):
        """Store feature importance scores."""
        df = importance_df.copy()
        df['model_name'] = model_name
        df['model_version'] = model_version
        df['feature_rank'] = range(1, len(df) + 1)
        df = df.rename(columns={
            'feature': 'feature_name',
            'importance': 'importance_score'
        })
        return self.load_data_to_db(df, 'feature_importance', if_exists='append')
    
    def get_recent_predictions(self, limit=100, model_name=None):
        """Get recent predictions."""
        query = "SELECT * FROM predictions WHERE 1=1"
        if model_name:
            query += f" AND model_name = '{model_name}'"
        query += f" ORDER BY prediction_date DESC LIMIT {limit}"
        return self.execute_query(query)
    
    def get_model_performance_history(self, model_name=None):
        """Get model performance history."""
        query = "SELECT * FROM model_performance"
        if model_name:
            query += f" WHERE model_name = '{model_name}'"
        query += " ORDER BY evaluation_date DESC"
        return self.execute_query(query)
    
    def close(self):
        """Close all connections."""
        if self.connection:
            self.connection.close()
            logger.info("psycopg2 connection closed")
        
        if self.engine:
            self.engine.dispose()
            logger.info("SQLAlchemy engine disposed")


def main():
    """Example usage."""
    logger.info("="*60)
    logger.info("POSTGRES INTEGRATION EXAMPLE")
    logger.info("="*60 + "\n")
    
    db = PostgresConnector(**DB_CONFIG)
    
    if db.connect():
        db.create_tables()
        
        # Example usage:
        # df = load_data()
        # db.load_data_to_db(df, 'customers')
        # 
        # metrics = {'accuracy': 0.85, 'precision': 0.75, ...}
        # db.store_model_performance(metrics, 'RandomForest', '1.0', 10000, 2000)
        # 
        # results = db.execute_query("SELECT * FROM customers LIMIT 10")
        # print(results)
        
        db.close()
    else:
        logger.error("Failed to connect")


if __name__ == "__main__":
    main()
