"""
Bank Marketing Propensity Model - PostgreSQL Integration

This script handles database operations including:
- Connection management with proper error handling
- Data ingestion and validation
- Prediction storage
- Query execution with logging
"""

import logging
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import psycopg2
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import DB_CONFIG, LOGGING_CONFIG, OUTPUT_CONFIG

# Configure logging
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
    """PostgreSQL database connector for propensity model."""
    
    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432
    ) -> None:
        """
        Initialize database connection parameters.
        
        Args:
            host: Database host
            database: Database name
            user: Database user
            password: Database password
            port: Database port (default: 5432)
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.engine: Optional[Engine] = None
        self.connection: Optional[Any] = None
        
        logger.info(f"PostgresConnector initialized for {database}@{host}:{port}")
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not PSYCOPG2_AVAILABLE or not SQLALCHEMY_AVAILABLE:
            logger.error("Required database libraries not available. "
                        "Install with: pip install psycopg2-binary sqlalchemy")
            return False
        
        try:
            # SQLAlchemy engine for pandas operations
            connection_string = (
                f"postgresql://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
            self.engine = create_engine(connection_string)
            
            # psycopg2 connection for raw SQL
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            
            logger.info("Database connection established successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    def create_tables(self) -> bool:
        """
        Create necessary tables for the propensity model.
        
        Returns:
            True if tables created successfully
        """
        if not self.connection:
            logger.error("No database connection. Call connect() first.")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Customers table
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
            
            # Campaigns table
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
            
            # Predictions table
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
            
            # Model performance table
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
            
            # Feature importance table
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
            logger.info("Database tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.connection.rollback()
            return False
    
    def load_data_to_db(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = 'append',
        index: bool = False
    ) -> bool:
        """
        Load pandas DataFrame to database table.
        
        Args:
            df: DataFrame to load
            table_name: Target table name
            if_exists: What to do if table exists ('append', 'replace', 'fail')
            index: Whether to write DataFrame index
            
        Returns:
            True if data loaded successfully
        """
        if not self.engine:
            logger.error("No database engine. Call connect() first.")
            return False
        
        try:
            df.to_sql(
                table_name, 
                self.engine, 
                if_exists=if_exists, 
                index=index
            )
            logger.info(f"Loaded {len(df)} rows to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data to {table_name}: {e}")
            return False
    
    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
            
        Returns:
            DataFrame with query results, or None on error
        """
        if not self.engine:
            logger.error("No database engine. Call connect() first.")
            return None
        
        try:
            df = pd.read_sql_query(query, self.engine, params=params)
            logger.info(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None
    
    def store_predictions(
        self,
        predictions_df: pd.DataFrame,
        model_name: str = "Unknown",
        model_version: str = "1.0"
    ) -> bool:
        """
        Store model predictions in database.
        
        Args:
            predictions_df: DataFrame with predictions
            model_name: Name of the model used
            model_version: Version of the model
            
        Returns:
            True if predictions stored successfully
        """
        # Add model metadata
        predictions_df = predictions_df.copy()
        predictions_df['model_name'] = model_name
        predictions_df['model_version'] = model_version
        
        return self.load_data_to_db(predictions_df, 'predictions', if_exists='append')
    
    def store_model_performance(
        self,
        metrics: Dict[str, float],
        model_name: str,
        model_version: str,
        training_samples: int,
        test_samples: int
    ) -> bool:
        """
        Store model performance metrics in database.
        
        Args:
            metrics: Dictionary of performance metrics
            model_name: Name of the model
            model_version: Version of the model
            training_samples: Number of training samples
            test_samples: Number of test samples
            
        Returns:
            True if metrics stored successfully
        """
        performance_df = pd.DataFrame([{
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
        
        return self.load_data_to_db(performance_df, 'model_performance', if_exists='append')
    
    def store_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str,
        model_version: str
    ) -> bool:
        """
        Store feature importance scores in database.
        
        Args:
            importance_df: DataFrame with feature importance
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            True if stored successfully
        """
        df = importance_df.copy()
        df['model_name'] = model_name
        df['model_version'] = model_version
        df['feature_rank'] = range(1, len(df) + 1)
        df = df.rename(columns={
            'feature': 'feature_name',
            'importance': 'importance_score'
        })
        
        return self.load_data_to_db(df, 'feature_importance', if_exists='append')
    
    def get_recent_predictions(
        self,
        limit: int = 100,
        model_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get recent predictions from database.
        
        Args:
            limit: Maximum number of records
            model_name: Filter by model name (optional)
            
        Returns:
            DataFrame with predictions
        """
        query = """
            SELECT * FROM predictions 
            WHERE 1=1
        """
        
        if model_name:
            query += f" AND model_name = '{model_name}'"
        
        query += f" ORDER BY prediction_date DESC LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_model_performance_history(
        self,
        model_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get model performance history.
        
        Args:
            model_name: Filter by model name (optional)
            
        Returns:
            DataFrame with performance history
        """
        query = "SELECT * FROM model_performance"
        
        if model_name:
            query += f" WHERE model_name = '{model_name}'"
        
        query += " ORDER BY evaluation_date DESC"
        
        return self.execute_query(query)
    
    def close(self) -> None:
        """Close database connections."""
        if self.connection:
            self.connection.close()
            logger.info("psycopg2 connection closed")
        
        if self.engine:
            self.engine.dispose()
            logger.info("SQLAlchemy engine disposed")
        
        logger.info("All database connections closed")


def main() -> None:
    """Example usage of PostgreSQL connector."""
    logger.info("="*60)
    logger.info("POSTGRESQL INTEGRATION EXAMPLE")
    logger.info("="*60 + "\n")
    
    # Initialize connector with config
    db = PostgresConnector(**DB_CONFIG)
    
    # Connect to database
    if db.connect():
        # Create tables
        db.create_tables()
        
        # Example: Load sample data
        # from train_model import load_data
        # df = load_data()
        # db.load_data_to_db(df, 'customers')
        
        # Example: Store performance metrics
        # metrics = {'accuracy': 0.85, 'precision': 0.75, 'recall': 0.80, 'f1': 0.77, 'roc_auc': 0.90}
        # db.store_model_performance(metrics, 'RandomForest', '1.0', 10000, 2000)
        
        # Example: Query data
        # results = db.execute_query("SELECT * FROM customers LIMIT 10")
        # print(results)
        
        # Close connection
        db.close()
    else:
        logger.error("Failed to connect to database")


if __name__ == "__main__":
    main()
