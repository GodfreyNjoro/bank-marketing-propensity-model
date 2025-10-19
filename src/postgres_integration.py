
"""
Bank Marketing Propensity Model - PostgreSQL Integration

This script handles database operations including:
- Connection management
- Data ingestion
- Prediction storage
- Query execution
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')


class PostgresConnector:
    """PostgreSQL database connector for propensity model"""
    
    def __init__(self, host, database, user, password, port=5432):
        """Initialize database connection parameters"""
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.engine = None
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            # SQLAlchemy engine for pandas operations
            connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string)
            
            # psycopg2 connection for raw SQL
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            
            print("Database connection established successfully")
            return True
        
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def create_tables(self):
        """Create necessary tables for the propensity model"""
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
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                performance_id SERIAL PRIMARY KEY,
                model_version VARCHAR(50),
                accuracy FLOAT,
                precision_score FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                roc_auc FLOAT,
                evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        print("Tables created successfully")
    
    def load_data_to_db(self, df, table_name, if_exists='append'):
        """Load pandas DataFrame to database table"""
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            print(f"Data loaded to {table_name} successfully")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def execute_query(self, query):
        """Execute SQL query and return results as DataFrame"""
        try:
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
    
    def store_predictions(self, predictions_df):
        """Store model predictions in database"""
        return self.load_data_to_db(predictions_df, 'predictions', if_exists='append')
    
    def close(self):
        """Close database connections"""
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        print("Database connections closed")


def main():
    """Example usage of PostgreSQL connector"""
    # Database credentials (replace with actual values)
    db_config = {
        'host': 'localhost',
        'database': 'bank_marketing',
        'user': 'your_username',
        'password': 'your_password',
        'port': 5432
    }
    
    # Initialize connector
    db = PostgresConnector(**db_config)
    
    # Connect to database
    if db.connect():
        # Create tables
        db.create_tables()
        
        # Example: Load data
        # df = pd.read_csv('../data/bank.csv')
        # db.load_data_to_db(df, 'customers')
        
        # Example: Query data
        # results = db.execute_query("SELECT * FROM customers LIMIT 10")
        # print(results)
        
        # Close connection
        db.close()


if __name__ == "__main__":
    main()
