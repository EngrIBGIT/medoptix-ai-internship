import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and data loading manager"""
    
    def __init__(self):
        load_dotenv()
        self.engine = None
        self._connect()
    
    def _connect(self):
        """Create database connection"""
        try:
            db_url = (
                f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@"
                f"{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
            )
            self.engine = create_engine(db_url)
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def load_all_data(self):
        """Load all required data from database"""
        try:
            # Define queries for each table
            queries = {
                "clinics": "SELECT * FROM clinics;",
                "patients": "SELECT * FROM patients;", 
                "sessions": "SELECT * FROM sessions;",
                "feedback": "SELECT * FROM feedback;",
                "dropout_flags": "SELECT * FROM dropout_flags;",
                "interventions": "SELECT * FROM interventions;"
            }
            
            # Load data from each table
            data = {}
            for table_name, query in queries.items():
                data[table_name] = pd.read_sql(query, self.engine)
                logger.info(f"Loaded {len(data[table_name])} rows from {table_name}")
            
            return (
                data['patients'], 
                data['sessions'], 
                data['dropout_flags'],
                data['feedback'],
                data['interventions'],
                data['clinics']
            )
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_patient_data(self, patient_id: int = None):
        """Get specific patient data or all patients"""
        try:
            if patient_id:
                query = f"SELECT * FROM patients WHERE patient_id = {patient_id};"
            else:
                query = "SELECT * FROM patients;"
            
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Error getting patient data: {str(e)}")
            raise
    
    def get_sessions_data(self, patient_id: int = None):
        """Get sessions data for specific patient or all sessions"""
        try:
            if patient_id:
                query = f"SELECT * FROM sessions WHERE patient_id = {patient_id};"
            else:
                query = "SELECT * FROM sessions;"
            
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Error getting sessions data: {str(e)}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

# Global database manager instance
db_manager = DatabaseManager()