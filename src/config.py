import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env

# --- Database Configuration ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "db": os.getenv("DB_NAME"),
}

# --- S3 Configuration ---
S3_BUCKET = "amdari-demo-etl1"
S3_PREFIX = "/medoptix/raw/"

# --- Default Columns for Modeling ---
NUMERIC_FEATURES = [
    "age", "bmi", "n_sessions", "avg_session_duration",
    "mean_pain", "mean_pain_delta", "satisfaction_mean"
]

CATEGORICAL_FEATURES = [
    "gender", "smoker", "chronic_cond", "injury_type"
]

# --- Target Columns ---
TARGET_DROPOUT = "dropout"
TARGET_ADHERENCE_SCORE = "home_adherence_mean"
TARGET_ADHERENCE_CLASS = "adherence_class"

# --- Model Directory ---
MODEL_DIR = "models/saved_models"
