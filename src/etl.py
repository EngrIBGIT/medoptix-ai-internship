import os
import pandas as pd
import logging
import boto3
from sqlalchemy import create_engine, text
from io import StringIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Environment config
DB_PARAMS = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "db": os.getenv("DB_NAME")
}
S3_BUCKET = "amdari-demo-etl1"
S3_FOLDER = "/medoptix/raw/"
LOCAL_DATA_FOLDER = "./medoptix_data/raw"

# Initialize boto3 client
s3 = boto3.client("s3")


# --- Cleaning Utilities ---
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    logging.info(f"Removed {before - len(df)} duplicates")
    return df

def handle_missing_values(df, strategy="drop"):
    if strategy == "drop":
        return df.dropna()
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    return df

def validate_schema(df, required_columns, file_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logging.warning(f"{file_name} missing columns: {missing}")
        return False
    return True

def clean_dataframe(df, required_columns, na_strategy="drop", file_name=""):
    df = clean_column_names(df)
    if not validate_schema(df, required_columns, file_name):
        return None
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy=na_strategy)
    return df


# --- S3 & DB Utilities ---
def upload_to_s3(folder, bucket, prefix):
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            key = f"{prefix}{file}"
            s3.upload_file(path, bucket, key)
            logging.info(f"Uploaded {file} to s3://{bucket}/{key}")

def download_from_s3(files, bucket, prefix, local_folder):
    for file in files:
        local_path = os.path.join(local_folder, file)
        s3.download_file(bucket, f"{prefix}{file}", local_path)
        logging.info(f"Downloaded {file} to {local_path}")

def load_csv_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(content))

def get_db_engine():
    url = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['db']}"
    return create_engine(url)

def insert_to_postgres(df, table_name, engine):
    df.to_sql(table_name, engine, chunksize=1000, method="multi", index=False, if_exists="append")
    logging.info(f"Inserted {len(df)} rows into '{table_name}'")


# --- Full ETL Workflow ---
def run_etl():
    schemas = {
        "clinics": ["clinic_id", "city", "country", "type", "postcode", "capacity", "staff_count", "speciality", "avg_rating"],
        "patients": ["patient_id", "age", "gender", "bmi", "smoker", "chronic_cond", "injury_type", "signup_date", "consent", "clinic_id", "referral_source", "insurance_type"],
        "sessions": ["session_id", "patient_id", "date", "week", "duration", "pain_level", "exercise_type", "home_adherence_pc", "satisfaction", "therapist_id"],
        "feedback": ["feedback_id", "session_id", "sentiment", "comments"],
        "dropout_flags": ["patient_id", "dropout", "dropout_week"],
        "interventions": ["intervention_id", "patient_id", "sent_at", "channel", "message", "responded"]
    }

    files = [f"{k}.csv" for k in schemas]
    upload_to_s3(LOCAL_DATA_FOLDER, S3_BUCKET, S3_FOLDER)

    engine = get_db_engine()

    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    for file in files:
        table = file.replace(".csv", "")
        logging.info(f"Processing {file} → {table}")
        df = load_csv_from_s3(S3_BUCKET, f"{S3_FOLDER}{file}")
        df_clean = clean_dataframe(df, schemas[table], na_strategy="drop", file_name=file)
        if df_clean is not None:
            # Save cleaned file to processed directory
            processed_path = os.path.join(processed_dir, file)
            df_clean.to_csv(processed_path, index=False)
            logging.info(f"Saved cleaned {file} to {processed_path}")

            insert_to_postgres(df_clean, table, engine)
        else:
            logging.warning(f"Skipping {table}: Schema issues.")
    engine.dispose()
