import pandas as pd
from src import etl
import os

def test_clean_dataframe_basic():
    # Sample data with noise
    df = pd.DataFrame({
        " Patient ID ": [1, 1, 2],
        "Age": [25, 25, None],
        "BMI": [23.4, 23.4, 27.5],
    })

    schema = ["patient_id", "age", "bmi"]
    cleaned = etl.clean_dataframe(df, required_columns=schema, na_strategy="drop", file_name="test.csv")

    assert cleaned is not None
    assert "patient_id" in cleaned.columns
    assert cleaned.shape[0] == 1  # Only the first row remains after dropna + deduplication

def test_upload_and_download_s3_mock(tmp_path):
    # This test will only run locally if AWS credentials are configured
    test_file = tmp_path / "mock.csv"
    test_file.write_text("id,value\n1,10\n2,20")

    # Upload test
    bucket = etl.S3_BUCKET
    prefix = "test-folder/"
    etl.s3.upload_file(str(test_file), bucket, f"{prefix}mock.csv")

    # Download test
    etl.s3.download_file(bucket, f"{prefix}mock.csv", str(tmp_path / "downloaded.csv"))
    assert os.path.exists(tmp_path / "downloaded.csv")

def test_db_connection():
    engine = etl.get_db_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            assert result.scalar() == 1
    finally:
        engine.dispose()
