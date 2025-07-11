import pandas as pd
from src.forecasting import train_regressor, train_classifier
import os

def test_adherence_forecasting():
    # Sample data
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "bmi": [25.5, 27.3, 29.1],
        "gender": ["M", "F", "F"],
        "injury_type": ["knee", "hip", "shoulder"],
        "adherence": [80, 60, 90],
        "class": ["High", "Medium", "High"]
    })

    num_cols = ["age", "bmi"]
    cat_cols = ["gender", "injury_type"]

    train_regressor(df[num_cols + cat_cols], df["adherence"], num_cols, cat_cols)
    assert os.path.exists("models/saved_models/adherence_regressor.joblib")

    train_classifier(df[num_cols + cat_cols], df["class"], num_cols, cat_cols)
    assert os.path.exists("models/saved_models/adherence_classifier.joblib")
