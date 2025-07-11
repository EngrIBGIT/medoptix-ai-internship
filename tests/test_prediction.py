import pandas as pd
from src.prediction import train_model
import os

def test_train_and_save_model():
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "bmi": [22.5, 27.8, 30.0],
        "gender": ["M", "F", "F"],
        "injury_type": ["knee", "hip", "shoulder"],
        "avg_duration": [25, 30, 35]
    })
    y = [0, 1, 1]

    model = train_model(df, y, model_name="Random Forest")
    assert model is not None
    assert os.path.exists("models/saved_models/random_forest_dropout_model.joblib")
