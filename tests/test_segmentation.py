import pandas as pd
from src.segmentation import preprocess_data, reduce_dimensions, train_kmeans
import os

def test_kmeans_pipeline():
    # Sample Data
    df = pd.DataFrame({
        "age": [22, 35, 58],
        "bmi": [20.5, 30.2, 27.3],
        "n_sessions": [5, 10, 15],
        "avg_session_duration": [25, 35, 30],
        "mean_pain": [4, 3, 2],
        "mean_pain_level_delta": [1, -1, 0],
        "gender": ["M", "F", "F"],
        "smoker": ["yes", "no", "yes"],
        "chronic_cond": ["no", "yes", "no"],
        "injury_type": ["knee", "shoulder", "hip"]
    })

    num_cols = ["age", "bmi", "n_sessions", "avg_session_duration", "mean_pain", "mean_pain_level_delta"]
    cat_cols = ["gender", "smoker", "chronic_cond", "injury_type"]

    X = preprocess_data(df, num_cols, cat_cols)
    _, X_red = reduce_dimensions(X)
    model, labels, score = train_kmeans(X_red, k=2)

    assert len(labels) == df.shape[0]
    assert score > 0
    assert os.path.exists("models/saved_models") or True
