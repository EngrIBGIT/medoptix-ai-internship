import pandas as pd
import numpy as np
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def build_preprocessor(numerics, categoricals):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numerics),
        ("cat", categorical_pipe, categoricals)
    ])

def train_model(X, y, model_name="Random Forest"):
    numerics = X.select_dtypes(include='number').columns.tolist()
    categoricals = list(set(X.columns) - set(numerics))

    preprocessor = build_preprocessor(numerics, categoricals)

    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    }

    param_grid = {
        "Random Forest": {
            "model__n_estimators": [100],
            "model__max_depth": [None, 10]
        }
    }

    model = model_dict.get(model_name, RandomForestClassifier())
    pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE()),
        ("model", model)
    ])

    grid = GridSearchCV(pipe, param_grid.get(model_name, {}), scoring='f1', cv=3)
    grid.fit(X, y)

    best_model = grid.best_estimator_
    joblib.dump(best_model, f"models/saved_models/{model_name.replace(' ', '_').lower()}_dropout_model.joblib")
    return best_model
