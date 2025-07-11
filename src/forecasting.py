import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

def build_preprocessor(num_cols, cat_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])

def train_regressor(X, y, num_cols, cat_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    rf = RandomForestRegressor(random_state=42)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf)
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print("Regressor MAE:", mean_absolute_error(y_test, preds))
    print("R² Score:", r2_score(y_test, preds))

    os.makedirs("models/saved_models", exist_ok=True)
    joblib.dump(pipe, "models/saved_models/adherence_regressor.joblib")

def train_classifier(X, y, num_cols, cat_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    clf = RandomForestClassifier(random_state=42)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", clf)
    ])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))

    os.makedirs("models/saved_models", exist_ok=True)
    joblib.dump(pipe, "models/saved_models/adherence_classifier.joblib")
