import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from joblib import dump
import os

def preprocess_data(df, num_cols, cat_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scalar", RobustScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])
    return preprocess.fit_transform(df)

def reduce_dimensions(X, n_components=0.85):
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    return pca, X_reduced

def train_kmeans(X, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", algorithm="lloyd")
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    return kmeans, labels, sil_score

def train_dbscan(X, eps=1.5, min_samples=3):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    if np.sum(labels != -1) > 1:
        sil_score = silhouette_score(X[labels != -1], labels[labels != -1])
    else:
        sil_score = -1
    return db, labels, sil_score

def save_model(model, name):
    os.makedirs("models/saved_models", exist_ok=True)
    dump(model, f"models/saved_models/{name}.joblib")
