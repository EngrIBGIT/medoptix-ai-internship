import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Import database manager
import sys
#sys.path.append('../app')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedOptixClustering:
    def __init__(self):
        self.preproc = None
        self.pca = None
        self.kmeans = None
        self.num_col = None
        self.cat_col = None
        self.db_manager = DatabaseManager()

    def load_data(self):
        """Load data from database"""
        logger.info("Loading data from database...")
        patients, sessions, dropout_flag, feedback, interventions, clinics = self.db_manager.load_all_data()
        return patients, sessions, dropout_flag

    def create_session_features(self, sessions):
        """Create aggregated session features"""
        sess_agg = (
            sessions
            .sort_values(['patient_id', 'date'])    
            .assign(pain_delta=lambda d: d.groupby("patient_id")["pain_level"].diff())
            .groupby("patient_id") 
            .agg(
                n_sessions=("session_id", "count"),          
                avg_session_duration=("duration", "mean"),
                first_week=("week", "min"),
                last_week=("week", "max"),
                mean_pain=("pain_level", "mean"),
                mean_pain_delta=("pain_delta", "mean"),
                home_adherence_mean=("home_adherence_pc", "mean"),
                satisfaction_mean=("satisfaction", "mean")
            )
        )
        return sess_agg

    def prepare_patient(self, patients, sessions):
        """Prepare patient data by combining with session features"""
        pat_sel = patients[
            [
                "patient_id", "age", "gender", "bmi", "smoker", "chronic_cond",
                "injury_type", "referral_source", "insurance_type"
            ]
        ].set_index("patient_id")

        sess_agg = self.create_session_features(sessions)
        X_raw = pat_sel.join(sess_agg, how="left").reset_index()
        
        return X_raw
    
    def setup_preprocessing(self):
        """Setup preprocessing pipelines"""
        self.num_col = [
            "age", "bmi", "n_sessions", "avg_session_duration",
            "mean_pain", "mean_pain_delta"
        ]

        self.cat_col = [
            "gender", "smoker", "chronic_cond", "injury_type"
        ]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.preproc = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.num_col),
                ("cat", categorical_pipe, self.cat_col)
            ]
        )
        
        return self.preproc

    def find_optimal_k(self, X_reduced, k_range=range(2,11)):
        """Find optimal number of clusters using elbow and silhouette methods"""
        wcss, sils = [], []
    
        for k in k_range:                    
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",          
                n_init="auto",
                random_state=42,
                algorithm="lloyd",
            )
            preds = kmeans.fit_predict(X_reduced)           
            wcss.append(kmeans.inertia_)            
            sils.append(silhouette_score(X_reduced, preds))
        
        # Create output directory
        os.makedirs("reports/figures/segmentation", exist_ok=True)
        
        # Plot the elbow method
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(k_range, wcss, marker="o")
        ax1.set_title("Elbow Method for Optimal K")
        ax1.set_xlabel("Number of clusters (k)")
        ax1.set_ylabel("WCSS")

        ax2.plot(k_range, sils, marker="o")
        ax2.set_title("Silhouette Score for Optimal K")
        ax2.set_xlabel("Number of clusters (k)")
        ax2.set_ylabel("Silhouette Score")

        plt.tight_layout()
        plt.savefig("reports/figures/segmentation/segmentation_optimal_k_selection.png", dpi=300, bbox_inches='tight')
        plt.show()

        return wcss, sils

    def perform_clustering(self, X_raw, k_opt=3):
        """Complete clustering process"""
        logger.info(f"Performing clustering with k={k_opt}...")

        # Preprocessing
        logger.info("Preprocessing data...")
        self.setup_preprocessing()
        X_clean = self.preproc.fit_transform(X_raw)

        # PCA for dimensionality reduction
        logger.info("Fitting PCA...")
        self.pca = PCA(n_components=0.95, random_state=42)
        X_reduced = self.pca.fit_transform(X_clean)
        logger.info(f"Reduced data shape: {X_reduced.shape}")

        # Find optimal number of clusters
        logger.info("Finding optimal number of clusters...")
        wcss, sils = self.find_optimal_k(X_reduced)

        # Final clustering with KMeans
        logger.info("Fitting KMeans...")
        self.kmeans = KMeans(
            n_clusters=k_opt, 
            init="k-means++",
            n_init="auto",
            random_state=42, 
            algorithm="lloyd"
        )
        clusters = self.kmeans.fit_predict(X_reduced)

        # Evaluate clustering 
        sil_score = silhouette_score(X_reduced, clusters)
        logger.info(f"Silhouette Score: {sil_score:.3f}")

        # Visualize clusters
        self.visualize_clusters(X_reduced, clusters)
        
        return clusters, X_reduced, X_clean
    
    def visualize_clusters(self, X_reduced, clusters):
        """Visualize clusters using PCA and t-SNE"""
        # Create output directory
        os.makedirs("reports/figures/segmentation", exist_ok=True)
        
        # PCA visualization
        pca2 = PCA(n_components=2, random_state=42)
        X_pca2d = pca2.fit_transform(X_reduced)

        # t-SNE visualization
        tsne2 = TSNE(n_components=2, perplexity=40, init="pca", random_state=42)
        X_tsne2d = tsne2.fit_transform(X_reduced)

        # Plot clusters
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scatter1 = ax1.scatter(X_pca2d[:, 0], X_pca2d[:, 1], c=clusters, cmap="viridis", s=50)
        ax1.set_title("Patient Segmentation - PCA Clusters")
        ax1.set_xlabel("PCA Component 1") 
        ax1.set_ylabel("PCA Component 2")
        plt.colorbar(scatter1, ax=ax1, label="Cluster")

        scatter2 = ax2.scatter(X_tsne2d[:, 0], X_tsne2d[:, 1], c=clusters, cmap="viridis", s=50)
        ax2.set_title("Patient Segmentation - t-SNE Clusters")
        ax2.set_xlabel("t-SNE Component 1")
        ax2.set_ylabel("t-SNE Component 2")
        plt.colorbar(scatter2, ax=ax2, label="Cluster")

        plt.tight_layout()
        plt.savefig("reports/figures/segmentation/segmentation_cluster_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()

    def save_models(self, filepath_prefix="models/segmentation/medoptix_segmentation"):
        """Save trained models"""
        os.makedirs("models/segmentation", exist_ok=True)
        
        joblib.dump(self.preproc, f"{filepath_prefix}_preprocessor.pkl")
        joblib.dump(self.pca, f"{filepath_prefix}_pca.pkl")
        joblib.dump(self.kmeans, f"{filepath_prefix}_kmeans.pkl")
        
        logger.info(f"Segmentation models saved with prefix: {filepath_prefix}")

if __name__ == "__main__":
    # Create output directories
    os.makedirs("models/segmentation", exist_ok=True)
    os.makedirs("reports/figures/segmentation", exist_ok=True)
    
    clustering = MedOptixClustering()
    
    # Load data from database
    patients, sessions, dropout_flag = clustering.load_data()
    
    # Prepare features
    X_raw = clustering.prepare_patient(patients, sessions)
    logger.info(f"Raw data shape: {X_raw.shape}")
    logger.info(f"Feature columns: {X_raw.columns.tolist()}")
    
    # Perform clustering
    clusters, X_reduced, X_clean = clustering.perform_clustering(X_raw, k_opt=3)
    
    # Create labels
    label = dropout_flag.set_index("patient_id")["dropout"]
    
    # Save results for prediction
    results = {
        'X_raw': X_raw,
        "clusters": clusters,
        'label': label,
        "X_reduced": X_reduced,
        "X_clean": X_clean
    }
    
    joblib.dump(results, "models/segmentation/medoptix_segmentation_results.pkl")
    
    # Save the models
    clustering.save_models()
    
    logger.info("Clustering completed and results saved.")
    logger.info("Cluster distribution:")
    print(pd.Series(clusters).value_counts())
    logger.info("Files saved in models/segmentation/ and reports/figures/segmentation/")