import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
import shap
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Import database manager
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedOptixAdherenceForecasting:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.label_encoder = None
        self.db_manager = DatabaseManager()
        self.feature_names = None

    def load_data(self):
        """Load data from database"""
        logger.info("Loading data from database...")
        patients, sessions, dropout_flags, feedback, interventions, clinics = self.db_manager.load_all_data()
        return patients, sessions, feedback, dropout_flags, interventions

    def aggregate_dynamic_features(self, sessions, feedback, interventions):
        """Create aggregated features from dynamic data"""
        
        # Session aggregations
        sess_agg = (
            sessions
            .sort_values(["patient_id", "date"])
            .assign(
                pain_delta=lambda d: d.groupby("patient_id")["pain_level"].diff(),
                session_gap=lambda d: d.groupby("patient_id")["date"].diff().dt.days
            )
            .groupby("patient_id")
            .agg(
                n_sessions=("session_id", "count"),
                avg_session_duration=("duration", "mean"),
                mean_pain=("pain_level", "mean"),
                mean_pain_delta=("pain_delta", "mean"),
                max_pain_delta=("pain_delta", "max"),
                min_pain_delta=("pain_delta", "min"),
                std_pain_delta=("pain_delta", "std"),
                home_adherence_mean=("home_adherence_pc", "mean"),
                home_adherence_std=("home_adherence_pc", "std"),
                satisfaction_mean=("satisfaction", "mean"),
                avg_session_gap=("session_gap", "mean"),
                missed_sessions=("session_gap", lambda x: (x > 7).sum())
            )
        )
        
        # Bin adherence into classes
        bins = [-1, 40, 60, np.inf]
        labels = ["Low", "Medium", "High"]
        sess_agg["adherence_class"] = pd.cut(sess_agg["home_adherence_mean"], bins=bins, labels=labels)

        # Feedback aggregations (if sentiment column exists)
        feedback_agg = pd.DataFrame()
        if not feedback.empty and "sentiment" in feedback.columns:
            feedback_agg = (
                feedback
                .merge(sessions[["session_id", "patient_id"]], on="session_id", how="left")
                .groupby("patient_id")
                .agg(
                    feedback_sentiment_mean=("sentiment", "mean"),
                    feedback_sentiment_std=("sentiment", "std"),
                    feedback_count=("sentiment", "count")
                )
            )

        # Interventions aggregations
        intervention_agg = pd.DataFrame()
        if not interventions.empty:
            intervention_agg = (
                interventions
                .groupby("patient_id")
                .agg(
                    n_interventions=("intervention_id", "count"),
                    unique_interventions=("channel", pd.Series.nunique)
                )
            )

        return sess_agg, feedback_agg, intervention_agg

    def prepare_features(self, patients, sessions, feedback, interventions):
        """Prepare all features for training"""
        
        # Get aggregated features
        sess_agg, feedback_agg, intervention_agg = self.aggregate_dynamic_features(sessions, feedback, interventions)
        
        # Set patient_id as index
        patients_indexed = patients.set_index("patient_id")
        
        # Join all features
        features = patients_indexed.join(sess_agg, how="left")
        
        if not feedback_agg.empty:
            features = features.join(feedback_agg, how="left")
        
        if not intervention_agg.empty:
            features = features.join(intervention_agg, how="left")

        # Define feature lists
        num_cols = [
            "age", "bmi", "n_sessions", "avg_session_duration",
            "mean_pain", "mean_pain_delta", "max_pain_delta", "min_pain_delta", "std_pain_delta",
            "home_adherence_mean", "home_adherence_std", "satisfaction_mean",
            "avg_session_gap", "missed_sessions"
        ]
        
        # Add feedback features if available
        if not feedback_agg.empty:
            num_cols.extend(["feedback_sentiment_mean", "feedback_sentiment_std", "feedback_count"])
        
        # Add intervention features if available
        if not intervention_agg.empty:
            num_cols.extend(["n_interventions", "unique_interventions"])

        cat_cols = ["gender", "smoker", "chronic_cond", "injury_type", "referral_source", "insurance_type"]
        target_col = "adherence_class"

        # Clean data
        features[num_cols] = features[num_cols].apply(pd.to_numeric, errors="coerce")
        features[cat_cols] = features[cat_cols].astype("string")
        
        # Drop rows missing target and replace categorical NAs
        data = features.dropna(subset=[target_col]).copy()
        for c in cat_cols:
            data[c] = data[c].fillna("Unknown")

        return data, num_cols, cat_cols, target_col

    def feature_selection(self, data, num_cols, cat_cols, target_col, top_k=10):
        """Select top features using RandomForest importance"""
        
        # Encode all features for feature selection
        fs_df = data.copy()
        le = LabelEncoder()
        fs_df[target_col] = le.fit_transform(fs_df[target_col])
        
        for c in cat_cols:
            fs_df[c] = LabelEncoder().fit_transform(fs_df[c])

        # Handle missing values - fill numeric NaNs with median and categorical with mode
        for col in num_cols:
            if col in fs_df.columns:
                fs_df[col] = fs_df[col].fillna(fs_df[col].median())
        
        for col in cat_cols:
            if col in fs_df.columns:
                fs_df[col] = fs_df[col].fillna(fs_df[col].mode()[0])

        # Feature importance with RandomForest
        rf_fs = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_fs.fit(fs_df[num_cols + cat_cols], fs_df[target_col])
        importances = rf_fs.feature_importances_

        # Select top features
        feat_names = num_cols + cat_cols
        top_idx = np.argsort(importances)[::-1][:top_k]
        selected_feats = [feat_names[i] for i in top_idx]
        
        logger.info(f"Top {top_k} selected features: {selected_feats}")
        
        return selected_feats

    def setup_preprocessing(self, selected_feats, num_cols, cat_cols):
        """Setup preprocessing pipeline"""
        
        num_sel = [f for f in selected_feats if f in num_cols]
        cat_sel = [f for f in selected_feats if f in cat_cols]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])
        
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer([
            ("num", numeric_pipe, num_sel),
            ("cat", categorical_pipe, cat_sel)
        ])
        
        return self.preprocessor

    def train_classifiers(self, X_train, y_train, X_test, y_test):
        """Train multiple classifiers and evaluate performance"""
        
        # Balance training data with SMOTE
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        
        # Define classifiers
        classifiers = {
            "RandomForest": RandomForestClassifier(random_state=42),
            "LogisticReg": LogisticRegression(max_iter=1000, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
            "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
            "MLP": MLPClassifier(max_iter=500, random_state=42)
        }

        best_models = {}
        results = {}
        
        # Create output directory
        os.makedirs("reports/figures/adherence_forecasting", exist_ok=True)
        
        for name, clf in classifiers.items():
            logger.info(f"\nTraining {name}...")
            
            # Train model
            clf.fit(X_train_bal, y_train_bal)
            y_pred = clf.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"{name} Accuracy: {accuracy:.3f}")
            print(f"\n== {name} Classification Report ==")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
            
            # Store results
            best_models[name] = clf
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            # Create confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=self.label_encoder.classes_, ax=ax, cmap='Blues'
            )
            plt.title(f"Adherence Forecasting - {name} Confusion Matrix")
            plt.savefig(f"reports/figures/adherence_forecasting/adherence_forecasting_{name.lower()}_confusion_matrix.png", 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close figure to prevent memory issues
        
        # Select best model based on accuracy
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.model = best_models[best_model_name]
        
        logger.info(f"\n🏆 Best model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.3f})")
        
        return best_models, best_model_name

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for RandomForest"""
        
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10]
        }
        
        rf_pipe = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", RandomForestClassifier(random_state=42))
        ])
        
        logger.info("Performing hyperparameter tuning...")
        
        # Disable parallel processing to avoid pickling issues
        grid = GridSearchCV(
            rf_pipe, 
            param_grid, 
            scoring="f1_weighted", 
            cv=3, 
            n_jobs=1,  # Set to 1 to avoid parallel processing issues
            verbose=2
        )
        
        grid.fit(X_train, y_train)
        
        logger.info(f"Best RF Params: {grid.best_params_}")
        
        return grid.best_estimator_

    def create_shap_summary(self, X_test):
        """Create SHAP summary plot"""
        
        try:
            explainer = shap.Explainer(self.model, X_test)
            shap_values = explainer(X_test)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, 
                            feature_names=self.preprocessor.get_feature_names_out(),
                            show=False)
            plt.title("SHAP Summary: Adherence Forecasting")
            plt.tight_layout()
            plt.savefig("reports/figures/adherence_forecasting/adherence_forecasting_shap_summary.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP plot: {str(e)}")

    def save_models(self, filepath_prefix="models/adherence_forecasting/medoptix_adherence_forecasting"):
        """Save trained models and metadata"""
        
        os.makedirs("models/adherence_forecasting", exist_ok=True)
        
        # Save main components
        joblib.dump(self.preprocessor, f"{filepath_prefix}_preprocessor.pkl")
        joblib.dump(self.model, f"{filepath_prefix}_model.pkl")
        joblib.dump(self.label_encoder, f"{filepath_prefix}_label_encoder.pkl")
        
        # Save feature names
        if self.feature_names is not None:
            joblib.dump(self.feature_names, f"{filepath_prefix}_feature_names.pkl")
        
        logger.info(f"Adherence forecasting models saved with prefix: {filepath_prefix}")

if __name__ == "__main__":
    # Create output directories
    os.makedirs("models/adherence_forecasting", exist_ok=True)
    os.makedirs("reports/figures/adherence_forecasting", exist_ok=True)
    
    forecaster = MedOptixAdherenceForecasting()
    
    # Load data from database
    patients, sessions, feedback, dropout_flags, interventions = forecaster.load_data()
    
    # Prepare features
    data, num_cols, cat_cols, target_col = forecaster.prepare_features(
        patients, sessions, feedback, interventions
    )
    
    logger.info(f"Prepared data shape: {data.shape}")
    
    # Feature selection
    selected_feats = forecaster.feature_selection(data, num_cols, cat_cols, target_col, top_k=10)
    
    # Prepare X and y
    X = data[selected_feats]
    y = data[target_col]
    
    # Encode target
    forecaster.label_encoder = LabelEncoder()
    y_enc = forecaster.label_encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, stratify=y_enc, test_size=0.2, random_state=42
    )
    
    # Setup preprocessing
    forecaster.setup_preprocessing(selected_feats, num_cols, cat_cols)
    X_train_proc = forecaster.preprocessor.fit_transform(X_train)
    X_test_proc = forecaster.preprocessor.transform(X_test)
    
    # Store feature names
    forecaster.feature_names = forecaster.preprocessor.get_feature_names_out()
    
    # Train classifiers
    best_models, best_model_name = forecaster.train_classifiers(X_train_proc, y_train, X_test_proc, y_test)
    
    # Hyperparameter tuning (with parallel processing disabled)
    best_rf_pipe = forecaster.hyperparameter_tuning(X_train, y_train)
    joblib.dump(best_rf_pipe, "models/adherence_forecasting/adherence_forecasting_random_forest_best.pkl")
    
    # SHAP analysis
    forecaster.create_shap_summary(X_test_proc)
    
    # Save models
    forecaster.save_models()
    
    logger.info("✅ Adherence forecasting pipeline completed successfully!")
    logger.info("Models and visualizations saved in models/adherence_forecasting/ and reports/figures/adherence_forecasting/")