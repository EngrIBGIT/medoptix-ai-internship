import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
import shap
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Import database manager
import sys
sys.path.append('../app')
from database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedOptixPrediction:
    def __init__(self):
        self.preproc = None
        self.model = None
        self.num_col = None
        self.cat_col = None
        self.shap_explainer = None
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
        """Prepare patient data for prediction"""
        pat_sel = patients[
            [
                "patient_id", "age", "gender", "bmi", "smoker", "chronic_cond",
                "injury_type", "referral_source", "insurance_type"
            ]
        ].set_index("patient_id")

        sess = self.create_session_features(sessions)
        X_raw = pat_sel.join(sess, how="left").reset_index()
        
        return X_raw
    
    def setup_preprocessing(self):
        """Define preprocessing pipelines"""
        self.num_col = [
            "age", "bmi", "n_sessions", "avg_session_duration", 
            "first_week", "mean_pain", "mean_pain_delta",
            "home_adherence_mean"
        ]

        self.cat_col = ["gender", "smoker"]

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
    
    def train_and_select_best(self, X_train, y_train, X_val, y_val, metric="roc_auc"):
        """Train multiple models and select the best one"""
        # Preprocess data
        self.setup_preprocessing()
        X_clean_train = self.preproc.fit_transform(X_train)
        X_clean_val = self.preproc.transform(X_val)

        # Define candidate models
        candidates = {
            "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "GradientBoost": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=8,
                class_weight="balanced", random_state=42
            ),
        }

        # Cross validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = {}
        
        for name, clf in candidates.items():
            cv_score = cross_val_score(
                clf, X_clean_train, y_train, cv=cv, scoring=metric
            ).mean()
            scores[name] = cv_score
            logger.info(f"{name}: mean {metric} = {cv_score:.3f}")

        # Pick the best model
        best_name = max(scores, key=scores.get)
        logger.info(f"🏆 Best model: {best_name}")

        # Train best model
        best_model = candidates[best_name]
        best_model.fit(X_clean_train, y_train)
        self.model = best_model

        # Setup SHAP explainer
        if best_name in ["RandomForest", "GradientBoost", "XGBoost"]:
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            self.shap_explainer = shap.LinearExplainer(
                self.model, X_clean_train, feature_perturbation="interventional"
            )

        logger.info("✅ Training complete and best model stored.")
        
        # Generate model performance plots
        self._plot_model_performance(X_clean_val, y_val, best_name)
        
        return best_name, scores

    def _plot_model_performance(self, X_val, y_val, model_name):
        """Create performance visualization plots"""
        os.makedirs("reports/figures/dropout_prediction", exist_ok=True)
        
        # Predictions
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax1, cmap='Blues')
        ax1.set_title(f'Confusion Matrix - {model_name}')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'ROC Curve - {model_name}')
        ax2.legend()
        
        # 3. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.preproc.get_feature_names_out()
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            ax3.bar(range(len(indices)), importances[indices])
            ax3.set_title(f'Top 10 Feature Importances - {model_name}')
            ax3.set_xticks(range(len(indices)))
            ax3.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        # 4. Prediction Distribution
        ax4.hist(y_pred_proba[y_val == 0], alpha=0.5, label='No Dropout', bins=20)
        ax4.hist(y_pred_proba[y_val == 1], alpha=0.5, label='Dropout', bins=20)
        ax4.set_xlabel('Predicted Dropout Probability')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Prediction Distribution - {model_name}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"reports/figures/dropout_prediction/dropout_prediction_{model_name.lower()}_performance.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        X_clean_test = self.preproc.transform(X_test)
        preds = self.model.predict(X_clean_test)
        
        logger.info("Classification Report:")
        print(classification_report(y_test, preds))
        logger.info("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    def predict(self, X_new):
        """Predict dropout probability with SHAP explanations"""
        X_clean = self.preproc.transform(X_new)
        dropout_prob = self.model.predict_proba(X_clean)[:, 1]

        # SHAP explanations
        if self.shap_explainer:
            shap_raw = self.shap_explainer.shap_values(X_clean)
            if len(shap_raw.shape) == 3:
                shap_raw = shap_raw[:, :, 1]

            features = self.preproc.get_feature_names_out()
            shap_df = pd.DataFrame(shap_raw, columns=features)
        else:
            shap_df = pd.DataFrame()

        # Assemble results
        results = X_new.reset_index(drop=True)
        results['dropout_probability'] = dropout_prob
        if not shap_df.empty:
            results = pd.concat([results, shap_df], axis=1)

        return results

    def save_models(self, filepath_prefix="models/dropout_prediction/medoptix_dropout_prediction"):
        """Save trained models and metadata"""
        os.makedirs("models/dropout_prediction", exist_ok=True)
        
        joblib.dump(self.preproc, f"{filepath_prefix}_preprocessor.pkl")
        joblib.dump(self.model, f"{filepath_prefix}_model.pkl")
        
        # Save feature names and column info
        feature_names = self.preproc.get_feature_names_out()
        joblib.dump(feature_names, f"{filepath_prefix}_feature_names.pkl")
        
        column_info = {
            'numerical_columns': self.num_col,
            'categorical_columns': self.cat_col,
            'all_columns': self.num_col + self.cat_col
        }
        joblib.dump(column_info, f"{filepath_prefix}_columns.pkl")
        
        logger.info(f"Dropout prediction models saved with prefix: {filepath_prefix}")

if __name__ == "__main__":
    # Create output directories
    os.makedirs("models/dropout_prediction", exist_ok=True)
    os.makedirs("reports/figures/dropout_prediction", exist_ok=True)
    
    predictor = MedOptixPrediction()

    # Load data from database
    patients, sessions, dropout_flag = predictor.load_data()

    # Prepare features
    X_all = predictor.prepare_patient(patients, sessions)
    logger.info(f"Prepared data shape: {X_all.shape}")
    logger.info(f"Features: {X_all.columns.tolist()}")

    # Define target variable
    y_all = dropout_flag.set_index("patient_id")["dropout"]
    
    # Align X and y data
    common_patients = X_all.set_index('patient_id').index.intersection(y_all.index)
    X_aligned = X_all.set_index('patient_id').loc[common_patients].reset_index()
    y_aligned = y_all.loc[common_patients].values
    
    logger.info(f"Aligned data shape: X={X_aligned.shape}, y={y_aligned.shape}")

    # Data splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_aligned, y_aligned, test_size=0.2, random_state=42, stratify=y_aligned
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Train the model
    best_model_name, scores = predictor.train_and_select_best(X_train, y_train, X_val, y_val)

    # Evaluate the model
    logger.info("\n=== Validation Set Evaluation ===")
    predictor.evaluate_model(X_val, y_val)
    
    logger.info("\n=== Test Set Evaluation ===")
    predictor.evaluate_model(X_test, y_test)

    # Save the model
    predictor.save_models()
    
    logger.info("✅ Dropout prediction pipeline completed successfully!")
    logger.info("Models and visualizations saved in models/dropout_prediction/ and reports/figures/dropout_prediction/")