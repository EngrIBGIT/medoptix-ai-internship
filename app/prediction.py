import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, Tuple, List
import os

logger = logging.getLogger(__name__)

class MedoptixPredictor:
    """Handles all ML predictions for Medoptix"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self) -> bool:
        """Load all trained models"""
        try:
            # Dropout prediction models
            dropout_path = "models/dropout_prediction/"
            if os.path.exists(f"{dropout_path}medoptix_dropout_prediction_preprocessor.pkl"):
                self.models["dropout_preprocessor"] = joblib.load(f"{dropout_path}medoptix_dropout_prediction_preprocessor.pkl")
                self.models["dropout_model"] = joblib.load(f"{dropout_path}medoptix_dropout_prediction_model.pkl")
                self.models["dropout_feature_names"] = joblib.load(f"{dropout_path}medoptix_dropout_prediction_feature_names.pkl")
                self.models["dropout_columns"] = joblib.load(f"{dropout_path}medoptix_dropout_prediction_columns.pkl")
                logger.info("Dropout prediction models loaded successfully")
            
            # Segmentation models
            seg_path = "models/segmentation/"
            if os.path.exists(f"{seg_path}medoptix_segmentation_preprocessor.pkl"):
                self.models["segmentation_preprocessor"] = joblib.load(f"{seg_path}medoptix_segmentation_preprocessor.pkl")
                self.models["segmentation_pca"] = joblib.load(f"{seg_path}medoptix_segmentation_pca.pkl")
                self.models["segmentation_kmeans"] = joblib.load(f"{seg_path}medoptix_segmentation_kmeans.pkl")
                logger.info("Segmentation models loaded successfully")
            
            # Adherence forecasting models
            adh_path = "models/adherence_forecasting/"
            if os.path.exists(f"{adh_path}medoptix_adherence_forecasting_preprocessor.pkl"):
                self.models["adherence_preprocessor"] = joblib.load(f"{adh_path}medoptix_adherence_forecasting_preprocessor.pkl")
                self.models["adherence_model"] = joblib.load(f"{adh_path}medoptix_adherence_forecasting_model.pkl")
                self.models["adherence_label_encoder"] = joblib.load(f"{adh_path}medoptix_adherence_forecasting_label_encoder.pkl")
                logger.info("Adherence forecasting models loaded successfully")
            
            logger.info(f"Available models: {list(self.models.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_dropout(self, patient_data: Dict[str, Any]) -> Tuple[float, int, str, List[str]]:
        """Predict dropout probability and return recommendations"""
        try:
            if "dropout_model" not in self.models:
                raise ValueError("Dropout model not loaded")
            
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Preprocess and predict
            X_processed = self.models["dropout_preprocessor"].transform(df)
            dropout_prob_raw = self.models["dropout_model"].predict_proba(X_processed)[0, 1]
            dropout_prob = float(dropout_prob_raw)
            
            # Get risk level and recommendations
            risk_level, recommendations = self._get_dropout_risk_recommendations(dropout_prob)
            
            return dropout_prob, 0, risk_level, recommendations
            
        except Exception as e:
            logger.error(f"Error in dropout prediction: {str(e)}")
            return 0.5, 0, "Medium", ["Unable to generate recommendations due to prediction error"]
    
    def predict_segment(self, patient_data: Dict[str, Any]) -> Tuple[int, str, List[str]]:
        """Predict patient segment and return characteristics"""
        try:
            if "segmentation_kmeans" not in self.models:
                raise ValueError("Segmentation model not loaded")
            
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Preprocess
            X_processed = self.models["segmentation_preprocessor"].transform(df)
            
            # Apply PCA
            X_reduced = self.models["segmentation_pca"].transform(X_processed)
            
            # Predict cluster
            cluster = self.models["segmentation_kmeans"].predict(X_reduced)[0]
            
            # Get cluster characteristics
            cluster_name, characteristics = self._get_cluster_characteristics(cluster)
            
            return int(cluster), cluster_name, characteristics
            
        except Exception as e:
            logger.error(f"Error in segmentation prediction: {str(e)}")
            return 0, "Unknown", ["Unable to determine patient segment"]
    
    def predict_adherence(self, patient_data: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """Predict adherence level and return recommendations"""
        try:
            if "adherence_model" not in self.models:
                raise ValueError("Adherence model not loaded")
            
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Preprocess and predict
            X_processed = self.models["adherence_preprocessor"].transform(df)
            pred_encoded = self.models["adherence_model"].predict(X_processed)[0]
            pred_proba = self.models["adherence_model"].predict_proba(X_processed)[0]
            
            # Decode prediction
            adherence_level = self.models["adherence_label_encoder"].inverse_transform([pred_encoded])[0]
            confidence = float(max(pred_proba))
            
            # Get recommendations
            recommendations = self._get_adherence_recommendations(adherence_level)
            
            return adherence_level, confidence, recommendations
            
        except Exception as e:
            logger.error(f"Error in adherence prediction: {str(e)}")
            return "Medium", 0.33, ["Unable to generate adherence recommendations"]
    
    def _get_dropout_risk_recommendations(self, dropout_prob: float) -> Tuple[str, List[str]]:
        """Generate risk level and recommendations for dropout"""
        if dropout_prob > 0.7:
            return "High", [
                "Schedule immediate follow-up call within 24 hours",
                "Assign dedicated support specialist",
                "Offer flexible scheduling options",
                "Consider telehealth sessions",
                "Implement motivational interviewing techniques"
            ]
        elif dropout_prob > 0.3:
            return "Medium", [
                "Send weekly check-in messages",
                "Monitor attendance closely",
                "Offer additional support if needed",
                "Provide educational materials about treatment benefits",
                "Consider group therapy sessions"
            ]
        else:
            return "Low", [
                "Continue standard care procedures",
                "Maintain regular check-ins",
                "Celebrate progress milestones",
                "Provide positive reinforcement"
            ]
    
    def _get_cluster_characteristics(self, cluster: int) -> Tuple[str, List[str]]:
        """Get characteristics for each cluster"""
        cluster_info = {
            0: ("High Engagement", [
                "High session attendance",
                "Good home exercise adherence", 
                "Positive treatment outcomes",
                "Low dropout risk"
            ]),
            1: ("Moderate Engagement", [
                "Average session attendance",
                "Variable home exercise adherence",
                "Mixed treatment outcomes",
                "Moderate dropout risk"
            ]),
            2: ("Low Engagement", [
                "Poor session attendance",
                "Low home exercise adherence",
                "Concerning treatment outcomes",
                "High dropout risk"
            ])
        }
        
        return cluster_info.get(cluster, ("Unknown", ["Unable to determine characteristics"]))
    
    def _get_adherence_recommendations(self, adherence_level: str) -> List[str]:
        """Generate recommendations based on adherence level"""
        recommendations = {
            "High": [
                "Maintain current engagement strategies",
                "Use as peer mentor for other patients",
                "Consider advanced treatment protocols",
                "Schedule less frequent check-ins"
            ],
            "Medium": [
                "Provide additional motivation techniques",
                "Send reminder notifications",
                "Offer flexible exercise options",
                "Monitor progress more closely"
            ],
            "Low": [
                "Implement intensive support program",
                "Schedule frequent check-ins",
                "Simplify exercise protocols",
                "Consider motivational interviewing",
                "Address barriers to adherence"
            ]
        }
        
        return recommendations.get(adherence_level, ["Standard care recommendations"])
    
    def get_comprehensive_prediction(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get all predictions for a patient"""
        try:
            results = {
                "patient_id": patient_data.get("patient_id", "unknown"),
                "predictions": {}
            }
            
            # Dropout prediction
            if "dropout_model" in self.models:
                dropout_prob, _, dropout_risk, dropout_recs = self.predict_dropout(patient_data)
                results["predictions"]["dropout"] = {
                    "probability": dropout_prob,
                    "risk_level": dropout_risk,
                    "recommendations": dropout_recs
                }
            
            # Segmentation prediction
            if "segmentation_kmeans" in self.models:
                cluster_id, cluster_name, cluster_chars = self.predict_segment(patient_data)
                results["predictions"]["segmentation"] = {
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "characteristics": cluster_chars
                }
            
            # Adherence prediction
            if "adherence_model" in self.models:
                adh_level, confidence, adh_recs = self.predict_adherence(patient_data)
                results["predictions"]["adherence"] = {
                    "level": adh_level,
                    "confidence": confidence,
                    "recommendations": adh_recs
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive prediction: {str(e)}")
            return {
                "patient_id": patient_data.get("patient_id", "unknown"),
                "error": str(e),
                "predictions": {}
            }