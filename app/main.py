from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
import os
from datetime import datetime
import hashlib

# Import local modules
from .prediction import MedoptixPredictor
from .schemas import (
    PatientInput, DropoutPredictionResponse, SegmentationResponse, 
    AdherenceResponse, ComprehensiveResponse, HealthResponse, 
    ErrorResponse, BatchPatientInput, BatchPredictionResponse
)
from .database import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
predictor = MedoptixPredictor()
db_manager = DatabaseManager()

# Initialize FastAPI app
app = FastAPI(
    title="MedOptix AI Platform",
    description="AI-powered patient analytics for physical therapy - dropout prediction, patient segmentation, and adherence forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for error handling
async def get_predictor():
    """Dependency to get predictor instance with error handling"""
    if not predictor.models:
        raise HTTPException(status_code=503, detail="ML models not loaded. Please check server status.")
    return predictor

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models and initialize services on startup"""
    logger.info("Starting MedOptix API...")
    success = predictor.load_models()
    if not success:
        logger.error("Failed to load ML models")
    else:
        logger.info("ML models loaded successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down MedOptix API...")
    db_manager.close_connection()

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint to verify API status and model availability
    """
    models_loaded = bool(predictor.models)
    
    model_info = {
        "total_models": len(predictor.models),
        "dropout_model_available": "dropout_model" in predictor.models,
        "segmentation_model_available": "segmentation_kmeans" in predictor.models,
        "adherence_model_available": "adherence_model" in predictor.models,
    }
    
    return HealthResponse(
        status="Healthy" if models_loaded else "Degraded",
        models_loaded=models_loaded,
        available_models=list(predictor.models.keys()),
        model_info=model_info
    )

# Utility function to generate patient ID
def generate_patient_id(patient_data: Dict[str, Any]) -> str:
    """Generate a unique patient ID based on input data"""
    data_str = str(sorted(patient_data.items()))
    hash_object = hashlib.md5(data_str.encode())
    return f"pt_{hash_object.hexdigest()[:8]}"

# Dropout prediction endpoint
@app.post("/predict/dropout", response_model=DropoutPredictionResponse, tags=["Predictions"])
async def predict_dropout(
    patient_data: PatientInput,
    predictor: MedoptixPredictor = Depends(get_predictor)
):
    """
    Predict patient dropout probability and risk level
    
    Returns:
    - Dropout probability (0-1)
    - Risk level (Low/Medium/High)
    - Personalized recommendations
    """
    try:
        # Convert to dict and predict
        data_dict = patient_data.dict()
        dropout_prob, _, risk_level, recommendations = predictor.predict_dropout(data_dict)
        
        # Generate patient ID
        patient_id = generate_patient_id(data_dict)
        
        return DropoutPredictionResponse(
            patient_id=patient_id,
            dropout_probability=dropout_prob,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in dropout prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Patient segmentation endpoint
@app.post("/predict/segmentation", response_model=SegmentationResponse, tags=["Predictions"])
async def predict_segmentation(
    patient_data: PatientInput,
    predictor: MedoptixPredictor = Depends(get_predictor)
):
    """
    Predict patient segment/cluster based on characteristics
    
    Returns:
    - Cluster ID and name
    - Cluster characteristics
    """
    try:
        data_dict = patient_data.dict()
        cluster_id, cluster_name, characteristics = predictor.predict_segment(data_dict)
        
        patient_id = generate_patient_id(data_dict)
        
        return SegmentationResponse(
            patient_id=patient_id,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            characteristics=characteristics
        )
        
    except Exception as e:
        logger.error(f"Error in segmentation prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

# Adherence prediction endpoint
@app.post("/predict/adherence", response_model=AdherenceResponse, tags=["Predictions"])
async def predict_adherence(
    patient_data: PatientInput,
    predictor: MedoptixPredictor = Depends(get_predictor)
):
    """
    Predict patient adherence level
    
    Returns:
    - Adherence level (Low/Medium/High)
    - Prediction confidence
    - Adherence improvement recommendations
    """
    try:
        data_dict = patient_data.dict()
        adherence_level, confidence, recommendations = predictor.predict_adherence(data_dict)
        
        patient_id = generate_patient_id(data_dict)
        
        return AdherenceResponse(
            patient_id=patient_id,
            adherence_level=adherence_level,
            confidence=confidence,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in adherence prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Adherence prediction failed: {str(e)}")

# Comprehensive prediction endpoint
@app.post("/predict/comprehensive", response_model=ComprehensiveResponse, tags=["Predictions"])
async def predict_comprehensive(
    patient_data: PatientInput,
    predictor: MedoptixPredictor = Depends(get_predictor)
):
    """
    Get all available predictions for a patient
    
    Returns:
    - Dropout prediction
    - Segmentation results  
    - Adherence forecast
    """
    try:
        data_dict = patient_data.dict()
        results = predictor.get_comprehensive_prediction(data_dict)
        
        patient_id = generate_patient_id(data_dict)
        
        return ComprehensiveResponse(
            patient_id=patient_id,
            predictions=results["predictions"]
        )
        
    except Exception as e:
        logger.error(f"Error in comprehensive prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    batch_input: BatchPatientInput,
    background_tasks: BackgroundTasks,
    predictor: MedoptixPredictor = Depends(get_predictor)
):
    """
    Run predictions on multiple patients
    
    Supports:
    - Batch dropout prediction
    - Batch segmentation
    - Batch adherence forecasting
    """
    try:
        results = []
        errors = []
        successful = 0
        
        for i, patient_data in enumerate(batch_input.patients):
            try:
                data_dict = patient_data.dict()
                patient_id = generate_patient_id(data_dict)
                
                patient_results = {"patient_id": patient_id}
                
                # Run requested predictions
                if "dropout" in batch_input.prediction_types:
                    dropout_prob, _, risk_level, recs = predictor.predict_dropout(data_dict)
                    patient_results["dropout"] = {
                        "probability": dropout_prob,
                        "risk_level": risk_level,
                        "recommendations": recs
                    }
                
                if "segmentation" in batch_input.prediction_types:
                    cluster_id, cluster_name, chars = predictor.predict_segment(data_dict)
                    patient_results["segmentation"] = {
                        "cluster_id": cluster_id,
                        "cluster_name": cluster_name,
                        "characteristics": chars
                    }
                
                if "adherence" in batch_input.prediction_types:
                    adh_level, confidence, adh_recs = predictor.predict_adherence(data_dict)
                    patient_results["adherence"] = {
                        "level": adh_level,
                        "confidence": confidence,
                        "recommendations": adh_recs
                    }
                
                results.append(patient_results)
                successful += 1
                
            except Exception as e:
                errors.append({
                    "patient_index": i,
                    "error": str(e)
                })
        
        return BatchPredictionResponse(
            total_patients=len(batch_input.patients),
            successful_predictions=successful,
            failed_predictions=len(errors),
            results=results,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Database endpoints
@app.get("/patients/{patient_id}", tags=["Database"])
async def get_patient(patient_id: int):
    """Get patient data from database"""
    try:
        patient_data = db_manager.get_patient_data(patient_id)
        if patient_data.empty:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient_data.to_dict('records')[0]
    except Exception as e:
        logger.error(f"Error retrieving patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}/sessions", tags=["Database"])
async def get_patient_sessions(patient_id: int):
    """Get patient session history from database"""
    try:
        sessions_data = db_manager.get_sessions_data(patient_id)
        if sessions_data.empty:
            raise HTTPException(status_code=404, detail="No sessions found for patient")
        return sessions_data.to_dict('records')
    except Exception as e:
        logger.error(f"Error retrieving sessions for patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model information endpoints
@app.get("/models/info", tags=["System"])
async def get_model_info():
    """Get detailed information about loaded models"""
    try:
        model_info = {
            "loaded_models": list(predictor.models.keys()),
            "model_details": {}
        }
        
        # Add model-specific information
        if "dropout_model" in predictor.models:
            model_info["model_details"]["dropout"] = {
                "type": str(type(predictor.models["dropout_model"]).__name__),
                "features": len(predictor.models.get("dropout_feature_names", [])),
            }
        
        if "segmentation_kmeans" in predictor.models:
            model_info["model_details"]["segmentation"] = {
                "type": "KMeans",
                "n_clusters": getattr(predictor.models["segmentation_kmeans"], 'n_clusters', 'unknown')
            }
        
        if "adherence_model" in predictor.models:
            model_info["model_details"]["adherence"] = {
                "type": str(type(predictor.models["adherence_model"]).__name__),
            }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "MedOptix AI Platform API",
        "version": "1.0.0",
        "description": "AI-powered patient analytics for physical therapy",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "dropout_prediction": "/predict/dropout",
            "segmentation": "/predict/segmentation",
            "adherence": "/predict/adherence",
            "comprehensive": "/predict/comprehensive",
            "batch": "/predict/batch"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )