from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class PatientInput(BaseModel):
    """Input schema for patient data"""
    # Patient basic information
    age: float = Field(..., ge=0, le=120, description="Patient's age")
    gender: str = Field(..., description="Gender (Male/Female)")
    bmi: float = Field(..., ge=10, le=100, description="Body Mass Index")
    smoker: str = Field(..., description="Smoking status (Yes/No)")
    chronic_cond: str = Field(..., description="Chronic condition status (Yes/No)")
    injury_type: str = Field(..., description="Type of injury")
    referral_source: str = Field(..., description="Source of referral")
    insurance_type: str = Field(..., description="Type of insurance")

    # Session information
    n_sessions: int = Field(..., ge=0, description="Number of sessions attended")
    avg_session_duration: float = Field(..., ge=0, description="Average session duration in minutes")
    first_week: int = Field(default=0, ge=0, description="Sessions in first week")
    last_week: int = Field(default=0, ge=0, description="Sessions in last week")
    mean_pain: float = Field(..., ge=0, le=10, description="Mean pain score (0-10)")
    mean_pain_delta: float = Field(..., description="Change in pain score")
    home_adherence_mean: float = Field(..., ge=0, le=100, description="Mean home exercise adherence percentage")
    satisfaction_mean: float = Field(..., ge=0, le=5, description="Mean patient satisfaction score")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35,
                "gender": "Female",
                "bmi": 24.5,
                "smoker": "No",
                "chronic_cond": "No",
                "injury_type": "Knee Injury",
                "referral_source": "Doctor",
                "insurance_type": "Private",
                "n_sessions": 8,
                "avg_session_duration": 45.0,
                "first_week": 2,
                "last_week": 1,
                "mean_pain": 4.5,
                "mean_pain_delta": -1.2,
                "home_adherence_mean": 75.0,
                "satisfaction_mean": 4.2
            }
        }
    )

class PatientBase(BaseModel):
    """Base patient model for database operations"""
    patient_id: Optional[int] = None
    age: float
    gender: str
    bmi: float
    smoker: str
    chronic_cond: str
    injury_type: str
    referral_source: str
    insurance_type: str
    
    model_config = ConfigDict(from_attributes=True)

class Patient(PatientBase):
    """Full patient model with ID"""
    patient_id: int

# Response Models
class DropoutPredictionResponse(BaseModel):
    """Response model for dropout prediction"""
    patient_id: str
    dropout_probability: float = Field(..., ge=0, le=1, description="Dropout probability (0-1)")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    recommendations: List[str] = Field(..., description="List of recommendations")

class SegmentationResponse(BaseModel):
    """Response model for patient segmentation"""
    patient_id: str
    cluster_id: int = Field(..., description="Cluster ID")
    cluster_name: str = Field(..., description="Cluster name/description")
    characteristics: List[str] = Field(..., description="Cluster characteristics")

class AdherenceResponse(BaseModel):
    """Response model for adherence prediction"""
    patient_id: str
    adherence_level: str = Field(..., description="Adherence level (Low/Medium/High)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    recommendations: List[str] = Field(..., description="Adherence recommendations")

class ComprehensiveResponse(BaseModel):
    """Response model for comprehensive predictions"""
    patient_id: str
    predictions: Dict[str, Any] = Field(..., description="All prediction results")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: bool
    available_models: List[str]
    model_info: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Batch prediction models
class BatchPatientInput(BaseModel):
    """Input for batch predictions"""
    patients: List[PatientInput]
    prediction_types: List[str] = Field(
        default=["dropout", "segmentation", "adherence"],
        description="Types of predictions to run"
    )

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    total_patients: int
    successful_predictions: int
    failed_predictions: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, str]] = Field(default_factory=list)