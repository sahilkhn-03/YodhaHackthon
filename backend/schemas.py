"""
Pydantic schemas for request validation and response serialization.

Why Pydantic?
- Validates incoming data automatically
- Converts database models to JSON for API responses
- Provides clear API documentation in OpenAPI/Swagger
"""

from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


# ============= USER SCHEMAS =============

class UserCreate(BaseModel):
    """Schema for creating a new user (registration)"""
    email: EmailStr  # Validates email format
    username: str
    password: str  # Will be hashed before storing
    full_name: Optional[str] = None
    role: str = "clinician"


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str


class UserResponse(BaseModel):
    """Schema for user data in API responses"""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True  # Allows converting SQLAlchemy models to Pydantic


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"


# ============= PATIENT SCHEMAS =============

class PatientCreate(BaseModel):
    """Schema for creating a new patient"""
    age_range: str  # "25-30" - privacy: no exact age
    gender: Optional[str] = None
    baseline_heart_rate: int  # Used for comparison in assessments


class PatientResponse(BaseModel):
    """Schema for patient data in API responses"""
    id: int
    patient_id: str  # PT-20260109-ABC123
    age_range: str
    gender: Optional[str]
    baseline_heart_rate: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============= ASSESSMENT SCHEMAS =============

class AssessmentCreate(BaseModel):
    """
    Schema for creating a new assessment.
    Contains ONLY analysis results - no raw biometric data!
    """
    patient_id: int
    duration_seconds: int
    
    # Analysis scores
    stress_score: float  # 0.0 to 1.0
    anxiety_score: float  # 0.0 to 1.0
    
    # Heart rate metrics (processed only, no raw ECG)
    heart_rate_avg: int
    heart_rate_min: int
    heart_rate_max: int
    heart_rate_change_percent: float
    
    # Text explanations
    analysis_summary: str  # "Heart rate elevated by 35%..."
    likely_reasons: str  # "Acute stress response..."
    recommendations: str  # "Deep breathing exercises..."


class AssessmentResponse(BaseModel):
    """Schema for assessment data in API responses"""
    id: int
    assessment_id: str  # AS-20260109-XYZ
    patient_id: int
    timestamp: datetime
    duration_seconds: int
    stress_score: float
    anxiety_score: float
    heart_rate_avg: int
    heart_rate_min: int
    heart_rate_max: int
    heart_rate_change_percent: float
    analysis_summary: str
    likely_reasons: str
    recommendations: str
    status: str
    
    class Config:
        from_attributes = True


# Why schemas?
# 1. REQUEST VALIDATION: Pydantic automatically validates incoming data
#    - Wrong email format? Rejected before hitting database
#    - Missing required field? Clear error message
#
# 2. RESPONSE SERIALIZATION: Converts database objects to JSON
#    - SQLAlchemy model → Pydantic schema → JSON
#
# 3. API DOCUMENTATION: FastAPI auto-generates docs from these schemas
#    - Visit /docs to see interactive API documentation
#
# 4. TYPE SAFETY: Catch errors during development, not production
