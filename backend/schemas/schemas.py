from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: str = "clinician"


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None


# Patient Schemas
class EmergencyContact(BaseModel):
    name: str
    relationship: str
    phone: str


class PatientBase(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    medical_history: Optional[str] = None
    emergency_contact: Optional[Dict[str, Any]] = None


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    medical_history: Optional[str] = None
    emergency_contact: Optional[Dict[str, Any]] = None


class PatientResponse(PatientBase):
    id: int
    patient_id: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Assessment Schemas
class AssessmentBase(BaseModel):
    assessment_type: str = "initial"
    patient_id: int


class AssessmentCreate(AssessmentBase):
    pass


class AssessmentUpdate(BaseModel):
    status: Optional[str] = None
    stress_score: Optional[float] = None
    anxiety_score: Optional[float] = None
    emotion_classification: Optional[str] = None
    facial_analysis: Optional[Dict[str, Any]] = None
    voice_analysis: Optional[Dict[str, Any]] = None
    posture_analysis: Optional[Dict[str, Any]] = None
    combined_metrics: Optional[Dict[str, Any]] = None
    clinician_notes: Optional[str] = None
    recommendations: Optional[str] = None


class AssessmentResponse(AssessmentBase):
    id: int
    assessment_id: str
    status: str
    duration_seconds: Optional[int] = None
    stress_score: Optional[float] = None
    anxiety_score: Optional[float] = None
    emotion_classification: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class AssessmentDetailResponse(AssessmentResponse):
    facial_analysis: Optional[Dict[str, Any]] = None
    voice_analysis: Optional[Dict[str, Any]] = None
    posture_analysis: Optional[Dict[str, Any]] = None
    combined_metrics: Optional[Dict[str, Any]] = None
    clinician_notes: Optional[str] = None
    recommendations: Optional[str] = None


# Real-time Data Schemas
class RealTimeDataCreate(BaseModel):
    assessment_id: int
    data_type: str
    features: Dict[str, Any]
    stress_level: float
    confidence_score: float


class RealTimeDataResponse(RealTimeDataCreate):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True


# Analytics Schemas
class AnalyticsLogCreate(BaseModel):
    event_type: str
    event_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class AnalyticsLogResponse(AnalyticsLogCreate):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True


class StressTrendResponse(BaseModel):
    patient_id: int
    date: datetime
    average_stress: float
    max_stress: float
    min_stress: float
    assessment_count: int


class PatientSummaryResponse(BaseModel):
    total_patients: int
    active_patients: int
    recent_assessments: int
    high_stress_alerts: int


# Vital Signs Schemas
class VitalSignsCreate(BaseModel):
    patient_id: int
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None
    temperature: Optional[float] = None


class VitalSignsResponse(VitalSignsCreate):
    id: int
    recorded_at: datetime
    
    class Config:
        from_attributes = True
