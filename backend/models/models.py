from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, default="clinician")  # clinician, admin, patient
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    patients = relationship("Patient", back_populates="clinician")
    assessments_created = relationship("Assessment", back_populates="clinician")


class Patient(Base):
    """Patient model for storing patient information."""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    date_of_birth = Column(DateTime)
    gender = Column(String)
    email = Column(String)
    phone = Column(String)
    medical_history = Column(Text)
    emergency_contact = Column(JSON)
    
    # Foreign Keys
    clinician_id = Column(Integer, ForeignKey("users.id"))
    
    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    clinician = relationship("User", back_populates="patients")
    assessments = relationship("Assessment", back_populates="patient")
    vital_signs = relationship("VitalSigns", back_populates="patient")


class Assessment(Base):
    """Assessment model for storing assessment sessions and results."""
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(String, unique=True, index=True, nullable=False)
    
    # Foreign Keys
    patient_id = Column(Integer, ForeignKey("patients.id"))
    clinician_id = Column(Integer, ForeignKey("users.id"))
    
    # Assessment Data
    assessment_type = Column(String)  # initial, follow-up, emergency
    status = Column(String, default="in_progress")  # in_progress, completed, cancelled
    duration_seconds = Column(Integer)
    
    # AI Analysis Results
    stress_score = Column(Float)
    anxiety_score = Column(Float)
    emotion_classification = Column(String)
    
    # Multimodal Analysis
    facial_analysis = Column(JSON)  # Facial expression data
    voice_analysis = Column(JSON)  # Voice stress indicators
    posture_analysis = Column(JSON)  # Body language data
    combined_metrics = Column(JSON)  # Fused multimodal data
    
    # Clinical Notes
    clinician_notes = Column(Text)
    recommendations = Column(Text)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="assessments")
    clinician = relationship("User", back_populates="assessments_created")
    real_time_data = relationship("RealTimeData", back_populates="assessment")


class RealTimeData(Base):
    """Real-time streaming data during assessments."""
    __tablename__ = "real_time_data"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"))
    
    # Time series data
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    data_type = Column(String)  # facial, audio, posture
    features = Column(JSON)
    stress_level = Column(Float)
    confidence_score = Column(Float)
    
    # Relationship
    assessment = relationship("Assessment", back_populates="real_time_data")


class VitalSigns(Base):
    """Patient vital signs tracking."""
    __tablename__ = "vital_signs"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    
    # Vital measurements
    heart_rate = Column(Integer)
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    respiratory_rate = Column(Integer)
    temperature = Column(Float)
    
    # Timestamps
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    patient = relationship("Patient", back_populates="vital_signs")


class AnalyticsLog(Base):
    """Analytics and reporting logs."""
    __tablename__ = "analytics_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String)  # assessment_completed, stress_alert, etc.
    event_data = Column(JSON)
    metadata = Column(JSON)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
