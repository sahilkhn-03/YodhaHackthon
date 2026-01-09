"""
Database models for NeuroBalance AI.
PRIVACY-FIRST: Only stores analysis results, never raw video/audio/biometric data.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    """
    User model for clinicians and admins.
    Stores authentication and profile info.
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)  # Never store plain passwords!
    full_name = Column(String)
    role = Column(String, default="clinician")  # clinician or admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship: One clinician has many patients
    patients = relationship("Patient", back_populates="clinician")


class Patient(Base):
    """
    Patient model - stores ONLY non-PII metadata.
    NO names, addresses, or identifiable info stored here per privacy requirements.
    """
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)  # Anonymous ID: PT-20260109-ABC123
    age_range = Column(String)  # "25-30" instead of exact age
    gender = Column(String, nullable=True)  # Optional
    baseline_heart_rate = Column(Integer)  # Resting HR for comparison
    clinician_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    clinician = relationship("User", back_populates="patients")
    assessments = relationship("Assessment", back_populates="patient")


class Assessment(Base):
    """
    Assessment model - stores ONLY analysis results and summaries.
    PRIVACY: NO facial landmarks, NO audio files, NO video frames stored here!
    """
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(String, unique=True, index=True)  # AS-20260109-XYZ
    patient_id = Column(Integer, ForeignKey("patients.id"))
    clinician_id = Column(Integer, ForeignKey("users.id"))
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    duration_seconds = Column(Integer)
    
    # Analysis Results (processed metrics only)
    stress_score = Column(Float)  # 0.0 to 1.0
    anxiety_score = Column(Float)  # 0.0 to 1.0
    
    # Heart Rate Analysis (aggregated metrics only, no raw ECG)
    heart_rate_avg = Column(Integer)  # Average BPM
    heart_rate_min = Column(Integer)
    heart_rate_max = Column(Integer)
    heart_rate_baseline = Column(Integer)  # Patient's normal resting HR
    heart_rate_change_percent = Column(Float)  # % change from baseline
    
    # Text Summaries (human-readable explanations)
    analysis_summary = Column(Text)  # "Heart rate elevated by 35%, vocal stress indicators present"
    likely_reasons = Column(Text)  # "Acute stress response, possible anxiety trigger"
    recommendations = Column(Text)  # "Deep breathing exercises, follow-up in 24 hours"
    
    # Status
    status = Column(String, default="completed")  # completed, in_progress, cancelled
    
    # Relationships
    patient = relationship("Patient", back_populates="assessments")


# Why these models?
# 1. User: Manage clinician logins and access control
# 2. Patient: Track patients with anonymous IDs (privacy-first)
# 3. Assessment: Store ONLY final analysis results, never raw biometric data
# 
# What we DON'T store (privacy protection):
# - Raw video frames
# - Audio recordings
# - Facial landmark coordinates
# - Voice waveforms
# - Personal names/addresses
