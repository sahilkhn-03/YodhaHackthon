"""
Assessment routes: Store and retrieve AI analysis results.

CRITICAL PRIVACY NOTE:
These endpoints store ONLY analysis results (stress scores, HR changes, text summaries).
NO raw biometric data (video, audio, facial landmarks) is ever stored here!

Data flow:
1. Frontend: Capture video/audio
2. Frontend: AI processing (facial analysis, voice stress detection)
3. Frontend: Extract metrics (stress score, HR changes)
4. Frontend → Backend: Send ONLY final metrics via this API
5. Backend: Store in Supabase
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime

from database import get_db
from models import User, Patient, Assessment
from schemas import AssessmentCreate, AssessmentResponse
from routes_auth import get_current_user

router = APIRouter()


def generate_assessment_id() -> str:
    """
    Generate unique assessment ID.
    Format: AS-20260109143025-ABC
    
    Contains timestamp for easy sorting and debugging.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = str(uuid.uuid4())[:3].upper()
    return f"AS-{timestamp}-{random_part}"


@router.post("/", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(
    assessment: AssessmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Store AI analysis results for an assessment.
    
    What this stores:
    ✅ Stress score (0-1 float)
    ✅ Anxiety score (0-1 float)
    ✅ Heart rate metrics (avg, min, max, % change)
    ✅ Text summaries (analysis, reasons, recommendations)
    
    What this NEVER stores (privacy):
    ❌ Video frames
    ❌ Audio recordings
    ❌ Facial landmark coordinates
    ❌ Raw ECG/heart rate data
    ❌ Voice waveforms
    
    Frontend responsibility:
    - Process all video/audio locally
    - Extract only the final metrics
    - Send ONLY those metrics to this endpoint
    - Discard raw biometric data immediately
    
    Why this approach?
    - HIPAA compliance: No biometric PII stored
    - Privacy-first: Raw data never leaves user's device
    - Storage efficient: Small text/numbers vs large video files
    - Faster: No large file uploads
    """
    # Verify patient exists and belongs to current clinician
    patient = db.query(Patient).filter(
        Patient.id == assessment.patient_id,
        Patient.clinician_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found or access denied"
        )
    
    # Generate assessment ID
    assessment_id = generate_assessment_id()
    
    # Create assessment record with ONLY analysis results
    db_assessment = Assessment(
        assessment_id=assessment_id,
        patient_id=assessment.patient_id,
        clinician_id=current_user.id,
        duration_seconds=assessment.duration_seconds,
        stress_score=assessment.stress_score,
        anxiety_score=assessment.anxiety_score,
        heart_rate_avg=assessment.heart_rate_avg,
        heart_rate_min=assessment.heart_rate_min,
        heart_rate_max=assessment.heart_rate_max,
        heart_rate_baseline=patient.baseline_heart_rate,
        heart_rate_change_percent=assessment.heart_rate_change_percent,
        analysis_summary=assessment.analysis_summary,
        likely_reasons=assessment.likely_reasons,
        recommendations=assessment.recommendations,
        status="completed"
    )
    
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)
    
    return db_assessment


@router.get("/", response_model=List[AssessmentResponse])
def get_assessments(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    patient_id: int = None,
    skip: int = 0,
    limit: int = 100
):
    """
    Get assessment history.
    
    Optional filter by patient_id:
    - GET /assessments?patient_id=5  → All assessments for patient 5
    - GET /assessments                → All assessments for all patients
    
    Use cases:
    - Dashboard: Recent assessments across all patients
    - Patient detail page: Assessment history for one patient
    - Analytics: Trend analysis over time
    
    Security: Only returns assessments from current clinician's patients
    """
    query = db.query(Assessment).filter(Assessment.clinician_id == current_user.id)
    
    # Optional filter by patient
    if patient_id:
        query = query.filter(Assessment.patient_id == patient_id)
    
    # Order by most recent first
    assessments = query.order_by(Assessment.timestamp.desc()).offset(skip).limit(limit).all()
    
    return assessments


@router.get("/{assessment_id}", response_model=AssessmentResponse)
def get_assessment(
    assessment_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed assessment results.
    
    Returns complete analysis including:
    - All stress/anxiety scores
    - Heart rate analysis
    - Text summaries and recommendations
    
    Use case: Detailed view when clinician clicks on assessment
    """
    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id,
        Assessment.clinician_id == current_user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    return assessment


@router.get("/patient/{patient_id}/trend")
def get_patient_trend(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 30
):
    """
    Get stress/anxiety trend for a patient.
    
    Returns:
    - Last N assessments
    - Ordered by date (oldest to newest for charting)
    - Only key metrics for visualization
    
    Frontend can use this to create:
    - Line charts showing stress over time
    - Trend arrows (improving vs worsening)
    - Alerts for high stress periods
    
    Example response:
    [
        {"date": "2026-01-01", "stress": 0.4, "anxiety": 0.3},
        {"date": "2026-01-05", "stress": 0.6, "anxiety": 0.5},
        {"date": "2026-01-09", "stress": 0.7, "anxiety": 0.6}
    ]
    """
    # Verify patient belongs to current clinician
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.clinician_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Get recent assessments
    assessments = db.query(Assessment).filter(
        Assessment.patient_id == patient_id
    ).order_by(Assessment.timestamp.asc()).limit(limit).all()
    
    # Format for frontend charting
    trend_data = [
        {
            "date": a.timestamp.strftime("%Y-%m-%d"),
            "timestamp": a.timestamp.isoformat(),
            "stress_score": a.stress_score,
            "anxiety_score": a.anxiety_score,
            "heart_rate_avg": a.heart_rate_avg
        }
        for a in assessments
    ]
    
    return {
        "patient_id": patient_id,
        "data_points": len(trend_data),
        "trend": trend_data
    }


# Route summary:
#
# POST /assessments/
# - Store AI analysis results (no raw data!)
# - Links to patient and clinician
# - Generates unique assessment ID
#
# GET /assessments/
# - List assessments with optional patient filter
# - Ordered by most recent
#
# GET /assessments/{id}
# - Get detailed assessment results
#
# GET /assessments/patient/{id}/trend
# - Get stress/anxiety trend over time
# - Perfect for charts and visualizations
