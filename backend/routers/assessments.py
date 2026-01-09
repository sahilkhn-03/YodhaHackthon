from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime
from database import get_db
from models import Assessment, Patient, User, RealTimeData
from schemas import (
    AssessmentCreate, AssessmentUpdate, AssessmentResponse,
    AssessmentDetailResponse, RealTimeDataCreate, RealTimeDataResponse
)
from utils.dependencies import get_current_active_clinician
from utils.helpers import generate_assessment_id

router = APIRouter()


@router.post("/", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
async def create_assessment(
    assessment: AssessmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Create a new assessment session."""
    
    # Verify patient exists and belongs to clinician
    patient = db.query(Patient).filter(
        Patient.id == assessment.patient_id,
        Patient.clinician_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found or not authorized"
        )
    
    # Generate assessment ID
    assessment_id = generate_assessment_id()
    
    # Create assessment
    db_assessment = Assessment(
        assessment_id=assessment_id,
        patient_id=assessment.patient_id,
        clinician_id=current_user.id,
        assessment_type=assessment.assessment_type,
        status="in_progress",
        started_at=datetime.utcnow()
    )
    
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)
    
    return db_assessment


@router.get("/", response_model=List[AssessmentResponse])
async def get_assessments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    patient_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Get all assessments for the current clinician."""
    
    query = db.query(Assessment).filter(Assessment.clinician_id == current_user.id)
    
    if patient_id:
        query = query.filter(Assessment.patient_id == patient_id)
    
    if status_filter:
        query = query.filter(Assessment.status == status_filter)
    
    assessments = query.order_by(desc(Assessment.created_at)).offset(skip).limit(limit).all()
    return assessments


@router.get("/{assessment_id}", response_model=AssessmentDetailResponse)
async def get_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Get detailed information for a specific assessment."""
    
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


@router.put("/{assessment_id}", response_model=AssessmentDetailResponse)
async def update_assessment(
    assessment_id: int,
    assessment_update: AssessmentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Update assessment results and analysis."""
    
    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id,
        Assessment.clinician_id == current_user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    # Update fields
    update_data = assessment_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(assessment, field, value)
    
    # If status is completed, set completion time
    if assessment_update.status == "completed" and not assessment.completed_at:
        assessment.completed_at = datetime.utcnow()
        if assessment.started_at:
            duration = (assessment.completed_at - assessment.started_at).total_seconds()
            assessment.duration_seconds = int(duration)
    
    assessment.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(assessment)
    
    return assessment


@router.delete("/{assessment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_assessment(
    assessment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Delete an assessment record."""
    
    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id,
        Assessment.clinician_id == current_user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    db.delete(assessment)
    db.commit()
    return None


@router.post("/{assessment_id}/realtime", response_model=RealTimeDataResponse)
async def add_realtime_data(
    assessment_id: int,
    data: RealTimeDataCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Add real-time data point to an assessment."""
    
    # Verify assessment exists and belongs to clinician
    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id,
        Assessment.clinician_id == current_user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    # Create real-time data entry
    db_data = RealTimeData(
        assessment_id=assessment_id,
        data_type=data.data_type,
        features=data.features,
        stress_level=data.stress_level,
        confidence_score=data.confidence_score
    )
    
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    
    return db_data


@router.get("/{assessment_id}/realtime", response_model=List[RealTimeDataResponse])
async def get_realtime_data(
    assessment_id: int,
    data_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Get real-time data for an assessment."""
    
    # Verify assessment exists
    assessment = db.query(Assessment).filter(
        Assessment.id == assessment_id,
        Assessment.clinician_id == current_user.id
    ).first()
    
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    query = db.query(RealTimeData).filter(RealTimeData.assessment_id == assessment_id)
    
    if data_type:
        query = query.filter(RealTimeData.data_type == data_type)
    
    data = query.order_by(desc(RealTimeData.timestamp)).limit(limit).all()
    return data
