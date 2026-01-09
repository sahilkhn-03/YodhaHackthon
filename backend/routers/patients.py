from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from database import get_db
from models import Patient, User
from schemas import PatientCreate, PatientUpdate, PatientResponse
from utils.dependencies import get_current_active_clinician
from utils.helpers import generate_patient_id

router = APIRouter()


@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Create a new patient record."""
    
    # Generate unique patient ID
    patient_id = generate_patient_id()
    
    # Create patient
    db_patient = Patient(
        patient_id=patient_id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        date_of_birth=patient.date_of_birth,
        gender=patient.gender,
        email=patient.email,
        phone=patient.phone,
        medical_history=patient.medical_history,
        emergency_contact=patient.emergency_contact,
        clinician_id=current_user.id
    )
    
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    
    return db_patient


@router.get("/", response_model=List[PatientResponse])
async def get_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = None,
    is_active: Optional[bool] = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Get all patients for the current clinician."""
    
    query = db.query(Patient).filter(Patient.clinician_id == current_user.id)
    
    if is_active is not None:
        query = query.filter(Patient.is_active == is_active)
    
    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            (Patient.first_name.ilike(search_filter)) |
            (Patient.last_name.ilike(search_filter)) |
            (Patient.patient_id.ilike(search_filter))
        )
    
    patients = query.offset(skip).limit(limit).all()
    return patients


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Get a specific patient by ID."""
    
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.clinician_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int,
    patient_update: PatientUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Update patient information."""
    
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.clinician_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Update fields
    update_data = patient_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(patient, field, value)
    
    patient.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(patient)
    
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: int,
    permanent: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Delete or deactivate a patient record."""
    
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.clinician_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    if permanent:
        # Permanent deletion
        db.delete(patient)
    else:
        # Soft delete (deactivate)
        patient.is_active = False
        patient.updated_at = datetime.utcnow()
    
    db.commit()
    return None


@router.get("/{patient_id}/history")
async def get_patient_history(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_clinician)
):
    """Get complete patient history including assessments."""
    from models import Assessment, VitalSigns
    
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.clinician_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    # Get assessments
    assessments = db.query(Assessment).filter(
        Assessment.patient_id == patient_id
    ).order_by(Assessment.created_at.desc()).all()
    
    # Get vital signs
    vital_signs = db.query(VitalSigns).filter(
        VitalSigns.patient_id == patient_id
    ).order_by(VitalSigns.recorded_at.desc()).limit(10).all()
    
    return {
        "patient": patient,
        "total_assessments": len(assessments),
        "recent_assessments": assessments[:5],
        "recent_vital_signs": vital_signs
    }
