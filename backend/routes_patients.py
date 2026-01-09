"""
Patient management routes: CRUD operations for patients.

CRUD = Create, Read, Update, Delete
Why CRUD? Standard pattern for database operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime

from database import get_db
from models import User, Patient
from schemas import PatientCreate, PatientResponse
from routes_auth import get_current_user

router = APIRouter()


def generate_patient_id() -> str:
    """
    Generate anonymous patient ID for privacy.
    Format: PT-20260109-ABC123
    
    Why anonymous IDs?
    - Privacy: No patient names in database
    - HIPAA compliance: De-identified data
    - Security: Can't guess other patient IDs
    """
    date_part = datetime.now().strftime("%Y%m%d")
    random_part = str(uuid.uuid4())[:6].upper()
    return f"PT-{date_part}-{random_part}"


@router.post("/", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
def create_patient(
    patient: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new patient record.
    
    Why this endpoint?
    - Clinicians can register new patients
    - Automatically generates anonymous patient ID
    - Links patient to the clinician who created it
    
    Protected: Requires authentication (current_user dependency)
    
    Privacy features:
    - No patient names stored (only age range)
    - Anonymous ID generated
    - Only basic baseline metrics stored
    """
    # Generate unique patient ID
    patient_id = generate_patient_id()
    
    # Create patient record
    db_patient = Patient(
        patient_id=patient_id,
        age_range=patient.age_range,
        gender=patient.gender,
        baseline_heart_rate=patient.baseline_heart_rate,
        clinician_id=current_user.id  # Link to current clinician
    )
    
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    
    return db_patient


@router.get("/", response_model=List[PatientResponse])
def get_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    Get all patients for the current clinician.
    
    Why pagination (skip/limit)?
    - Performance: Don't load 10,000 patients at once
    - User experience: Load data in chunks
    
    Query params:
    - skip: Number of records to skip (for pagination)
    - limit: Max records to return (default 100)
    
    Example:
    - GET /patients?skip=0&limit=20  → First 20 patients
    - GET /patients?skip=20&limit=20 → Next 20 patients
    
    Security: Only returns patients belonging to current clinician
    """
    patients = db.query(Patient).filter(
        Patient.clinician_id == current_user.id
    ).offset(skip).limit(limit).all()
    
    return patients


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific patient by ID.
    
    Path parameter: /patients/123 → patient_id = 123
    
    Security check:
    - Verify patient belongs to current clinician
    - Prevents clinicians from accessing other clinicians' patients
    """
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.clinician_id == current_user.id  # Security: only own patients
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_patient(
    patient_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a patient record.
    
    Why 204 No Content?
    - HTTP standard for successful deletion
    - No response body needed
    
    Note: In production, consider "soft delete" (mark as inactive)
    instead of permanent deletion for data retention compliance.
    """
    patient = db.query(Patient).filter(
        Patient.id == patient_id,
        Patient.clinician_id == current_user.id
    ).first()
    
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found"
        )
    
    db.delete(patient)
    db.commit()
    
    return None


# Route summary:
#
# POST /patients/
# - Create new patient
# - Generates anonymous ID
# - Links to current clinician
#
# GET /patients/
# - List all patients for current clinician
# - Supports pagination
#
# GET /patients/{id}
# - Get specific patient details
# - Security: only own patients
#
# DELETE /patients/{id}
# - Delete patient record
# - Security: only own patients
