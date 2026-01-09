import uuid
from datetime import datetime


def generate_patient_id() -> str:
    """Generate unique patient ID."""
    timestamp = datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8].upper()
    return f"PT-{timestamp}-{unique_id}"


def generate_assessment_id() -> str:
    """Generate unique assessment ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:6].upper()
    return f"AS-{timestamp}-{unique_id}"
