"""
Quick Test Script - Verify Backend Works

WHY: Test database connection and create sample data.
Run this after starting the server to verify everything works.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


def test_register():
    """Test user registration."""
    print("\n2. Testing user registration...")
    data = {
        "email": "doctor@test.com",
        "username": "doctor1",
        "password": "SecurePass123",
        "full_name": "Dr. Test",
        "role": "clinician"
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()


def test_login():
    """Test login and get token."""
    print("\n3. Testing login...")
    data = {
        "username": "doctor1",
        "password": "SecurePass123"
    }
    response = requests.post(f"{BASE_URL}/auth/login", data=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Token received: {result['access_token'][:50]}...")
    return result["access_token"]


def test_create_patient(token):
    """Test creating a patient."""
    print("\n4. Testing patient creation...")
    data = {
        "age_range": "25-30",
        "gender": "female",
        "baseline_heart_rate": 72,
        "notes": "Test patient"
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/patients/", json=data, headers=headers)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Patient ID: {result['patient_id']}")
    return result["id"]


def test_create_assessment(patient_id, token):
    """Test creating an assessment."""
    print("\n5. Testing assessment creation...")
    data = {
        "patient_id": patient_id,
        "duration_seconds": 180,
        "stress_score": 0.72,
        "anxiety_score": 0.65,
        "heart_rate_avg": 95,
        "heart_rate_max": 105,
        "heart_rate_baseline": 70,
        "analysis_summary": "Test assessment - elevated heart rate",
        "likely_reasons": "Stress response detected",
        "recommendations": "Follow-up recommended"
    }
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/assessments/", json=data, headers=headers)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Assessment ID: {result['assessment_id']}")


if __name__ == "__main__":
    print("="*60)
    print("NeuroBalance AI Backend - Quick Test")
    print("="*60)
    print("\nMake sure the server is running: python main.py")
    print("Press Enter to start tests...")
    input()
    
    try:
        test_health()
        user = test_register()
        token = test_login()
        patient_id = test_create_patient(token)
        test_create_assessment(patient_id, token)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now:")
        print("1. Visit http://localhost:8000/docs for interactive API docs")
        print("2. Test endpoints with Postman or curl")
        print("3. Connect your frontend application")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nTroubleshooting:")
        print("1. Is the server running? (python main.py)")
        print("2. Is Supabase/PostgreSQL configured? (check .env)")
        print("3. Are dependencies installed? (pip install -r requirements.txt)")
