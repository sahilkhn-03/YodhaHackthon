"""
Test script for baseline/normal heart rate mapping.
Demonstrates how to set and retrieve normal resting heart rate.
"""

import requests
import time

API_BASE = "http://localhost:8001"

print("="*70)
print("❤️  TESTING NORMAL/BASELINE HEART RATE MAPPING")
print("="*70)

# Start simulation
print("\n1. Starting simulation with default baseline...")
response = requests.post(f"{API_BASE}/simulation/start")
print(f"   ✅ {response.json()['message']}")
time.sleep(2)

# Get current baseline
print("\n2. Getting current baseline heart rate...")
response = requests.get(f"{API_BASE}/heartbeat/baseline")
data = response.json()
print(f"   Baseline BPM: {data['baseline_bpm']}")
print(f"   Current BPM: {data['current_bpm']}")
print(f"   Profile: {data['profile']}")
print(f"   Stress Level: {data['current_stress_level']}")

time.sleep(2)

# Test different person profiles
profiles = [
    {"name": "Elite Athlete", "bpm": 48, "reason": "profile"},
    {"name": "Fitness Enthusiast", "bpm": 62, "reason": "profile"},
    {"name": "Average Person", "bpm": 75, "reason": "profile"},
    {"name": "Sedentary Individual", "bpm": 85, "reason": "profile"},
    {"name": "Anxious Person", "bpm": 95, "reason": "ml_model"}
]

for profile in profiles:
    print(f"\n3. Setting baseline for: {profile['name']} ({profile['bpm']} BPM)")
    
    # Set baseline
    response = requests.post(
        f"{API_BASE}/input/set-normal-heartrate",
        params={"baseline_bpm": profile['bpm'], "reason": profile['reason']}
    )
    result = response.json()
    
    print(f"   ✅ {result['message']}")
    print(f"   Old baseline: {result['old_baseline_bpm']} BPM")
    print(f"   New baseline: {result['new_baseline_bpm']} BPM")
    print(f"   Profile: {result['profile']}")
    print(f"   Reason: {result['reason']}")
    
    time.sleep(1)
    
    # Check current heart rate
    response = requests.get(f"{API_BASE}/heartbeat/current")
    data = response.json()
    print(f"   Current BPM: {data['bpm']} (should be near {profile['bpm']})")
    print(f"   Blood Pressure: {data['systolic']}/{data['diastolic']}")
    
    time.sleep(2)

# Test with stress
print("\n4. Testing stress response with different baselines...")
print("\n   Setting baseline to 65 BPM (athlete)...")
requests.post(f"{API_BASE}/input/set-normal-heartrate", params={"baseline_bpm": 65})
time.sleep(1)

print("   Triggering stress via video input...")
requests.post(f"{API_BASE}/input/video-stress", params={"stress_detected": True, "intensity": 0.8})
time.sleep(2)

response = requests.get(f"{API_BASE}/heartbeat/current")
data = response.json()
print(f"   Athlete under stress: {data['bpm']} BPM (baseline 65 + stress)")

time.sleep(2)

print("\n   Setting baseline to 90 BPM (anxious person)...")
requests.post(f"{API_BASE}/input/set-normal-heartrate", params={"baseline_bpm": 90})
time.sleep(1)

print("   Triggering same stress level...")
requests.post(f"{API_BASE}/input/video-stress", params={"stress_detected": True, "intensity": 0.8})
time.sleep(2)

response = requests.get(f"{API_BASE}/heartbeat/current")
data = response.json()
print(f"   Anxious person under stress: {data['bpm']} BPM (baseline 90 + stress)")

# Final state
print("\n5. Final baseline check...")
response = requests.get(f"{API_BASE}/heartbeat/baseline")
data = response.json()
print(f"   Current baseline: {data['baseline_bpm']} BPM")
print(f"   Current heart rate: {data['current_bpm']} BPM")
print(f"   Profile: {data['profile']}")

print("\n" + "="*70)
print("✅ BASELINE HEART RATE MAPPING TEST COMPLETE")
print("="*70)

print("\n📊 Use Cases:")
print("   1. User Profile: Set baseline based on age, fitness, health")
print("   2. Calibration: Set from actual heart rate measurement")
print("   3. ML Model: Estimate baseline from historical data")
print("   4. Simulation: Test different person types (athlete vs sedentary)")
print("\n💡 Stress increases are RELATIVE to baseline:")
print("   - Athlete (65 BPM) + stress → ~100-110 BPM")
print("   - Anxious (90 BPM) + stress → ~125-135 BPM")
