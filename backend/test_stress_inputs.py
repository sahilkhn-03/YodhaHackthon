"""
Test script for external stress input endpoints.
Demonstrates how video/audio analysis ML models can trigger stress.
"""

import requests
import time

API_BASE = "http://localhost:8001"

print("="*60)
print("🧪 TESTING EXTERNAL STRESS INPUT ENDPOINTS")
print("="*60)

# Start simulation first
print("\n1. Starting simulation...")
response = requests.post(f"{API_BASE}/simulation/start")
print(f"✅ {response.json()['message']}")
time.sleep(2)

# Get baseline
print("\n2. Getting baseline heartbeat...")
response = requests.get(f"{API_BASE}/heartbeat/current")
data = response.json()
print(f"   BPM: {data['bpm']}, BP: {data['systolic']}/{data['diastolic']}, Stress: {data['stress_level']}")

time.sleep(3)

# Test video stress input
print("\n3. Testing VIDEO STRESS INPUT (simulating facial stress detection)...")
response = requests.post(
    f"{API_BASE}/input/video-stress",
    params={"stress_detected": True, "intensity": 0.75}
)
result = response.json()
print(f"   ✅ {result['message']}")
print(f"   Applied stress level: {result['current_stress_level']}")
print(f"   Current BPM: {result['current_bpm']}")

time.sleep(3)

# Check increased heart rate
print("\n4. Checking heart rate increase...")
response = requests.get(f"{API_BASE}/heartbeat/current")
data = response.json()
print(f"   BPM: {data['bpm']} (should be elevated)")
print(f"   BP: {data['systolic']}/{data['diastolic']} (should be elevated)")
print(f"   Stress: {data['stress_level']}")

time.sleep(3)

# Test audio stress input
print("\n5. Testing AUDIO STRESS INPUT (simulating voice stress detection)...")
response = requests.post(
    f"{API_BASE}/input/audio-stress",
    params={"stress_detected": True, "intensity": 0.6}
)
result = response.json()
print(f"   ✅ {result['message']}")
print(f"   Applied stress level: {result['current_stress_level']}")
print(f"   Current BPM: {result['current_bpm']}")

time.sleep(3)

# Test combined input
print("\n6. Testing COMBINED INPUT (video + audio)...")
response = requests.post(
    f"{API_BASE}/input/combined-stress",
    params={
        "video_stress": True,
        "audio_stress": True,
        "video_intensity": 0.8,
        "audio_intensity": 0.7
    }
)
result = response.json()
print(f"   ✅ {result['message']}")
print(f"   Video stress: {result['video_stress']}")
print(f"   Audio stress: {result['audio_stress']}")
print(f"   Applied intensity: {result['applied_intensity']} (max of both)")
print(f"   Current stress level: {result['current_stress_level']}")
print(f"   Current BPM: {result['current_bpm']}")

time.sleep(3)

# Final check
print("\n7. Final heartbeat check...")
response = requests.get(f"{API_BASE}/heartbeat/current")
data = response.json()
print(f"   BPM: {data['bpm']}")
print(f"   BP: {data['systolic']}/{data['diastolic']}")
print(f"   Stress: {data['stress_level']}")

print("\n" + "="*60)
print("✅ TEST COMPLETE")
print("="*60)
print("\nNOTE: Stress will gradually decay back to baseline over time.")
print("Random stress events also occur automatically (5-8% chance per update).")
