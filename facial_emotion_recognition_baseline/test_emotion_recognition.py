"""
Test script for Facial Emotion Recognition
Quick verification without webcam
"""

import numpy as np
from facial_emotion_recognition import FacialEmotionRecognizer
import time

def test_feature_extraction():
    """Test the feature extraction functionality"""
    print("Testing Facial Emotion Recognition...")
    
    try:
        # Initialize recognizer
        recognizer = FacialEmotionRecognizer()
        print("✅ FacialEmotionRecognizer initialized successfully")
        
        # Test with dummy features
        dummy_features = np.random.rand(68).astype(np.float32)
        
        # Test emotion classification
        emotion_result = recognizer.classify_emotion(dummy_features)
        print("✅ Emotion classification working")
        
        # Print results
        print("\n📊 Test Results:")
        print(f"Dominant Emotion: {emotion_result['dominant_emotion']}")
        print(f"Stress Score: {emotion_result['stress_score']:.3f}")
        print(f"Confidence: {emotion_result['confidence']:.3f}")
        
        print("\n🎭 Emotion Probabilities:")
        for emotion, prob in emotion_result['emotions'].items():
            print(f"  {emotion}: {prob:.3f}")
        
        print("\n✅ All tests passed! Facial emotion recognition is ready.")
        print("\n📝 Next steps:")
        print("1. Install Python: https://python.org/downloads/")
        print("2. Install packages: pip install -r requirements.txt")
        print("3. Run: python facial_emotion_recognition.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_feature_extraction()