"""
Quick test script for the simplified emotion recognition system
"""

import sys
import subprocess

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT FOUND")
        return False

def test_system():
    """Test the emotion recognition system"""
    print("🔍 Checking Python and package installation...")
    print(f"Python version: {sys.version}")
    print()
    
    # Check required packages
    packages = ['cv2', 'mediapipe', 'numpy', 'sklearn']
    all_ok = True
    
    for package in packages:
        if not check_package(package):
            all_ok = False
    
    print()
    
    if all_ok:
        print("🎉 All packages are installed!")
        print()
        
        # Test the emotion recognition
        try:
            from simplified_emotion_recognition import SimplifiedFacialEmotionRecognizer
            recognizer = SimplifiedFacialEmotionRecognizer()
            print("✅ Emotion recognition system initialized successfully!")
            print()
            print("🚀 You can now run:")
            print("   python simplified_emotion_recognition.py")
            print()
            return True
        except Exception as e:
            print(f"❌ Error initializing emotion recognition: {e}")
            return False
    else:
        print("❌ Some packages are missing. Please install them with:")
        print("   pip install mediapipe opencv-python numpy scikit-learn matplotlib")
        return False

if __name__ == "__main__":
    test_system()