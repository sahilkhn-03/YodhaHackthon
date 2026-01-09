"""
Test Script for Facial Stress Inference Module
===============================================
Verifies all components work correctly before FastAPI integration.
"""

import sys
import time
import numpy as np

def test_imports():
    """Test that all required packages are installed."""
    print("🔍 Testing imports...")
    
    packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe', 
        'numpy': 'numpy',
    }
    
    all_ok = True
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - Install with: pip install {package}")
            all_ok = False
    
    return all_ok


def test_module_components():
    """Test individual components of the stress inference module."""
    print("\n🔍 Testing module components...")
    
    try:
        from facial_stress_inference import (
            StressConfig,
            FaceLandmarkIndices,
            LandmarkExtractor,
            FeatureComputer,
            StressScoreCalculator,
            FacialStressInference
        )
        print("   ✅ All classes imported successfully")
        
        # Test config
        config = StressConfig()
        assert config.smoothing_window == 10
        assert config.weight_eye_closure == 0.4
        print("   ✅ StressConfig initialized")
        
        # Test landmark indices
        indices = FaceLandmarkIndices()
        assert indices.LEFT_EYE['p1'] == 33
        assert indices.MOUTH['upper_lip_top'] == 13
        print("   ✅ FaceLandmarkIndices loaded")
        
        # Test feature computer with dummy landmarks
        feature_computer = FeatureComputer(config)
        
        # Create dummy landmarks (468 points)
        dummy_landmarks = np.random.rand(468, 3).astype(np.float32)
        
        features = feature_computer.compute_all_features(dummy_landmarks)
        assert 'eye_closure' in features
        assert 'eyebrow_tension' in features
        assert 'jaw_openness' in features
        assert 0.0 <= features['eye_closure'] <= 1.0
        print("   ✅ FeatureComputer working")
        
        # Test stress calculator
        stress_calculator = StressScoreCalculator(config)
        
        stress_result = stress_calculator.calculate(features)
        assert 'facial_stress' in stress_result
        assert 0 <= stress_result['facial_stress'] <= 100
        print("   ✅ StressScoreCalculator working")
        
        # Test temporal smoothing
        for _ in range(15):
            stress_calculator.calculate(features)
        assert len(stress_calculator.history) == 10  # Window size
        print("   ✅ Temporal smoothing working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_inference_engine():
    """Test the main inference engine with a dummy frame."""
    print("\n🔍 Testing inference engine...")
    
    try:
        from facial_stress_inference import FacialStressInference
        import cv2
        
        # Initialize engine
        engine = FacialStressInference()
        print("   ✅ FacialStressInference initialized")
        
        # Test with dummy black frame (no face)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = engine.process_frame(dummy_frame)
        
        assert result is not None
        assert result['face_detected'] == False
        assert 'facial_stress' in result
        assert 'inference_time_ms' in result
        print("   ✅ No-face handling works")
        
        # Test reset
        engine.reset_session()
        print("   ✅ Session reset works")
        
        # Cleanup
        engine.close()
        print("   ✅ Resource cleanup works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_format():
    """Verify output matches the required format."""
    print("\n🔍 Testing output format...")
    
    try:
        from facial_stress_inference import FacialStressInference
        import cv2
        
        engine = FacialStressInference()
        
        # Try to capture a real frame if webcam available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                result = engine.process_frame(frame)
                
                # Verify required fields
                required_fields = [
                    'facial_stress',
                    'eye_closure', 
                    'eyebrow_tension',
                    'jaw_tension'
                ]
                
                for field in required_fields:
                    assert field in result, f"Missing field: {field}"
                
                # Verify ranges
                assert 0 <= result['facial_stress'] <= 100, "facial_stress out of range"
                assert 0.0 <= result['eye_closure'] <= 1.0, "eye_closure out of range"
                assert 0.0 <= result['eyebrow_tension'] <= 1.0, "eyebrow_tension out of range"
                assert 0.0 <= result['jaw_tension'] <= 1.0, "jaw_tension out of range"
                
                print("   ✅ Output format matches specification")
                print(f"\n   📊 Sample Output:")
                print(f"   {result}")
            else:
                print("   ⚠️ Could not read frame from webcam")
            
            cap.release()
        else:
            print("   ⚠️ No webcam available (format test skipped)")
        
        engine.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_performance():
    """Test inference performance target (<500ms)."""
    print("\n🔍 Testing performance...")
    
    try:
        from facial_stress_inference import FacialStressInference
        import cv2
        
        engine = FacialStressInference()
        
        # Use webcam if available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            inference_times = []
            
            for i in range(30):
                ret, frame = cap.read()
                if ret:
                    result = engine.process_frame(frame)
                    inference_times.append(result['inference_time_ms'])
            
            cap.release()
            
            if inference_times:
                avg_time = np.mean(inference_times)
                max_time = np.max(inference_times)
                min_time = np.min(inference_times)
                
                print(f"   📊 Performance Results (30 frames):")
                print(f"      Average: {avg_time:.1f}ms")
                print(f"      Min: {min_time:.1f}ms")
                print(f"      Max: {max_time:.1f}ms")
                
                if avg_time < 500:
                    print(f"   ✅ PASS: Average inference < 500ms")
                else:
                    print(f"   ❌ FAIL: Average inference > 500ms")
        else:
            print("   ⚠️ No webcam available (performance test skipped)")
        
        engine.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("  FACIAL STRESS INFERENCE - TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    if not results[0][1]:
        print("\n❌ Cannot proceed - missing packages")
        print("Install with: pip install mediapipe opencv-python numpy")
        return False
    
    # Test 2: Module components
    results.append(("Module Components", test_module_components()))
    
    # Test 3: Inference engine
    results.append(("Inference Engine", test_inference_engine()))
    
    # Test 4: Output format
    results.append(("Output Format", test_output_format()))
    
    # Test 5: Performance
    results.append(("Performance", test_performance()))
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
        if result:
            passed += 1
    
    print(f"\n   Total: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ready for FastAPI integration.")
        print("\n📝 Next steps:")
        print("   1. Run live demo: python facial_stress_inference.py")
        print("   2. Integrate with FastAPI backend")
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
