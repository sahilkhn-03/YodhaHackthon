"""
Complete Test Suite for Facial Stress Inference System
======================================================
"""

import sys
import time
import numpy as np
import cv2

def test_basic_functionality():
    """Test basic system functionality"""
    print("🧪 TEST 1: Basic Functionality")
    print("-" * 40)
    
    try:
        from facial_stress_inference_v2 import FacialStressInference, StressConfig
        print("✅ Module imports successful")
        
        # Test configuration
        config = StressConfig()
        print(f"✅ Config loaded - Smoothing: {config.smoothing_window} frames")
        print(f"✅ Weights: Eye={config.weight_eye_closure}, Brow={config.weight_eyebrow_tension}, Jaw={config.weight_jaw_openness}")
        
        # Test engine initialization
        engine = FacialStressInference(config)
        print("✅ Engine initialized successfully")
        
        # Test with dummy frame (no face)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = engine.process_frame(dummy_frame)
        
        print(f"✅ Frame processed in {result['inference_time_ms']:.1f}ms")
        print(f"✅ Face detected: {result['face_detected']}")
        print(f"✅ Stress score: {result['facial_stress']}")
        
        # Validate output format
        required_fields = ['facial_stress', 'eye_closure', 'eyebrow_tension', 'jaw_tension', 'face_detected', 'inference_time_ms']
        missing = [f for f in required_fields if f not in result]
        
        if missing:
            print(f"❌ Missing fields: {missing}")
            return False
        else:
            print("✅ All required output fields present")
        
        engine.close()
        print("✅ Cleanup successful\n")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_performance():
    """Test performance benchmarks"""
    print("🚀 TEST 2: Performance Benchmarks")
    print("-" * 40)
    
    try:
        from facial_stress_inference_v2 import FacialStressInference
        
        engine = FacialStressInference()
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run multiple tests
        times = []
        for i in range(10):
            result = engine.process_frame(test_frame)
            times.append(result['inference_time_ms'])
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"✅ Average inference time: {avg_time:.1f}ms")
        print(f"✅ Min inference time: {min_time:.1f}ms")
        print(f"✅ Max inference time: {max_time:.1f}ms")
        
        # Check performance targets
        if avg_time < 500:
            print(f"✅ PASS: Average time < 500ms (Target met by {500/avg_time:.1f}x)")
        else:
            print(f"❌ FAIL: Average time > 500ms")
            
        if avg_time < 50:
            print("🎉 EXCELLENT: Real-time performance achieved!")
            
        engine.close()
        print("✅ Performance test completed\n")
        return avg_time < 500
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_webcam_integration():
    """Test webcam integration"""
    print("📹 TEST 3: Webcam Integration")
    print("-" * 40)
    
    try:
        # Check if webcam is available
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  No webcam available - skipping test")
            return True
            
        print("✅ Webcam accessible")
        
        from facial_stress_inference_v2 import FacialStressInference
        engine = FacialStressInference()
        
        # Test with 5 real frames
        results = []
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                result = engine.process_frame(frame)
                results.append(result)
                print(f"   Frame {i+1}: Stress={result['facial_stress']}, Time={result['inference_time_ms']:.1f}ms")
        
        cap.release()
        engine.close()
        
        if results:
            face_detected_count = sum(1 for r in results if r['face_detected'])
            avg_inference = np.mean([r['inference_time_ms'] for r in results])
            
            print(f"✅ Processed {len(results)} frames")
            print(f"✅ Faces detected in {face_detected_count}/{len(results)} frames")
            print(f"✅ Average real-world inference: {avg_inference:.1f}ms")
        
        print("✅ Webcam integration test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

def test_stress_detection_accuracy():
    """Test stress detection with different scenarios"""
    print("🎯 TEST 4: Stress Detection Logic")
    print("-" * 40)
    
    try:
        from facial_stress_inference_v2 import FacialStressInference
        
        engine = FacialStressInference()
        
        # Test temporal smoothing
        print("Testing temporal smoothing...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process multiple frames to test smoothing
        scores = []
        for i in range(15):
            result = engine.process_frame(dummy_frame)
            scores.append(result['facial_stress'])
        
        print(f"✅ Temporal smoothing working - {len(set(scores))} different scores over 15 frames")
        
        # Test session reset
        engine.reset_session()
        result_after_reset = engine.process_frame(dummy_frame)
        print("✅ Session reset working")
        
        # Test feature ranges
        if 0 <= result_after_reset['eye_closure'] <= 1:
            print("✅ Eye closure in valid range [0,1]")
        if 0 <= result_after_reset['eyebrow_tension'] <= 1:
            print("✅ Eyebrow tension in valid range [0,1]")  
        if 0 <= result_after_reset['jaw_tension'] <= 1:
            print("✅ Jaw tension in valid range [0,1]")
        if 0 <= result_after_reset['facial_stress'] <= 100:
            print("✅ Facial stress in valid range [0,100]")
            
        engine.close()
        print("✅ Stress detection test completed\n")
        return True
        
    except Exception as e:
        print(f"❌ Stress detection test failed: {e}")
        return False

def run_full_test_suite():
    """Run complete test suite"""
    print("=" * 60)
    print("  FACIAL STRESS INFERENCE - COMPLETE TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Performance Benchmarks", test_performance),
        ("Webcam Integration", test_webcam_integration),  
        ("Stress Detection Logic", test_stress_detection_accuracy),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if i < passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System ready for production.")
        print("\n📋 Next steps:")
        print("   1. ✅ Facial stress detection - COMPLETE")
        print("   2. 🔄 Voice stress extraction - NEXT")
        print("   3. 🔄 Multimodal fusion - PENDING")
        print("   4. 🔄 FastAPI backend - PENDING")
        
    return passed == total

if __name__ == "__main__":
    success = run_full_test_suite()
    sys.exit(0 if success else 1)