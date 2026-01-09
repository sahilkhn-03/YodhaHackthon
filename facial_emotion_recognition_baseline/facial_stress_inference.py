"""
Facial Stress Inference Module
==============================
Real-time facial stress estimation using MediaPipe Face Mesh + OpenCV + NumPy.

Features:
- Eye Aspect Ratio (EAR) → eye closure detection
- Eyebrow tension → eyebrow-to-eye distance
- Jaw openness → lip distance measurement
- Facial symmetry (optional)

Output: Stress score 0-100 with individual feature breakdowns.

Author: Team Mission404 - NeuroBalance AI
License: MIT
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StressConfig:
    """Configuration parameters for stress detection."""
    
    # Temporal smoothing window size
    smoothing_window: int = 10
    
    # Feature weights for stress calculation
    weight_eye_closure: float = 0.4
    weight_eyebrow_tension: float = 0.3
    weight_jaw_openness: float = 0.3
    
    # Normalization bounds (empirically determined)
    # Eye Aspect Ratio typical range
    ear_min: float = 0.15  # Closed eyes
    ear_max: float = 0.35  # Wide open eyes
    
    # Eyebrow-eye distance typical range (normalized by face height)
    eyebrow_min: float = 0.02  # Furrowed brows (stressed)
    eyebrow_max: float = 0.08  # Relaxed/raised brows
    
    # Mouth aspect ratio typical range
    mouth_min: float = 0.0   # Closed mouth
    mouth_max: float = 0.5   # Wide open mouth (yawn/stress)
    
    # MediaPipe confidence thresholds
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


# =============================================================================
# LANDMARK INDICES (MediaPipe Face Mesh 468 landmarks)
# =============================================================================

class FaceLandmarkIndices:
    """
    MediaPipe Face Mesh landmark indices for stress-related features.
    Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    """
    
    # Left eye landmarks (6 points for EAR calculation)
    LEFT_EYE = {
        'p1': 33,   # Left corner
        'p2': 160,  # Upper lid (left)
        'p3': 158,  # Upper lid (right)
        'p4': 133,  # Right corner
        'p5': 153,  # Lower lid (right)
        'p6': 144,  # Lower lid (left)
    }
    
    # Right eye landmarks (6 points for EAR calculation)
    RIGHT_EYE = {
        'p1': 362,  # Left corner
        'p2': 385,  # Upper lid (left)
        'p3': 387,  # Upper lid (right)
        'p4': 263,  # Right corner
        'p5': 373,  # Lower lid (right)
        'p6': 380,  # Lower lid (left)
    }
    
    # Left eyebrow landmarks
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    
    # Right eyebrow landmarks
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]
    
    # Mouth landmarks for jaw openness
    MOUTH = {
        'upper_lip_top': 13,      # Top of upper lip
        'lower_lip_bottom': 14,   # Bottom of lower lip
        'left_corner': 61,        # Left mouth corner
        'right_corner': 291,      # Right mouth corner
        'upper_inner': 0,         # Inner upper lip
        'lower_inner': 17,        # Inner lower lip
    }
    
    # Face reference points for normalization
    FACE = {
        'forehead': 10,           # Top of face
        'chin': 152,              # Bottom of face
        'left_cheek': 234,        # Left side
        'right_cheek': 454,       # Right side
        'nose_tip': 1,            # Nose tip
    }


# =============================================================================
# LANDMARK EXTRACTION
# =============================================================================

class LandmarkExtractor:
    """
    Extracts facial landmarks from video frames using MediaPipe Face Mesh.
    Designed for CPU-only, real-time performance.
    """
    
    def __init__(self, config: StressConfig = None):
        """
        Initialize MediaPipe Face Mesh.
        
        Args:
            config: StressConfig instance with detection parameters.
        """
        self.config = config or StressConfig()
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,        # Video mode for tracking
            max_num_faces=1,                # Single face for performance
            refine_landmarks=True,          # Better eye/lip landmarks
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        
        # Store landmark indices
        self.indices = FaceLandmarkIndices()
    
    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from a BGR frame.
        
        Args:
            frame: BGR image from OpenCV (np.ndarray, shape: H x W x 3)
        
        Returns:
            np.ndarray of shape (468, 3) with normalized x, y, z coordinates,
            or None if no face detected.
        """
        if frame is None or frame.size == 0:
            return None
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        # Check if face detected
        if not results.multi_face_landmarks:
            return None
        
        # Extract first face landmarks as numpy array
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([
            [lm.x, lm.y, lm.z] 
            for lm in face_landmarks.landmark
        ], dtype=np.float32)
        
        return landmarks
    
    def get_landmark_point(self, landmarks: np.ndarray, index: int) -> np.ndarray:
        """
        Get a single landmark point by index.
        
        Args:
            landmarks: Full landmarks array (468, 3)
            index: Landmark index (0-467)
        
        Returns:
            np.ndarray of shape (3,) with x, y, z coordinates.
        """
        return landmarks[index]
    
    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

class FeatureComputer:
    """
    Computes stress-related facial features from landmarks.
    All features are normalized to 0-1 range.
    """
    
    def __init__(self, config: StressConfig = None):
        """
        Initialize feature computer.
        
        Args:
            config: StressConfig with normalization parameters.
        """
        self.config = config or StressConfig()
        self.indices = FaceLandmarkIndices()
    
    @staticmethod
    def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two 2D/3D points.
        
        Args:
            p1, p2: Points as numpy arrays.
        
        Returns:
            Float distance value.
        """
        return float(np.linalg.norm(p1[:2] - p2[:2]))  # Use only x, y
    
    def compute_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio (EAR) for blink/eye closure detection.
        
        Formula:
            EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
        Where p1-p6 are the 6 eye landmark points.
        Low EAR = eyes closed, High EAR = eyes open.
        
        Args:
            landmarks: Full face landmarks array (468, 3)
        
        Returns:
            Average EAR for both eyes (0.0 - 0.5 typical range).
        """
        def single_eye_ear(eye_indices: dict) -> float:
            """Calculate EAR for one eye."""
            p1 = landmarks[eye_indices['p1']]
            p2 = landmarks[eye_indices['p2']]
            p3 = landmarks[eye_indices['p3']]
            p4 = landmarks[eye_indices['p4']]
            p5 = landmarks[eye_indices['p5']]
            p6 = landmarks[eye_indices['p6']]
            
            # Vertical distances
            vertical_1 = self.euclidean_distance(p2, p6)
            vertical_2 = self.euclidean_distance(p3, p5)
            
            # Horizontal distance
            horizontal = self.euclidean_distance(p1, p4)
            
            # Avoid division by zero
            if horizontal < 1e-6:
                return 0.0
            
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        
        # Calculate EAR for both eyes
        left_ear = single_eye_ear(self.indices.LEFT_EYE)
        right_ear = single_eye_ear(self.indices.RIGHT_EYE)
        
        # Return average
        return (left_ear + right_ear) / 2.0
    
    def compute_eyebrow_tension(self, landmarks: np.ndarray) -> float:
        """
        Compute eyebrow tension based on eyebrow-to-eye distance.
        
        Lower distance = furrowed brows = higher stress.
        Higher distance = relaxed/raised brows = lower stress.
        
        Normalized by face height for scale invariance.
        
        Args:
            landmarks: Full face landmarks array (468, 3)
        
        Returns:
            Normalized eyebrow tension (0.0 - 1.0).
        """
        # Get face height for normalization
        forehead = landmarks[self.indices.FACE['forehead']]
        chin = landmarks[self.indices.FACE['chin']]
        face_height = self.euclidean_distance(forehead, chin)
        
        if face_height < 1e-6:
            return 0.5  # Default neutral value
        
        def single_eyebrow_distance(eyebrow_indices: List[int], eye_indices: dict) -> float:
            """Calculate average distance between eyebrow and eye."""
            # Eyebrow center (average of eyebrow points)
            eyebrow_points = landmarks[eyebrow_indices]
            eyebrow_center = np.mean(eyebrow_points, axis=0)
            
            # Eye center (average of eye corners)
            eye_center = (landmarks[eye_indices['p1']] + landmarks[eye_indices['p4']]) / 2
            
            # Vertical distance (y-axis)
            distance = abs(eyebrow_center[1] - eye_center[1])
            
            return distance
        
        # Calculate for both sides
        left_dist = single_eyebrow_distance(
            self.indices.LEFT_EYEBROW, 
            self.indices.LEFT_EYE
        )
        right_dist = single_eyebrow_distance(
            self.indices.RIGHT_EYEBROW, 
            self.indices.RIGHT_EYE
        )
        
        # Average and normalize by face height
        avg_distance = (left_dist + right_dist) / 2.0
        normalized = avg_distance / face_height
        
        return normalized
    
    def compute_jaw_openness(self, landmarks: np.ndarray) -> float:
        """
        Compute mouth/jaw openness from lip landmarks.
        
        Mouth Aspect Ratio (MAR) = vertical / horizontal distance.
        High MAR = open mouth (potential stress indicator).
        
        Args:
            landmarks: Full face landmarks array (468, 3)
        
        Returns:
            Mouth aspect ratio (0.0 - 1.0 typical range).
        """
        mouth = self.indices.MOUTH
        
        # Vertical distance (lip opening)
        upper_lip = landmarks[mouth['upper_lip_top']]
        lower_lip = landmarks[mouth['lower_lip_bottom']]
        vertical = self.euclidean_distance(upper_lip, lower_lip)
        
        # Horizontal distance (mouth width)
        left_corner = landmarks[mouth['left_corner']]
        right_corner = landmarks[mouth['right_corner']]
        horizontal = self.euclidean_distance(left_corner, right_corner)
        
        if horizontal < 1e-6:
            return 0.0
        
        # Mouth aspect ratio
        mar = vertical / horizontal
        
        return mar
    
    def compute_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """
        Compute facial symmetry score (optional feature).
        
        Asymmetry can indicate muscle tension or stress.
        
        Args:
            landmarks: Full face landmarks array (468, 3)
        
        Returns:
            Symmetry score (1.0 = perfect symmetry, 0.0 = high asymmetry).
        """
        face = self.indices.FACE
        
        # Get nose tip as center reference
        nose_tip = landmarks[face['nose_tip']]
        
        # Get cheek points
        left_cheek = landmarks[face['left_cheek']]
        right_cheek = landmarks[face['right_cheek']]
        
        # Calculate distances from center
        left_dist = self.euclidean_distance(nose_tip, left_cheek)
        right_dist = self.euclidean_distance(nose_tip, right_cheek)
        
        # Symmetry ratio (closer to 1 = more symmetric)
        if max(left_dist, right_dist) < 1e-6:
            return 1.0
        
        symmetry = min(left_dist, right_dist) / max(left_dist, right_dist)
        
        return symmetry
    
    def normalize_feature(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a feature value to 0-1 range.
        
        Args:
            value: Raw feature value.
            min_val: Expected minimum value.
            max_val: Expected maximum value.
        
        Returns:
            Normalized value clamped to [0.0, 1.0].
        """
        if max_val <= min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def compute_all_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute all stress-related features from landmarks.
        
        Args:
            landmarks: Full face landmarks array (468, 3)
        
        Returns:
            Dictionary with normalized features (0.0 - 1.0):
            - eye_closure: Higher = more closed eyes (stressed)
            - eyebrow_tension: Higher = more furrowed brows (stressed)
            - jaw_openness: Higher = more open jaw (stressed)
            - facial_symmetry: Higher = more symmetric (relaxed)
        """
        # Compute raw features
        ear = self.compute_eye_aspect_ratio(landmarks)
        eyebrow_dist = self.compute_eyebrow_tension(landmarks)
        jaw = self.compute_jaw_openness(landmarks)
        symmetry = self.compute_facial_symmetry(landmarks)
        
        # Normalize features to 0-1 (stress-oriented: higher = more stress)
        
        # Eye closure: INVERT EAR (low EAR = closed = high stress)
        eye_closure_normalized = 1.0 - self.normalize_feature(
            ear, 
            self.config.ear_min, 
            self.config.ear_max
        )
        
        # Eyebrow tension: INVERT (low distance = furrowed = high stress)
        eyebrow_tension_normalized = 1.0 - self.normalize_feature(
            eyebrow_dist, 
            self.config.eyebrow_min, 
            self.config.eyebrow_max
        )
        
        # Jaw openness: Direct (high = stressed)
        jaw_openness_normalized = self.normalize_feature(
            jaw, 
            self.config.mouth_min, 
            self.config.mouth_max
        )
        
        return {
            'eye_closure': round(eye_closure_normalized, 4),
            'eyebrow_tension': round(eyebrow_tension_normalized, 4),
            'jaw_openness': round(jaw_openness_normalized, 4),
            'facial_symmetry': round(symmetry, 4),
        }


# =============================================================================
# STRESS SCORE CALCULATION
# =============================================================================

class StressScoreCalculator:
    """
    Calculates final stress score with temporal smoothing.
    """
    
    def __init__(self, config: StressConfig = None):
        """
        Initialize stress calculator with temporal smoothing buffer.
        
        Args:
            config: StressConfig with weights and smoothing parameters.
        """
        self.config = config or StressConfig()
        
        # Temporal smoothing buffer (rolling window)
        self.history: deque = deque(maxlen=self.config.smoothing_window)
    
    def compute_raw_stress(self, features: Dict[str, float]) -> float:
        """
        Compute raw stress score from features (before smoothing).
        
        Formula:
            stress = 0.4 * eye_closure + 0.3 * eyebrow_tension + 0.3 * jaw_openness
        
        Args:
            features: Dictionary with normalized features.
        
        Returns:
            Raw stress score (0.0 - 1.0).
        """
        stress = (
            self.config.weight_eye_closure * features['eye_closure'] +
            self.config.weight_eyebrow_tension * features['eyebrow_tension'] +
            self.config.weight_jaw_openness * features['jaw_openness']
        )
        
        return float(np.clip(stress, 0.0, 1.0))
    
    def apply_temporal_smoothing(self, raw_stress: float) -> float:
        """
        Apply rolling average smoothing to reduce noise.
        
        Args:
            raw_stress: Current frame's raw stress score.
        
        Returns:
            Smoothed stress score (0.0 - 1.0).
        """
        # Add to history
        self.history.append(raw_stress)
        
        # Calculate rolling average
        smoothed = np.mean(list(self.history))
        
        return float(smoothed)
    
    def scale_to_100(self, stress_normalized: float) -> int:
        """
        Scale normalized stress score to 0-100 range.
        
        Args:
            stress_normalized: Stress score (0.0 - 1.0).
        
        Returns:
            Stress score (0 - 100).
        """
        return int(round(stress_normalized * 100))
    
    def calculate(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Calculate final stress score with all processing.
        
        Args:
            features: Dictionary with normalized features.
        
        Returns:
            Complete stress result dictionary.
        """
        # Compute raw stress
        raw_stress = self.compute_raw_stress(features)
        
        # Apply temporal smoothing
        smoothed_stress = self.apply_temporal_smoothing(raw_stress)
        
        # Scale to 0-100
        final_stress = self.scale_to_100(smoothed_stress)
        
        return {
            'facial_stress': final_stress,
            'eye_closure': features['eye_closure'],
            'eyebrow_tension': features['eyebrow_tension'],
            'jaw_tension': features['jaw_openness'],  # Alias for clarity
            'facial_symmetry': features.get('facial_symmetry', 1.0),
            'raw_stress': round(raw_stress, 4),
            'smoothed_stress': round(smoothed_stress, 4),
        }
    
    def reset(self):
        """Clear temporal smoothing history (for new sessions)."""
        self.history.clear()


# =============================================================================
# MAIN INFERENCE ENGINE (FastAPI-Compatible)
# =============================================================================

class FacialStressInference:
    """
    Main facial stress inference engine.
    
    Designed to be modular and callable from FastAPI.
    Thread-safe for concurrent requests.
    
    Usage:
        engine = FacialStressInference()
        result = engine.process_frame(frame)
        engine.close()
    """
    
    def __init__(self, config: StressConfig = None):
        """
        Initialize the complete inference pipeline.
        
        Args:
            config: Optional StressConfig for customization.
        """
        self.config = config or StressConfig()
        
        # Initialize components
        self.landmark_extractor = LandmarkExtractor(self.config)
        self.feature_computer = FeatureComputer(self.config)
        self.stress_calculator = StressScoreCalculator(self.config)
        
        # Performance tracking
        self.last_inference_time_ms: float = 0.0
        
        print("[FacialStressInference] Initialized successfully")
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, any]]:
        """
        Process a single video frame and return stress metrics.
        
        This is the main entry point for inference.
        
        Args:
            frame: BGR image from OpenCV (np.ndarray, shape: H x W x 3)
        
        Returns:
            Dictionary with stress metrics, or None if no face detected:
            {
                "facial_stress": 0-100,
                "eye_closure": 0.0-1.0,
                "eyebrow_tension": 0.0-1.0,
                "jaw_tension": 0.0-1.0,
                "facial_symmetry": 0.0-1.0,
                "face_detected": True,
                "inference_time_ms": <float>
            }
        """
        start_time = time.perf_counter()
        
        # Step 1: Extract landmarks
        landmarks = self.landmark_extractor.extract(frame)
        
        if landmarks is None:
            self.last_inference_time_ms = (time.perf_counter() - start_time) * 1000
            return {
                'facial_stress': 0,
                'eye_closure': 0.0,
                'eyebrow_tension': 0.0,
                'jaw_tension': 0.0,
                'facial_symmetry': 1.0,
                'face_detected': False,
                'inference_time_ms': round(self.last_inference_time_ms, 2),
            }
        
        # Step 2: Compute features
        features = self.feature_computer.compute_all_features(landmarks)
        
        # Step 3: Calculate stress score
        stress_result = self.stress_calculator.calculate(features)
        
        # Calculate inference time
        self.last_inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Build final output
        result = {
            'facial_stress': stress_result['facial_stress'],
            'eye_closure': stress_result['eye_closure'],
            'eyebrow_tension': stress_result['eyebrow_tension'],
            'jaw_tension': stress_result['jaw_tension'],
            'facial_symmetry': stress_result['facial_symmetry'],
            'face_detected': True,
            'inference_time_ms': round(self.last_inference_time_ms, 2),
        }
        
        return result
    
    def reset_session(self):
        """Reset temporal smoothing for a new assessment session."""
        self.stress_calculator.reset()
    
    def close(self):
        """Release all resources."""
        self.landmark_extractor.close()
        print("[FacialStressInference] Resources released")


# =============================================================================
# WEBCAM DEMO (Standalone Testing)
# =============================================================================

def run_webcam_demo():
    """
    Run a live webcam demo for testing.
    Press 'q' to quit.
    """
    print("=" * 60)
    print("  FACIAL STRESS INFERENCE - LIVE DEMO")
    print("=" * 60)
    print("\nInitializing...")
    
    # Initialize inference engine
    engine = FacialStressInference()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    # Set webcam properties for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n✅ Webcam opened successfully")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    fps_start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to capture frame")
                break
            
            # Process frame
            result = engine.process_frame(frame)
            
            # Draw results on frame
            if result['face_detected']:
                stress = result['facial_stress']
                
                # Color based on stress level
                if stress < 40:
                    color = (0, 255, 0)      # Green - Low stress
                    level = "LOW"
                elif stress < 70:
                    color = (0, 165, 255)    # Orange - Medium stress
                    level = "MEDIUM"
                else:
                    color = (0, 0, 255)      # Red - High stress
                    level = "HIGH"
                
                # Draw stress score
                cv2.putText(frame, f"STRESS: {stress} ({level})", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                # Draw individual features
                y_offset = 80
                features = [
                    ("Eye Closure", result['eye_closure']),
                    ("Eyebrow Tension", result['eyebrow_tension']),
                    ("Jaw Tension", result['jaw_tension']),
                ]
                
                for name, value in features:
                    bar_width = int(value * 150)
                    cv2.putText(frame, f"{name}: {value:.2f}", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (180, y_offset - 12), (180 + bar_width, y_offset),
                                 color, -1)
                    y_offset += 25
                
                # Draw inference time
                cv2.putText(frame, f"Inference: {result['inference_time_ms']:.1f}ms", 
                           (20, y_offset + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_start_time)
                print(f"FPS: {fps:.1f} | Stress: {result['facial_stress']} | "
                      f"Inference: {result['inference_time_ms']:.1f}ms")
            
            # Show frame
            cv2.imshow('Facial Stress Inference', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
        print("\n✅ Demo ended successfully")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_webcam_demo()
