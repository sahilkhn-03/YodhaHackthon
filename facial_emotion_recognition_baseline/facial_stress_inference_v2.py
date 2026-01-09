"""
Facial Stress Inference Module v2 (MediaPipe Tasks API)
========================================================
Real-time facial stress estimation using MediaPipe FaceLandmarker + OpenCV + NumPy.

Compatible with MediaPipe 0.10.31+ (Tasks Vision API for Python 3.14+)

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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from typing import Dict, Optional, List
from dataclasses import dataclass
import time
import os
import urllib.request


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StressConfig:
    """Configuration parameters for stress detection."""
    
    # Temporal smoothing window size
    smoothing_window: int = 10
    
    # Feature weights for stress calculation (must sum to 1.0)
    weight_eye_closure: float = 0.4
    weight_eyebrow_tension: float = 0.3
    weight_jaw_openness: float = 0.3
    
    # Normalization bounds (empirically determined)
    ear_min: float = 0.15  # Closed eyes
    ear_max: float = 0.35  # Wide open eyes
    eyebrow_min: float = 0.02  # Furrowed brows (stressed)
    eyebrow_max: float = 0.08  # Relaxed/raised brows
    mouth_min: float = 0.0   # Closed mouth
    mouth_max: float = 0.5   # Wide open mouth
    
    # MediaPipe confidence thresholds
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Model path
    model_path: str = "face_landmarker.task"
    
    # Visualization settings
    show_landmarks: bool = True
    show_connections: bool = True
    landmark_size: int = 1
    connection_thickness: int = 1


# =============================================================================
# LANDMARK INDICES (MediaPipe Face Landmarker - 478 landmarks)
# =============================================================================

class FaceLandmarkIndices:
    """MediaPipe Face Landmarker landmark indices for stress-related features."""
    
    # Left eye landmarks (6 points for EAR calculation)
    LEFT_EYE = {
        'p1': 33,   # Left corner
        'p2': 160,  # Upper lid (left)
        'p3': 158,  # Upper lid (right)
        'p4': 133,  # Right corner
        'p5': 153,  # Lower lid (right)
        'p6': 144,  # Lower lid (left)
    }
    
    # Right eye landmarks
    RIGHT_EYE = {
        'p1': 362,  # Left corner
        'p2': 385,  # Upper lid (left)
        'p3': 387,  # Upper lid (right)
        'p4': 263,  # Right corner
        'p5': 373,  # Lower lid (right)
        'p6': 380,  # Lower lid (left)
    }
    
    # Eyebrow landmarks
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]
    
    # Mouth landmarks
    MOUTH = {
        'upper_lip_top': 13,
        'lower_lip_bottom': 14,
        'left_corner': 61,
        'right_corner': 291,
    }
    
    # Face reference points
    FACE = {
        'forehead': 10,
        'chin': 152,
        'left_cheek': 234,
        'right_cheek': 454,
        'nose_tip': 1,
    }

    # Landmark visualization groups (for different colors)
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    LEFT_EYE_FULL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_FULL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    LEFT_EYEBROW_FULL = [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305]
    RIGHT_EYEBROW_FULL = [276, 283, 282, 281, 278, 344, 360, 363, 331, 279, 440, 75]
    
    NOSE_FULL = [1, 2, 5, 4, 6, 19, 94, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 278, 344, 360, 363, 331, 279]
    
    MOUTH_FULL = [0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 269, 270, 267, 271, 272, 271, 272, 269, 270, 267, 271, 272, 17, 18, 200, 199, 175, 61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
    
    CHIN_JAW = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
    
    # All landmark groups for visualization
    LANDMARK_GROUPS = {
        'face_oval': {'indices': FACE_OVAL, 'color': (255, 255, 255), 'name': 'Face Outline'},  # White
        'left_eye': {'indices': LEFT_EYE_FULL, 'color': (0, 255, 0), 'name': 'Left Eye'},      # Green
        'right_eye': {'indices': RIGHT_EYE_FULL, 'color': (0, 255, 255), 'name': 'Right Eye'},  # Cyan
        'left_eyebrow': {'indices': LEFT_EYEBROW_FULL, 'color': (255, 0, 0), 'name': 'Left Eyebrow'},  # Red
        'right_eyebrow': {'indices': RIGHT_EYEBROW_FULL, 'color': (255, 0, 255), 'name': 'Right Eyebrow'},  # Magenta
        'nose': {'indices': NOSE_FULL, 'color': (0, 165, 255), 'name': 'Nose'},                # Orange
        'mouth': {'indices': MOUTH_FULL, 'color': (255, 255, 0), 'name': 'Mouth'},             # Yellow
        'chin_jaw': {'indices': CHIN_JAW, 'color': (128, 0, 128), 'name': 'Chin/Jaw'},         # Purple
    }


# =============================================================================
# MODEL DOWNLOADER
# =============================================================================

def download_model(model_path: str = "face_landmarker.task") -> str:
    """
    Download the Face Landmarker model if not present.
    
    Args:
        model_path: Path to save the model.
    
    Returns:
        Path to the model file.
    """
    if os.path.exists(model_path):
        print(f"[INFO] Model found at {model_path}")
        return model_path
    
    print(f"[INFO] Downloading Face Landmarker model...")
    
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[INFO] Model downloaded to {model_path}")
        return model_path
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        raise


# =============================================================================
# LANDMARK EXTRACTION
# =============================================================================

class LandmarkExtractor:
    """
    Extracts facial landmarks using MediaPipe FaceLandmarker (Tasks API).
    CPU-only, real-time performance.
    """
    
    def __init__(self, config: StressConfig = None):
        """Initialize MediaPipe Face Landmarker."""
        self.config = config or StressConfig()
        
        # Download model if needed
        model_path = download_model(self.config.model_path)
        
        # Configure Face Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_tracking_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.indices = FaceLandmarkIndices()
        
        print("[LandmarkExtractor] Initialized successfully")
    
    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from a BGR frame.
        
        Args:
            frame: BGR image from OpenCV (np.ndarray, shape: H x W x 3)
        
        Returns:
            np.ndarray of shape (478, 3) with x, y, z coordinates,
            or None if no face detected.
        """
        if frame is None or frame.size == 0:
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face landmarks
        result = self.face_landmarker.detect(mp_image)
        
        # Check if face detected
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None
        
        # Extract first face landmarks as numpy array
        face_landmarks = result.face_landmarks[0]
        landmarks = np.array([
            [lm.x, lm.y, lm.z] 
            for lm in face_landmarks
        ], dtype=np.float32)
        
        return landmarks
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw all 478 facial landmarks on the frame with different colors for different regions.
        
        Args:
            frame: BGR image from OpenCV
            landmarks: Facial landmarks array (478, 3)
        
        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None or not self.config.show_landmarks:
            return frame
        
        height, width = frame.shape[:2]
        
        # Draw landmark groups with different colors
        for group_name, group_info in self.indices.LANDMARK_GROUPS.items():
            color = group_info['color']
            
            for idx in group_info['indices']:
                if idx < len(landmarks):
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmarks[idx][0] * width)
                    y = int(landmarks[idx][1] * height)
                    
                    # Draw landmark point
                    cv2.circle(frame, (x, y), self.config.landmark_size, color, -1)
        
        # Draw connections if enabled
        if self.config.show_connections:
            self._draw_connections(frame, landmarks)
        
        # Draw legend
        self._draw_legend(frame)
        
        return frame
    
    def _draw_connections(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw connections between landmark points."""
        height, width = frame.shape[:2]
        
        # Define key connections for face structure
        connections = [
            # Face outline
            (self.indices.FACE_OVAL[:-1], self.indices.FACE_OVAL[1:]),
            # Eyes
            (self.indices.LEFT_EYE_FULL[:-1], self.indices.LEFT_EYE_FULL[1:]),
            (self.indices.RIGHT_EYE_FULL[:-1], self.indices.RIGHT_EYE_FULL[1:]),
            # Eyebrows
            (self.indices.LEFT_EYEBROW_FULL[:-1], self.indices.LEFT_EYEBROW_FULL[1:]),
            (self.indices.RIGHT_EYEBROW_FULL[:-1], self.indices.RIGHT_EYEBROW_FULL[1:]),
            # Mouth outline
            (self.indices.MOUTH_FULL[:-1], self.indices.MOUTH_FULL[1:]),
        ]
        
        for start_indices, end_indices in connections:
            for start_idx, end_idx in zip(start_indices, end_indices):
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (
                        int(landmarks[start_idx][0] * width),
                        int(landmarks[start_idx][1] * height)
                    )
                    end_point = (
                        int(landmarks[end_idx][0] * width),
                        int(landmarks[end_idx][1] * height)
                    )
                    cv2.line(frame, start_point, end_point, (100, 100, 100), self.config.connection_thickness)
    
    def _draw_legend(self, frame: np.ndarray):
        """Draw color legend for landmark groups."""
        height, width = frame.shape[:2]
        
        # Legend position (top right)
        legend_x = width - 150
        legend_y = 20
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (legend_x - 10, legend_y - 10), (width - 10, legend_y + len(self.indices.LANDMARK_GROUPS) * 15 + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw legend items
        for i, (group_name, group_info) in enumerate(self.indices.LANDMARK_GROUPS.items()):
            y_pos = legend_y + i * 15
            color = group_info['color']
            name = group_info['name']
            
            # Draw color circle
            cv2.circle(frame, (legend_x, y_pos), 4, color, -1)
            
            # Draw text
            cv2.putText(frame, name, (legend_x + 15, y_pos + 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

class FeatureComputer:
    """
    Computes stress-related facial features from landmarks.
    All features are normalized to 0-1 range.
    """
    
    def __init__(self, config: StressConfig = None):
        """Initialize feature computer."""
        self.config = config or StressConfig()
        self.indices = FaceLandmarkIndices()
    
    @staticmethod
    def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points (using x, y only)."""
        return float(np.linalg.norm(p1[:2] - p2[:2]))
    
    def compute_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """
        Compute Eye Aspect Ratio (EAR).
        
        Formula: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        Low EAR = eyes closed, High EAR = eyes open.
        """
        def single_eye_ear(eye_indices: dict) -> float:
            try:
                p1 = landmarks[eye_indices['p1']]
                p2 = landmarks[eye_indices['p2']]
                p3 = landmarks[eye_indices['p3']]
                p4 = landmarks[eye_indices['p4']]
                p5 = landmarks[eye_indices['p5']]
                p6 = landmarks[eye_indices['p6']]
                
                vertical_1 = self.euclidean_distance(p2, p6)
                vertical_2 = self.euclidean_distance(p3, p5)
                horizontal = self.euclidean_distance(p1, p4)
                
                if horizontal < 1e-6:
                    return 0.25
                
                return (vertical_1 + vertical_2) / (2.0 * horizontal)
            except IndexError:
                return 0.25
        
        left_ear = single_eye_ear(self.indices.LEFT_EYE)
        right_ear = single_eye_ear(self.indices.RIGHT_EYE)
        
        return (left_ear + right_ear) / 2.0
    
    def compute_eyebrow_tension(self, landmarks: np.ndarray) -> float:
        """
        Compute eyebrow tension (distance between eyebrow and eye).
        Lower distance = furrowed brows = higher stress.
        """
        try:
            forehead = landmarks[self.indices.FACE['forehead']]
            chin = landmarks[self.indices.FACE['chin']]
            face_height = self.euclidean_distance(forehead, chin)
            
            if face_height < 1e-6:
                return 0.5
            
            def eyebrow_distance(eyebrow_indices: List[int], eye_indices: dict) -> float:
                eyebrow_points = landmarks[eyebrow_indices]
                eyebrow_center = np.mean(eyebrow_points, axis=0)
                eye_center = (landmarks[eye_indices['p1']] + landmarks[eye_indices['p4']]) / 2
                return abs(eyebrow_center[1] - eye_center[1])
            
            left_dist = eyebrow_distance(self.indices.LEFT_EYEBROW, self.indices.LEFT_EYE)
            right_dist = eyebrow_distance(self.indices.RIGHT_EYEBROW, self.indices.RIGHT_EYE)
            
            avg_distance = (left_dist + right_dist) / 2.0
            return avg_distance / face_height
        except IndexError:
            return 0.5
    
    def compute_jaw_openness(self, landmarks: np.ndarray) -> float:
        """
        Compute mouth/jaw openness (MAR = vertical / horizontal).
        High MAR = open mouth.
        """
        try:
            mouth = self.indices.MOUTH
            
            upper_lip = landmarks[mouth['upper_lip_top']]
            lower_lip = landmarks[mouth['lower_lip_bottom']]
            vertical = self.euclidean_distance(upper_lip, lower_lip)
            
            left_corner = landmarks[mouth['left_corner']]
            right_corner = landmarks[mouth['right_corner']]
            horizontal = self.euclidean_distance(left_corner, right_corner)
            
            if horizontal < 1e-6:
                return 0.0
            
            return vertical / horizontal
        except IndexError:
            return 0.0
    
    def compute_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """Compute facial symmetry score (1.0 = symmetric)."""
        try:
            face = self.indices.FACE
            nose_tip = landmarks[face['nose_tip']]
            left_cheek = landmarks[face['left_cheek']]
            right_cheek = landmarks[face['right_cheek']]
            
            left_dist = self.euclidean_distance(nose_tip, left_cheek)
            right_dist = self.euclidean_distance(nose_tip, right_cheek)
            
            if max(left_dist, right_dist) < 1e-6:
                return 1.0
            
            return min(left_dist, right_dist) / max(left_dist, right_dist)
        except IndexError:
            return 1.0
    
    def normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-1 range."""
        if max_val <= min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def compute_all_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute all stress-related features (normalized to 0-1).
        Higher values = more stress.
        """
        # Compute raw features
        ear = self.compute_eye_aspect_ratio(landmarks)
        eyebrow_dist = self.compute_eyebrow_tension(landmarks)
        jaw = self.compute_jaw_openness(landmarks)
        symmetry = self.compute_facial_symmetry(landmarks)
        
        # Normalize (stress-oriented: higher = more stress)
        
        # Eye closure: INVERT EAR (low EAR = closed = stressed)
        eye_closure = 1.0 - self.normalize(ear, self.config.ear_min, self.config.ear_max)
        
        # Eyebrow tension: INVERT (low distance = furrowed = stressed)
        eyebrow_tension = 1.0 - self.normalize(eyebrow_dist, self.config.eyebrow_min, self.config.eyebrow_max)
        
        # Jaw openness: Direct (high = stressed)
        jaw_openness = self.normalize(jaw, self.config.mouth_min, self.config.mouth_max)
        
        return {
            'eye_closure': round(eye_closure, 4),
            'eyebrow_tension': round(eyebrow_tension, 4),
            'jaw_openness': round(jaw_openness, 4),
            'facial_symmetry': round(symmetry, 4),
        }


# =============================================================================
# STRESS SCORE CALCULATION
# =============================================================================

class StressScoreCalculator:
    """Calculates final stress score with temporal smoothing."""
    
    def __init__(self, config: StressConfig = None):
        """Initialize with temporal smoothing buffer."""
        self.config = config or StressConfig()
        self.history: deque = deque(maxlen=self.config.smoothing_window)
    
    def compute_raw_stress(self, features: Dict[str, float]) -> float:
        """
        Compute raw stress score.
        Formula: 0.4 * eye_closure + 0.3 * eyebrow_tension + 0.3 * jaw_openness
        """
        stress = (
            self.config.weight_eye_closure * features['eye_closure'] +
            self.config.weight_eyebrow_tension * features['eyebrow_tension'] +
            self.config.weight_jaw_openness * features['jaw_openness']
        )
        return float(np.clip(stress, 0.0, 1.0))
    
    def smooth(self, raw_stress: float) -> float:
        """Apply rolling average smoothing."""
        self.history.append(raw_stress)
        return float(np.mean(list(self.history)))
    
    def calculate(self, features: Dict[str, float]) -> Dict[str, any]:
        """Calculate final stress score (0-100)."""
        raw_stress = self.compute_raw_stress(features)
        smoothed_stress = self.smooth(raw_stress)
        final_stress = int(round(smoothed_stress * 100))
        
        return {
            'facial_stress': final_stress,
            'eye_closure': features['eye_closure'],
            'eyebrow_tension': features['eyebrow_tension'],
            'jaw_tension': features['jaw_openness'],
            'facial_symmetry': features.get('facial_symmetry', 1.0),
        }
    
    def reset(self):
        """Clear temporal smoothing history."""
        self.history.clear()


# =============================================================================
# MAIN INFERENCE ENGINE (FastAPI-Compatible)
# =============================================================================

class FacialStressInference:
    """
    Main facial stress inference engine.
    Modular and callable from FastAPI.
    
    Usage:
        engine = FacialStressInference()
        result = engine.process_frame(frame)
        engine.close()
    """
    
    def __init__(self, config: StressConfig = None):
        """Initialize the complete inference pipeline."""
        self.config = config or StressConfig()
        
        self.landmark_extractor = LandmarkExtractor(self.config)
        self.feature_computer = FeatureComputer(self.config)
        self.stress_calculator = StressScoreCalculator(self.config)
        
        self.last_inference_time_ms: float = 0.0
        
        print("[FacialStressInference] Initialized successfully")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Process a single video frame and return stress metrics.
        
        Args:
            frame: BGR image from OpenCV (np.ndarray, shape: H x W x 3)
        
        Returns:
            {
                "facial_stress": 0-100,
                "eye_closure": 0.0-1.0,
                "eyebrow_tension": 0.0-1.0,
                "jaw_tension": 0.0-1.0,
                "face_detected": True/False,
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
                'face_detected': False,
                'inference_time_ms': round(self.last_inference_time_ms, 2),
            }
        
        # Step 2: Compute features
        features = self.feature_computer.compute_all_features(landmarks)
        
        # Step 3: Calculate stress score
        stress_result = self.stress_calculator.calculate(features)
        
        # Calculate inference time
        self.last_inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'facial_stress': stress_result['facial_stress'],
            'eye_closure': stress_result['eye_closure'],
            'eyebrow_tension': stress_result['eyebrow_tension'],
            'jaw_tension': stress_result['jaw_tension'],
            'face_detected': True,
            'inference_time_ms': round(self.last_inference_time_ms, 2),
        }
    
    def process_frame_with_visualization(self, frame: np.ndarray) -> tuple:
        """
        Process a single video frame and return stress metrics with visualization.
        
        Args:
            frame: BGR image from OpenCV (np.ndarray, shape: H x W x 3)
        
        Returns:
            Tuple of (stress_result_dict, processed_frame_with_landmarks)
        """
        start_time = time.perf_counter()
        
        # Step 1: Extract landmarks
        landmarks = self.landmark_extractor.extract(frame)
        
        if landmarks is None:
            self.last_inference_time_ms = (time.perf_counter() - start_time) * 1000
            result = {
                'facial_stress': 0,
                'eye_closure': 0.0,
                'eyebrow_tension': 0.0,
                'jaw_tension': 0.0,
                'face_detected': False,
                'inference_time_ms': round(self.last_inference_time_ms, 2),
            }
            return result, frame
        
        # Step 2: Compute features
        features = self.feature_computer.compute_all_features(landmarks)
        
        # Step 3: Calculate stress score
        stress_result = self.stress_calculator.calculate(features)
        
        # Step 4: Draw landmarks on frame
        processed_frame = self.landmark_extractor.draw_landmarks(frame.copy(), landmarks)
        
        # Calculate inference time
        self.last_inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        result = {
            'facial_stress': stress_result['facial_stress'],
            'eye_closure': stress_result['eye_closure'],
            'eyebrow_tension': stress_result['eyebrow_tension'],
            'jaw_tension': stress_result['jaw_tension'],
            'face_detected': True,
            'inference_time_ms': round(self.last_inference_time_ms, 2),
        }
        
        return result, processed_frame
    
    def reset_session(self):
        """Reset temporal smoothing for a new session."""
        self.stress_calculator.reset()
    
    def close(self):
        """Release all resources."""
        self.landmark_extractor.close()
        print("[FacialStressInference] Resources released")


# =============================================================================
# WEBCAM DEMO
# =============================================================================

def run_webcam_demo():
    """Run a live webcam demo with facial landmark visualization. Press 'q' to quit."""
    print("=" * 60)
    print("  FACIAL STRESS INFERENCE - LIVE DEMO WITH LANDMARKS")
    print("=" * 60)
    print("\nInitializing...")
    
    # Create config with landmark visualization enabled
    config = StressConfig(
        show_landmarks=True,
        show_connections=True,
        landmark_size=1,
        connection_thickness=1
    )
    
    engine = FacialStressInference(config)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n✅ Webcam opened successfully")
    print("Press 'q' to quit, 'l' to toggle landmarks, 'c' to toggle connections\n")
    
    frame_count = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with visualization
            result, processed_frame = engine.process_frame_with_visualization(frame)
            
            # Draw stress information overlay
            if result['face_detected']:
                stress = result['facial_stress']
                stress_level = get_stress_level_label(stress)
                
                # Draw stress information box
                cv2.rectangle(processed_frame, (10, 10), (300, 120), (0, 0, 0), -1)
                cv2.rectangle(processed_frame, (10, 10), (300, 120), (255, 255, 255), 2)
                
                cv2.putText(processed_frame, f"Stress: {stress} - {stress_level}", 
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(processed_frame, f"Eye Closure: {result['eye_closure']:.2f}", 
                           (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.putText(processed_frame, f"Eyebrow Tension: {result['eyebrow_tension']:.2f}", 
                           (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.putText(processed_frame, f"Jaw Tension: {result['jaw_tension']:.2f}", 
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.putText(processed_frame, f"Time: {result['inference_time_ms']:.1f}ms", 
                           (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                
                # Color code by stress level
                color = (0, 255, 0) if stress < 30 else (0, 255, 255) if stress < 60 else (0, 0, 255)
                cv2.rectangle(processed_frame, (8, 8), (302, 122), color, 3)
            else:
                cv2.putText(processed_frame, "No face detected", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_start_time
                fps = 10 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
                
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                           (processed_frame.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Facial Stress Detection with Landmarks', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):  # Toggle landmarks
                config.show_landmarks = not config.show_landmarks
                print(f"Landmarks: {'ON' if config.show_landmarks else 'OFF'}")
            elif key == ord('c'):  # Toggle connections
                config.show_connections = not config.show_connections
                print(f"Connections: {'ON' if config.show_connections else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nStopping demo...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
        print("Demo ended successfully")


def get_stress_level_label(stress_score: int) -> str:
    """Convert stress score to descriptive label."""
    if stress_score < 20:
        return "VERY LOW"
    elif stress_score < 35:
        return "LOW"
    elif stress_score < 50:
        return "MODERATE"
    elif stress_score < 70:
        return "HIGH"
    else:
        return "VERY HIGH"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_webcam_demo()
