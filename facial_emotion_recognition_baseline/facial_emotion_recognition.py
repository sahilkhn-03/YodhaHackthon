"""
Facial Emotion Recognition Baseline using MediaPipe and TensorFlow
NeuroBalance AI - Team Mission404

Features:
- MediaPipe Face Mesh detection
- Facial feature extraction (eye closure, eyebrow position, mouth openness, facial symmetry)
- Real-time emotion classification
- Stress confidence scoring
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, List
import time
import math

class FacialEmotionRecognizer:
    def __init__(self):
        """Initialize MediaPipe face mesh and emotion classifier"""
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh with refined landmarks for better accuracy
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define facial landmark indices for feature extraction
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.left_eyebrow_indices = [46, 53, 52, 51, 48]
        self.right_eyebrow_indices = [276, 283, 282, 281, 278]
        self.mouth_indices = [0, 17, 18, 200, 199, 175, 13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Initialize lightweight emotion classifier (will be replaced with pre-trained model)
        self.emotion_model = None
        self._create_dummy_model()
        
        print("FacialEmotionRecognizer initialized successfully")

    def _create_dummy_model(self):
        """Create a dummy model for demonstration. Replace with pre-trained model."""
        # This is a placeholder - in production, load a pre-trained emotion classifier
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(68,)),  # 68 facial features
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.emotion_labels), activation='softmax')
        ])
        
        # Compile with dummy weights for demonstration
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.emotion_model = model
        print("Dummy emotion model created (replace with pre-trained model)")

    def extract_facial_features(self, landmarks) -> np.ndarray:
        """
        Extract facial features from MediaPipe landmarks
        
        Returns:
            np.ndarray: Feature vector with eye closure, eyebrow position, mouth openness, facial symmetry
        """
        features = []
        
        try:
            # Convert landmarks to numpy array
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            # 1. Eye closure features
            left_eye_closure = self._calculate_eye_closure(landmarks_array, self.left_eye_indices)
            right_eye_closure = self._calculate_eye_closure(landmarks_array, self.right_eye_indices)
            features.extend([left_eye_closure, right_eye_closure])
            
            # 2. Eyebrow position features
            left_eyebrow_height = self._calculate_eyebrow_height(landmarks_array, self.left_eyebrow_indices, self.left_eye_indices)
            right_eyebrow_height = self._calculate_eyebrow_height(landmarks_array, self.right_eyebrow_indices, self.right_eye_indices)
            features.extend([left_eyebrow_height, right_eyebrow_height])
            
            # 3. Mouth openness features
            mouth_openness = self._calculate_mouth_openness(landmarks_array, self.mouth_indices)
            mouth_width = self._calculate_mouth_width(landmarks_array, self.mouth_indices)
            features.extend([mouth_openness, mouth_width])
            
            # 4. Facial symmetry
            symmetry_score = self._calculate_facial_symmetry(landmarks_array)
            features.append(symmetry_score)
            
            # Add additional geometric features for better emotion detection
            features.extend(self._extract_additional_features(landmarks_array))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(68, dtype=np.float32)  # Return zero vector on error

    def _calculate_eye_closure(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """Calculate eye closure ratio (0 = closed, 1 = open)"""
        try:
            eye_points = landmarks[eye_indices]
            
            # Calculate vertical distances
            vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
            vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # Eye aspect ratio
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        except:
            return 0.2  # Default value

    def _calculate_eyebrow_height(self, landmarks: np.ndarray, eyebrow_indices: List[int], eye_indices: List[int]) -> float:
        """Calculate eyebrow height relative to eye"""
        try:
            eyebrow_points = landmarks[eyebrow_indices]
            eye_points = landmarks[eye_indices]
            
            eyebrow_center_y = np.mean(eyebrow_points[:, 1])
            eye_center_y = np.mean(eye_points[:, 1])
            
            return abs(eyebrow_center_y - eye_center_y)
        except:
            return 0.05  # Default value

    def _calculate_mouth_openness(self, landmarks: np.ndarray, mouth_indices: List[int]) -> float:
        """Calculate mouth openness ratio"""
        try:
            mouth_points = landmarks[mouth_indices]
            
            # Get mouth corners and center points
            top_lip = mouth_points[13]  # Top center
            bottom_lip = mouth_points[14]  # Bottom center
            left_corner = mouth_points[12]  # Left corner
            right_corner = mouth_points[16]  # Right corner
            
            # Calculate vertical and horizontal distances
            vertical_distance = np.linalg.norm(top_lip - bottom_lip)
            horizontal_distance = np.linalg.norm(left_corner - right_corner)
            
            # Mouth aspect ratio
            mar = vertical_distance / horizontal_distance if horizontal_distance > 0 else 0
            return mar
        except:
            return 0.3  # Default value

    def _calculate_mouth_width(self, landmarks: np.ndarray, mouth_indices: List[int]) -> float:
        """Calculate mouth width"""
        try:
            mouth_points = landmarks[mouth_indices]
            left_corner = mouth_points[12]
            right_corner = mouth_points[16]
            return np.linalg.norm(left_corner - right_corner)
        except:
            return 0.05  # Default value

    def _calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        try:
            # Get face center (nose tip)
            nose_tip = landmarks[1]  # Approximate nose tip
            
            # Calculate symmetry by comparing left and right side distances
            left_cheek = landmarks[234]  # Left cheek
            right_cheek = landmarks[454]  # Right cheek
            
            left_distance = np.linalg.norm(nose_tip - left_cheek)
            right_distance = np.linalg.norm(nose_tip - right_cheek)
            
            # Symmetry score (closer to 1 = more symmetric)
            symmetry = min(left_distance, right_distance) / max(left_distance, right_distance) if max(left_distance, right_distance) > 0 else 1.0
            return symmetry
        except:
            return 0.9  # Default high symmetry

    def _extract_additional_features(self, landmarks: np.ndarray) -> List[float]:
        """Extract additional geometric features for better emotion classification"""
        features = []
        
        try:
            # Nose-to-mouth distance
            nose_tip = landmarks[1]
            mouth_center = landmarks[13]
            nose_mouth_distance = np.linalg.norm(nose_tip - mouth_center)
            features.append(nose_mouth_distance)
            
            # Face width and height ratios
            face_width = np.linalg.norm(landmarks[234] - landmarks[454])  # Cheek to cheek
            face_height = np.linalg.norm(landmarks[10] - landmarks[152])  # Forehead to chin
            face_ratio = face_width / face_height if face_height > 0 else 1.0
            features.append(face_ratio)
            
            # Add more features to reach 68 total
            while len(features) < 60:  # Fill remaining slots with normalized coordinates
                if len(features) < len(landmarks):
                    features.extend([landmarks[len(features)][0], landmarks[len(features)][1]])
                else:
                    features.append(0.0)
                    
        except:
            # Fill with zeros if extraction fails
            while len(features) < 60:
                features.append(0.0)
        
        return features[:60]  # Ensure exactly 60 additional features

    def classify_emotion(self, features: np.ndarray) -> Dict[str, float]:
        """
        Classify emotion from facial features
        
        Returns:
            Dict[str, float]: Emotion probabilities and stress score
        """
        try:
            # Ensure features is the right shape
            if len(features) != 68:
                features = np.pad(features, (0, max(0, 68 - len(features))), mode='constant')[:68]
            
            features = features.reshape(1, -1)
            
            # Get emotion predictions from model
            predictions = self.emotion_model.predict(features, verbose=0)[0]
            
            # Create emotion dictionary
            emotion_probs = {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotion_labels, predictions)
            }
            
            # Calculate stress score based on emotion probabilities
            stress_score = self._calculate_stress_score(emotion_probs)
            
            result = {
                'emotions': emotion_probs,
                'dominant_emotion': max(emotion_probs, key=emotion_probs.get),
                'stress_score': stress_score,
                'confidence': float(np.max(predictions))
            }
            
            return result
            
        except Exception as e:
            print(f"Error in emotion classification: {e}")
            # Return default values
            return {
                'emotions': {emotion: 0.14 for emotion in self.emotion_labels},
                'dominant_emotion': 'neutral',
                'stress_score': 0.3,
                'confidence': 0.5
            }

    def _calculate_stress_score(self, emotion_probs: Dict[str, float]) -> float:
        """Calculate stress score from emotion probabilities"""
        # Stress weights for each emotion (0-1 scale)
        stress_weights = {
            'angry': 0.9,
            'disgust': 0.7,
            'fear': 0.85,
            'happy': 0.1,
            'sad': 0.6,
            'surprise': 0.4,
            'neutral': 0.2
        }
        
        # Calculate weighted stress score
        stress_score = sum(
            emotion_probs[emotion] * stress_weights[emotion]
            for emotion in emotion_probs
        )
        
        return min(1.0, max(0.0, stress_score))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Process a single frame for emotion detection
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple[np.ndarray, Optional[Dict]]: Processed frame and emotion results
        """
        if frame is None:
            return frame, None
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Draw face landmarks and get emotion results
        emotion_result = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Extract features and classify emotion
                features = self.extract_facial_features(face_landmarks)
                emotion_result = self.classify_emotion(features)
                
                # Draw emotion information on frame
                self._draw_emotion_info(frame, emotion_result)
                
                break  # Process only the first detected face
        
        return frame, emotion_result

    def _draw_emotion_info(self, frame: np.ndarray, emotion_result: Dict):
        """Draw emotion information on the frame"""
        if emotion_result is None:
            return
            
        height, width = frame.shape[:2]
        
        # Draw emotion text
        emotion = emotion_result['dominant_emotion']
        stress_score = emotion_result['stress_score']
        confidence = emotion_result['confidence']
        
        # Color based on stress level
        if stress_score > 0.7:
            color = (0, 0, 255)  # Red
        elif stress_score > 0.4:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Draw text
        cv2.putText(frame, f"Emotion: {emotion.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Stress: {stress_score:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    """Test the facial emotion recognition system"""
    print("Starting Facial Emotion Recognition Test...")
    
    # Initialize the recognizer
    recognizer = FacialEmotionRecognizer()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame, emotion_result = recognizer.process_frame(frame)
        
        # Display results
        cv2.imshow('Facial Emotion Recognition', processed_frame)
        
        # Print emotion results to console
        if emotion_result:
            print(f"Emotion: {emotion_result['dominant_emotion']}, "
                  f"Stress: {emotion_result['stress_score']:.2f}, "
                  f"Confidence: {emotion_result['confidence']:.2f}")
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Facial emotion recognition test completed")

if __name__ == "__main__":
    main()