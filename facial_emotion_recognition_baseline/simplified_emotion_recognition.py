"""
Simplified Facial Emotion Recognition without TensorFlow
Using scikit-learn for compatibility with Python 3.14
NeuroBalance AI - Team Mission404
"""

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import time
import math
import joblib
import os

class SimplifiedFacialEmotionRecognizer:
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
        
        # Initialize emotion classifier and scaler
        self.emotion_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self._create_and_train_model()
        
        print("SimplifiedFacialEmotionRecognizer initialized successfully")

    def _create_and_train_model(self):
        """Create and train a simple model with synthetic data"""
        # Generate synthetic training data for demonstration
        n_samples = 1000
        n_features = 25  # Reduced feature count for simplicity
        
        # Create synthetic features
        X = np.random.rand(n_samples, n_features)
        
        # Create labels based on feature patterns (simplified emotion rules)
        y = []
        for i in range(n_samples):
            # Simple rules based on facial features
            features = X[i]
            eye_closure = features[0]  # Eye aspect ratio
            mouth_openness = features[4]  # Mouth aspect ratio
            eyebrow_height = features[2]  # Eyebrow position
            
            if eye_closure < 0.3 and mouth_openness < 0.3:  # Eyes closed, mouth closed
                emotion = 5  # sad
            elif mouth_openness > 0.7:  # Wide open mouth
                emotion = 6 if eyebrow_height > 0.6 else 1  # surprise or disgust
            elif eyebrow_height < 0.3 and mouth_openness < 0.4:  # Low eyebrows
                emotion = 0  # angry
            elif mouth_openness > 0.5 and eyebrow_height > 0.5:  # Smile + raised eyebrows
                emotion = 3  # happy
            elif eye_closure > 0.6 and mouth_openness < 0.4:  # Wide eyes, closed mouth
                emotion = 2  # fear
            else:
                emotion = 6  # neutral
            
            y.append(emotion)
        
        y = np.array(y)
        
        # Train the model
        X_scaled = self.scaler.fit_transform(X)
        self.emotion_model.fit(X_scaled, y)
        
        print("Emotion model trained with synthetic data")

    def extract_facial_features(self, landmarks) -> np.ndarray:
        """
        Extract simplified facial features from MediaPipe landmarks
        
        Returns:
            np.ndarray: Feature vector with 25 key facial measurements
        """
        features = []
        
        try:
            # Convert landmarks to numpy array
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            
            # 1. Eye closure features (2 features)
            left_eye_closure = self._calculate_eye_closure(landmarks_array, self.left_eye_indices)
            right_eye_closure = self._calculate_eye_closure(landmarks_array, self.right_eye_indices)
            features.extend([left_eye_closure, right_eye_closure])
            
            # 2. Eyebrow position features (2 features)
            left_eyebrow_height = self._calculate_eyebrow_height(landmarks_array, self.left_eyebrow_indices, self.left_eye_indices)
            right_eyebrow_height = self._calculate_eyebrow_height(landmarks_array, self.right_eyebrow_indices, self.right_eye_indices)
            features.extend([left_eyebrow_height, right_eyebrow_height])
            
            # 3. Mouth features (3 features)
            mouth_openness = self._calculate_mouth_openness(landmarks_array, self.mouth_indices)
            mouth_width = self._calculate_mouth_width(landmarks_array, self.mouth_indices)
            mouth_curvature = self._calculate_mouth_curvature(landmarks_array, self.mouth_indices)
            features.extend([mouth_openness, mouth_width, mouth_curvature])
            
            # 4. Facial symmetry (1 feature)
            symmetry_score = self._calculate_facial_symmetry(landmarks_array)
            features.append(symmetry_score)
            
            # 5. Additional geometric features (17 more features to reach 25)
            additional = self._extract_geometric_features(landmarks_array)
            features.extend(additional)
            
            # Ensure exactly 25 features
            features = features[:25]
            while len(features) < 25:
                features.append(0.0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(25, dtype=np.float32)

    def _calculate_eye_closure(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """Calculate eye closure ratio (0 = closed, 1 = open)"""
        try:
            if len(eye_indices) < 6:
                return 0.3
            
            # Use first 6 points for calculation
            eye_points = landmarks[eye_indices[:6]]
            
            # Calculate vertical distances
            vertical_1 = np.linalg.norm(eye_points[1] - eye_points[4])
            vertical_2 = np.linalg.norm(eye_points[2] - eye_points[5])
            
            # Calculate horizontal distance
            horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # Eye aspect ratio
            if horizontal > 0:
                ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
                return min(1.0, max(0.0, ear))
            return 0.3
        except:
            return 0.3

    def _calculate_eyebrow_height(self, landmarks: np.ndarray, eyebrow_indices: List[int], eye_indices: List[int]) -> float:
        """Calculate eyebrow height relative to eye"""
        try:
            eyebrow_points = landmarks[eyebrow_indices]
            eye_points = landmarks[eye_indices[:6]]
            
            eyebrow_center_y = np.mean(eyebrow_points[:, 1])
            eye_center_y = np.mean(eye_points[:, 1])
            
            return abs(eyebrow_center_y - eye_center_y)
        except:
            return 0.05

    def _calculate_mouth_openness(self, landmarks: np.ndarray, mouth_indices: List[int]) -> float:
        """Calculate mouth openness ratio"""
        try:
            mouth_points = landmarks[mouth_indices[:8]]  # Use first 8 points
            
            # Approximate mouth opening
            if len(mouth_points) >= 4:
                top_lip = mouth_points[1]
                bottom_lip = mouth_points[3]
                left_corner = mouth_points[0]
                right_corner = mouth_points[2]
                
                vertical_distance = np.linalg.norm(top_lip - bottom_lip)
                horizontal_distance = np.linalg.norm(left_corner - right_corner)
                
                if horizontal_distance > 0:
                    mar = vertical_distance / horizontal_distance
                    return min(1.0, max(0.0, mar))
            return 0.3
        except:
            return 0.3

    def _calculate_mouth_width(self, landmarks: np.ndarray, mouth_indices: List[int]) -> float:
        """Calculate mouth width"""
        try:
            mouth_points = landmarks[mouth_indices[:4]]
            if len(mouth_points) >= 3:
                left_corner = mouth_points[0]
                right_corner = mouth_points[2]
                return np.linalg.norm(left_corner - right_corner)
            return 0.05
        except:
            return 0.05

    def _calculate_mouth_curvature(self, landmarks: np.ndarray, mouth_indices: List[int]) -> float:
        """Calculate mouth curvature (smile/frown detection)"""
        try:
            mouth_points = landmarks[mouth_indices[:6]]
            if len(mouth_points) >= 6:
                # Simple curvature based on mouth corners vs center
                left_corner = mouth_points[0]
                right_corner = mouth_points[2]
                center = mouth_points[1]
                
                corners_y = (left_corner[1] + right_corner[1]) / 2
                center_y = center[1]
                
                return center_y - corners_y  # Positive = smile, negative = frown
            return 0.0
        except:
            return 0.0

    def _calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        try:
            # Use nose tip and cheek points for symmetry
            if len(landmarks) > 454:
                nose_tip = landmarks[1]
                left_cheek = landmarks[234] if len(landmarks) > 234 else landmarks[50]
                right_cheek = landmarks[454] if len(landmarks) > 454 else landmarks[100]
                
                left_distance = np.linalg.norm(nose_tip - left_cheek)
                right_distance = np.linalg.norm(nose_tip - right_cheek)
                
                if max(left_distance, right_distance) > 0:
                    symmetry = min(left_distance, right_distance) / max(left_distance, right_distance)
                    return symmetry
            return 0.9
        except:
            return 0.9

    def _extract_geometric_features(self, landmarks: np.ndarray) -> List[float]:
        """Extract additional geometric features"""
        features = []
        
        try:
            # Face dimensions
            face_width = np.linalg.norm(landmarks[234] - landmarks[454]) if len(landmarks) > 454 else 0.1
            face_height = np.linalg.norm(landmarks[10] - landmarks[152]) if len(landmarks) > 152 else 0.1
            face_ratio = face_width / face_height if face_height > 0 else 1.0
            features.append(face_ratio)
            
            # Nose features
            nose_tip = landmarks[1] if len(landmarks) > 1 else np.array([0.5, 0.5, 0])
            nose_width = np.linalg.norm(landmarks[31] - landmarks[35]) if len(landmarks) > 35 else 0.02
            features.extend([nose_tip[0], nose_tip[1], nose_width])
            
            # Eye distance
            eye_distance = np.linalg.norm(landmarks[33] - landmarks[362]) if len(landmarks) > 362 else 0.1
            features.append(eye_distance)
            
            # Fill remaining slots with normalized landmark positions
            key_points = [10, 151, 9, 8, 168, 6, 1, 33, 362, 13, 14, 17]
            for i, point_idx in enumerate(key_points):
                if len(features) >= 17:
                    break
                if len(landmarks) > point_idx:
                    features.extend([landmarks[point_idx][0], landmarks[point_idx][1]])
                else:
                    features.extend([0.5, 0.5])
            
            # Ensure exactly 17 additional features
            while len(features) < 17:
                features.append(0.0)
                
        except Exception as e:
            # Fill with default values on error
            while len(features) < 17:
                features.append(0.5)
        
        return features[:17]

    def classify_emotion(self, features: np.ndarray) -> Dict[str, float]:
        """
        Classify emotion from facial features using Random Forest
        
        Returns:
            Dict[str, float]: Emotion probabilities and stress score
        """
        try:
            # Ensure features is the right shape
            if len(features) != 25:
                features = np.pad(features, (0, max(0, 25 - len(features))), mode='constant')[:25]
            
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get emotion predictions
            predictions = self.emotion_model.predict_proba(features_scaled)[0]
            
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
                'emotions': {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels},
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
    """Test the simplified facial emotion recognition system"""
    print("Starting Simplified Facial Emotion Recognition Test...")
    
    # Initialize the recognizer
    recognizer = SimplifiedFacialEmotionRecognizer()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame, emotion_result = recognizer.process_frame(frame)
        
        # Display results
        cv2.imshow('Facial Emotion Recognition', processed_frame)
        
        # Print emotion results every 30 frames (approximately once per second)
        frame_count += 1
        if frame_count % 30 == 0 and emotion_result:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.1f} | Emotion: {emotion_result['dominant_emotion']} | "
                  f"Stress: {emotion_result['stress_score']:.2f} | "
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