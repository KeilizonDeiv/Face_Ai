"""
Face Detection and Analysis Engine
Uses open-source models for face detection, emotion recognition, and analysis
"""

import cv2
import numpy as np
from deepface import DeepFace
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')


class FaceAnalyzer:
    """Face detection and emotion analysis using open-source models"""

    def __init__(self):
        """Initialize face analyzer with models"""
        if not os.path.exists(YUNET_MODEL_PATH):
            raise FileNotFoundError(
                f"Face detection model not found at {YUNET_MODEL_PATH}. "
                "Run 'python download_models.py' once to fetch it."
            )
        self.face_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3
        )

        # Emotion labels
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Color mappings for emotions
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 100, 0),     # Dark Green
            'fear': (128, 0, 128),      # Purple
            'happy': (0, 255, 0),       # Green
            'sad': (255, 0, 0),         # Blue
            'surprise': (0, 255, 255),  # Yellow
            'neutral': (200, 200, 200)  # Gray
        }
        
        print("✓ Face analyzer initialized")
    
    def detect_faces_raw(self, image: np.ndarray) -> np.ndarray:
        """
        Run the YuNet detector and return its raw per-face rows.

        Each row is [x, y, w, h, right_eye_x, right_eye_y, left_eye_x,
        left_eye_y, nose_x, nose_y, right_mouth_x, right_mouth_y,
        left_mouth_x, left_mouth_y, score] — the 5-point landmarks are
        needed for face alignment before recognition.

        Args:
            image: Input image (BGR format)

        Returns:
            Nx15 float array (empty if no faces found)
        """
        h, w = image.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(image)

        if faces is None:
            return np.empty((0, 15), dtype=np.float32)

        return faces

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using the YuNet DNN face detector

        Args:
            image: Input image (BGR format)

        Returns:
            List of face bounding boxes (x, y, width, height)
        """
        raw_faces = self.detect_faces_raw(image)
        img_h, img_w = image.shape[:2]

        boxes = []
        for row in raw_faces:
            x, y, w, h = row[:4]
            x = max(0, int(round(x)))
            y = max(0, int(round(y)))
            w = min(int(round(w)), img_w - x)
            h = min(int(round(h)), img_h - y)
            boxes.append((x, y, w, h))

        return boxes

    def analyze_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict:
        """
        Analyze a detected face for emotions and attributes
        
        Args:
            image: Full image
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Dictionary with analysis results
        """
        x, y, w, h = face_box
        
        # Extract face region with padding
        padding = 20
        y1 = max(0, y - padding)
        y2 = min(image.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(image.shape[1], x + w + padding)
        
        face_img = image[y1:y2, x1:x2]
        
        try:
            # Use DeepFace for emotion analysis
            analysis = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Handle both list and dict responses
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            emotion_scores = analysis['emotion']
            dominant_emotion = analysis['dominant_emotion']
            
            return {
                'emotion_scores': emotion_scores,
                'dominant_emotion': dominant_emotion,
                'confidence': emotion_scores[dominant_emotion],
                'box': face_box
            }
            
        except Exception as e:
            # Fallback if analysis fails
            return {
                'emotion_scores': {emotion: 0 for emotion in self.emotions},
                'dominant_emotion': 'neutral',
                'confidence': 0,
                'box': face_box,
                'error': str(e)
            }
    
    def analyze_image(self, image: np.ndarray) -> Dict:
        """
        Analyze all faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with all face analyses
        """
        faces = self.detect_faces(image)
        
        results = {
            'num_faces': len(faces),
            'faces': [],
            'image_shape': image.shape
        }
        
        for i, face_box in enumerate(faces):
            analysis = self.analyze_face(image, tuple(face_box))
            analysis['face_id'] = i + 1
            results['faces'].append(analysis)
        
        return results
    
    def draw_analysis(self, image: np.ndarray, analysis: Dict) -> np.ndarray:
        """
        Draw analysis results on image
        
        Args:
            image: Input image
            analysis: Analysis results
            
        Returns:
            Image with drawn annotations
        """
        output = image.copy()
        
        for face in analysis['faces']:
            x, y, w, h = face['box']
            emotion = face['dominant_emotion']
            confidence = face['confidence']
            
            # Get color for emotion
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label = f"{emotion} ({confidence:.1f}%)"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                output,
                (x, y - label_h - 10),
                (x + label_w + 10, y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                output,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw face ID
            cv2.putText(
                output,
                f"Face {face['face_id']}",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        # Draw summary
        summary_text = f"Faces Detected: {analysis['num_faces']}"
        cv2.putText(
            output,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return output
    
    def get_emotion_distribution(self, analysis: Dict) -> Dict:
        """
        Get aggregated emotion distribution across all faces
        
        Args:
            analysis: Analysis results
            
        Returns:
            Dictionary with emotion statistics
        """
        if analysis['num_faces'] == 0:
            return {emotion: 0 for emotion in self.emotions}
        
        # Aggregate emotions
        emotion_totals = {emotion: 0 for emotion in self.emotions}
        
        for face in analysis['faces']:
            for emotion, score in face['emotion_scores'].items():
                emotion_totals[emotion] += score
        
        # Average across faces
        for emotion in emotion_totals:
            emotion_totals[emotion] /= analysis['num_faces']
        
        return emotion_totals


class VideoAnalyzer:
    """Real-time video analysis for webcam feed"""
    
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.cap = None
        self.is_running = False
    
    def start_camera(self, camera_id: int = 0) -> bool:
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            print("✓ Camera started")
            return True
            
        except Exception as e:
            print(f"✗ Camera start failed: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            self.is_running = False
            print("✓ Camera stopped")
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Get and analyze single frame
        
        Returns:
            Tuple of (frame, analysis) or (None, None) if failed
        """
        if not self.cap or not self.is_running:
            return None, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None, None
        
        # Analyze frame
        analysis = self.face_analyzer.analyze_image(frame)
        
        # Draw annotations
        annotated_frame = self.face_analyzer.draw_analysis(frame, analysis)
        
        return annotated_frame, analysis
    
    def save_snapshot(self, frame: np.ndarray, filepath: str) -> bool:
        """Save current frame as image"""
        try:
            cv2.imwrite(filepath, frame)
            return True
        except Exception as e:
            print(f"✗ Save failed: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Face Analyzer Example")
    print("="*60)
    
    # Initialize analyzer
    analyzer = FaceAnalyzer()
    
    # Create test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (200, 200, 200)
    
    # Add text
    cv2.putText(
        test_img,
        "Point camera at face to start analysis",
        (50, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )
    
    print("\n✓ Face analyzer ready")
    print("Use VideoAnalyzer class for real-time camera analysis")
