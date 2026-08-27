"""
Face Detection and Analysis Engine
Uses open-source models for face detection, emotion recognition, and analysis
"""

import cv2
import numpy as np
from deepface import DeepFace
import os
import threading
import time
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from tracker import IOUTracker
from face_recognizer import FaceRecognizer


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')

# How often (in frames) a track's name/emotion gets refreshed once it has
# one. New tracks are always recognized immediately on their first frame.
RECOGNIZE_EVERY_N_FRAMES = 12


class FaceAnalyzer:
    """Face detection and emotion analysis using open-source models"""

    def __init__(self, recognizer: Optional[FaceRecognizer] = None):
        """
        Initialize face analyzer with models.

        Args:
            recognizer: an existing FaceRecognizer to share (e.g. so the
                video pipeline and the image-upload path see the same
                enrolled-people index). A new one is created if omitted.
        """
        if not os.path.exists(YUNET_MODEL_PATH):
            raise FileNotFoundError(
                f"Face detection model not found at {YUNET_MODEL_PATH}. "
                "Run 'python download_models.py' once to fetch it."
            )
        self.face_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3
        )
        self.recognizer = recognizer if recognizer is not None else FaceRecognizer()

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
        
        print("Face analyzer initialized")
    
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

    def recognize_face(self, image: np.ndarray, detection_row: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify a detected face against the enrolled people, given the
        image and that face's raw YuNet detection row (used for alignment).

        Returns:
            (name, match_score) — name is None if no enrolled person
            matched above the recognizer's threshold.
        """
        embedding = self.recognizer.embed(image, detection_row)
        return self.recognizer.recognize(embedding)

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
        Analyze all faces in a single still image (detection + emotion +
        recognition). Used for one-off uploads, where there's no video
        stream to track across, so faces are just numbered in detection
        order rather than assigned a persistent track id.

        Args:
            image: Input image (BGR format)

        Returns:
            Dictionary with all face analyses
        """
        raw_faces = self.detect_faces_raw(image)
        img_h, img_w = image.shape[:2]

        results = {
            'num_faces': len(raw_faces),
            'faces': [],
            'image_shape': image.shape
        }

        for i, row in enumerate(raw_faces):
            x, y, w, h = row[:4]
            x = max(0, int(round(x)))
            y = max(0, int(round(y)))
            w = min(int(round(w)), img_w - x)
            h = min(int(round(h)), img_h - y)
            face_box = (x, y, w, h)

            analysis = self.analyze_face(image, face_box)
            name, match_score = self.recognize_face(image, row)
            analysis['face_id'] = i + 1
            analysis['name'] = name
            analysis['match_score'] = match_score
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
            
            # Draw identity label: recognized name takes priority over the
            # bare face/track number, "Unknown" if recognition ran and found
            # nobody, or just the number if recognition hasn't run yet.
            name = face.get('name')
            if name:
                id_label = f"{name} ({face.get('match_score', 0):.2f})"
            elif 'track_id' in face:
                id_label = "Unknown" if face.get('recognized_once') else f"Track {face['track_id']}"
            else:
                id_label = f"Face {face['face_id']}"

            cv2.putText(
                output,
                id_label,
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
    """
    Real-time video analysis for webcam feed.

    Capture (reading camera frames) and inference (detection + tracking +
    throttled recognition/emotion) run in separate background threads so
    the streamed video's frame rate isn't capped by how long a full
    analysis pass takes.
    """

    def __init__(self, recognizer: Optional[FaceRecognizer] = None):
        self.face_analyzer = FaceAnalyzer(recognizer=recognizer)
        self.tracker = IOUTracker(iou_threshold=0.3, max_disappeared=15)
        self.cap = None
        self.is_running = False
        self.frame_count = 0

        self._raw_frame: Optional[np.ndarray] = None
        self._raw_frame_lock = threading.Lock()

        self._annotated_frame: Optional[np.ndarray] = None
        self._analysis: Dict = {'num_faces': 0, 'faces': []}
        self._result_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None

    def start_camera(self, camera_id: int = 0) -> bool:
        """Start camera capture and the background processing threads"""
        try:
            self.cap = cv2.VideoCapture(camera_id)

            if not self.cap.isOpened():
                return False

            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.tracker.reset()
            self.frame_count = 0
            with self._raw_frame_lock:
                self._raw_frame = None
            with self._result_lock:
                self._annotated_frame = None
                self._analysis = {'num_faces': 0, 'faces': []}

            self.is_running = True
            self._stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._capture_thread.start()
            self._inference_thread.start()

            print("Camera started")
            return True

        except Exception as e:
            print(f"Camera start failed: {e}")
            return False

    def stop_camera(self):
        """Stop the background threads and release the camera"""
        self.is_running = False
        self._stop_event.set()

        for t in (self._capture_thread, self._inference_thread):
            if t is not None:
                t.join(timeout=2)
        self._capture_thread = None
        self._inference_thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        print("Camera stopped")

    def _capture_loop(self):
        """Continuously read camera frames as fast as the camera allows"""
        while not self._stop_event.is_set():
            cap = self.cap
            if cap is None:
                break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._raw_frame_lock:
                self._raw_frame = frame

    def _inference_loop(self):
        """Continuously analyze the most recent frame at whatever rate the AI workload supports"""
        while not self._stop_event.is_set():
            with self._raw_frame_lock:
                frame = None if self._raw_frame is None else self._raw_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            self.frame_count += 1
            annotated_frame, analysis = self._process_frame(frame)

            with self._result_lock:
                self._annotated_frame = annotated_frame
                self._analysis = analysis

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect, track, and (throttled) recognize/emotion-classify faces in one frame"""
        raw_faces = self.face_analyzer.detect_faces_raw(frame)
        img_h, img_w = frame.shape[:2]

        boxes = []
        for row in raw_faces:
            x, y, w, h = row[:4]
            x = max(0, int(round(x)))
            y = max(0, int(round(y)))
            w = min(int(round(w)), img_w - x)
            h = min(int(round(h)), img_h - y)
            boxes.append((x, y, w, h))

        tracks = self.tracker.update(boxes)

        for track in tracks:
            if track.detection_index is None:
                continue  # coasting on its last known box this frame

            due_for_refresh = (
                track.last_recognized_frame == -1
                or self.frame_count - track.last_recognized_frame >= RECOGNIZE_EVERY_N_FRAMES
            )
            if not due_for_refresh:
                continue

            row = raw_faces[track.detection_index]
            emotion_result = self.face_analyzer.analyze_face(frame, track.box)
            name, score = self.face_analyzer.recognize_face(frame, row)

            track.dominant_emotion = emotion_result['dominant_emotion']
            track.emotion_confidence = emotion_result['confidence']
            track.emotion_scores = emotion_result['emotion_scores']
            track.name = name
            track.match_score = score
            track.last_recognized_frame = self.frame_count

        faces = [
            {
                'track_id': track.id,
                'box': track.box,
                'dominant_emotion': track.dominant_emotion or 'neutral',
                'confidence': track.emotion_confidence,
                'emotion_scores': track.emotion_scores or {e: 0 for e in self.face_analyzer.emotions},
                'name': track.name,
                'match_score': track.match_score,
                'recognized_once': track.last_recognized_frame != -1,
            }
            for track in tracks
        ]

        analysis = {
            'num_faces': len(faces),
            'faces': faces,
            'image_shape': frame.shape,
        }

        annotated_frame = self.face_analyzer.draw_analysis(frame, analysis)
        return annotated_frame, analysis

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Get the most recently processed frame + analysis. Capture and
        inference already run continuously in background threads, so this
        just reads the latest buffered result instead of blocking on a
        fresh camera read + analysis pass.
        """
        if not self.is_running:
            return None, None

        with self._result_lock:
            if self._annotated_frame is None:
                return None, None
            return self._annotated_frame.copy(), self._analysis

    def get_latest_analysis(self) -> Dict:
        """Thread-safe accessor for the latest analysis, for the /get_analysis route"""
        with self._result_lock:
            return self._analysis

    def save_snapshot(self, frame: np.ndarray, filepath: str) -> bool:
        """Save current frame as image"""
        try:
            cv2.imwrite(filepath, frame)
            return True
        except Exception as e:
            print(f"Save failed: {e}")
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
    
    print("\nFace analyzer ready")
    print("Use VideoAnalyzer class for real-time camera analysis")
