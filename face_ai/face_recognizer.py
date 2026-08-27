"""
Face recognition / enrollment engine, using OpenCV's SFace ONNX model.

This is the "training" step the app exposes to users: enrolling a person
from a handful of photos (name + photos -> stored face embeddings), and
later recognizing a live face by comparing its embedding against everyone
enrolled so far.
"""

import json
import os
import threading
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PEOPLE_DIR = os.path.join(DATA_DIR, 'people')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings.json')

# SFace's documented cosine-similarity threshold for a positive match
MATCH_THRESHOLD = 0.363


class FaceRecognizer:
    """Enrolls people from photos and recognizes faces against the enrolled set."""

    def __init__(self):
        if not os.path.exists(SFACE_MODEL_PATH):
            raise FileNotFoundError(
                f"Face recognition model not found at {SFACE_MODEL_PATH}. "
                "Run 'python download_models.py' once to fetch it."
            )
        if not os.path.exists(YUNET_MODEL_PATH):
            raise FileNotFoundError(
                f"Face detection model not found at {YUNET_MODEL_PATH}. "
                "Run 'python download_models.py' once to fetch it."
            )

        self._recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        # A dedicated small detector for enrollment photos, decoupled from
        # the live-video detector so this module works standalone.
        self._enroll_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320), score_threshold=0.6, nms_threshold=0.3
        )

        os.makedirs(PEOPLE_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)

        # RLock (not Lock): this instance is shared between the live-video
        # inference thread and Flask request threads (enrollment/recognition
        # routes), and enroll_person() re-enters the lock (embed -> index
        # update) from the same thread, which a plain Lock would deadlock on.
        self._lock = threading.RLock()
        self._embeddings: Dict[str, List[List[float]]] = {}
        self._load_index()

    # ---------- persistence ----------

    def _load_index(self):
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, 'r') as f:
                self._embeddings = json.load(f)
        else:
            self._embeddings = {}

    def _save_index_locked(self):
        """Caller must hold self._lock."""
        tmp_path = EMBEDDINGS_PATH + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(self._embeddings, f)
        os.replace(tmp_path, EMBEDDINGS_PATH)

    # ---------- embedding ----------

    def embed(self, image: np.ndarray, detection_row: np.ndarray) -> np.ndarray:
        """
        Align and embed a single face given the full image and its YuNet
        detection row (box + 5-point landmarks, from FaceDetectorYN.detect()).

        Locked because the underlying SFace net is shared between the live
        video inference thread and Flask request threads (enroll/recognize
        routes) — OpenCV DNN nets aren't guaranteed safe for concurrent
        inference calls from multiple threads.
        """
        row = np.asarray(detection_row, dtype=np.float32).reshape(1, -1)
        with self._lock:
            aligned = self._recognizer.alignCrop(image, row)
            feature = self._recognizer.feature(aligned)
        return np.asarray(feature).flatten()

    def _detect_and_embed_best_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        with self._lock:
            self._enroll_detector.setInputSize((w, h))
            _, faces = self._enroll_detector.detect(image)

        if faces is None or len(faces) == 0:
            return None

        best_row = max(faces, key=lambda r: r[-1])  # highest detection score
        return self.embed(image, best_row)

    # ---------- enrollment ----------

    def enroll_person(self, name: str, photos: List[np.ndarray]) -> Dict:
        """
        Enroll a person from a list of decoded BGR images. Each photo is
        detected + aligned + embedded independently; photos with no
        detectable face are skipped and reported back by index.
        """
        name = name.strip()
        if not name:
            raise ValueError("Name is required")

        new_embeddings = []
        failed_indices = []

        person_dir = os.path.join(PEOPLE_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        for i, photo in enumerate(photos):
            embedding = self._detect_and_embed_best_face(photo)
            if embedding is None:
                failed_indices.append(i)
                continue

            new_embeddings.append(embedding.tolist())
            photo_path = os.path.join(person_dir, f'{uuid.uuid4().hex}.jpg')
            cv2.imwrite(photo_path, photo)

        if new_embeddings:
            with self._lock:
                self._embeddings.setdefault(name, [])
                self._embeddings[name].extend(new_embeddings)
                self._save_index_locked()

        return {
            'name': name,
            'photos_enrolled': len(new_embeddings),
            'photos_failed': failed_indices,
        }

    def delete_person(self, name: str) -> bool:
        with self._lock:
            existed = self._embeddings.pop(name, None) is not None
            if existed:
                self._save_index_locked()

        person_dir = os.path.join(PEOPLE_DIR, name)
        if os.path.isdir(person_dir):
            for fname in os.listdir(person_dir):
                os.remove(os.path.join(person_dir, fname))
            os.rmdir(person_dir)

        return existed

    def list_people(self) -> List[Dict]:
        with self._lock:
            snapshot = {name: len(embs) for name, embs in self._embeddings.items()}

        people = []
        for name, num_photos in snapshot.items():
            person_dir = os.path.join(PEOPLE_DIR, name)
            thumbnail = None
            if os.path.isdir(person_dir):
                saved_photos = sorted(os.listdir(person_dir))
                if saved_photos:
                    thumbnail = saved_photos[0]
            people.append({'name': name, 'num_photos': num_photos, 'thumbnail': thumbnail})

        return people

    # ---------- recognition ----------

    def recognize(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Compare an embedding against every enrolled person's stored
        embeddings, returning the best-matching name and cosine similarity
        score if it clears MATCH_THRESHOLD, else (None, best_score_seen).
        """
        with self._lock:
            items = list(self._embeddings.items())

        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        best_name = None
        best_score = 0.0

        with self._lock:
            for name, stored_list in items:
                for stored in stored_list:
                    stored_arr = np.asarray(stored, dtype=np.float32).reshape(1, -1)
                    score = self._recognizer.match(emb, stored_arr, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_score = score
                        best_name = name

        if best_score >= MATCH_THRESHOLD:
            return best_name, best_score

        return None, best_score
