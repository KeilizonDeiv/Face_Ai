"""
Face recognition / enrollment engine, using OpenCV's SFace ONNX model.

This is the "training" step the app exposes to users: enrolling a person
from a handful of photos (name + photos -> stored face embeddings), and
later recognizing a live face by comparing its embedding against everyone
enrolled so far.
"""

import json
import os
import re
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from werkzeug.utils import secure_filename

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PEOPLE_DIR = os.path.join(DATA_DIR, 'people')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings.json')
LAST_SEEN_PATH = os.path.join(DATA_DIR, 'last_seen.json')

# SFace's documented cosine-similarity threshold for a positive match
MATCH_THRESHOLD = 0.363

# Names double as directory names under PEOPLE_DIR, so they're restricted to
# a safe charset up front — rejecting "../"-style names outright is far less
# error-prone than trying to sanitize them after the fact.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9 ._'-]{1,100}$")


def _validate_name(name: str) -> str:
    """Return the trimmed name if it's safe to use as a directory/key, else raise ValueError."""
    name = name.strip()
    if not name or not _NAME_PATTERN.match(name) or '..' in name:
        raise ValueError(
            "Name may only contain letters, numbers, spaces, and . _ ' - "
            "(1-100 characters)"
        )
    return name


def _person_dir(name: str) -> str:
    """Resolve a validated name to its directory, with a belt-and-suspenders containment check."""
    person_dir = os.path.join(PEOPLE_DIR, name)
    real_people_dir = os.path.realpath(PEOPLE_DIR)
    real_person_dir = os.path.realpath(person_dir)
    if os.path.commonpath([real_people_dir, real_person_dir]) != real_people_dir:
        raise ValueError("Invalid name")
    return person_dir


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
        # name -> [{'photo': filename_or_None, 'embedding': [floats]}, ...]
        self._embeddings: Dict[str, List[Dict]] = {}
        # name -> unix timestamp of the last time they were recognized.
        # Kept in a separate small file from the (potentially large)
        # embeddings so recording a sighting doesn't rewrite every stored
        # face vector on every throttled recognition hit.
        self._last_seen: Dict[str, float] = {}
        # Runtime-adjustable via /settings; starts at the module default.
        self.match_threshold: float = MATCH_THRESHOLD
        self._load_index()

    # ---------- persistence ----------

    def _load_index(self):
        if os.path.exists(EMBEDDINGS_PATH):
            with open(EMBEDDINGS_PATH, 'r') as f:
                raw = json.load(f)
            # Migrate the legacy format (name -> list of raw embedding
            # vectors, no photo linkage) to the current one item-by-item, so
            # photos enrolled before per-photo delete existed still load.
            self._embeddings = {
                name: [
                    entry if isinstance(entry, dict) else {'photo': None, 'embedding': entry}
                    for entry in entries
                ]
                for name, entries in raw.items()
            }
        else:
            self._embeddings = {}

        if os.path.exists(LAST_SEEN_PATH):
            with open(LAST_SEEN_PATH, 'r') as f:
                self._last_seen = json.load(f)
        else:
            self._last_seen = {}

    def _save_index_locked(self):
        """Caller must hold self._lock."""
        tmp_path = EMBEDDINGS_PATH + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(self._embeddings, f)
        os.replace(tmp_path, EMBEDDINGS_PATH)

    def _save_last_seen_locked(self):
        """Caller must hold self._lock."""
        tmp_path = LAST_SEEN_PATH + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(self._last_seen, f)
        os.replace(tmp_path, LAST_SEEN_PATH)

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
        name = _validate_name(name)

        new_entries = []
        failed_indices = []

        person_dir = _person_dir(name)
        os.makedirs(person_dir, exist_ok=True)

        for i, photo in enumerate(photos):
            embedding = self._detect_and_embed_best_face(photo)
            if embedding is None:
                failed_indices.append(i)
                continue

            photo_filename = f'{uuid.uuid4().hex}.jpg'
            cv2.imwrite(os.path.join(person_dir, photo_filename), photo)
            new_entries.append({'photo': photo_filename, 'embedding': embedding.tolist()})

        if new_entries:
            with self._lock:
                self._embeddings.setdefault(name, [])
                self._embeddings[name].extend(new_entries)
                self._save_index_locked()

        return {
            'name': name,
            'photos_enrolled': len(new_entries),
            'photos_failed': failed_indices,
        }

    def delete_person(self, name: str) -> bool:
        try:
            name = _validate_name(name)
        except ValueError:
            # Not a name this recognizer could ever have enrolled (e.g. a
            # path-traversal attempt) — nothing to delete, and importantly
            # nothing to touch on disk.
            return False

        with self._lock:
            existed = self._embeddings.pop(name, None) is not None
            if existed:
                self._save_index_locked()
            if self._last_seen.pop(name, None) is not None:
                self._save_last_seen_locked()

        person_dir = _person_dir(name)
        if os.path.isdir(person_dir):
            for fname in os.listdir(person_dir):
                os.remove(os.path.join(person_dir, fname))
            os.rmdir(person_dir)

        return existed

    def get_photos(self, name: str) -> List[str]:
        """Filenames of photos currently on file for an enrolled person (empty if unknown)."""
        try:
            name = _validate_name(name)
        except ValueError:
            return []
        with self._lock:
            entries = self._embeddings.get(name, [])
            return [e['photo'] for e in entries if e.get('photo')]

    def delete_photo(self, name: str, filename: str) -> Optional[bool]:
        """
        Remove one enrolled photo (and its embedding) from a person. If it
        was their last photo, the person is removed entirely — an entry
        with zero embeddings can never be recognized anyway.

        Returns None if that filename wasn't found, True if deleted and the
        person still has other photos, False if deleted and the person was
        removed entirely as a result.
        """
        try:
            name = _validate_name(name)
        except ValueError:
            return None
        filename = secure_filename(filename)

        with self._lock:
            entries = self._embeddings.get(name)
            if not entries:
                return None

            remaining = [e for e in entries if e.get('photo') != filename]
            if len(remaining) == len(entries):
                return None  # that filename wasn't one of theirs

            person_removed = not remaining
            if remaining:
                self._embeddings[name] = remaining
            else:
                self._embeddings.pop(name, None)
                self._last_seen.pop(name, None)
                self._save_last_seen_locked()
            self._save_index_locked()

        person_dir = _person_dir(name)
        photo_path = os.path.join(person_dir, filename)
        if os.path.isfile(photo_path):
            os.remove(photo_path)

        if person_removed and os.path.isdir(person_dir) and not os.listdir(person_dir):
            os.rmdir(person_dir)

        return not person_removed

    def list_people(self) -> List[Dict]:
        with self._lock:
            snapshot = {name: list(entries) for name, entries in self._embeddings.items()}
            last_seen = dict(self._last_seen)

        people = []
        for name, entries in snapshot.items():
            thumbnail = next((e['photo'] for e in entries if e.get('photo')), None)
            people.append({
                'name': name,
                'num_photos': len(entries),
                'thumbnail': thumbnail,
                'last_seen': last_seen.get(name),
            })

        return people

    # ---------- recognition ----------

    def recognize(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Compare an embedding against every enrolled person's stored
        embeddings, returning the best-matching name and cosine similarity
        score if it clears match_threshold, else (None, best_score_seen).
        """
        with self._lock:
            items = list(self._embeddings.items())

        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        best_name = None
        best_score = 0.0

        with self._lock:
            for name, entries in items:
                for entry in entries:
                    stored_arr = np.asarray(entry['embedding'], dtype=np.float32).reshape(1, -1)
                    score = self._recognizer.match(emb, stored_arr, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_score = score
                        best_name = name

        if best_score >= self.match_threshold:
            with self._lock:
                self._last_seen[best_name] = time.time()
                self._save_last_seen_locked()
            return best_name, best_score

        return None, best_score
