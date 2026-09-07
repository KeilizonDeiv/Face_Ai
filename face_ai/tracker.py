"""
Lightweight IOU-based multi-face tracker.

Assigns a stable track ID to each detected face across frames by greedily
matching new detections to existing tracks using intersection-over-union
(IOU), instead of the previous behaviour of assigning a fresh ID from
enumeration order every single frame.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# Number of frame-to-frame motion samples averaged into a liveness verdict,
# and the mean-pixel-diff (0-255 grayscale scale) below which a face is
# flagged as suspiciously static (e.g. a printed photo held up to the
# camera). This is a lightweight heuristic, not real anti-spoofing — it
# only catches the "perfectly still photo" case, not a replayed video or a
# photo someone is subtly moving.
LIVENESS_HISTORY_LEN = 30
LIVENESS_MOTION_THRESHOLD = 1.5


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class Track:
    """A single tracked face, persisted across frames."""

    def __init__(self, track_id: int, box: Tuple[int, int, int, int], detection_index: int):
        self.id = track_id
        self.box = box
        self.hits = 1
        self.frames_since_seen = 0

        # Index into this frame's detection list this track is currently
        # matched to (None if it wasn't matched to any detection this frame,
        # i.e. it's coasting on its last known box). Lets callers look up the
        # raw YuNet row (with landmarks) for a freshly-matched track.
        self.detection_index: Optional[int] = detection_index

        # Recognition/emotion state carried forward between throttled refreshes
        self.name: Optional[str] = None
        self.match_score: float = 0.0
        self.dominant_emotion: Optional[str] = None
        self.emotion_confidence: float = 0.0
        self.emotion_scores: Dict[str, float] = {}
        self.last_recognized_frame: int = -1

        # Liveness bookkeeping (see push_liveness_sample). The thumbnail is
        # an opaque small grayscale crop from the previous frame — this
        # module stays free of an OpenCV/numpy dependency and just holds
        # whatever face_analyzer.py hands it.
        self.liveness_history: "deque[float]" = deque(maxlen=LIVENESS_HISTORY_LEN)
        self.liveness_prev_thumb: Optional[Any] = None
        self.liveness_score: Optional[float] = None
        self.liveness_status: str = 'checking'  # 'checking' | 'live' | 'low_motion'

    def update_box(self, box: Tuple[int, int, int, int], detection_index: int):
        self.box = box
        self.detection_index = detection_index
        self.hits += 1
        self.frames_since_seen = 0

    def push_liveness_sample(self, motion: float):
        """Record one frame-to-frame motion sample; classify live/low_motion once enough have accumulated."""
        self.liveness_history.append(motion)
        if len(self.liveness_history) >= LIVENESS_HISTORY_LEN:
            self.liveness_score = sum(self.liveness_history) / len(self.liveness_history)
            self.liveness_status = 'live' if self.liveness_score >= LIVENESS_MOTION_THRESHOLD else 'low_motion'

    @property
    def liveness_progress(self) -> float:
        """Fraction (0-1) of the way through collecting enough samples for a liveness verdict."""
        return min(1.0, len(self.liveness_history) / LIVENESS_HISTORY_LEN)


class IOUTracker:
    """Greedy IOU-matching tracker for keeping stable face IDs across frames."""

    def __init__(self, iou_threshold: float = 0.3, max_disappeared: int = 15):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.tracks: List[Track] = []
        self._next_id = 1

    def reset(self):
        """Clear all tracks and restart ID numbering (call on new camera session)."""
        self.tracks = []
        self._next_id = 1

    def update(self, boxes: List[Tuple[int, int, int, int]]) -> List[Track]:
        """
        Match this frame's detected boxes against existing tracks, update
        matched tracks in place, create new tracks for unmatched boxes, and
        drop tracks that have disappeared for too long.

        Returns the current list of live tracks.
        """
        candidates = []
        for t_idx, track in enumerate(self.tracks):
            for b_idx, box in enumerate(boxes):
                iou = _iou(track.box, box)
                if iou >= self.iou_threshold:
                    candidates.append((iou, t_idx, b_idx))

        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_tracks = set()
        matched_boxes = set()
        for iou, t_idx, b_idx in candidates:
            if t_idx in matched_tracks or b_idx in matched_boxes:
                continue
            self.tracks[t_idx].update_box(boxes[b_idx], b_idx)
            matched_tracks.add(t_idx)
            matched_boxes.add(b_idx)

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.frames_since_seen += 1
                track.detection_index = None

        for b_idx, box in enumerate(boxes):
            if b_idx not in matched_boxes:
                self.tracks.append(Track(self._next_id, box, b_idx))
                self._next_id += 1

        self.tracks = [
            t for t in self.tracks if t.frames_since_seen <= self.max_disappeared
        ]

        return self.tracks
