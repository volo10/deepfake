"""
Motion Analysis Module.

Analyzes temporal stability and background motion to detect deepfakes.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from .base import to_grayscale, CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


class MotionAnalyzer:
    """
    Analyze motion patterns in video sequences.

    Detects:
    - Unnaturally stable frames (too little natural variation)
    - Static/frozen backgrounds while face moves
    """

    def __init__(self, frame_stability_threshold: float = 2.0):
        """
        Initialize motion analyzer.

        Args:
            frame_stability_threshold: Below this = too stable
        """
        self.frame_stability_threshold = frame_stability_threshold

    def analyze_temporal_stability(self, faces: List[np.ndarray]) -> Dict:
        """
        Check if frames are unnaturally stable.

        Real video has natural micro-movements and variations.
        """
        if len(faces) < 2:
            return {'anomaly_score': 0.0, 'mean_diff': 0.0}

        frame_diffs = []

        for i in range(1, len(faces)):
            if faces[i].size == 0 or faces[i-1].size == 0:
                continue

            prev = faces[i-1]
            curr = faces[i]

            if prev.shape != curr.shape:
                continue

            prev_gray = to_grayscale(prev)
            curr_gray = to_grayscale(curr)

            diff = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
            frame_diffs.append(diff)

        if not frame_diffs:
            return {'anomaly_score': 0.0, 'mean_diff': 0.0}

        mean_diff = np.mean(frame_diffs)

        if mean_diff < self.frame_stability_threshold:
            anomaly_score = min(
                (self.frame_stability_threshold - mean_diff) / 1.5, 0.5
            )
        else:
            anomaly_score = 0.0

        return {
            'anomaly_score': anomaly_score,
            'mean_diff': mean_diff,
            'std_diff': np.std(frame_diffs)
        }

    def analyze_background_motion(
        self,
        frames: List[np.ndarray],
        boxes: List[Optional[Tuple[int, int, int, int]]]
    ) -> Dict:
        """
        Analyze background motion consistency.

        In real videos, background should have natural motion relative to face.
        Deepfakes often have static/frozen backgrounds while the face moves.
        """
        if len(frames) < 10:
            return {'anomaly_score': 0.0, 'bg_motion': 0.0, 'face_motion': 0.0}

        face_motions = []
        bg_motions = []

        prev_frame = None
        prev_box = None

        for frame, box in zip(frames, boxes):
            if frame is None or box is None or prev_frame is None or prev_box is None:
                prev_frame = frame
                prev_box = box
                continue

            gray = to_grayscale(frame)
            prev_gray = to_grayscale(prev_frame)

            x, y, w, h = box
            px, py, pw, ph = prev_box

            # Face region motion
            face_curr = gray[y:y+h, x:x+w]
            face_prev = prev_gray[py:py+ph, px:px+pw]

            if face_curr.size > 0 and face_prev.size > 0:
                target_h = min(face_curr.shape[0], face_prev.shape[0])
                target_w = min(face_curr.shape[1], face_prev.shape[1])
                if target_h > 10 and target_w > 10:
                    face_curr_resized = cv2.resize(face_curr, (target_w, target_h))
                    face_prev_resized = cv2.resize(face_prev, (target_w, target_h))
                    face_diff = np.mean(np.abs(
                        face_curr_resized.astype(float) - face_prev_resized.astype(float)
                    ))
                    face_motions.append(face_diff)

            # Background region motion (exclude face area)
            mask = np.ones(gray.shape, dtype=bool)
            mask[y:y+h, x:x+w] = False

            prev_mask = np.ones(prev_gray.shape, dtype=bool)
            prev_mask[py:py+ph, px:px+pw] = False

            combined_mask = mask & prev_mask
            if np.sum(combined_mask) > 100:
                bg_diff = np.mean(np.abs(
                    gray[combined_mask].astype(float) - prev_gray[combined_mask].astype(float)
                ))
                bg_motions.append(bg_diff)

            prev_frame = frame
            prev_box = box

        if not face_motions or not bg_motions:
            return {'anomaly_score': 0.0, 'bg_motion': 0.0, 'face_motion': 0.0}

        mean_face_motion = np.mean(face_motions)
        mean_bg_motion = np.mean(bg_motions)

        anomaly_score = 0.0

        if mean_face_motion > 3.0 and mean_bg_motion < 1.0:
            anomaly_score = min((mean_face_motion - mean_bg_motion) / 10.0, 1.0)
        elif mean_face_motion > 2.0 and mean_bg_motion < mean_face_motion * 0.2:
            ratio = mean_bg_motion / (mean_face_motion + 1e-8)
            anomaly_score = min((1.0 - ratio) * 0.7, 0.8)

        return {
            'anomaly_score': anomaly_score,
            'face_motion': mean_face_motion,
            'bg_motion': mean_bg_motion,
            'motion_ratio': mean_bg_motion / (mean_face_motion + 1e-8)
        }
