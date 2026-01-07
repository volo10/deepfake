"""
Color Analysis Module.

Analyzes color consistency and saturation patterns to detect deepfakes.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from .base import CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


class ColorAnalyzer:
    """
    Analyze color characteristics in face images.

    Detects:
    - Face-neck color mismatches
    - Abnormal saturation patterns
    """

    def __init__(
        self,
        color_diff_threshold: float = 50.0,
        saturation_variance_high: float = 1400
    ):
        """
        Initialize color analyzer.

        Args:
            color_diff_threshold: Above this = color mismatch
            saturation_variance_high: Above this = color manipulation
        """
        self.color_diff_threshold = color_diff_threshold
        self.saturation_variance_high = saturation_variance_high

    def analyze_consistency(
        self,
        frames: List[np.ndarray],
        boxes: List[Tuple[int, int, int, int]]
    ) -> Dict:
        """
        Check color consistency between face and neck/background.

        Deepfakes often have color mismatches at boundaries.
        """
        color_diffs = []

        for frame, box in zip(frames, boxes):
            if frame.size == 0 or box is None:
                continue

            x, y, w, h = box

            # Get face region
            face_region = frame[y:y+h, x:x+w]

            # Get neck region (below face)
            neck_y_start = y + h
            neck_y_end = min(neck_y_start + h//3, frame.shape[0])
            neck_x_start = max(0, x + w//4)
            neck_x_end = min(x + 3*w//4, frame.shape[1])

            if neck_y_end <= neck_y_start or neck_x_end <= neck_x_start:
                continue

            neck_region = frame[neck_y_start:neck_y_end, neck_x_start:neck_x_end]

            if face_region.size == 0 or neck_region.size == 0:
                continue

            # Compare mean colors
            face_color = np.mean(face_region, axis=(0, 1))
            neck_color = np.mean(neck_region, axis=(0, 1))

            color_diff = np.linalg.norm(face_color - neck_color)
            color_diffs.append(color_diff)

        if not color_diffs:
            return {'anomaly_score': 0.0, 'mean_diff': 0.0}

        mean_diff = np.mean(color_diffs)

        if mean_diff > self.color_diff_threshold:
            gap = mean_diff - self.color_diff_threshold
            anomaly_score = min(gap / 25.0, 1.0)
        else:
            anomaly_score = 0.0

        return {
            'anomaly_score': anomaly_score,
            'mean_diff': mean_diff,
            'max_diff': np.max(color_diffs) if color_diffs else 0.0
        }

    def analyze_saturation(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze color saturation variance.

        Deepfakes often have abnormal saturation patterns due to
        color manipulation during face swapping.
        """
        sat_variances = []

        for face in faces:
            if face.size == 0 or len(face.shape) != 3:
                continue

            hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]

            sat_variance = np.var(saturation)
            sat_variances.append(sat_variance)

        if not sat_variances:
            return {'anomaly_score': 0.0, 'mean_variance': 0.0}

        mean_variance = np.mean(sat_variances)

        if mean_variance > self.saturation_variance_high:
            excess = mean_variance - self.saturation_variance_high
            anomaly_score = min(excess / 1000, 1.0)
        else:
            anomaly_score = 0.0

        return {
            'anomaly_score': anomaly_score,
            'mean_variance': mean_variance,
            'max_variance': np.max(sat_variances) if sat_variances else 0.0
        }
