"""
Edge Analysis Module.

Analyzes edge density and boundary sharpness to detect deepfake artifacts.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from .base import to_grayscale, CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


class EdgeAnalyzer:
    """
    Analyze edge characteristics in face images.

    Detects:
    - Abnormal edge density (too low or too high)
    - Unnatural face boundary sharpness
    """

    def __init__(
        self,
        density_low: float = 0.025,
        density_high: float = 0.09,
        boundary_sharpness_threshold: float = 7.0
    ):
        """
        Initialize edge analyzer.

        Args:
            density_low: Below this = lack of detail
            density_high: Above this = over-sharpened
            boundary_sharpness_threshold: Above this = artificial boundary
        """
        self.density_low = density_low
        self.density_high = density_high
        self.boundary_sharpness_threshold = boundary_sharpness_threshold

    def analyze_density(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze edge density - detect both too low AND too high.

        Real faces have natural edges from features, wrinkles, etc.
        """
        densities = []

        for face in faces:
            if face.size == 0:
                continue

            gray = to_grayscale(face)
            edges = cv2.Canny(gray, 50, 150)
            density = np.mean(edges) / 255.0
            densities.append(density)

        if not densities:
            return {'anomaly_score': 0.0, 'mean_density': 0.0, 'issue_type': 'none'}

        mean_density = np.mean(densities)
        anomaly_score = 0.0
        issue_type = 'normal'

        if mean_density < self.density_low:
            gap = self.density_low - mean_density
            anomaly_score = min(gap / 0.015, 1.0)
            issue_type = 'too few edges (smoothed)'
        elif mean_density > self.density_high:
            excess = mean_density - self.density_high
            anomaly_score = min(excess / 0.03, 1.0)
            issue_type = 'too many edges (sharpened)'

        return {
            'anomaly_score': anomaly_score,
            'mean_density': mean_density,
            'std_density': np.std(densities),
            'issue_type': issue_type
        }

    def analyze_boundary_sharpness(
        self,
        faces: List[np.ndarray],
        frames: Optional[List[np.ndarray]],
        boxes: Optional[List[Tuple[int, int, int, int]]]
    ) -> Dict:
        """
        Analyze face boundary sharpness.

        Deepfakes often have unnaturally sharp or artificial boundaries.
        """
        if not frames or not boxes:
            return {'anomaly_score': 0.0, 'mean_sharpness': 0.0}

        sharpness_values = []

        for frame, box in zip(frames, boxes):
            if frame is None or box is None or frame.size == 0:
                continue

            x, y, w, h = box

            # Create face mask
            face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            face_mask[y:y+h, x:x+w] = 255

            # Get boundary region
            boundary = cv2.Canny(face_mask, 100, 200)
            dilated = cv2.dilate(boundary, np.ones((5, 5), np.uint8))

            if np.sum(dilated) == 0:
                continue

            gray = to_grayscale(frame)
            laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)

            # Get sharpness at boundary
            boundary_sharpness = np.mean(np.abs(laplacian[dilated > 0]))
            sharpness_values.append(boundary_sharpness)

        if not sharpness_values:
            return {'anomaly_score': 0.0, 'mean_sharpness': 0.0}

        mean_sharpness = np.mean(sharpness_values)

        if mean_sharpness > self.boundary_sharpness_threshold:
            excess = mean_sharpness - self.boundary_sharpness_threshold
            anomaly_score = min(excess / 5.0, 1.0)
            if mean_sharpness > 15.0:
                anomaly_score = min(anomaly_score + 0.3, 1.0)
            if mean_sharpness > 25.0:
                anomaly_score = min(anomaly_score + 0.2, 1.0)
        else:
            anomaly_score = 0.0

        return {
            'anomaly_score': anomaly_score,
            'mean_sharpness': mean_sharpness,
            'max_sharpness': np.max(sharpness_values) if sharpness_values else 0.0
        }
