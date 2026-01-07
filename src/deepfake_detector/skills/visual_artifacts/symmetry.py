"""
Symmetry Analysis Module.

Analyzes facial symmetry to detect deepfake manipulation artifacts.
"""

from typing import List, Dict
import numpy as np

from .base import to_grayscale, CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


class SymmetryAnalyzer:
    """
    Analyze facial symmetry patterns.

    Real faces have high but not perfect symmetry.
    Some deepfakes exhibit unusual asymmetry patterns.
    """

    def __init__(self, symmetry_threshold: float = 0.82):
        """
        Initialize symmetry analyzer.

        Args:
            symmetry_threshold: Below this = asymmetry anomaly
        """
        self.symmetry_threshold = symmetry_threshold

    def analyze(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze facial symmetry anomalies.

        Args:
            faces: List of face images

        Returns:
            Analysis result with anomaly_score and metrics
        """
        symmetry_scores = []

        for face in faces:
            if face.size == 0:
                continue

            gray = to_grayscale(face)
            h, w = gray.shape

            # Split face into left and right halves
            mid = w // 2
            left_half = gray[:, :mid]
            right_half = gray[:, mid:mid + left_half.shape[1]]

            if left_half.shape[1] == 0 or right_half.shape[1] == 0:
                continue

            # Flip right half for comparison
            right_flipped = cv2.flip(right_half, 1)

            # Ensure same size
            min_w = min(left_half.shape[1], right_flipped.shape[1])
            if min_w < 10:
                continue

            left_half = left_half[:, :min_w]
            right_flipped = right_flipped[:, :min_w]

            # Compute symmetry as 1 - normalized difference
            diff = np.abs(left_half.astype(float) - right_flipped.astype(float))
            symmetry = 1.0 - np.mean(diff) / 255.0
            symmetry_scores.append(symmetry)

        if not symmetry_scores:
            return {'anomaly_score': 0.0, 'mean_symmetry': 1.0}

        mean_symmetry = np.mean(symmetry_scores)

        if mean_symmetry < self.symmetry_threshold:
            gap = self.symmetry_threshold - mean_symmetry
            anomaly_score = min(gap / 0.12, 1.0)
        else:
            anomaly_score = 0.0

        return {
            'anomaly_score': anomaly_score,
            'mean_symmetry': mean_symmetry,
            'std_symmetry': np.std(symmetry_scores),
            'min_symmetry': np.min(symmetry_scores)
        }
