"""
Texture Analysis Module.

Analyzes texture variance and local contrast in faces to detect
over-smoothed or over-sharpened deepfake artifacts.
"""

from typing import List, Dict
import numpy as np

from .base import to_grayscale, CV2_AVAILABLE

if CV2_AVAILABLE:
    import cv2


class TextureAnalyzer:
    """
    Analyze texture characteristics in face images.

    Detects:
    - Over-smoothed skin (GAN artifacts)
    - Over-sharpened/processed faces
    - Abnormal local contrast patterns
    """

    def __init__(
        self,
        variance_low: float = 20.0,
        variance_high: float = 200.0,
        local_contrast_high: float = 7.5
    ):
        """
        Initialize texture analyzer.

        Args:
            variance_low: Below this = over-smoothed
            variance_high: Above this = over-processed
            local_contrast_high: Above this = suspicious contrast
        """
        self.variance_low = variance_low
        self.variance_high = variance_high
        self.local_contrast_high = local_contrast_high

    def analyze_texture(self, faces: List[np.ndarray]) -> Dict:
        """
        Detect texture anomalies - both over-smoothed AND over-sharpened.

        Real faces have natural texture from pores, wrinkles, etc.
        """
        variances = []

        for face in faces:
            if face.size == 0:
                continue

            gray = to_grayscale(face)

            # Focus on skin regions (center of face)
            h, w = gray.shape
            skin_region = gray[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]

            if skin_region.size == 0:
                continue

            laplacian = cv2.Laplacian(skin_region, cv2.CV_64F)
            variance = np.var(laplacian)
            variances.append(variance)

        if not variances:
            return {'anomaly_score': 0.0, 'mean_variance': 0.0, 'issue_type': 'none'}

        mean_variance = np.mean(variances)
        anomaly_score = 0.0
        issue_type = 'normal'

        if mean_variance < self.variance_low:
            ratio = mean_variance / self.variance_low
            anomaly_score = min((1.0 - ratio) * 1.2, 1.0)
            issue_type = 'over-smoothed'
        elif mean_variance > self.variance_high:
            excess = (mean_variance - self.variance_high) / self.variance_high
            anomaly_score = min(excess * 1.5, 1.0)
            issue_type = 'over-sharpened'

        return {
            'anomaly_score': anomaly_score,
            'mean_variance': mean_variance,
            'std_variance': np.std(variances),
            'min_variance': np.min(variances),
            'issue_type': issue_type
        }

    def analyze_local_contrast(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze local contrast in faces.

        Over-processed deepfakes often have unnaturally high local contrast.
        """
        from scipy import ndimage

        contrast_values = []

        for face in faces:
            if face.size == 0:
                continue

            gray = to_grayscale(face)
            local_std = ndimage.generic_filter(gray.astype(float), np.std, size=5)
            local_contrast = np.mean(local_std)
            contrast_values.append(local_contrast)

        if not contrast_values:
            return {'anomaly_score': 0.0, 'mean_contrast': 0.0}

        mean_contrast = np.mean(contrast_values)

        if mean_contrast > self.local_contrast_high:
            excess = mean_contrast - self.local_contrast_high
            anomaly_score = min(excess / 2.5, 1.0)
        else:
            anomaly_score = 0.0

        return {
            'anomaly_score': anomaly_score,
            'mean_contrast': mean_contrast,
            'std_contrast': np.std(contrast_values)
        }
