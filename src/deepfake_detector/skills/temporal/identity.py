"""
Identity Stability Analysis Module.

Analyzes embedding stability over time to detect identity drift.
"""

from typing import List, Dict
import numpy as np


class IdentityAnalyzer:
    """
    Analyze identity embedding stability over time.

    Identity drift indicates potential face swapping manipulation.
    """

    def __init__(self, distance_threshold: float = 0.4):
        """
        Initialize identity analyzer.

        Args:
            distance_threshold: Threshold for significant identity change
        """
        self.distance_threshold = distance_threshold

    def analyze(self, embeddings: List[np.ndarray]) -> Dict:
        """
        Analyze identity stability over time.

        Args:
            embeddings: List of face embeddings

        Returns:
            Dict with stability metrics
        """
        if len(embeddings) < 2:
            return {"stability": 1.0, "drift": 0.0, "variance": 0.0}

        embeddings_array = np.array(embeddings)

        # Compute centroid
        centroid = np.mean(embeddings_array, axis=0)

        # Distance from centroid
        distances = [
            self._cosine_distance(e, centroid)
            for e in embeddings_array
        ]

        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        variance = np.var(distances)

        # Stability score
        stability = 1.0 - min(mean_distance / 0.5, 1.0)

        # Detect drift (change in centroid over windows)
        drift = self._compute_drift(embeddings_array)

        return {
            "stability": stability,
            "drift": drift,
            "mean_distance": mean_distance,
            "max_distance": max_distance,
            "variance": variance
        }

    def _compute_drift(self, embeddings_array: np.ndarray) -> float:
        """Compute identity drift across time windows."""
        window_size = min(30, len(embeddings_array) // 3)
        if window_size <= 5:
            return 0.0

        window_centroids = []
        for i in range(0, len(embeddings_array) - window_size, window_size // 2):
            window = embeddings_array[i:i + window_size]
            window_centroids.append(np.mean(window, axis=0))

        if len(window_centroids) <= 1:
            return 0.0

        drift_distances = [
            self._cosine_distance(window_centroids[i], window_centroids[i+1])
            for i in range(len(window_centroids) - 1)
        ]
        return np.max(drift_distances)

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 1.0

        return 1.0 - dot / (norm_a * norm_b)
