"""
Blink Analysis Module.

Analyzes blink patterns to detect temporal anomalies in deepfakes.
"""

from typing import List, Dict
import numpy as np


class BlinkAnalyzer:
    """
    Analyze blink patterns in facial video sequences.

    Normal blink rate: 15-20 per minute.
    Abnormal patterns may indicate manipulation.
    """

    def __init__(self, ear_threshold: float = 0.2):
        """
        Initialize blink analyzer.

        Args:
            ear_threshold: Eye Aspect Ratio threshold for blink detection
        """
        self.ear_threshold = ear_threshold

    def analyze(
        self,
        landmarks_sequence: List[np.ndarray],
        fps: float
    ) -> Dict:
        """
        Analyze blink patterns for anomalies.

        Args:
            landmarks_sequence: Facial landmarks over time
            fps: Video frame rate

        Returns:
            Dict with blink analysis results
        """
        if len(landmarks_sequence) < 30 or fps <= 0:
            return {"blink_rate": 0, "anomaly_score": 0, "blinks": []}

        # Compute Eye Aspect Ratio over time
        ear_values = []
        for landmarks in landmarks_sequence:
            if len(landmarks) >= 468:  # MediaPipe
                ear = self._compute_ear_mediapipe(landmarks)
            elif len(landmarks) >= 68:  # dlib
                ear = self._compute_ear_dlib(landmarks)
            else:
                ear = 0.3
            ear_values.append(ear)

        ear_values = np.array(ear_values)
        blinks = self._detect_blinks(ear_values)

        # Calculate blink rate
        duration_minutes = len(landmarks_sequence) / fps / 60
        blink_rate = len(blinks) / duration_minutes if duration_minutes > 0 else 0

        # Analyze blink durations
        blink_durations = [b["duration"] for b in blinks] if blinks else []

        # Anomaly scoring
        anomaly_score, reasons = self._compute_anomaly_score(
            blink_rate, blink_durations
        )

        return {
            "blink_rate": blink_rate,
            "num_blinks": len(blinks),
            "blinks": blinks,
            "anomaly_score": anomaly_score,
            "reasons": reasons,
            "ear_values": ear_values.tolist()
        }

    def _compute_ear_mediapipe(self, landmarks: np.ndarray) -> float:
        """Compute Eye Aspect Ratio from MediaPipe landmarks."""
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [263, 387, 385, 362, 380, 373]

        def eye_aspect_ratio(pts):
            v1 = np.linalg.norm(landmarks[pts[1]] - landmarks[pts[5]])
            v2 = np.linalg.norm(landmarks[pts[2]] - landmarks[pts[4]])
            h = np.linalg.norm(landmarks[pts[0]] - landmarks[pts[3]])
            return (v1 + v2) / (2.0 * h) if h > 0 else 0

        return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

    def _compute_ear_dlib(self, landmarks: np.ndarray) -> float:
        """Compute Eye Aspect Ratio from dlib landmarks."""
        def eye_aspect_ratio(eye_pts):
            v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
            v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
            h = np.linalg.norm(eye_pts[0] - eye_pts[3])
            return (v1 + v2) / (2.0 * h) if h > 0 else 0

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

    def _detect_blinks(self, ear_values: np.ndarray) -> List[Dict]:
        """Detect blink events from EAR time series."""
        blinks = []
        in_blink = False
        blink_start = 0

        for i, ear in enumerate(ear_values):
            if ear < self.ear_threshold:
                if not in_blink:
                    in_blink = True
                    blink_start = i
            else:
                if in_blink:
                    blink_end = i
                    duration = (blink_end - blink_start) / 30.0

                    if 0.05 < duration < 0.5:
                        blinks.append({
                            "start": blink_start,
                            "end": blink_end,
                            "duration": duration
                        })
                    in_blink = False

        return blinks

    def _compute_anomaly_score(
        self,
        blink_rate: float,
        blink_durations: List[float]
    ) -> tuple:
        """Compute anomaly score based on blink patterns."""
        anomaly_score = 0.0
        reasons = []

        if blink_rate < 5:
            anomaly_score += 0.4
            reasons.append("very_low_blink_rate")
        elif blink_rate < 10:
            anomaly_score += 0.2
            reasons.append("low_blink_rate")
        elif blink_rate > 40:
            anomaly_score += 0.3
            reasons.append("high_blink_rate")

        if blink_durations and len(blink_durations) > 2:
            duration_std = np.std(blink_durations)
            if duration_std > 0.15:
                anomaly_score += 0.2
                reasons.append("inconsistent_blink_duration")

        return min(anomaly_score, 1.0), reasons
