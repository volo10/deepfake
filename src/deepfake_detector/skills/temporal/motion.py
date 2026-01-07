"""
Motion Analysis Module.

Analyzes expression flow, texture stability, and motion smoothness.
"""

from typing import List, Dict
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class MotionAnalyzer:
    """
    Analyze motion patterns in facial video sequences.

    Detects:
    - Expression discontinuities
    - Texture flickering
    - Motion jitter
    """

    def analyze_expression_flow(
        self,
        landmarks_sequence: List[np.ndarray]
    ) -> Dict:
        """
        Analyze expression transition smoothness.

        Args:
            landmarks_sequence: Facial landmarks over time

        Returns:
            Dict with smoothness metrics
        """
        if len(landmarks_sequence) < 3:
            return {"smoothness": 1.0, "discontinuities": []}

        landmarks_array = np.array([
            lm[:68] if len(lm) >= 68 else lm
            for lm in landmarks_sequence
        ])

        if len(landmarks_array[0]) == 0:
            return {"smoothness": 1.0, "discontinuities": []}

        velocities = np.diff(landmarks_array, axis=0)
        accelerations = np.diff(velocities, axis=0)

        acc_magnitudes = np.linalg.norm(accelerations, axis=-1)
        mean_acc = np.mean(acc_magnitudes, axis=-1)

        threshold = np.mean(mean_acc) + 2 * np.std(mean_acc)
        discontinuities = np.where(mean_acc > threshold)[0].tolist()

        smoothness = 1.0 - min(len(discontinuities) / len(mean_acc), 1.0)

        return {
            "smoothness": smoothness,
            "discontinuities": discontinuities,
            "mean_acceleration": float(np.mean(mean_acc))
        }

    def analyze_texture_stability(
        self,
        face_sequence: List[np.ndarray]
    ) -> Dict:
        """
        Analyze texture stability across frames.

        Args:
            face_sequence: List of face images

        Returns:
            Dict with stability metrics
        """
        if len(face_sequence) < 2:
            return {"stability": 1.0, "flicker_score": 0.0}

        hf_sequence = []
        for face in face_sequence:
            gray = face if len(face.shape) == 2 else np.mean(face, axis=2)

            if CV2_AVAILABLE:
                blurred = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
            else:
                blurred = gray.astype(np.float32)

            hf = gray.astype(np.float32) - blurred
            hf_sequence.append(hf)

        diffs = []
        for i in range(len(hf_sequence) - 1):
            diff = np.abs(hf_sequence[i+1] - hf_sequence[i])
            diffs.append(np.mean(diff))

        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        stability = 1.0 - min(mean_diff / 30.0, 1.0)

        return {
            "stability": stability,
            "flicker_score": mean_diff,
            "max_flicker": max_diff
        }

    def analyze_motion_smoothness(
        self,
        landmarks_sequence: List[np.ndarray]
    ) -> Dict:
        """
        Analyze head/face motion smoothness.

        Args:
            landmarks_sequence: Facial landmarks over time

        Returns:
            Dict with motion smoothness metrics
        """
        if len(landmarks_sequence) < 3:
            return {"smoothness": 1.0, "jitter_score": 0.0}

        positions = []
        for landmarks in landmarks_sequence:
            if len(landmarks) >= 468:  # MediaPipe nose tip
                positions.append(landmarks[1][:2])
            elif len(landmarks) >= 68:  # dlib nose tip
                positions.append(landmarks[30])
            elif len(landmarks) > 0:
                positions.append(np.mean(landmarks, axis=0)[:2])

        if len(positions) < 3:
            return {"smoothness": 1.0, "jitter_score": 0.0}

        positions = np.array(positions)

        velocities = np.diff(positions, axis=0)
        speed = np.linalg.norm(velocities, axis=1)

        accelerations = np.diff(velocities, axis=0)
        acc_magnitude = np.linalg.norm(accelerations, axis=1)

        mean_speed = np.mean(speed) + 1e-6
        jitter_score = np.mean(acc_magnitude) / mean_speed

        smoothness = 1.0 - min(jitter_score / 2.0, 1.0)

        return {
            "smoothness": smoothness,
            "jitter_score": jitter_score,
            "mean_speed": float(mean_speed)
        }
