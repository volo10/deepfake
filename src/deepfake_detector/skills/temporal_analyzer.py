"""
Temporal Consistency Analyzer

Analyzes frame-to-frame coherence to detect temporal artifacts
characteristic of deepfakes.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import signal
from scipy.stats import pearsonr

from ..models import Finding, Severity, SkillResult

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Analyze temporal consistency in facial video sequences.
    
    Detects:
    - Identity drift over time
    - Unnatural blink patterns
    - Expression discontinuities
    - Texture flickering
    - Motion jitter
    """
    
    def __init__(self):
        self.blink_threshold = 0.2  # Eye aspect ratio threshold
        self.identity_threshold = 0.4  # Embedding distance threshold
        
    def analyze(self, face_sequence: List[np.ndarray],
                landmarks_sequence: List[np.ndarray],
                embeddings_sequence: List[np.ndarray],
                fps: float) -> SkillResult:
        """
        Perform temporal consistency analysis.
        
        Args:
            face_sequence: Aligned face crops over time
            landmarks_sequence: Facial landmarks over time
            embeddings_sequence: Face embeddings over time
            fps: Video frame rate
            
        Returns:
            SkillResult with temporal consistency score and findings
        """
        findings = []
        scores = {}
        
        # Identity stability analysis
        identity_result = self._analyze_identity_stability(embeddings_sequence)
        scores["identity"] = identity_result["stability"]
        if identity_result["drift"] > 0.3:
            findings.append(Finding(
                category="temporal",
                description=f"Identity drift detected: {identity_result['drift']:.2f}",
                severity=Severity.HIGH if identity_result["drift"] > 0.5 else Severity.MEDIUM,
                confidence=min(identity_result["drift"] / 0.5, 1.0),
                evidence={"drift": identity_result["drift"]}
            ))
        
        # Blink analysis
        blink_result = self._analyze_blinks(landmarks_sequence, fps)
        scores["blink"] = 1.0 - blink_result["anomaly_score"]
        if blink_result["anomaly_score"] > 0.3:
            findings.append(Finding(
                category="temporal",
                description=f"Abnormal blink pattern: {blink_result['blink_rate']:.1f} blinks/min",
                severity=Severity.MEDIUM,
                confidence=blink_result["anomaly_score"],
                evidence=blink_result
            ))
        
        # Expression flow analysis
        expression_result = self._analyze_expression_flow(landmarks_sequence)
        scores["expression"] = expression_result["smoothness"]
        if expression_result["smoothness"] < 0.7:
            findings.append(Finding(
                category="temporal",
                description="Expression discontinuities detected",
                severity=Severity.MEDIUM,
                confidence=1.0 - expression_result["smoothness"]
            ))
        
        # Texture stability analysis
        texture_result = self._analyze_texture_stability(face_sequence)
        scores["texture"] = texture_result["stability"]
        if texture_result["stability"] < 0.6:
            findings.append(Finding(
                category="temporal",
                description="Texture flickering detected",
                severity=Severity.HIGH if texture_result["stability"] < 0.4 else Severity.MEDIUM,
                confidence=1.0 - texture_result["stability"]
            ))
        
        # Motion smoothness
        motion_result = self._analyze_motion_smoothness(landmarks_sequence)
        scores["motion"] = motion_result["smoothness"]
        if motion_result["smoothness"] < 0.7:
            findings.append(Finding(
                category="temporal",
                description="Motion jitter detected",
                severity=Severity.LOW,
                confidence=1.0 - motion_result["smoothness"]
            ))
        
        # Compute overall score (higher = more suspicious)
        weights = {
            "identity": 1.5,
            "blink": 1.2,
            "expression": 1.3,
            "texture": 1.4,
            "motion": 1.1
        }
        
        weighted_sum = sum((1.0 - scores[k]) * weights[k] for k in scores)
        total_weight = sum(weights.values())
        overall_score = weighted_sum / total_weight
        
        # Confidence based on data quality
        confidence = min(len(face_sequence) / 100, 1.0)  # More frames = more confidence
        
        return SkillResult(
            skill_name="temporal_consistency",
            score=overall_score,
            confidence=confidence,
            findings=findings,
            raw_data={
                "scores": scores,
                "identity_result": identity_result,
                "blink_result": blink_result
            }
        )
    
    def _analyze_identity_stability(self, embeddings: List[np.ndarray]) -> Dict:
        """Analyze identity embedding stability over time."""
        if len(embeddings) < 2:
            return {"stability": 1.0, "drift": 0.0, "variance": 0.0}
        
        embeddings_array = np.array(embeddings)
        
        # Compute centroid
        centroid = np.mean(embeddings_array, axis=0)
        
        # Distance from centroid
        distances = [self._cosine_distance(e, centroid) for e in embeddings_array]
        
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        variance = np.var(distances)
        
        # Stability score
        stability = 1.0 - min(mean_distance / 0.5, 1.0)
        
        # Detect drift (change in centroid over windows)
        window_size = min(30, len(embeddings) // 3)
        if window_size > 5:
            window_centroids = []
            for i in range(0, len(embeddings) - window_size, window_size // 2):
                window = embeddings_array[i:i + window_size]
                window_centroids.append(np.mean(window, axis=0))
            
            if len(window_centroids) > 1:
                drift_distances = [
                    self._cosine_distance(window_centroids[i], window_centroids[i+1])
                    for i in range(len(window_centroids) - 1)
                ]
                drift = np.max(drift_distances)
            else:
                drift = 0.0
        else:
            drift = 0.0
        
        return {
            "stability": stability,
            "drift": drift,
            "mean_distance": mean_distance,
            "max_distance": max_distance,
            "variance": variance
        }
    
    def _analyze_blinks(self, landmarks_sequence: List[np.ndarray], 
                       fps: float) -> Dict:
        """Analyze blink patterns for anomalies."""
        if len(landmarks_sequence) < 30 or fps <= 0:
            return {"blink_rate": 0, "anomaly_score": 0, "blinks": []}
        
        # Compute Eye Aspect Ratio (EAR) over time
        ear_values = []
        for landmarks in landmarks_sequence:
            if len(landmarks) >= 468:  # MediaPipe
                ear = self._compute_ear_mediapipe(landmarks)
            elif len(landmarks) >= 68:  # dlib
                ear = self._compute_ear_dlib(landmarks)
            else:
                ear = 0.3  # Default
            ear_values.append(ear)
        
        ear_values = np.array(ear_values)
        
        # Detect blink events
        blinks = self._detect_blinks(ear_values, self.blink_threshold)
        
        # Calculate blink rate
        duration_minutes = len(landmarks_sequence) / fps / 60
        blink_rate = len(blinks) / duration_minutes if duration_minutes > 0 else 0
        
        # Analyze blink durations
        blink_durations = [b["duration"] for b in blinks] if blinks else []
        
        # Anomaly scoring
        anomaly_score = 0.0
        reasons = []
        
        # Normal blink rate: 15-20 per minute
        if blink_rate < 5:
            anomaly_score += 0.4
            reasons.append("very_low_blink_rate")
        elif blink_rate < 10:
            anomaly_score += 0.2
            reasons.append("low_blink_rate")
        elif blink_rate > 40:
            anomaly_score += 0.3
            reasons.append("high_blink_rate")
        
        # Blink duration consistency
        if blink_durations and len(blink_durations) > 2:
            duration_std = np.std(blink_durations)
            if duration_std > 0.15:  # High variance in duration
                anomaly_score += 0.2
                reasons.append("inconsistent_blink_duration")
        
        anomaly_score = min(anomaly_score, 1.0)
        
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
        # Left eye landmarks
        left_eye = [33, 160, 158, 133, 153, 144]
        # Right eye landmarks  
        right_eye = [263, 387, 385, 362, 380, 373]
        
        def eye_aspect_ratio(pts):
            # Vertical distances
            v1 = np.linalg.norm(landmarks[pts[1]] - landmarks[pts[5]])
            v2 = np.linalg.norm(landmarks[pts[2]] - landmarks[pts[4]])
            # Horizontal distance
            h = np.linalg.norm(landmarks[pts[0]] - landmarks[pts[3]])
            return (v1 + v2) / (2.0 * h) if h > 0 else 0
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        return (left_ear + right_ear) / 2.0
    
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
    
    def _detect_blinks(self, ear_values: np.ndarray, 
                      threshold: float) -> List[Dict]:
        """Detect blink events from EAR time series."""
        blinks = []
        in_blink = False
        blink_start = 0
        
        for i, ear in enumerate(ear_values):
            if ear < threshold:
                if not in_blink:
                    in_blink = True
                    blink_start = i
            else:
                if in_blink:
                    blink_end = i
                    duration = (blink_end - blink_start) / 30.0  # Assume 30fps
                    
                    if 0.05 < duration < 0.5:  # Valid blink duration
                        blinks.append({
                            "start": blink_start,
                            "end": blink_end,
                            "duration": duration
                        })
                    in_blink = False
        
        return blinks
    
    def _analyze_expression_flow(self, landmarks_sequence: List[np.ndarray]) -> Dict:
        """Analyze expression transition smoothness."""
        if len(landmarks_sequence) < 3:
            return {"smoothness": 1.0, "discontinuities": []}
        
        # Compute landmark velocities and accelerations
        landmarks_array = np.array([lm[:68] if len(lm) >= 68 else lm 
                                    for lm in landmarks_sequence])
        
        if len(landmarks_array[0]) == 0:
            return {"smoothness": 1.0, "discontinuities": []}
        
        velocities = np.diff(landmarks_array, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Compute magnitude of accelerations
        acc_magnitudes = np.linalg.norm(accelerations, axis=-1)
        mean_acc = np.mean(acc_magnitudes, axis=-1)
        
        # Detect discontinuities (high acceleration)
        threshold = np.mean(mean_acc) + 2 * np.std(mean_acc)
        discontinuities = np.where(mean_acc > threshold)[0].tolist()
        
        # Smoothness score
        smoothness = 1.0 - min(len(discontinuities) / len(mean_acc), 1.0)
        
        return {
            "smoothness": smoothness,
            "discontinuities": discontinuities,
            "mean_acceleration": float(np.mean(mean_acc))
        }
    
    def _analyze_texture_stability(self, face_sequence: List[np.ndarray]) -> Dict:
        """Analyze texture stability across frames."""
        if len(face_sequence) < 2:
            return {"stability": 1.0, "flicker_score": 0.0}
        
        # Extract high-frequency components
        hf_sequence = []
        for face in face_sequence:
            gray = face if len(face.shape) == 2 else np.mean(face, axis=2)
            # High-pass filter
            blurred = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0) \
                      if 'cv2' in dir() else gray.astype(np.float32)
            hf = gray.astype(np.float32) - blurred
            hf_sequence.append(hf)
        
        # Compute frame-to-frame differences
        diffs = []
        for i in range(len(hf_sequence) - 1):
            diff = np.abs(hf_sequence[i+1] - hf_sequence[i])
            diffs.append(np.mean(diff))
        
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        
        # Stability score (lower diff = more stable)
        stability = 1.0 - min(mean_diff / 30.0, 1.0)
        
        return {
            "stability": stability,
            "flicker_score": mean_diff,
            "max_flicker": max_diff
        }
    
    def _analyze_motion_smoothness(self, landmarks_sequence: List[np.ndarray]) -> Dict:
        """Analyze head/face motion smoothness."""
        if len(landmarks_sequence) < 3:
            return {"smoothness": 1.0, "jitter_score": 0.0}
        
        # Use nose tip or face center as motion proxy
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
        
        # Compute velocities
        velocities = np.diff(positions, axis=0)
        speed = np.linalg.norm(velocities, axis=1)
        
        # Compute accelerations
        accelerations = np.diff(velocities, axis=0)
        acc_magnitude = np.linalg.norm(accelerations, axis=1)
        
        # Jitter = high acceleration relative to speed
        mean_speed = np.mean(speed) + 1e-6
        jitter_score = np.mean(acc_magnitude) / mean_speed
        
        # Smoothness
        smoothness = 1.0 - min(jitter_score / 2.0, 1.0)
        
        return {
            "smoothness": smoothness,
            "jitter_score": jitter_score,
            "mean_speed": float(mean_speed)
        }
    
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity


# Import cv2 at module level for texture analysis
try:
    import cv2
except ImportError:
    cv2 = None

