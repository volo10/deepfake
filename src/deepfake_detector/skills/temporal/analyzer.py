"""
Temporal Consistency Analyzer - Main Module.

Coordinates temporal analysis for deepfake detection.
"""

import logging
from typing import List, Dict
import numpy as np

from ...models import Finding, Severity, SkillResult
from .blink import BlinkAnalyzer
from .identity import IdentityAnalyzer
from .motion import MotionAnalyzer

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Analyze temporal consistency in facial video sequences.

    Coordinates multiple specialized analyzers:
    - BlinkAnalyzer: Blink pattern analysis
    - IdentityAnalyzer: Identity stability over time
    - MotionAnalyzer: Expression flow and motion smoothness
    """

    def __init__(self):
        """Initialize temporal analyzer."""
        self.blink_analyzer = BlinkAnalyzer()
        self.identity_analyzer = IdentityAnalyzer()
        self.motion_analyzer = MotionAnalyzer()

        self.weights = {
            "identity": 1.5,
            "blink": 1.2,
            "expression": 1.3,
            "texture": 1.4,
            "motion": 1.1
        }

    def analyze(
        self,
        face_sequence: List[np.ndarray],
        landmarks_sequence: List[np.ndarray],
        embeddings_sequence: List[np.ndarray],
        fps: float
    ) -> SkillResult:
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
        identity_result = self.identity_analyzer.analyze(embeddings_sequence)
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
        blink_result = self.blink_analyzer.analyze(landmarks_sequence, fps)
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
        expression_result = self.motion_analyzer.analyze_expression_flow(
            landmarks_sequence
        )
        scores["expression"] = expression_result["smoothness"]
        if expression_result["smoothness"] < 0.7:
            findings.append(Finding(
                category="temporal",
                description="Expression discontinuities detected",
                severity=Severity.MEDIUM,
                confidence=1.0 - expression_result["smoothness"]
            ))

        # Texture stability analysis
        texture_result = self.motion_analyzer.analyze_texture_stability(face_sequence)
        scores["texture"] = texture_result["stability"]
        if texture_result["stability"] < 0.6:
            findings.append(Finding(
                category="temporal",
                description="Texture flickering detected",
                severity=Severity.HIGH if texture_result["stability"] < 0.4 else Severity.MEDIUM,
                confidence=1.0 - texture_result["stability"]
            ))

        # Motion smoothness
        motion_result = self.motion_analyzer.analyze_motion_smoothness(landmarks_sequence)
        scores["motion"] = motion_result["smoothness"]
        if motion_result["smoothness"] < 0.7:
            findings.append(Finding(
                category="temporal",
                description="Motion jitter detected",
                severity=Severity.LOW,
                confidence=1.0 - motion_result["smoothness"]
            ))

        # Compute overall score
        overall_score = self._compute_overall_score(scores)
        confidence = min(len(face_sequence) / 100, 1.0)

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

    def _compute_overall_score(self, scores: Dict[str, float]) -> float:
        """Compute weighted overall score."""
        weighted_sum = sum(
            (1.0 - scores[k]) * self.weights[k]
            for k in scores
        )
        total_weight = sum(self.weights.values())
        return weighted_sum / total_weight
