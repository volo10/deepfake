"""
Frequency Artifact Analyzer - Main Module.

Coordinates frequency domain analysis for deepfake detection.
"""

import logging
from typing import List, Optional
import numpy as np

from ...models import Finding, Severity, SkillResult
from .gan_detection import GANDetector
from .artifacts import ArtifactDetector

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """
    Analyze frequency domain artifacts characteristic of deepfakes.

    Coordinates specialized detectors:
    - GANDetector: GAN fingerprint detection
    - ArtifactDetector: Checkerboard and noise artifacts
    """

    def __init__(self):
        """Initialize frequency analyzer."""
        self.gan_detector = GANDetector()
        self.artifact_detector = ArtifactDetector()

    def analyze(
        self,
        face_crops: List[np.ndarray],
        full_frames: Optional[List[np.ndarray]] = None
    ) -> SkillResult:
        """
        Analyze frequency domain artifacts.

        Args:
            face_crops: Aligned face crops
            full_frames: Optional full video frames for context

        Returns:
            SkillResult with frequency analysis
        """
        if len(face_crops) == 0:
            return self._empty_result()

        # Sample faces for analysis
        sample_indices = np.linspace(
            0, len(face_crops) - 1,
            min(30, len(face_crops)), dtype=int
        )
        sampled_faces = [face_crops[i] for i in sample_indices]

        findings = []

        # GAN fingerprint detection
        gan_score, gan_details = self.gan_detector.detect(sampled_faces)
        if gan_score > 0.4:
            findings.append(Finding(
                category="frequency",
                description=f"Potential GAN fingerprint detected (score: {gan_score:.2f})",
                severity=Severity.HIGH if gan_score > 0.7 else Severity.MEDIUM,
                confidence=gan_score,
                evidence=gan_details
            ))

        # Checkerboard artifact detection
        checker_score = self.artifact_detector.detect_checkerboard(sampled_faces)
        if checker_score > 0.3:
            findings.append(Finding(
                category="frequency",
                description=f"Checkerboard artifacts detected (score: {checker_score:.2f})",
                severity=Severity.MEDIUM,
                confidence=checker_score
            ))

        # High-frequency analysis
        hf_result = self.artifact_detector.analyze_high_frequency(sampled_faces)
        if hf_result["anomaly_score"] > 0.4:
            findings.append(Finding(
                category="frequency",
                description="Unnatural high-frequency characteristics",
                severity=Severity.MEDIUM,
                confidence=hf_result["anomaly_score"]
            ))

        # Noise consistency
        noise_score = 1.0
        if full_frames and len(full_frames) > 0:
            sampled_frames = [full_frames[i] for i in sample_indices]
            noise_score = self.artifact_detector.analyze_noise_consistency(
                sampled_faces, sampled_frames
            )
            if noise_score < 0.5:
                findings.append(Finding(
                    category="frequency",
                    description="Noise inconsistency between face and background",
                    severity=Severity.MEDIUM,
                    confidence=1.0 - noise_score
                ))

        # Compute overall anomaly score
        anomaly_score = (
            gan_score * 0.4 +
            checker_score * 0.25 +
            hf_result["anomaly_score"] * 0.2 +
            (1.0 - noise_score) * 0.15
        )

        return SkillResult(
            skill_name="frequency",
            score=anomaly_score,
            confidence=0.7,
            findings=findings,
            raw_data={
                "gan_score": gan_score,
                "checkerboard_score": checker_score,
                "hf_analysis": hf_result,
                "noise_consistency": noise_score
            }
        )

    def _empty_result(self) -> SkillResult:
        """Return empty result for no input."""
        return SkillResult(
            skill_name="frequency",
            score=0.5,
            confidence=0.1,
            findings=[],
            raw_data={}
        )
