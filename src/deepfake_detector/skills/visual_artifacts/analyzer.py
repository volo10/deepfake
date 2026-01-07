"""
Visual Artifact Analyzer - Main Orchestrator.

Coordinates all visual artifact detection modules to provide
comprehensive deepfake detection through visual analysis.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np

from ...models import Finding, Severity, SkillResult
from .base import is_cv2_available
from .texture import TextureAnalyzer
from .symmetry import SymmetryAnalyzer
from .edge import EdgeAnalyzer
from .color import ColorAnalyzer
from .motion import MotionAnalyzer

logger = logging.getLogger(__name__)


class VisualArtifactAnalyzer:
    """
    Analyze visual artifacts characteristic of deepfakes.

    Coordinates multiple specialized analyzers:
    - TextureAnalyzer: texture variance and local contrast
    - SymmetryAnalyzer: facial symmetry patterns
    - EdgeAnalyzer: edge density and boundary sharpness
    - ColorAnalyzer: color consistency and saturation
    - MotionAnalyzer: temporal stability and background motion
    """

    def __init__(self):
        """Initialize all sub-analyzers with empirically-tuned thresholds."""
        self.texture = TextureAnalyzer()
        self.symmetry = SymmetryAnalyzer()
        self.edge = EdgeAnalyzer()
        self.color = ColorAnalyzer()
        self.motion = MotionAnalyzer()

        # Score weights for final aggregation
        self.weights = {
            'texture': 1.5,
            'symmetry': 1.8,
            'edge_density': 1.4,
            'boundary': 1.8,
            'local_contrast': 1.3,
            'saturation': 1.2,
            'color_consistency': 1.5,
            'background_motion': 1.7,
            'temporal_stability': 0.6
        }

    def analyze(
        self,
        face_sequence: List[np.ndarray],
        full_frames: Optional[List[np.ndarray]] = None,
        face_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> SkillResult:
        """
        Analyze visual artifacts in face sequence.

        Args:
            face_sequence: Aligned face crops
            full_frames: Optional full video frames
            face_boxes: Optional face bounding boxes (x, y, w, h)

        Returns:
            SkillResult with score, confidence, and findings
        """
        if not is_cv2_available():
            return self._unavailable_result("OpenCV not available")

        if len(face_sequence) < 5:
            return self._unavailable_result("Insufficient frames")

        findings = []
        scores = {}

        # Run all analyses
        self._analyze_texture(face_sequence, findings, scores)
        self._analyze_symmetry(face_sequence, findings, scores)
        self._analyze_edges(face_sequence, full_frames, face_boxes, findings, scores)
        self._analyze_colors(face_sequence, full_frames, face_boxes, findings, scores)
        self._analyze_motion(face_sequence, full_frames, face_boxes, findings, scores)

        # Compute weighted score
        overall_score = self._compute_overall_score(scores)

        return SkillResult(
            skill_name="visual_artifacts",
            score=overall_score,
            confidence=min(len(face_sequence) / 50, 1.0),
            findings=findings,
            raw_data={'scores': scores}
        )

    def _unavailable_result(self, error: str) -> SkillResult:
        """Return result when analysis cannot be performed."""
        return SkillResult(
            skill_name="visual_artifacts",
            score=0.5,
            confidence=0.1,
            findings=[],
            raw_data={"error": error}
        )

    def _analyze_texture(self, faces, findings, scores):
        """Run texture and local contrast analysis."""
        texture_result = self.texture.analyze_texture(faces)
        scores['texture'] = texture_result['anomaly_score']
        if texture_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Texture anomaly: {texture_result.get('issue_type', 'abnormal')} "
                           f"(variance: {texture_result['mean_variance']:.1f})",
                severity=Severity.HIGH if texture_result['anomaly_score'] > 0.6 else Severity.MEDIUM,
                confidence=texture_result['anomaly_score'],
                evidence=texture_result
            ))

        contrast_result = self.texture.analyze_local_contrast(faces)
        scores['local_contrast'] = contrast_result['anomaly_score']
        if contrast_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Abnormal local contrast ({contrast_result['mean_contrast']:.2f})",
                severity=Severity.MEDIUM,
                confidence=contrast_result['anomaly_score'],
                evidence=contrast_result
            ))

    def _analyze_symmetry(self, faces, findings, scores):
        """Run facial symmetry analysis."""
        symmetry_result = self.symmetry.analyze(faces)
        scores['symmetry'] = symmetry_result['anomaly_score']
        if symmetry_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Facial symmetry anomaly (symmetry: {symmetry_result['mean_symmetry']:.2f})",
                severity=Severity.HIGH if symmetry_result['anomaly_score'] > 0.5 else Severity.MEDIUM,
                confidence=symmetry_result['anomaly_score'],
                evidence=symmetry_result
            ))

    def _analyze_edges(self, faces, frames, boxes, findings, scores):
        """Run edge density and boundary sharpness analysis."""
        edge_result = self.edge.analyze_density(faces)
        scores['edge_density'] = edge_result['anomaly_score']
        if edge_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Edge density anomaly: {edge_result.get('issue_type', 'abnormal')} "
                           f"({edge_result['mean_density']:.4f})",
                severity=Severity.MEDIUM,
                confidence=edge_result['anomaly_score'],
                evidence=edge_result
            ))

        boundary_result = self.edge.analyze_boundary_sharpness(faces, frames, boxes)
        scores['boundary'] = boundary_result['anomaly_score']
        if boundary_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Unnatural face boundary sharpness ({boundary_result['mean_sharpness']:.2f})",
                severity=Severity.HIGH if boundary_result['anomaly_score'] > 0.6 else Severity.MEDIUM,
                confidence=boundary_result['anomaly_score'],
                evidence=boundary_result
            ))

    def _analyze_colors(self, faces, frames, boxes, findings, scores):
        """Run color consistency and saturation analysis."""
        if frames and boxes and len(boxes) > 0:
            valid_pairs = [(f, b) for f, b in zip(frames, boxes) if b is not None]
            if valid_pairs:
                valid_frames, valid_boxes = zip(*valid_pairs)
                color_result = self.color.analyze_consistency(
                    list(valid_frames), list(valid_boxes)
                )
                scores['color_consistency'] = color_result['anomaly_score']
                if color_result['anomaly_score'] > 0.2:
                    findings.append(Finding(
                        category="visual_artifacts",
                        description=f"Face-neck color mismatch ({color_result['mean_diff']:.1f})",
                        severity=Severity.HIGH if color_result['anomaly_score'] > 0.5 else Severity.MEDIUM,
                        confidence=color_result['anomaly_score'],
                        evidence=color_result
                    ))

        saturation_result = self.color.analyze_saturation(faces)
        scores['saturation'] = saturation_result['anomaly_score']
        if saturation_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Abnormal color saturation variance ({saturation_result['mean_variance']:.1f})",
                severity=Severity.MEDIUM,
                confidence=saturation_result['anomaly_score'],
                evidence=saturation_result
            ))

    def _analyze_motion(self, faces, frames, boxes, findings, scores):
        """Run temporal stability and background motion analysis."""
        stability_result = self.motion.analyze_temporal_stability(faces)
        scores['temporal_stability'] = stability_result['anomaly_score']
        if stability_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Unnaturally stable frames (variation: {stability_result['mean_diff']:.2f})",
                severity=Severity.LOW,
                confidence=stability_result['anomaly_score'],
                evidence=stability_result
            ))

        if frames and boxes and len(frames) > 10:
            bg_result = self.motion.analyze_background_motion(frames, boxes)
            scores['background_motion'] = bg_result['anomaly_score']
            if bg_result['anomaly_score'] > 0.3:
                findings.append(Finding(
                    category="visual_artifacts",
                    description=f"Static/frozen background detected "
                               f"({bg_result['bg_motion']:.2f} vs face {bg_result['face_motion']:.2f})",
                    severity=Severity.HIGH if bg_result['anomaly_score'] > 0.6 else Severity.MEDIUM,
                    confidence=bg_result['anomaly_score'],
                    evidence=bg_result
                ))

    def _compute_overall_score(self, scores: dict) -> float:
        """Compute weighted overall score with synergy boost."""
        weighted_sum = sum(
            scores.get(k, 0) * self.weights[k]
            for k in self.weights if k in scores
        )
        total_weight = sum(
            self.weights[k] for k in self.weights if k in scores
        )

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Synergy boost for multiple triggered indicators
        triggered_count = sum(1 for s in scores.values() if s > 0.2)
        if triggered_count >= 4:
            overall_score = min(overall_score * 1.5, 1.0)
        elif triggered_count >= 3:
            overall_score = min(overall_score * 1.3, 1.0)
        elif triggered_count >= 2:
            overall_score = min(overall_score * 1.15, 1.0)

        return overall_score
