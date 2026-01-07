"""
Physiological Signal Analyzer - Main Module.

Coordinates rPPG extraction and signal analysis for deepfake detection.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np

from ...models import Finding, Severity, SkillResult
from .rppg import RPPGExtractor
from .signal_analysis import SignalAnalyzer

logger = logging.getLogger(__name__)


class PhysiologicalAnalyzer:
    """
    Extract and analyze remote photoplethysmography (rPPG) signals.

    Real faces exhibit subtle color changes synchronized with heartbeat.
    Deepfakes typically lack or have inconsistent physiological signals.
    """

    def __init__(self):
        """Initialize physiological analyzer."""
        self.rppg_extractor = RPPGExtractor()
        self.signal_analyzer = SignalAnalyzer()

    def analyze(
        self,
        face_sequence: List[np.ndarray],
        fps: float,
        forehead_regions: Optional[List[np.ndarray]] = None,
        cheek_regions: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> SkillResult:
        """
        Analyze physiological signals in face video.

        Args:
            face_sequence: Aligned face crops over time
            fps: Video frame rate
            forehead_regions: Optional forehead ROIs
            cheek_regions: Optional (left_cheek, right_cheek) ROI pairs

        Returns:
            SkillResult with physiological analysis
        """
        # Need sufficient frames for rPPG analysis
        min_frames = int(fps * 5)
        if len(face_sequence) < min_frames:
            return self._insufficient_frames_result()

        # Extract rPPG signal
        rppg_signal = self.rppg_extractor.extract_chrom(face_sequence, fps)

        # Analyze signal
        hr, hr_confidence = self.signal_analyzer.estimate_heart_rate(rppg_signal, fps)
        signal_present, signal_quality = self.signal_analyzer.assess_signal_quality(
            rppg_signal, fps
        )

        # Spatial and bilateral analysis
        spatial_coherence = 1.0
        if forehead_regions or cheek_regions:
            spatial_coherence = self.signal_analyzer.analyze_spatial_coherence(
                face_sequence, forehead_regions, cheek_regions, fps
            )

        lr_correlation = self.signal_analyzer.analyze_bilateral_symmetry(
            face_sequence, fps
        )

        # Generate findings and score
        findings, anomaly_score = self._evaluate_signals(
            signal_present, hr, hr_confidence, signal_quality,
            spatial_coherence, lr_correlation
        )

        return SkillResult(
            skill_name="physiological",
            score=anomaly_score,
            confidence=min(signal_quality + 0.3, 0.9),
            findings=findings,
            raw_data={
                "heart_rate": hr,
                "hr_confidence": hr_confidence,
                "signal_present": signal_present,
                "signal_quality": signal_quality,
                "spatial_coherence": spatial_coherence,
                "lr_correlation": lr_correlation,
                "rppg_signal": rppg_signal[:1000].tolist() if len(rppg_signal) > 1000 else rppg_signal.tolist()
            }
        )

    def _insufficient_frames_result(self) -> SkillResult:
        """Return result for insufficient video length."""
        return SkillResult(
            skill_name="physiological",
            score=0.5,
            confidence=0.2,
            findings=[Finding(
                category="physiological",
                description="Insufficient video length for rPPG analysis",
                severity=Severity.LOW,
                confidence=1.0
            )],
            raw_data={"reason": "insufficient_frames"}
        )

    def _evaluate_signals(
        self,
        signal_present: bool,
        hr: float,
        hr_confidence: float,
        signal_quality: float,
        spatial_coherence: float,
        lr_correlation: float
    ) -> Tuple[List[Finding], float]:
        """
        Evaluate physiological signals and generate findings.

        Returns:
            Tuple of (findings, anomaly_score)
        """
        findings = []
        anomaly_score = 0.0

        if not signal_present:
            findings.append(Finding(
                category="physiological",
                description="No detectable rPPG signal",
                severity=Severity.HIGH,
                confidence=0.8
            ))
            anomaly_score += 0.4

        min_hr, max_hr = self.signal_analyzer.min_hr, self.signal_analyzer.max_hr
        if signal_present and (hr < min_hr or hr > max_hr):
            findings.append(Finding(
                category="physiological",
                description=f"Unrealistic heart rate estimate: {hr:.0f} BPM",
                severity=Severity.MEDIUM,
                confidence=hr_confidence
            ))
            anomaly_score += 0.2

        if spatial_coherence < 0.5:
            findings.append(Finding(
                category="physiological",
                description=f"Spatial incoherence in physiological signals ({spatial_coherence:.2f})",
                severity=Severity.HIGH,
                confidence=0.7
            ))
            anomaly_score += 0.3

        if lr_correlation < 0.6:
            findings.append(Finding(
                category="physiological",
                description=f"Bilateral asymmetry in physiological signals ({lr_correlation:.2f})",
                severity=Severity.MEDIUM,
                confidence=0.6
            ))
            anomaly_score += 0.25

        if signal_quality < 0.3:
            findings.append(Finding(
                category="physiological",
                description="Low quality physiological signal",
                severity=Severity.LOW,
                confidence=0.5
            ))
            anomaly_score += 0.15

        return findings, min(anomaly_score, 1.0)
