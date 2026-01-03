"""
Deepfake Detection Agent
========================

A comprehensive AI agent for detecting deepfake videos through multi-modal analysis.

Components:
- Face tracking and landmark extraction
- Temporal consistency analysis
- Physiological signal detection (rPPG)
- Frequency domain artifact detection
- Audio-visual alignment verification
- Identity reasoning and verification
- Explainable AI outputs

Usage:
    from deepfake_detector import DeepfakeDetector
    
    detector = DeepfakeDetector()
    result = detector.analyze("video.mp4")
    
    print(result.verdict)      # REAL | DEEPFAKE | UNCERTAIN
    print(result.confidence)   # 0.0 - 1.0
    print(result.explanation)  # Human-readable explanation
"""

from .detector import DeepfakeDetector
from .models import DetectionResult, Verdict, Finding
from .skills import (
    FaceTracker,
    TemporalAnalyzer,
    PhysiologicalAnalyzer,
    FrequencyAnalyzer,
    AudioVisualAnalyzer,
    IdentityAnalyzer,
    ExplainabilityEngine
)

__version__ = "1.0.0"
__author__ = "Deepfake Detection Team"

__all__ = [
    "DeepfakeDetector",
    "DetectionResult",
    "Verdict",
    "Finding",
    "FaceTracker",
    "TemporalAnalyzer",
    "PhysiologicalAnalyzer",
    "FrequencyAnalyzer",
    "AudioVisualAnalyzer",
    "IdentityAnalyzer",
    "ExplainabilityEngine",
]

