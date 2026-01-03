"""
Deepfake Detection Agent
========================

A comprehensive, multi-modal AI agent for detecting deepfake videos.

This package provides production-grade deepfake detection through analysis of:
- Visual artifacts (texture, symmetry, edges, boundaries)
- Temporal consistency (identity drift, expression continuity)
- Physiological signals (rPPG/heart rate patterns)
- Frequency domain artifacts (GAN fingerprints)
- Audio-visual alignment (lip sync verification)
- Identity reasoning (face embedding stability)

Quick Start
-----------
>>> from deepfake_detector import DeepfakeDetector
>>> 
>>> detector = DeepfakeDetector()
>>> result = detector.analyze("video.mp4")
>>> 
>>> print(result.verdict)      # REAL | DEEPFAKE | UNCERTAIN
>>> print(result.confidence)   # 0.0 - 1.0
>>> print(result.explanation)  # Human-readable explanation

Configuration
-------------
>>> from deepfake_detector import DeepfakeDetector, AnalysisConfig
>>> 
>>> config = AnalysisConfig(
...     max_frames=300,
...     deepfake_threshold=0.40,
...     visual_artifacts_weight=4.0
... )
>>> detector = DeepfakeDetector(config=config)

Batch Processing
----------------
>>> results = detector.analyze_batch(
...     ["video1.mp4", "video2.mp4"],
...     max_workers=4
... )

For more information, see:
- API Documentation: docs/API.md
- Architecture: docs/ARCHITECTURE.md
- Configuration: config/config.yaml
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Deepfake Detection Team"
__license__ = "MIT"

# Core classes
from .detector import DeepfakeDetector, analyze_video
from .models import (
    # Enums
    Verdict,
    Severity,
    # Data structures
    AnalysisConfig,
    DetectionResult,
    SkillResult,
    Finding,
    TimeSegment,
    VideoMetadata,
    # Face detection types
    DetectedFace,
    BoundingBox,
    HeadPose,
    FrameAnalysis,
)

# Skill modules
from .skills import (
    FaceTracker,
    TemporalAnalyzer,
    PhysiologicalAnalyzer,
    FrequencyAnalyzer,
    AudioVisualAnalyzer,
    IdentityAnalyzer,
    VisualArtifactAnalyzer,
    ExplainabilityEngine,
)

# Exceptions
from .exceptions import (
    DeepfakeError,
    VideoLoadError,
    FaceDetectionError,
    AnalysisError,
    ConfigurationError,
)

# Configuration loading
from .config import load_config, configure_logging

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Main entry points
    "DeepfakeDetector",
    "analyze_video",
    # Configuration
    "AnalysisConfig",
    "load_config",
    "configure_logging",
    # Result types
    "Verdict",
    "Severity",
    "DetectionResult",
    "SkillResult",
    "Finding",
    "TimeSegment",
    "VideoMetadata",
    # Face types
    "DetectedFace",
    "BoundingBox",
    "HeadPose",
    "FrameAnalysis",
    # Skills
    "FaceTracker",
    "TemporalAnalyzer",
    "PhysiologicalAnalyzer",
    "FrequencyAnalyzer",
    "AudioVisualAnalyzer",
    "IdentityAnalyzer",
    "VisualArtifactAnalyzer",
    "ExplainabilityEngine",
    # Exceptions
    "DeepfakeError",
    "VideoLoadError",
    "FaceDetectionError",
    "AnalysisError",
    "ConfigurationError",
]
