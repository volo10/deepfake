"""
Frequency Analysis Subpackage.

Modular frequency domain analysis for deepfake detection.
"""

from .analyzer import FrequencyAnalyzer
from .gan_detection import GANDetector
from .artifacts import ArtifactDetector

__all__ = [
    "FrequencyAnalyzer",
    "GANDetector",
    "ArtifactDetector",
]
