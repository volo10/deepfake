"""
Temporal Analysis Subpackage.

Modular temporal consistency analysis for deepfake detection.
"""

from .analyzer import TemporalAnalyzer
from .blink import BlinkAnalyzer
from .identity import IdentityAnalyzer
from .motion import MotionAnalyzer

__all__ = [
    "TemporalAnalyzer",
    "BlinkAnalyzer",
    "IdentityAnalyzer",
    "MotionAnalyzer",
]
