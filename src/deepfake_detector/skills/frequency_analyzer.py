"""
Frequency Artifact Analyzer - Backward Compatibility Module.

This module re-exports FrequencyAnalyzer from the frequency subpackage
to maintain backward compatibility with existing imports.

The implementation has been split into smaller, focused modules:
- frequency/gan_detection.py: GAN fingerprint detection
- frequency/artifacts.py: Checkerboard and noise artifact detection
- frequency/analyzer.py: Main orchestrator

For new code, prefer importing directly:
    from deepfake_detector.skills.frequency import FrequencyAnalyzer
"""

from .frequency import FrequencyAnalyzer

__all__ = ["FrequencyAnalyzer"]
