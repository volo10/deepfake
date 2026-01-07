"""
Visual Artifact Analyzer - Backward Compatibility Module.

This module re-exports VisualArtifactAnalyzer from the visual_artifacts subpackage
to maintain backward compatibility with existing imports.

The implementation has been split into smaller, focused modules:
- visual_artifacts/texture.py: Texture and local contrast analysis
- visual_artifacts/symmetry.py: Facial symmetry analysis
- visual_artifacts/edge.py: Edge density and boundary sharpness
- visual_artifacts/color.py: Color consistency and saturation
- visual_artifacts/motion.py: Temporal stability and background motion
- visual_artifacts/analyzer.py: Main orchestrator

For new code, prefer importing directly:
    from deepfake_detector.skills.visual_artifacts import VisualArtifactAnalyzer
"""

from .visual_artifacts import VisualArtifactAnalyzer

__all__ = ["VisualArtifactAnalyzer"]
