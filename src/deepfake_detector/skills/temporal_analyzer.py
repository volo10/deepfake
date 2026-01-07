"""
Temporal Consistency Analyzer - Backward Compatibility Module.

This module re-exports TemporalAnalyzer from the temporal subpackage
to maintain backward compatibility with existing imports.

The implementation has been split into smaller, focused modules:
- temporal/blink.py: Blink pattern analysis
- temporal/identity.py: Identity stability analysis
- temporal/motion.py: Expression flow and motion smoothness
- temporal/analyzer.py: Main orchestrator

For new code, prefer importing directly:
    from deepfake_detector.skills.temporal import TemporalAnalyzer
"""

from .temporal import TemporalAnalyzer

__all__ = ["TemporalAnalyzer"]
