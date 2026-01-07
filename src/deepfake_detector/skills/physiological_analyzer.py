"""
Physiological Signal Analyzer - Backward Compatibility Module.

This module re-exports PhysiologicalAnalyzer from the physiological subpackage
to maintain backward compatibility with existing imports.

The implementation has been split into smaller, focused modules:
- physiological/rppg.py: rPPG signal extraction using CHROM method
- physiological/signal_analysis.py: HR estimation and signal quality analysis
- physiological/analyzer.py: Main orchestrator

For new code, prefer importing directly:
    from deepfake_detector.skills.physiological import PhysiologicalAnalyzer
"""

from .physiological import PhysiologicalAnalyzer

__all__ = ["PhysiologicalAnalyzer"]
