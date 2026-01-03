#!/usr/bin/env python3
"""
Demo script for the Deepfake Detection Agent.

This script demonstrates how to use the detector with sample code.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import DeepfakeDetector
from src.models import AnalysisConfig, Verdict


def demo_basic_usage():
    """Demonstrate basic usage of the detector."""
    print("=" * 60)
    print("DEEPFAKE DETECTION AGENT - DEMO")
    print("=" * 60)
    print()
    
    # Initialize with default configuration
    print("1. Initializing detector with default configuration...")
    detector = DeepfakeDetector()
    print("   ✓ Detector initialized")
    print()
    
    # Show configuration
    print("2. Default configuration:")
    config = detector.config
    print(f"   - Max frames: {config.max_frames}")
    print(f"   - Sample rate: {config.sample_rate}")
    print(f"   - Deepfake threshold: {config.deepfake_threshold}")
    print(f"   - Uncertain threshold: {config.uncertain_threshold}")
    print()
    
    # Show skill weights
    print("3. Skill weights:")
    for skill, weight in config.skill_weights.items():
        print(f"   - {skill}: {weight}")
    print()
    
    print("=" * 60)
    print("USAGE EXAMPLE")
    print("=" * 60)
    print()
    print("To analyze a video, use:")
    print()
    print("```python")
    print("from src import DeepfakeDetector")
    print()
    print("detector = DeepfakeDetector()")
    print("result = detector.analyze('path/to/video.mp4')")
    print()
    print("print(result.verdict)      # REAL | DEEPFAKE | UNCERTAIN")
    print("print(result.confidence)   # 0.0 - 1.0")
    print("print(result.explanation)  # Human-readable explanation")
    print()
    print("# Access findings")
    print("for finding in result.findings:")
    print("    print(f'- {finding.description}')")
    print("```")
    print()
    
    print("=" * 60)
    print("COMMAND LINE USAGE")
    print("=" * 60)
    print()
    print("# Basic analysis")
    print("python -m src.cli analyze video.mp4")
    print()
    print("# Generate HTML report")
    print("python -m src.cli analyze video.mp4 --output report.html --format html")
    print()
    print("# With reference face")
    print("python -m src.cli analyze video.mp4 --reference known_face.jpg")
    print()
    print("# Video info")
    print("python -m src.cli info video.mp4")
    print()


def demo_custom_configuration():
    """Demonstrate custom configuration."""
    print("=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    print()
    
    # Create custom config
    config = AnalysisConfig(
        max_frames=500,
        sample_rate=2,  # Every 2nd frame
        deepfake_threshold=0.7,  # More conservative
        uncertain_threshold=0.3,
        enable_rppg=True,
        enable_audio_analysis=True,
        skill_weights={
            "face_tracking": 1.0,
            "temporal": 2.0,  # Emphasize temporal analysis
            "physiological": 1.5,
            "frequency": 1.2,
            "audio_visual": 1.3,
            "identity": 1.1
        },
        verbose=True
    )
    
    print("Custom configuration created:")
    print(f"  max_frames: {config.max_frames}")
    print(f"  sample_rate: {config.sample_rate}")
    print(f"  deepfake_threshold: {config.deepfake_threshold}")
    print()
    
    detector = DeepfakeDetector(config=config)
    print("✓ Custom detector initialized")
    print()


def demo_result_structure():
    """Show the structure of detection results."""
    print("=" * 60)
    print("DETECTION RESULT STRUCTURE")
    print("=" * 60)
    print()
    
    print("""
DetectionResult:
├── verdict: Verdict          # REAL | DEEPFAKE | UNCERTAIN
├── confidence: float         # 0.0 - 1.0
├── overall_score: float      # Aggregated anomaly score
├── skill_results: Dict       # Per-skill results
│   ├── temporal: SkillResult
│   ├── physiological: SkillResult
│   ├── frequency: SkillResult
│   ├── audio_visual: SkillResult
│   └── identity: SkillResult
├── findings: List[Finding]   # Detected anomalies
│   └── Finding:
│       ├── category: str
│       ├── description: str
│       ├── severity: Severity (LOW|MEDIUM|HIGH|CRITICAL)
│       ├── confidence: float
│       ├── frame_range: Optional[Tuple]
│       └── evidence: Dict
├── frame_scores: List[float] # Per-frame suspicion
├── suspicious_segments: List[TimeSegment]
├── explanation: str          # Human-readable
├── attention_maps: Dict      # Visual attention
└── metadata:
    ├── video_duration
    ├── fps
    ├── total_frames
    ├── faces_detected
    └── processing_time
    """)


def main():
    """Run all demos."""
    demo_basic_usage()
    demo_custom_configuration()
    demo_result_structure()
    
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print()
    print("To analyze a real video, run:")
    print("  python -m src.cli analyze <video_path>")
    print()


if __name__ == "__main__":
    main()

