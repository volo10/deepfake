"""
Command Line Interface for the Deepfake Detection Agent.

Usage:
    python -m src.cli analyze video.mp4
    python -m src.cli analyze video.mp4 --output report.html --format html
"""

import argparse
import sys
import json
import logging
from pathlib import Path

from .detector import DeepfakeDetector
from .models import AnalysisConfig
from .utils import setup_logging, get_video_info


def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection Agent - Analyze videos for manipulation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a video")
    analyze_parser.add_argument("video", type=str, help="Path to video file")
    analyze_parser.add_argument("--reference", type=str, help="Reference face image")
    analyze_parser.add_argument("--output", "-o", type=str, help="Output file path")
    analyze_parser.add_argument(
        "--format", "-f", 
        choices=["text", "json", "html"],
        default="text",
        help="Output format"
    )
    analyze_parser.add_argument(
        "--max-frames", type=int, default=300,
        help="Maximum frames to analyze"
    )
    analyze_parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Deepfake threshold (0-1)"
    )
    analyze_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show video information")
    info_parser.add_argument("video", type=str, help="Path to video file")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "info":
        run_info(args)
    else:
        parser.print_help()


def run_analyze(args):
    """Run video analysis."""
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print(f"🔍 Analyzing: {video_path.name}")
    print("-" * 50)
    
    # Configure detector
    config = AnalysisConfig(
        max_frames=args.max_frames,
        deepfake_threshold=args.threshold,
        verbose=args.verbose
    )
    
    # Initialize detector
    detector = DeepfakeDetector(config=config)
    
    # Load reference if provided
    reference_faces = None
    if args.reference:
        ref_path = Path(args.reference)
        if ref_path.exists():
            try:
                import cv2
                ref_image = cv2.imread(str(ref_path))
                if ref_image is not None:
                    reference_faces = [cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)]
                    print(f"📷 Using reference: {ref_path.name}")
            except ImportError:
                logger.warning("OpenCV not available for reference image loading")
    
    # Run analysis
    try:
        result = detector.analyze(str(video_path), reference_faces=reference_faces)
    except Exception as e:
        print(f"Error during analysis: {e}")
        logger.exception("Analysis failed")
        sys.exit(1)
    
    # Output results
    print()
    print("=" * 50)
    print(f"VERDICT: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.1%}")
    print("=" * 50)
    
    if args.format == "text":
        print()
        print(result.explanation)
        
    elif args.format == "json":
        output = json.dumps(result.to_dict(), indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Results saved to: {args.output}")
        else:
            print(output)
            
    elif args.format == "html":
        from .skills import ExplainabilityEngine
        engine = ExplainabilityEngine()
        html = engine.generate_report(result, output_format="html")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(html)
            print(f"Report saved to: {args.output}")
        else:
            print(html)
    
    # Summary stats
    print()
    print("-" * 50)
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Frames analyzed: {result.total_frames}")
    print(f"Faces detected: {result.faces_detected}")
    
    if result.findings:
        print(f"\nTop findings ({len(result.findings)} total):")
        for finding in result.findings[:3]:
            print(f"  • {finding.description}")
    
    # Exit code based on verdict
    if result.verdict.value == "DEEPFAKE":
        sys.exit(2)
    elif result.verdict.value == "UNCERTAIN":
        sys.exit(1)
    else:
        sys.exit(0)


def run_info(args):
    """Show video information."""
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    info = get_video_info(str(video_path))
    
    if not info:
        print("Could not read video information")
        sys.exit(1)
    
    print(f"Video: {video_path.name}")
    print("-" * 40)
    print(f"Resolution: {info['width']}x{info['height']}")
    print(f"Frame rate: {info['fps']:.2f} fps")
    print(f"Total frames: {info['total_frames']}")
    print(f"Duration: {info['duration']:.2f}s")


if __name__ == "__main__":
    main()

