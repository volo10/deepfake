#!/usr/bin/env python3
"""
Analyze videos for deepfakes.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detector import DeepfakeDetector
from src.models import AnalysisConfig

def analyze_video(video_path):
    """Analyze a single video."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(video_path)}")
    print('='*60)
    
    # Configure for analysis
    config = AnalysisConfig(
        max_frames=200,  # Analyze up to 200 frames
        sample_rate=2,   # Every 2nd frame for speed
        deepfake_threshold=0.35,  # Adjusted: real videos ~0.11-0.14, need margin
        uncertain_threshold=0.25,
    )
    
    detector = DeepfakeDetector(config=config)
    
    try:
        result = detector.analyze(video_path)
        
        # Print results
        verdict_icon = "⚠️" if result.verdict.value == "DEEPFAKE" else "✓" if result.verdict.value == "REAL" else "❓"
        print(f"\n{verdict_icon} VERDICT: {result.verdict.value}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Overall Score: {result.overall_score:.2f}")
        
        # Show skill scores
        print(f"\n   Skill Scores:")
        for skill_name, skill_result in result.skill_results.items():
            indicator = "🔴" if skill_result.score > 0.5 else "🟡" if skill_result.score > 0.3 else "🟢"
            print(f"   {indicator} {skill_name}: {skill_result.score:.2f}")
        
        # Show top findings
        if result.findings:
            print(f"\n   Top Findings:")
            for finding in result.findings[:5]:
                print(f"   • {finding.description}")
        
        print(f"\n   Processing time: {result.processing_time:.2f}s")
        print(f"   Frames analyzed: {result.total_frames}")
        print(f"   Faces detected: {result.faces_detected}")
        
        return result
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Video files to analyze
    videos = [
        "/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/ssstwitter.com_1710426862820.mp4",
        "/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/ssstwitter.com_1722408787274.mp4",
        "/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/The_mann_in_202601031346_3qkij.mp4"
    ]
    
    print("🔍 DEEPFAKE DETECTION ANALYSIS")
    print("="*60)
    
    results = {}
    
    for video_path in videos:
        if os.path.exists(video_path):
            result = analyze_video(video_path)
            results[os.path.basename(video_path)] = result
        else:
            print(f"Video not found: {video_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for video_name, result in results.items():
        if result:
            icon = "⚠️ DEEPFAKE" if result.verdict.value == "DEEPFAKE" else "✓ REAL" if result.verdict.value == "REAL" else "❓ UNCERTAIN"
            print(f"\n{video_name}:")
            print(f"  {icon} ({result.confidence:.1%} confidence)")
        else:
            print(f"\n{video_name}:")
            print(f"  ❌ Analysis failed")

if __name__ == "__main__":
    main()

