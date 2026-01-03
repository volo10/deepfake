#!/usr/bin/env python3
"""
Analyze the 2 new videos for deepfakes.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detector import DeepfakeDetector
from src.models import AnalysisConfig

def analyze_video(video_path, detector):
    """Analyze a single video."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(video_path)}")
    print('='*60)
    
    try:
        result = detector.analyze(video_path)
        
        verdict_icon = "⚠️" if result.verdict.value == "DEEPFAKE" else "✓" if result.verdict.value == "REAL" else "❓"
        print(f"\n{verdict_icon} VERDICT: {result.verdict.value}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Overall Score: {result.overall_score:.2f}")
        
        print(f"\n   Skill Scores:")
        for skill_name, skill_result in result.skill_results.items():
            indicator = "🔴" if skill_result.score > 0.5 else "🟡" if skill_result.score > 0.3 else "🟢"
            print(f"   {indicator} {skill_name}: {skill_result.score:.2f}")
        
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
    # New videos to analyze
    videos = [
        "/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/Ben_Volovelsky_Summer2026_Video.mp4",
        "/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/The_man_in_202601031425_w8qo6.mp4"
    ]
    
    print("🔍 DEEPFAKE DETECTION - NEW VIDEOS")
    print("="*60)
    
    config = AnalysisConfig(
        max_frames=200,
        sample_rate=2,
        deepfake_threshold=0.35,
        uncertain_threshold=0.25,
    )
    
    detector = DeepfakeDetector(config=config)
    
    results = {}
    
    for video_path in videos:
        if os.path.exists(video_path):
            result = analyze_video(video_path, detector)
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

