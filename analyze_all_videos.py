#!/usr/bin/env python3
"""
Analyze ALL videos for deepfakes.
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
            for finding in result.findings[:6]:
                print(f"   • {finding.description}")
        
        print(f"\n   Processing time: {result.processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    videos = [
        # Real videos
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/ssstwitter.com_1710426862820.mp4", "REAL"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/ssstwitter.com_1722408787274.mp4", "REAL"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/Ben_Volovelsky_Summer2026_Video.mp4", "REAL"),
        # Fake videos
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/The_mann_in_202601031346_3qkij.mp4", "FAKE"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/The_man_in_202601031425_w8qo6.mp4", "FAKE"),
    ]
    
    print("🔍 DEEPFAKE DETECTION - ALL VIDEOS")
    print("="*60)
    
    config = AnalysisConfig(
        max_frames=200,
        sample_rate=2,
        deepfake_threshold=0.35,
        uncertain_threshold=0.25,
    )
    
    detector = DeepfakeDetector(config=config)
    
    results = {}
    
    for video_path, ground_truth in videos:
        if os.path.exists(video_path):
            result = analyze_video(video_path, detector)
            results[os.path.basename(video_path)] = (ground_truth, result)
        else:
            print(f"Video not found: {video_path}")
    
    # Summary with accuracy
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    correct = 0
    total = 0
    
    for video_name, (ground_truth, result) in results.items():
        if result:
            total += 1
            predicted = result.verdict.value
            is_correct = (ground_truth == "REAL" and predicted == "REAL") or \
                        (ground_truth == "FAKE" and predicted == "DEEPFAKE")
            
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            icon = "⚠️" if predicted == "DEEPFAKE" else "✓" if predicted == "REAL" else "❓"
            print(f"\n{status} {video_name}:")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Prediction:   {icon} {predicted} ({result.confidence:.1%})")
    
    print(f"\n{'='*60}")
    print(f"ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()

