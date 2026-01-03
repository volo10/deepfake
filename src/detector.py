"""
Main Deepfake Detection Agent.

This module implements the core detection pipeline that aggregates
evidence from multiple analysis skills.
"""

import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

from .models import (
    DetectionResult, Verdict, Finding, Severity,
    SkillResult, TimeSegment, VideoMetadata, AnalysisConfig
)
from .skills import (
    FaceTracker,
    TemporalAnalyzer, 
    PhysiologicalAnalyzer,
    FrequencyAnalyzer,
    AudioVisualAnalyzer,
    IdentityAnalyzer,
    VisualArtifactAnalyzer,
    ExplainabilityEngine
)
from .utils import load_video, extract_audio


logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Main deepfake detection agent.
    
    Aggregates evidence from multiple analysis skills to determine
    whether a video is REAL, DEEPFAKE, or UNCERTAIN.
    
    Usage:
        detector = DeepfakeDetector()
        result = detector.analyze("video.mp4")
        print(result.verdict)
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the detector.
        
        Args:
            config: Analysis configuration. Uses defaults if not provided.
        """
        self.config = config or AnalysisConfig()
        
        # Initialize skills
        self.face_tracker = FaceTracker()
        self.temporal_analyzer = TemporalAnalyzer()
        self.physiological_analyzer = PhysiologicalAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.audio_visual_analyzer = AudioVisualAnalyzer()
        self.identity_analyzer = IdentityAnalyzer()
        self.visual_artifact_analyzer = VisualArtifactAnalyzer()
        self.explainability = ExplainabilityEngine()
        
        logger.info("DeepfakeDetector initialized")
    
    def analyze(self, video_path: str, 
                reference_faces: Optional[List[np.ndarray]] = None) -> DetectionResult:
        """
        Analyze a video for deepfake indicators.
        
        Args:
            video_path: Path to the video file
            reference_faces: Optional list of reference face images for identity verification
            
        Returns:
            DetectionResult with verdict, confidence, and detailed findings
        """
        start_time = time.time()
        logger.info(f"Starting analysis of: {video_path}")
        
        # Load video
        video_meta = self._load_video_metadata(video_path)
        frames, fps = load_video(video_path, 
                                  max_frames=self.config.max_frames,
                                  sample_rate=self.config.sample_rate)
        
        # Extract audio if available
        audio = None
        audio_sr = None
        if video_meta.has_audio and self.config.enable_audio_analysis:
            audio, audio_sr = extract_audio(video_path)
        
        # Stage 1: Face Detection and Tracking
        logger.info("Stage 1: Face tracking")
        face_results = self.face_tracker.track(frames)
        
        if not face_results or len(face_results) == 0:
            logger.warning("No faces detected in video")
            return self._create_no_face_result(video_meta, time.time() - start_time)
        
        # Extract face sequences for analysis
        face_sequence = self._extract_face_sequence(face_results)
        landmarks_sequence = self._extract_landmarks_sequence(face_results)
        embeddings_sequence = self._extract_embeddings(face_sequence)
        
        # Stage 2: Run Analysis Skills
        logger.info("Stage 2: Running analysis skills")
        skill_results: Dict[str, SkillResult] = {}
        
        # Temporal Consistency Analysis
        skill_results["temporal"] = self.temporal_analyzer.analyze(
            face_sequence=face_sequence,
            landmarks_sequence=landmarks_sequence,
            embeddings_sequence=embeddings_sequence,
            fps=fps
        )
        
        # Physiological Signal Analysis
        if self.config.enable_rppg and len(face_sequence) >= 150:  # Need ~5s at 30fps
            skill_results["physiological"] = self.physiological_analyzer.analyze(
                face_sequence=face_sequence,
                fps=fps
            )
        
        # Frequency Artifact Analysis
        if self.config.enable_frequency_analysis:
            skill_results["frequency"] = self.frequency_analyzer.analyze(
                face_crops=face_sequence,
                full_frames=frames
            )
        
        # Audio-Visual Alignment
        if audio is not None:
            mouth_regions = self._extract_mouth_regions(face_results)
            skill_results["audio_visual"] = self.audio_visual_analyzer.analyze(
                mouth_sequence=mouth_regions,
                audio=audio,
                audio_sr=audio_sr,
                fps=fps
            )
        
        # Identity Analysis
        skill_results["identity"] = self.identity_analyzer.analyze(
            embeddings_sequence=embeddings_sequence,
            reference_faces=reference_faces
        )
        
        # Visual Artifact Analysis (texture, symmetry, color)
        face_boxes = self._extract_face_boxes(face_results)
        skill_results["visual_artifacts"] = self.visual_artifact_analyzer.analyze(
            face_sequence=face_sequence,
            full_frames=frames,
            face_boxes=face_boxes
        )
        
        # Stage 3: Aggregate Evidence
        logger.info("Stage 3: Aggregating evidence")
        overall_score, findings = self._aggregate_evidence(skill_results)
        
        # Stage 4: Make Decision
        verdict, confidence = self._make_decision(overall_score, findings)
        
        # Stage 5: Generate Explanation
        logger.info("Stage 4: Generating explanation")
        frame_scores = self._compute_frame_scores(face_results, skill_results, len(frames))
        suspicious_segments = self._identify_suspicious_segments(frame_scores, fps)
        
        explanation = self.explainability.generate_explanation(
            verdict=verdict,
            confidence=confidence,
            findings=findings,
            skill_results=skill_results
        )
        
        attention_maps = {}
        if self.config.generate_attention_maps:
            attention_maps = self.explainability.generate_attention_maps(
                frames=frames,
                face_results=face_results,
                findings=findings
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis complete in {processing_time:.2f}s - Verdict: {verdict.value}")
        
        return DetectionResult(
            verdict=verdict,
            confidence=confidence,
            overall_score=overall_score,
            skill_results=skill_results,
            findings=findings,
            frame_scores=frame_scores,
            suspicious_segments=suspicious_segments,
            explanation=explanation,
            attention_maps=attention_maps,
            video_duration=video_meta.duration,
            fps=fps,
            total_frames=len(frames),
            faces_detected=len(face_results),
            processing_time=processing_time
        )
    
    def _load_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata."""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Check for audio (basic check)
        has_audio = True  # Assume true, will be verified during extraction
        
        cap.release()
        
        return VideoMetadata(
            path=video_path,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            has_audio=has_audio
        )
    
    def _extract_face_sequence(self, face_results: List) -> List[np.ndarray]:
        """Extract aligned face crops from tracking results."""
        faces = []
        for frame_result in face_results:
            if frame_result.faces:
                # Use the primary face (highest confidence or largest)
                primary_face = max(frame_result.faces, key=lambda f: f.confidence)
                if primary_face.aligned_face is not None:
                    faces.append(primary_face.aligned_face)
        return faces
    
    def _extract_landmarks_sequence(self, face_results: List) -> List[np.ndarray]:
        """Extract landmark sequences from tracking results."""
        landmarks = []
        for frame_result in face_results:
            if frame_result.faces:
                primary_face = max(frame_result.faces, key=lambda f: f.confidence)
                landmarks.append(primary_face.landmarks)
        return landmarks
    
    def _extract_embeddings(self, face_sequence: List[np.ndarray]) -> List[np.ndarray]:
        """Extract face embeddings for identity analysis."""
        return self.face_tracker.extract_embeddings(face_sequence)
    
    def _extract_mouth_regions(self, face_results: List) -> List[np.ndarray]:
        """Extract mouth region crops from tracking results."""
        mouths = []
        for frame_result in face_results:
            if frame_result.faces:
                primary_face = max(frame_result.faces, key=lambda f: f.confidence)
                if "mouth" in primary_face.regions:
                    mouths.append(primary_face.regions["mouth"])
        return mouths
    
    def _extract_face_boxes(self, face_results: List):
        """Extract face bounding boxes from tracking results."""
        boxes = []
        for frame_result in face_results:
            if frame_result.faces:
                primary_face = max(frame_result.faces, key=lambda f: f.confidence)
                bbox = primary_face.bbox
                boxes.append((bbox.x, bbox.y, bbox.width, bbox.height))
            else:
                boxes.append(None)
        return boxes
    
    def _aggregate_evidence(self, skill_results: Dict[str, SkillResult]) -> tuple:
        """
        Aggregate evidence from all skills into an overall score.
        
        Returns:
            Tuple of (overall_score, combined_findings)
        """
        all_findings: List[Finding] = []
        weighted_scores = []
        total_weight = 0
        max_skill_score = 0.0
        
        # Get visual artifacts score - it's our most reliable indicator
        visual_artifacts_score = skill_results.get("visual_artifacts", SkillResult("", 0.5, 0.5, [])).score
        
        for skill_name, result in skill_results.items():
            weight = self.config.skill_weights.get(skill_name, 1.0)
            
            # If visual artifacts is very clean (<0.1) and this skill has high score,
            # reduce its weight (likely a noisy/unreliable signal)
            if visual_artifacts_score < 0.1 and result.score > 0.5 and skill_name != "visual_artifacts":
                weight = weight * 0.3  # Reduce impact of likely false positive
            
            # Weight by confidence
            effective_weight = weight * result.confidence
            weighted_scores.append(result.score * effective_weight)
            total_weight += effective_weight
            
            # Track max score from any skill
            if result.score > max_skill_score:
                max_skill_score = result.score
            
            all_findings.extend(result.findings)
        
        # Compute weighted average
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
        else:
            overall_score = 0.5  # Uncertain
        
        # Boost overall score if any single skill has high confidence detection
        # This prevents a single strong signal from being diluted by other skills
        if max_skill_score > 0.5:
            # Blend weighted average with max score more aggressively
            boost_factor = (max_skill_score - 0.5) / 0.5  # 0 to 1 for scores 0.5 to 1.0
            # Stronger blending - up to 50% contribution from max score
            overall_score = overall_score * (1 - boost_factor * 0.5) + max_skill_score * (boost_factor * 0.5)
        
        # If visual artifacts is very clean, trust it more and reduce overall score
        if visual_artifacts_score < 0.1:
            overall_score = overall_score * 0.6  # Strong evidence it's real
        elif visual_artifacts_score < 0.15:
            overall_score = overall_score * 0.75  # Good evidence it's real
        
        # Sort findings by severity and confidence
        all_findings.sort(key=lambda f: (f.severity.value, f.confidence), reverse=True)
        
        return overall_score, all_findings
    
    def _make_decision(self, overall_score: float, 
                       findings: List[Finding]) -> tuple:
        """
        Make final verdict based on aggregated evidence.
        
        Uses conservative thresholds - prefers UNCERTAIN over false accusation.
        """
        # Check for high-severity findings
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        high_findings = [f for f in findings if f.severity == Severity.HIGH]
        
        # Strong evidence path - multiple high-severity findings
        if len(critical_findings) >= 2 or (len(critical_findings) >= 1 and len(high_findings) >= 2):
            if overall_score >= 0.3:
                return Verdict.DEEPFAKE, min(0.9, 0.5 + overall_score * 0.4)
        
        # Score-based decision
        if overall_score >= self.config.deepfake_threshold:
            # Scale confidence based on how far above threshold
            margin = overall_score - self.config.deepfake_threshold
            base_confidence = 0.4 + min(margin / 0.3, 0.5)  # 0.4 to 0.9
            
            # Boost confidence if multiple high findings
            if len(high_findings) >= 2:
                base_confidence = min(base_confidence + 0.15, 0.95)
            
            return Verdict.DEEPFAKE, base_confidence
        
        elif overall_score <= self.config.uncertain_threshold:
            # High confidence for clearly real videos
            margin = self.config.uncertain_threshold - overall_score
            confidence = min(0.95, 0.7 + margin * 1.5)
            return Verdict.REAL, confidence
        
        else:
            # Uncertain zone
            confidence = 0.3 + abs(overall_score - 0.3) * 0.3
            return Verdict.UNCERTAIN, confidence
    
    def _compute_frame_scores(self, face_results: List, 
                              skill_results: Dict[str, SkillResult],
                              total_frames: int) -> List[float]:
        """Compute per-frame suspicion scores."""
        frame_scores = [0.0] * total_frames
        
        for skill_name, result in skill_results.items():
            if "frame_scores" in result.raw_data:
                skill_frame_scores = result.raw_data["frame_scores"]
                weight = self.config.skill_weights.get(skill_name, 1.0)
                
                for i, score in enumerate(skill_frame_scores):
                    if i < len(frame_scores):
                        frame_scores[i] += score * weight
        
        # Normalize
        max_weight = sum(self.config.skill_weights.values())
        frame_scores = [s / max_weight for s in frame_scores]
        
        return frame_scores
    
    def _identify_suspicious_segments(self, frame_scores: List[float], 
                                      fps: float) -> List[TimeSegment]:
        """Identify time segments with high suspicion scores."""
        segments = []
        threshold = 0.5
        
        in_segment = False
        segment_start = 0
        segment_scores = []
        
        for i, score in enumerate(frame_scores):
            if score >= threshold:
                if not in_segment:
                    in_segment = True
                    segment_start = i
                    segment_scores = []
                segment_scores.append(score)
            else:
                if in_segment:
                    # End segment
                    segments.append(TimeSegment(
                        start=segment_start / fps,
                        end=i / fps,
                        suspicion_score=np.mean(segment_scores),
                        reason="Elevated suspicion scores"
                    ))
                    in_segment = False
        
        # Handle segment at end
        if in_segment:
            segments.append(TimeSegment(
                start=segment_start / fps,
                end=len(frame_scores) / fps,
                suspicion_score=np.mean(segment_scores),
                reason="Elevated suspicion scores"
            ))
        
        # Sort by suspicion score
        segments.sort(key=lambda s: s.suspicion_score, reverse=True)
        
        return segments[:5]  # Return top 5 suspicious segments
    
    def _create_no_face_result(self, video_meta: VideoMetadata, 
                                processing_time: float) -> DetectionResult:
        """Create result when no faces are detected."""
        return DetectionResult(
            verdict=Verdict.UNCERTAIN,
            confidence=0.1,
            overall_score=0.5,
            findings=[Finding(
                category="face_tracking",
                description="No faces detected in video",
                severity=Severity.LOW,
                confidence=1.0
            )],
            explanation="Unable to analyze: No faces detected in the video.",
            video_duration=video_meta.duration,
            fps=video_meta.fps,
            total_frames=video_meta.total_frames,
            faces_detected=0,
            processing_time=processing_time
        )


def analyze_video(video_path: str, **kwargs) -> DetectionResult:
    """
    Convenience function to analyze a video.
    
    Args:
        video_path: Path to video file
        **kwargs: Additional arguments passed to DeepfakeDetector
        
    Returns:
        DetectionResult
    """
    detector = DeepfakeDetector()
    return detector.analyze(video_path, **kwargs)

