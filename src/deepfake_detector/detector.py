"""
Main Deepfake Detection Agent.

This module implements the core detection pipeline that aggregates
evidence from multiple analysis skills.
"""

import time
import logging
from typing import Optional, List, Dict
import numpy as np

from .models import (
    DetectionResult, Verdict, Finding, Severity,
    SkillResult, VideoMetadata, AnalysisConfig
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
from .extraction import (
    load_video_metadata,
    extract_face_sequence,
    extract_landmarks_sequence,
    extract_mouth_regions,
    extract_face_boxes
)
from .aggregation import (
    aggregate_evidence,
    make_decision,
    compute_frame_scores,
    identify_suspicious_segments
)


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

    def analyze(
        self,
        video_path: str,
        reference_faces: Optional[List[np.ndarray]] = None
    ) -> DetectionResult:
        """
        Analyze a video for deepfake indicators.

        Args:
            video_path: Path to the video file
            reference_faces: Optional list of reference face images

        Returns:
            DetectionResult with verdict, confidence, and detailed findings
        """
        start_time = time.time()
        logger.info(f"Starting analysis of: {video_path}")

        # Load video
        video_meta = load_video_metadata(video_path)
        frames, fps = load_video(
            video_path,
            max_frames=self.config.max_frames,
            sample_rate=self.config.sample_rate
        )

        # Extract audio if available
        audio, audio_sr = self._extract_audio_if_available(video_path, video_meta)

        # Stage 1: Face Detection and Tracking
        logger.info("Stage 1: Face tracking")
        face_results = self.face_tracker.track(frames)

        if not face_results or len(face_results) == 0:
            logger.warning("No faces detected in video")
            return self._create_no_face_result(video_meta, time.time() - start_time)

        # Extract data sequences
        face_sequence = extract_face_sequence(face_results)
        landmarks_sequence = extract_landmarks_sequence(face_results)
        embeddings_sequence = self.face_tracker.extract_embeddings(face_sequence)

        # Stage 2: Run Analysis Skills
        logger.info("Stage 2: Running analysis skills")
        skill_results = self._run_analysis_skills(
            frames, face_sequence, landmarks_sequence, embeddings_sequence,
            face_results, fps, audio, audio_sr, reference_faces
        )

        # Stage 3: Aggregate Evidence
        logger.info("Stage 3: Aggregating evidence")
        overall_score, findings = aggregate_evidence(skill_results, self.config)

        # Stage 4: Make Decision
        verdict, confidence = make_decision(overall_score, findings, self.config)

        # Stage 5: Generate Explanation
        logger.info("Stage 5: Generating explanation")
        return self._build_result(
            verdict, confidence, overall_score, skill_results, findings,
            face_results, frames, fps, video_meta, start_time
        )

    def _extract_audio_if_available(self, video_path, video_meta):
        """Extract audio from video if available and enabled."""
        if video_meta.has_audio and self.config.enable_audio_analysis:
            return extract_audio(video_path)
        return None, None

    def _run_analysis_skills(
        self, frames, face_sequence, landmarks_sequence, embeddings_sequence,
        face_results, fps, audio, audio_sr, reference_faces
    ) -> Dict[str, SkillResult]:
        """Run all analysis skills and collect results."""
        skill_results: Dict[str, SkillResult] = {}

        # Temporal Consistency Analysis
        skill_results["temporal"] = self.temporal_analyzer.analyze(
            face_sequence=face_sequence,
            landmarks_sequence=landmarks_sequence,
            embeddings_sequence=embeddings_sequence,
            fps=fps
        )

        # Physiological Signal Analysis
        if self.config.enable_rppg and len(face_sequence) >= 150:
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
            mouth_regions = extract_mouth_regions(face_results)
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

        # Visual Artifact Analysis
        face_boxes = extract_face_boxes(face_results)
        skill_results["visual_artifacts"] = self.visual_artifact_analyzer.analyze(
            face_sequence=face_sequence,
            full_frames=frames,
            face_boxes=face_boxes
        )

        return skill_results

    def _build_result(
        self, verdict, confidence, overall_score, skill_results, findings,
        face_results, frames, fps, video_meta, start_time
    ) -> DetectionResult:
        """Build the final DetectionResult."""
        frame_scores = compute_frame_scores(
            skill_results, len(frames), self.config
        )
        suspicious_segments = identify_suspicious_segments(frame_scores, fps)

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
        logger.info(
            f"Analysis complete in {processing_time:.2f}s - Verdict: {verdict.value}"
        )

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

    def _create_no_face_result(
        self, video_meta: VideoMetadata, processing_time: float
    ) -> DetectionResult:
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
