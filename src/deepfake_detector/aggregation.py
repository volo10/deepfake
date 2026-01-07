"""
Evidence Aggregation Module.

Provides functions for aggregating skill results and making detection decisions.
"""

from typing import List, Dict, Tuple
import numpy as np

from .models import (
    Finding, Severity, SkillResult, TimeSegment, Verdict, AnalysisConfig
)


def aggregate_evidence(
    skill_results: Dict[str, SkillResult],
    config: AnalysisConfig
) -> Tuple[float, List[Finding]]:
    """
    Aggregate evidence from all skills into an overall score.

    Args:
        skill_results: Dictionary of skill name to SkillResult
        config: Analysis configuration with skill weights

    Returns:
        Tuple of (overall_score, combined_findings)
    """
    all_findings: List[Finding] = []
    weighted_scores = []
    total_weight = 0
    max_skill_score = 0.0

    # Get visual artifacts score - it's our most reliable indicator
    visual_artifacts_result = skill_results.get(
        "visual_artifacts", SkillResult("", 0.5, 0.5, [])
    )
    visual_artifacts_score = visual_artifacts_result.score

    for skill_name, result in skill_results.items():
        weight = config.skill_weights.get(skill_name, 1.0)

        # If visual artifacts is very clean (<0.1) and this skill has high score,
        # reduce its weight (likely a noisy/unreliable signal)
        if visual_artifacts_score < 0.1 and result.score > 0.5:
            if skill_name != "visual_artifacts":
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
    if max_skill_score > 0.5:
        boost_factor = (max_skill_score - 0.5) / 0.5
        overall_score = (
            overall_score * (1 - boost_factor * 0.5) +
            max_skill_score * (boost_factor * 0.5)
        )

    # If visual artifacts is very clean, trust it more and reduce overall score
    if visual_artifacts_score < 0.1:
        overall_score = overall_score * 0.6  # Strong evidence it's real
    elif visual_artifacts_score < 0.15:
        overall_score = overall_score * 0.75  # Good evidence it's real

    # Sort findings by severity and confidence
    all_findings.sort(key=lambda f: (f.severity.value, f.confidence), reverse=True)

    return overall_score, all_findings


def make_decision(
    overall_score: float,
    findings: List[Finding],
    config: AnalysisConfig
) -> Tuple[Verdict, float]:
    """
    Make final verdict based on aggregated evidence.

    Uses conservative thresholds - prefers UNCERTAIN over false accusation.

    Args:
        overall_score: Aggregated score from all skills
        findings: List of all findings
        config: Analysis configuration with thresholds

    Returns:
        Tuple of (Verdict, confidence)
    """
    # Check for high-severity findings
    critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
    high_findings = [f for f in findings if f.severity == Severity.HIGH]

    # Strong evidence path - multiple high-severity findings
    if len(critical_findings) >= 2:
        if overall_score >= 0.3:
            return Verdict.DEEPFAKE, min(0.9, 0.5 + overall_score * 0.4)
    elif len(critical_findings) >= 1 and len(high_findings) >= 2:
        if overall_score >= 0.3:
            return Verdict.DEEPFAKE, min(0.9, 0.5 + overall_score * 0.4)

    # Score-based decision
    if overall_score >= config.deepfake_threshold:
        margin = overall_score - config.deepfake_threshold
        base_confidence = 0.4 + min(margin / 0.3, 0.5)

        if len(high_findings) >= 2:
            base_confidence = min(base_confidence + 0.15, 0.95)

        return Verdict.DEEPFAKE, base_confidence

    elif overall_score <= config.uncertain_threshold:
        margin = config.uncertain_threshold - overall_score
        confidence = min(0.95, 0.7 + margin * 1.5)
        return Verdict.REAL, confidence

    else:
        confidence = 0.3 + abs(overall_score - 0.3) * 0.3
        return Verdict.UNCERTAIN, confidence


def compute_frame_scores(
    skill_results: Dict[str, SkillResult],
    total_frames: int,
    config: AnalysisConfig
) -> List[float]:
    """
    Compute per-frame suspicion scores.

    Args:
        skill_results: Dictionary of skill results
        total_frames: Total number of video frames
        config: Analysis configuration with skill weights

    Returns:
        List of per-frame suspicion scores
    """
    frame_scores = [0.0] * total_frames

    for skill_name, result in skill_results.items():
        if "frame_scores" in result.raw_data:
            skill_frame_scores = result.raw_data["frame_scores"]
            weight = config.skill_weights.get(skill_name, 1.0)

            for i, score in enumerate(skill_frame_scores):
                if i < len(frame_scores):
                    frame_scores[i] += score * weight

    # Normalize
    max_weight = sum(config.skill_weights.values())
    frame_scores = [s / max_weight for s in frame_scores]

    return frame_scores


def identify_suspicious_segments(
    frame_scores: List[float],
    fps: float,
    threshold: float = 0.5
) -> List[TimeSegment]:
    """
    Identify time segments with high suspicion scores.

    Args:
        frame_scores: Per-frame suspicion scores
        fps: Video frames per second
        threshold: Score threshold for suspicious frames

    Returns:
        List of top 5 suspicious time segments
    """
    segments = []
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
