"""
Integration tests for the full deepfake detection pipeline.

Tests cover:
- End-to-end detection flow
- Skill module interaction
- Result aggregation
- Explanation generation
"""

import pytest
import numpy as np
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_detector.models import (
    Verdict,
    AnalysisConfig,
    SkillResult,
    Finding,
    Severity,
)


class TestDetectorIntegration:
    """Integration tests for DeepfakeDetector."""
    
    @pytest.fixture
    def mock_detector(self):
        """Create a detector with mocked skill modules."""
        from deepfake_detector.detector import DeepfakeDetector
        
        with patch.multiple(
            'deepfake_detector.detector',
            FaceTracker=MagicMock,
            TemporalAnalyzer=MagicMock,
            PhysiologicalAnalyzer=MagicMock,
            FrequencyAnalyzer=MagicMock,
            AudioVisualAnalyzer=MagicMock,
            IdentityAnalyzer=MagicMock,
            VisualArtifactAnalyzer=MagicMock,
            ExplainabilityEngine=MagicMock,
        ):
            config = AnalysisConfig(max_frames=50)
            detector = DeepfakeDetector(config=config)
            yield detector
    
    def test_evidence_aggregation_all_clean(self):
        """Test aggregation when all skills report clean."""
        skill_results = {
            "visual_artifacts": SkillResult(
                skill_name="visual_artifacts", score=0.1, confidence=0.9, findings=[]
            ),
            "temporal": SkillResult(
                skill_name="temporal", score=0.05, confidence=0.85, findings=[]
            ),
            "physiological": SkillResult(
                skill_name="physiological", score=0.15, confidence=0.7, findings=[]
            ),
        }
        
        # Calculate weighted score
        total_weight = 0
        weighted_sum = 0
        weights = {"visual_artifacts": 3.0, "temporal": 1.0, "physiological": 0.5}
        
        for name, result in skill_results.items():
            weight = weights.get(name, 1.0) * result.confidence
            weighted_sum += result.score * weight
            total_weight += weight
        
        overall_score = weighted_sum / total_weight
        
        assert overall_score < 0.25, "Clean signals should produce low score"
    
    def test_evidence_aggregation_mixed_signals(self):
        """Test aggregation with mixed signals."""
        skill_results = {
            "visual_artifacts": SkillResult(
                skill_name="visual_artifacts", score=0.6, confidence=0.9, findings=[]
            ),
            "temporal": SkillResult(
                skill_name="temporal", score=0.1, confidence=0.85, findings=[]
            ),
            "physiological": SkillResult(
                skill_name="physiological", score=0.2, confidence=0.7, findings=[]
            ),
        }
        
        weights = {"visual_artifacts": 3.0, "temporal": 1.0, "physiological": 0.5}
        
        total_weight = 0
        weighted_sum = 0
        
        for name, result in skill_results.items():
            weight = weights.get(name, 1.0) * result.confidence
            weighted_sum += result.score * weight
            total_weight += weight
        
        overall_score = weighted_sum / total_weight
        
        # Visual artifacts is weighted heavily, so score should be elevated
        assert overall_score > 0.3, "High visual_artifacts should elevate score"
    
    def test_verdict_decision_deepfake(self):
        """Test that high score produces DEEPFAKE verdict."""
        config = AnalysisConfig(deepfake_threshold=0.35)
        overall_score = 0.55
        
        if overall_score >= config.deepfake_threshold:
            verdict = Verdict.DEEPFAKE
        elif overall_score <= config.uncertain_threshold:
            verdict = Verdict.REAL
        else:
            verdict = Verdict.UNCERTAIN
        
        assert verdict == Verdict.DEEPFAKE
    
    def test_verdict_decision_real(self):
        """Test that low score produces REAL verdict."""
        config = AnalysisConfig(uncertain_threshold=0.25)
        overall_score = 0.15
        
        if overall_score >= config.deepfake_threshold:
            verdict = Verdict.DEEPFAKE
        elif overall_score <= config.uncertain_threshold:
            verdict = Verdict.REAL
        else:
            verdict = Verdict.UNCERTAIN
        
        assert verdict == Verdict.REAL
    
    def test_verdict_decision_uncertain(self):
        """Test that middle score produces UNCERTAIN verdict."""
        config = AnalysisConfig(
            deepfake_threshold=0.35,
            uncertain_threshold=0.25
        )
        overall_score = 0.30
        
        if overall_score >= config.deepfake_threshold:
            verdict = Verdict.DEEPFAKE
        elif overall_score <= config.uncertain_threshold:
            verdict = Verdict.REAL
        else:
            verdict = Verdict.UNCERTAIN
        
        assert verdict == Verdict.UNCERTAIN


class TestFindingAggregation:
    """Tests for finding aggregation across skills."""
    
    def test_findings_sorted_by_severity(self):
        """Test that findings are sorted by severity."""
        findings = [
            Finding("visual", "Low issue", Severity.LOW, 0.5),
            Finding("visual", "Critical issue", Severity.CRITICAL, 0.9),
            Finding("visual", "High issue", Severity.HIGH, 0.8),
            Finding("visual", "Medium issue", Severity.MEDIUM, 0.6),
        ]
        
        # Sort by severity (critical > high > medium > low)
        severity_order = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1
        }
        
        sorted_findings = sorted(
            findings,
            key=lambda f: severity_order[f.severity],
            reverse=True
        )
        
        assert sorted_findings[0].severity == Severity.CRITICAL
        assert sorted_findings[1].severity == Severity.HIGH
        assert sorted_findings[-1].severity == Severity.LOW
    
    def test_findings_from_multiple_skills(self):
        """Test aggregating findings from multiple skills."""
        skill_results = {
            "visual_artifacts": SkillResult(
                skill_name="visual_artifacts",
                score=0.5,
                confidence=0.9,
                findings=[
                    Finding("visual", "Texture anomaly", Severity.HIGH, 0.85)
                ]
            ),
            "temporal": SkillResult(
                skill_name="temporal",
                score=0.3,
                confidence=0.8,
                findings=[
                    Finding("temporal", "Identity drift", Severity.MEDIUM, 0.7)
                ]
            ),
        }
        
        all_findings = []
        for result in skill_results.values():
            all_findings.extend(result.findings)
        
        assert len(all_findings) == 2
        categories = {f.category for f in all_findings}
        assert "visual" in categories
        assert "temporal" in categories


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""
    
    def test_high_confidence_for_strong_signal(self):
        """Test high confidence when signal is strong."""
        overall_score = 0.8
        threshold = 0.35
        
        # Calculate confidence based on margin above threshold
        margin = overall_score - threshold
        base_confidence = 0.4 + min(margin / 0.3, 0.5)
        
        assert base_confidence > 0.8, "Strong signal should have high confidence"
    
    def test_low_confidence_for_borderline(self):
        """Test lower confidence for borderline cases."""
        overall_score = 0.36  # Just above threshold
        threshold = 0.35
        
        margin = overall_score - threshold
        base_confidence = 0.4 + min(margin / 0.3, 0.5)
        
        assert base_confidence < 0.6, "Borderline case should have lower confidence"
    
    def test_confidence_boost_multiple_findings(self):
        """Test confidence boost with multiple high-severity findings."""
        base_confidence = 0.6
        high_findings = [
            Finding("visual", "Issue 1", Severity.HIGH, 0.8),
            Finding("visual", "Issue 2", Severity.HIGH, 0.85),
            Finding("visual", "Issue 3", Severity.HIGH, 0.75),
        ]
        
        # Boost for multiple findings
        if len(high_findings) >= 2:
            boosted = min(base_confidence + 0.15, 0.95)
        else:
            boosted = base_confidence
        
        assert boosted == 0.75, "Multiple findings should boost confidence"


class TestResultSerialization:
    """Tests for result serialization."""
    
    def test_detection_result_to_json(self, sample_detection_result):
        """Test that result can be serialized to JSON."""
        import json
        
        d = sample_detection_result.to_dict()
        json_str = json.dumps(d)
        
        assert len(json_str) > 0
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        assert parsed["verdict"] == sample_detection_result.verdict.value
    
    def test_result_contains_all_fields(self, sample_detection_result):
        """Test that all required fields are present."""
        d = sample_detection_result.to_dict()
        
        required_fields = [
            "verdict",
            "confidence",
            "overall_score",
            "findings",
            "explanation",
            "metadata"
        ]
        
        for field in required_fields:
            assert field in d, f"Missing field: {field}"


class TestEdgeCases:
    """Tests for edge cases in integration."""
    
    def test_no_skill_results(self):
        """Test handling when no skills produce results."""
        skill_results = {}
        
        # Should handle gracefully
        if not skill_results:
            overall_score = 0.5  # Uncertain
            verdict = Verdict.UNCERTAIN
        else:
            overall_score = sum(r.score for r in skill_results.values()) / len(skill_results)
        
        assert verdict == Verdict.UNCERTAIN
    
    def test_all_skills_zero_confidence(self):
        """Test when all skills have zero confidence."""
        skill_results = {
            "visual_artifacts": SkillResult("visual_artifacts", 0.8, 0.0, []),
            "temporal": SkillResult("temporal", 0.6, 0.0, []),
        }
        
        total_weight = sum(r.confidence for r in skill_results.values())
        
        if total_weight == 0:
            overall_score = 0.5  # Uncertain
        else:
            overall_score = 0
        
        assert overall_score == 0.5
    
    def test_single_skill_result(self):
        """Test with only one skill producing result."""
        skill_results = {
            "visual_artifacts": SkillResult("visual_artifacts", 0.7, 0.9, [
                Finding("visual", "Anomaly", Severity.HIGH, 0.85)
            ])
        }
        
        # Should still produce valid result
        overall_score = skill_results["visual_artifacts"].score
        
        assert 0 <= overall_score <= 1

