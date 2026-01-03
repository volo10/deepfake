"""
Unit tests for data models.

Tests cover:
- Model instantiation
- Field validation
- Serialization/deserialization
- Enum behavior
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_detector.models import (
    Verdict,
    Severity,
    BoundingBox,
    HeadPose,
    DetectedFace,
    FrameAnalysis,
    Finding,
    TimeSegment,
    SkillResult,
    DetectionResult,
    VideoMetadata,
    AnalysisConfig,
)


class TestVerdict:
    """Tests for Verdict enum."""
    
    def test_verdict_values(self):
        """Test that all verdict values are correct."""
        assert Verdict.REAL.value == "REAL"
        assert Verdict.DEEPFAKE.value == "DEEPFAKE"
        assert Verdict.UNCERTAIN.value == "UNCERTAIN"
    
    def test_verdict_from_string(self):
        """Test creating verdict from string."""
        assert Verdict("REAL") == Verdict.REAL
        assert Verdict("DEEPFAKE") == Verdict.DEEPFAKE
        assert Verdict("UNCERTAIN") == Verdict.UNCERTAIN
    
    def test_verdict_invalid_string(self):
        """Test that invalid string raises error."""
        with pytest.raises(ValueError):
            Verdict("INVALID")


class TestSeverity:
    """Tests for Severity enum."""
    
    def test_severity_values(self):
        """Test that all severity values are correct."""
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""
    
    def test_bbox_creation(self):
        """Test basic bounding box creation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 150
    
    def test_bbox_to_tuple(self):
        """Test converting bbox to tuple."""
        bbox = BoundingBox(x=10, y=20, width=100, height=150)
        assert bbox.to_tuple() == (10, 20, 100, 150)
    
    def test_bbox_area(self):
        """Test area calculation."""
        bbox = BoundingBox(x=0, y=0, width=100, height=200)
        assert bbox.area() == 20000
    
    def test_bbox_zero_area(self):
        """Test zero-area bounding box."""
        bbox = BoundingBox(x=0, y=0, width=0, height=100)
        assert bbox.area() == 0


class TestHeadPose:
    """Tests for HeadPose dataclass."""
    
    def test_head_pose_creation(self):
        """Test basic head pose creation."""
        pose = HeadPose(pitch=10.5, yaw=-5.3, roll=2.1)
        assert pose.pitch == 10.5
        assert pose.yaw == -5.3
        assert pose.roll == 2.1


class TestFinding:
    """Tests for Finding dataclass."""
    
    def test_finding_creation(self):
        """Test basic finding creation."""
        finding = Finding(
            category="visual",
            description="Test finding",
            severity=Severity.HIGH,
            confidence=0.85
        )
        assert finding.category == "visual"
        assert finding.severity == Severity.HIGH
        assert finding.confidence == 0.85
    
    def test_finding_with_frame_range(self):
        """Test finding with frame range."""
        finding = Finding(
            category="temporal",
            description="Identity drift",
            severity=Severity.MEDIUM,
            confidence=0.7,
            frame_range=(10, 50)
        )
        assert finding.frame_range == (10, 50)
    
    def test_finding_to_dict(self):
        """Test finding serialization."""
        finding = Finding(
            category="visual",
            description="Test finding",
            severity=Severity.HIGH,
            confidence=0.85,
            evidence={"metric": 0.5}
        )
        d = finding.to_dict()
        assert d["category"] == "visual"
        assert d["severity"] == "high"
        assert d["confidence"] == 0.85
        assert d["evidence"]["metric"] == 0.5


class TestSkillResult:
    """Tests for SkillResult dataclass."""
    
    def test_skill_result_creation(self):
        """Test basic skill result creation."""
        result = SkillResult(
            skill_name="visual_artifacts",
            score=0.65,
            confidence=0.9,
            findings=[]
        )
        assert result.skill_name == "visual_artifacts"
        assert result.score == 0.65
        assert result.confidence == 0.9
    
    def test_skill_result_with_raw_data(self):
        """Test skill result with raw data."""
        result = SkillResult(
            skill_name="temporal",
            score=0.3,
            confidence=0.8,
            findings=[],
            raw_data={"drift": 0.15, "variance": 0.02}
        )
        assert "drift" in result.raw_data
        assert result.raw_data["drift"] == 0.15


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""
    
    def test_detection_result_creation(self):
        """Test basic detection result creation."""
        result = DetectionResult(
            verdict=Verdict.REAL,
            confidence=0.95,
            overall_score=0.15
        )
        assert result.verdict == Verdict.REAL
        assert result.confidence == 0.95
        assert result.overall_score == 0.15
    
    def test_detection_result_to_dict(self):
        """Test detection result serialization."""
        finding = Finding(
            category="test",
            description="Test",
            severity=Severity.LOW,
            confidence=0.5
        )
        result = DetectionResult(
            verdict=Verdict.DEEPFAKE,
            confidence=0.82,
            overall_score=0.58,
            findings=[finding],
            explanation="Test explanation",
            video_duration=10.5,
            fps=30.0,
            total_frames=315,
            processing_time=5.2
        )
        d = result.to_dict()
        assert d["verdict"] == "DEEPFAKE"
        assert d["confidence"] == 0.82
        assert len(d["findings"]) == 1
        assert d["metadata"]["fps"] == 30.0
    
    def test_detection_result_summary(self):
        """Test summary generation."""
        result = DetectionResult(
            verdict=Verdict.REAL,
            confidence=0.9,
            overall_score=0.1
        )
        summary = result.summary()
        assert "REAL" in summary
        assert "90" in summary or "0.9" in summary


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AnalysisConfig()
        assert config.max_frames == 300
        assert config.sample_rate == 1
        assert 0 < config.deepfake_threshold < 1
        assert 0 < config.uncertain_threshold < 1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AnalysisConfig(
            max_frames=500,
            sample_rate=2,
            deepfake_threshold=0.5,
            uncertain_threshold=0.2
        )
        assert config.max_frames == 500
        assert config.sample_rate == 2
        assert config.deepfake_threshold == 0.5
    
    def test_config_skill_weights(self):
        """Test skill weights in configuration."""
        config = AnalysisConfig()
        assert "visual_artifacts" in config.skill_weights
        assert config.skill_weights["visual_artifacts"] > 0
    
    def test_config_features(self):
        """Test feature flags in configuration."""
        config = AnalysisConfig(
            enable_rppg=False,
            enable_audio_analysis=False
        )
        assert config.enable_rppg is False
        assert config.enable_audio_analysis is False


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""
    
    def test_video_metadata_creation(self):
        """Test video metadata creation."""
        meta = VideoMetadata(
            path="/path/to/video.mp4",
            duration=30.5,
            fps=30.0,
            width=1920,
            height=1080,
            total_frames=915,
            has_audio=True
        )
        assert meta.path == "/path/to/video.mp4"
        assert meta.duration == 30.5
        assert meta.fps == 30.0
        assert meta.has_audio is True


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_finding_empty_evidence(self):
        """Test finding with empty evidence."""
        finding = Finding(
            category="test",
            description="Test",
            severity=Severity.LOW,
            confidence=0.0,
            evidence={}
        )
        d = finding.to_dict()
        assert d["evidence"] == {}
    
    def test_detection_result_empty_findings(self):
        """Test detection result with no findings."""
        result = DetectionResult(
            verdict=Verdict.REAL,
            confidence=1.0,
            overall_score=0.0,
            findings=[]
        )
        assert len(result.findings) == 0
        d = result.to_dict()
        assert d["findings"] == []
    
    def test_config_extreme_values(self):
        """Test configuration with extreme values."""
        # Very strict config
        config = AnalysisConfig(
            deepfake_threshold=0.99,
            uncertain_threshold=0.01
        )
        assert config.deepfake_threshold == 0.99
        assert config.uncertain_threshold == 0.01

