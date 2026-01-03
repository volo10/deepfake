"""
Unit tests for custom exceptions.

Tests cover:
- Exception creation
- Exception attributes
- Exception string representation
- Exception serialization
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_detector.exceptions import (
    DeepfakeError,
    VideoLoadError,
    FaceDetectionError,
    AnalysisError,
    ConfigurationError,
    AudioExtractionError,
    TimeoutError,
    ResourceError,
)


class TestDeepfakeError:
    """Tests for base DeepfakeError."""
    
    def test_basic_creation(self):
        """Test basic exception creation."""
        error = DeepfakeError("Test error message")
        assert str(error) == "[E000] Test error message"
        assert error.message == "Test error message"
        assert error.code == "E000"
    
    def test_with_custom_code(self):
        """Test exception with custom error code."""
        error = DeepfakeError("Custom error", code="E999")
        assert error.code == "E999"
        assert "[E999]" in str(error)
    
    def test_with_details(self):
        """Test exception with additional details."""
        error = DeepfakeError(
            "Error with details",
            details={"key": "value", "count": 42}
        )
        assert error.details["key"] == "value"
        assert error.details["count"] == 42
    
    def test_to_dict(self):
        """Test exception serialization."""
        error = DeepfakeError(
            "Test error",
            code="E001",
            details={"test": True}
        )
        d = error.to_dict()
        
        assert d["error_type"] == "DeepfakeError"
        assert d["code"] == "E001"
        assert d["message"] == "Test error"
        assert d["details"]["test"] is True
    
    def test_can_be_raised(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(DeepfakeError) as exc_info:
            raise DeepfakeError("Test raise")
        
        assert "Test raise" in str(exc_info.value)
    
    def test_inherits_from_exception(self):
        """Test that DeepfakeError inherits from Exception."""
        error = DeepfakeError("Test")
        assert isinstance(error, Exception)


class TestVideoLoadError:
    """Tests for VideoLoadError."""
    
    def test_creation(self):
        """Test VideoLoadError creation."""
        error = VideoLoadError(
            "Cannot load video",
            video_path="/path/to/video.mp4"
        )
        assert error.video_path == "/path/to/video.mp4"
        assert error.code == "E001"
    
    def test_with_details(self):
        """Test VideoLoadError with details."""
        error = VideoLoadError(
            "Corrupted video",
            video_path="/path/to/video.mp4",
            details={"codec": "unknown", "size": 0}
        )
        assert error.details["codec"] == "unknown"
    
    def test_inherits_from_deepfake_error(self):
        """Test inheritance chain."""
        error = VideoLoadError("Test", video_path="/test.mp4")
        assert isinstance(error, DeepfakeError)
        assert isinstance(error, Exception)


class TestFaceDetectionError:
    """Tests for FaceDetectionError."""
    
    def test_creation(self):
        """Test FaceDetectionError creation."""
        error = FaceDetectionError(
            "No faces found",
            frame_count=100,
            face_count=0
        )
        assert error.frame_count == 100
        assert error.face_count == 0
        assert error.code == "E002"
    
    def test_default_values(self):
        """Test default attribute values."""
        error = FaceDetectionError("Face too small")
        assert error.frame_count == 0
        assert error.face_count == 0
    
    def test_to_dict(self):
        """Test serialization preserves attributes."""
        error = FaceDetectionError(
            "Detection failed",
            frame_count=50,
            face_count=25
        )
        d = error.to_dict()
        
        assert d["error_type"] == "FaceDetectionError"
        assert d["code"] == "E002"


class TestAnalysisError:
    """Tests for AnalysisError."""
    
    def test_creation(self):
        """Test AnalysisError creation."""
        error = AnalysisError(
            "Skill failed",
            skill_name="visual_artifacts",
            stage="analysis"
        )
        assert error.skill_name == "visual_artifacts"
        assert error.stage == "analysis"
        assert error.code == "E003"
    
    def test_without_skill(self):
        """Test AnalysisError without specific skill."""
        error = AnalysisError("General failure")
        assert error.skill_name is None
        assert error.stage is None


class TestConfigurationError:
    """Tests for ConfigurationError."""
    
    def test_creation(self):
        """Test ConfigurationError creation."""
        error = ConfigurationError(
            "Invalid threshold",
            config_key="deepfake_threshold",
            expected_type="float",
            actual_value="invalid"
        )
        assert error.config_key == "deepfake_threshold"
        assert error.expected_type == "float"
        assert error.actual_value == "invalid"
        assert error.code == "E010"
    
    def test_minimal_creation(self):
        """Test ConfigurationError with minimal info."""
        error = ConfigurationError("Config file not found")
        assert error.config_key is None


class TestAudioExtractionError:
    """Tests for AudioExtractionError."""
    
    def test_creation(self):
        """Test AudioExtractionError creation."""
        error = AudioExtractionError(
            "FFmpeg not found",
            video_path="/path/to/video.mp4"
        )
        assert error.video_path == "/path/to/video.mp4"
        assert error.code == "E006"


class TestTimeoutError:
    """Tests for TimeoutError."""
    
    def test_creation(self):
        """Test TimeoutError creation."""
        error = TimeoutError(
            "Analysis timed out",
            timeout_seconds=300,
            elapsed_seconds=305.5
        )
        assert error.timeout_seconds == 300
        assert error.elapsed_seconds == 305.5
        assert error.code == "E020"


class TestResourceError:
    """Tests for ResourceError."""
    
    def test_creation(self):
        """Test ResourceError creation."""
        error = ResourceError(
            "Out of memory",
            resource_type="memory",
            limit=4096,
            usage=5000
        )
        assert error.resource_type == "memory"
        assert error.limit == 4096
        assert error.usage == 5000
        assert error.code == "E030"
    
    def test_gpu_memory_error(self):
        """Test GPU memory error."""
        error = ResourceError(
            "CUDA out of memory",
            resource_type="gpu_memory",
            limit=8192,
            usage=8500
        )
        assert error.resource_type == "gpu_memory"


class TestExceptionHierarchy:
    """Tests for exception hierarchy and catching."""
    
    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        try:
            raise VideoLoadError("Test", video_path="/test.mp4")
        except VideoLoadError as e:
            assert e.video_path == "/test.mp4"
    
    def test_catch_base_exception(self):
        """Test catching by base exception type."""
        exceptions = [
            VideoLoadError("Test", video_path="/test.mp4"),
            FaceDetectionError("Test"),
            AnalysisError("Test"),
            ConfigurationError("Test"),
        ]
        
        for exc in exceptions:
            try:
                raise exc
            except DeepfakeError as e:
                assert isinstance(e, DeepfakeError)
    
    def test_exception_chaining(self):
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DeepfakeError("Wrapped error") from e
        except DeepfakeError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)

