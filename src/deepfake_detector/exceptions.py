"""
Custom exceptions for the Deepfake Detection Agent.

This module defines a hierarchy of exceptions for handling errors
throughout the detection pipeline.
"""

from __future__ import annotations

from typing import Any


class DeepfakeError(Exception):
    """
    Base exception for all deepfake detector errors.
    
    All exceptions raised by this package inherit from this class,
    making it easy to catch any detector-related error.
    
    Attributes:
        message: Human-readable error message
        code: Error code for programmatic handling
        details: Additional context about the error
    
    Example:
        >>> try:
        ...     result = detector.analyze("video.mp4")
        ... except DeepfakeError as e:
        ...     print(f"Detection failed: {e}")
    """
    
    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "E000"
        self.details = details or {}
    
    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class VideoLoadError(DeepfakeError):
    """
    Error loading or reading a video file.
    
    Raised when:
        - Video file doesn't exist
        - Video file is corrupted
        - Video format is not supported
        - Video cannot be decoded
    
    Attributes:
        video_path: Path to the problematic video file
    
    Example:
        >>> try:
        ...     result = detector.analyze("missing.mp4")
        ... except VideoLoadError as e:
        ...     print(f"Cannot load video: {e.video_path}")
    """
    
    def __init__(
        self,
        message: str,
        video_path: str,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="E001", details=details)
        self.video_path = video_path


class FaceDetectionError(DeepfakeError):
    """
    Error during face detection or tracking.
    
    Raised when:
        - No faces are detected in the video
        - Face detector fails to initialize
        - Face is too small for analysis
        - Face tracking loses the target
    
    Attributes:
        frame_count: Number of frames processed before failure
        face_count: Number of faces detected before failure
    
    Example:
        >>> try:
        ...     result = detector.analyze("no_face_video.mp4")
        ... except FaceDetectionError as e:
        ...     print(f"Face detection failed after {e.frame_count} frames")
    """
    
    def __init__(
        self,
        message: str,
        frame_count: int = 0,
        face_count: int = 0,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="E002", details=details)
        self.frame_count = frame_count
        self.face_count = face_count


class AnalysisError(DeepfakeError):
    """
    Error during video analysis.
    
    Raised when:
        - A skill module fails
        - Evidence aggregation fails
        - Unexpected error during processing
    
    Attributes:
        skill_name: Name of the skill that failed (if applicable)
        stage: Analysis stage where failure occurred
    
    Example:
        >>> try:
        ...     result = detector.analyze("video.mp4")
        ... except AnalysisError as e:
        ...     print(f"Analysis failed in skill: {e.skill_name}")
    """
    
    def __init__(
        self,
        message: str,
        skill_name: str | None = None,
        stage: str | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="E003", details=details)
        self.skill_name = skill_name
        self.stage = stage


class ConfigurationError(DeepfakeError):
    """
    Error in configuration.
    
    Raised when:
        - Configuration file is invalid
        - Required configuration is missing
        - Configuration value is out of range
        - Environment variable has invalid value
    
    Attributes:
        config_key: Name of the problematic configuration key
        expected_type: Expected type of the configuration value
        actual_value: Actual value that caused the error
    
    Example:
        >>> try:
        ...     config = AnalysisConfig.from_file("invalid.yaml")
        ... except ConfigurationError as e:
        ...     print(f"Invalid config key: {e.config_key}")
    """
    
    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        expected_type: str | None = None,
        actual_value: Any = None,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="E010", details=details)
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class AudioExtractionError(DeepfakeError):
    """
    Error extracting audio from video.
    
    Raised when:
        - FFmpeg is not installed
        - Audio track is missing
        - Audio cannot be decoded
    
    Note: This is typically a warning-level issue, as
    analysis can proceed without audio.
    """
    
    def __init__(
        self,
        message: str,
        video_path: str,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="E006", details=details)
        self.video_path = video_path


class TimeoutError(DeepfakeError):
    """
    Analysis timed out.
    
    Raised when:
        - Video analysis exceeds the configured timeout
        - A skill module takes too long to complete
    
    Attributes:
        timeout_seconds: Configured timeout that was exceeded
        elapsed_seconds: Actual time elapsed before timeout
    """
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        elapsed_seconds: float,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="E020", details=details)
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class ResourceError(DeepfakeError):
    """
    System resource error.
    
    Raised when:
        - Memory limit exceeded
        - GPU memory exhausted
        - File system full
    
    Attributes:
        resource_type: Type of resource that was exhausted
        limit: Configured limit that was exceeded
        usage: Actual usage that caused the error
    """
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        limit: float | int | None = None,
        usage: float | int | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="E030", details=details)
        self.resource_type = resource_type
        self.limit = limit
        self.usage = usage

