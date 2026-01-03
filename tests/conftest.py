"""
Pytest configuration and fixtures for the deepfake detector test suite.

This module provides reusable fixtures for testing various components
of the deepfake detection system.
"""

import numpy as np
import pytest
from pathlib import Path
from typing import List, Tuple, Generator
from unittest.mock import MagicMock, patch

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Return default analysis configuration."""
    from deepfake_detector.models import AnalysisConfig
    return AnalysisConfig()


@pytest.fixture
def strict_config():
    """Return strict configuration for high-precision detection."""
    from deepfake_detector.models import AnalysisConfig
    return AnalysisConfig(
        max_frames=500,
        sample_rate=1,
        deepfake_threshold=0.40,
        uncertain_threshold=0.20,
    )


@pytest.fixture
def fast_config():
    """Return fast configuration for quick testing."""
    from deepfake_detector.models import AnalysisConfig
    return AnalysisConfig(
        max_frames=50,
        sample_rate=5,
        deepfake_threshold=0.35,
        uncertain_threshold=0.25,
    )


# =============================================================================
# Image and Video Fixtures
# =============================================================================

@pytest.fixture
def sample_frame() -> np.ndarray:
    """Generate a sample video frame (RGB image)."""
    # Create a 480x640 RGB image with random noise
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_face_crop() -> np.ndarray:
    """Generate a sample aligned face crop (224x224)."""
    # Create a face-sized crop with skin-tone-like colors
    face = np.zeros((224, 224, 3), dtype=np.uint8)
    face[:, :, 0] = 180  # B
    face[:, :, 1] = 160  # G
    face[:, :, 2] = 200  # R
    # Add some variation
    noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
    face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return face


@pytest.fixture
def sample_face_sequence(sample_face_crop) -> List[np.ndarray]:
    """Generate a sequence of face crops simulating video frames."""
    faces = []
    for i in range(30):
        # Create slight variations
        variation = np.random.randint(-5, 5, sample_face_crop.shape, dtype=np.int16)
        frame_face = np.clip(
            sample_face_crop.astype(np.int16) + variation, 0, 255
        ).astype(np.uint8)
        faces.append(frame_face)
    return faces


@pytest.fixture
def sample_frames() -> List[np.ndarray]:
    """Generate a sequence of video frames."""
    frames = []
    for i in range(30):
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def synthetic_deepfake_face() -> np.ndarray:
    """Generate a face with deepfake-like artifacts (over-smoothed)."""
    # Very smooth texture (low variance)
    face = np.ones((224, 224, 3), dtype=np.uint8) * 180
    # Add very subtle noise
    noise = np.random.randint(-2, 2, (224, 224, 3), dtype=np.int16)
    face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return face


@pytest.fixture
def synthetic_real_face() -> np.ndarray:
    """Generate a face with natural texture (realistic variance)."""
    # Base skin tone
    face = np.zeros((224, 224, 3), dtype=np.uint8)
    face[:, :, 0] = 180  # B
    face[:, :, 1] = 160  # G
    face[:, :, 2] = 200  # R
    
    # Add natural texture (higher variance)
    noise = np.random.randint(-30, 30, (224, 224, 3), dtype=np.int16)
    face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some structure (simulating natural features)
    face[80:100, 90:110] = [150, 140, 160]  # Eye region
    face[80:100, 114:134] = [150, 140, 160]  # Other eye
    face[130:150, 100:124] = [170, 150, 180]  # Nose
    
    return face


# =============================================================================
# Bounding Box and Landmark Fixtures
# =============================================================================

@pytest.fixture
def sample_bbox() -> Tuple[int, int, int, int]:
    """Return a sample face bounding box."""
    return (100, 50, 200, 250)  # x, y, width, height


@pytest.fixture
def sample_landmarks() -> np.ndarray:
    """Generate sample facial landmarks (68-point)."""
    # Generate 68 landmark points
    landmarks = np.zeros((68, 2), dtype=np.float32)
    
    # Face contour (0-16)
    for i in range(17):
        landmarks[i] = [50 + i * 8, 80 + abs(i - 8) * 3]
    
    # Left eyebrow (17-21)
    for i in range(5):
        landmarks[17 + i] = [60 + i * 6, 50]
    
    # Right eyebrow (22-26)
    for i in range(5):
        landmarks[22 + i] = [100 + i * 6, 50]
    
    # Nose (27-35)
    for i in range(9):
        landmarks[27 + i] = [92, 60 + i * 5]
    
    # Left eye (36-41)
    landmarks[36:42] = [[65, 65], [70, 62], [78, 62], [82, 65], [78, 68], [70, 68]]
    
    # Right eye (42-47)
    landmarks[42:48] = [[105, 65], [110, 62], [118, 62], [122, 65], [118, 68], [110, 68]]
    
    # Mouth outer (48-59)
    for i in range(12):
        angle = i * 30 * np.pi / 180
        landmarks[48 + i] = [92 + 15 * np.cos(angle), 130 + 8 * np.sin(angle)]
    
    # Mouth inner (60-67)
    for i in range(8):
        angle = i * 45 * np.pi / 180
        landmarks[60 + i] = [92 + 10 * np.cos(angle), 130 + 5 * np.sin(angle)]
    
    return landmarks


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_video_capture():
    """Create a mock for cv2.VideoCapture."""
    with patch("cv2.VideoCapture") as mock:
        cap_instance = MagicMock()
        cap_instance.get.side_effect = lambda x: {
            5: 30.0,    # FPS
            3: 640.0,   # WIDTH
            4: 480.0,   # HEIGHT
            7: 100.0,   # FRAME_COUNT
        }.get(x, 0.0)
        cap_instance.isOpened.return_value = True
        cap_instance.read.return_value = (
            True,
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        )
        mock.return_value = cap_instance
        yield mock


@pytest.fixture
def mock_face_cascade():
    """Create a mock for face detection cascade."""
    with patch("cv2.CascadeClassifier") as mock:
        cascade_instance = MagicMock()
        cascade_instance.detectMultiScale.return_value = np.array([[100, 50, 200, 250]])
        mock.return_value = cascade_instance
        yield mock


# =============================================================================
# Data Model Fixtures
# =============================================================================

@pytest.fixture
def sample_finding():
    """Create a sample Finding object."""
    from deepfake_detector.models import Finding, Severity
    return Finding(
        category="visual",
        description="Over-smoothed texture detected",
        severity=Severity.HIGH,
        confidence=0.85,
        frame_range=(10, 20),
        evidence={"texture_variance": 25.3}
    )


@pytest.fixture
def sample_skill_result(sample_finding):
    """Create a sample SkillResult object."""
    from deepfake_detector.models import SkillResult
    return SkillResult(
        skill_name="visual_artifacts",
        score=0.65,
        confidence=0.85,
        findings=[sample_finding],
        raw_data={"texture_variance": 25.3, "symmetry": 0.75}
    )


@pytest.fixture
def sample_detection_result(sample_skill_result):
    """Create a sample DetectionResult object."""
    from deepfake_detector.models import DetectionResult, Verdict
    return DetectionResult(
        verdict=Verdict.DEEPFAKE,
        confidence=0.82,
        overall_score=0.58,
        skill_results={"visual_artifacts": sample_skill_result},
        findings=sample_skill_result.findings,
        explanation="Video shows signs of manipulation",
        video_duration=10.5,
        fps=30.0,
        total_frames=315,
        faces_detected=300,
        processing_time=15.3
    )


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_video_path(tmp_path) -> Path:
    """Create a temporary video file path."""
    return tmp_path / "test_video.mp4"


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_logging():
    """Reset logging configuration after each test."""
    import logging
    yield
    # Reset loggers
    logging.getLogger("deepfake_detector").handlers = []

