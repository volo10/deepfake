"""
Data models and structures for the Deepfake Detection Agent.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
import numpy as np


class Verdict(Enum):
    """Detection verdict."""
    REAL = "REAL"
    DEEPFAKE = "DEEPFAKE"
    UNCERTAIN = "UNCERTAIN"


class Severity(Enum):
    """Finding severity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BoundingBox:
    """Face bounding box."""
    x: int
    y: int
    width: int
    height: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    def area(self) -> int:
        return self.width * self.height


@dataclass
class HeadPose:
    """Head pose estimation."""
    pitch: float  # Up/down rotation
    yaw: float    # Left/right rotation
    roll: float   # Tilt


@dataclass
class DetectedFace:
    """Single detected face in a frame."""
    face_id: int
    bbox: BoundingBox
    confidence: float
    landmarks: np.ndarray  # (N, 2) or (N, 3) facial landmarks
    aligned_face: Optional[np.ndarray] = None  # 224x224 aligned face crop
    embedding: Optional[np.ndarray] = None  # Face embedding vector
    pose: Optional[HeadPose] = None
    regions: Dict[str, np.ndarray] = field(default_factory=dict)  # ROIs


@dataclass
class FrameAnalysis:
    """Analysis results for a single frame."""
    frame_idx: int
    timestamp: float
    faces: List[DetectedFace]
    suspicion_score: float = 0.0
    findings: List['Finding'] = field(default_factory=list)


@dataclass
class Finding:
    """A single detection finding/anomaly."""
    category: str  # e.g., "temporal", "frequency", "audio_visual"
    description: str
    severity: Severity
    confidence: float  # 0-1
    frame_range: Optional[Tuple[int, int]] = None  # Start and end frame
    time_range: Optional[Tuple[float, float]] = None  # Start and end time
    location: Optional[BoundingBox] = None  # Spatial location if applicable
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "frame_range": self.frame_range,
            "time_range": self.time_range,
            "evidence": self.evidence
        }


@dataclass
class TimeSegment:
    """A suspicious time segment in the video."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    suspicion_score: float
    reason: str
    findings: List[Finding] = field(default_factory=list)


@dataclass
class SkillResult:
    """Result from a single analysis skill."""
    skill_name: str
    score: float  # 0-1, where 1 = definitely fake
    confidence: float  # Confidence in the score
    findings: List[Finding]
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DetectionResult:
    """Complete detection result."""
    verdict: Verdict
    confidence: float  # 0-1, confidence in verdict
    
    # Aggregated scores
    overall_score: float  # 0-1, where 1 = definitely fake
    
    # Per-skill results
    skill_results: Dict[str, SkillResult] = field(default_factory=dict)
    
    # Aggregated findings
    findings: List[Finding] = field(default_factory=list)
    
    # Temporal analysis
    frame_scores: List[float] = field(default_factory=list)
    suspicious_segments: List[TimeSegment] = field(default_factory=list)
    
    # Explainability
    explanation: str = ""
    attention_maps: Dict[int, np.ndarray] = field(default_factory=dict)
    
    # Metadata
    video_duration: float = 0.0
    fps: float = 0.0
    total_frames: int = 0
    faces_detected: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "overall_score": self.overall_score,
            "findings": [f.to_dict() for f in self.findings],
            "suspicious_segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "score": s.suspicion_score,
                    "reason": s.reason
                }
                for s in self.suspicious_segments
            ],
            "explanation": self.explanation,
            "metadata": {
                "video_duration": self.video_duration,
                "fps": self.fps,
                "total_frames": self.total_frames,
                "faces_detected": self.faces_detected,
                "processing_time": self.processing_time
            }
        }
    
    def summary(self) -> str:
        """Generate a brief summary."""
        icon = "⚠️" if self.verdict == Verdict.DEEPFAKE else "✓" if self.verdict == Verdict.REAL else "❓"
        return f"{icon} {self.verdict.value} (confidence: {self.confidence:.1%})"


@dataclass
class VideoMetadata:
    """Video file metadata."""
    path: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    has_audio: bool
    codec: str = ""
    
    
@dataclass
class AnalysisConfig:
    """Configuration for the analysis pipeline."""
    # General settings
    max_frames: int = 300  # Maximum frames to analyze
    sample_rate: int = 1   # Analyze every Nth frame
    
    # Face detection
    min_face_size: int = 64
    max_faces: int = 5
    
    # Thresholds (calibrated on real vs fake video analysis)
    deepfake_threshold: float = 0.35  # Score above this = DEEPFAKE
    uncertain_threshold: float = 0.25  # Score below this = REAL
    
    # Skill weights
    skill_weights: Dict[str, float] = field(default_factory=lambda: {
        "face_tracking": 1.0,
        "temporal": 1.3,
        "physiological": 0.5,  # Very low weight - often noisy/unreliable without proper setup
        "frequency": 1.1,
        "audio_visual": 1.2,
        "identity": 1.0,
        "visual_artifacts": 3.0  # Highest weight - most reliable discriminator for modern deepfakes
    })
    
    # Features to enable/disable
    enable_rppg: bool = True
    enable_audio_analysis: bool = True
    enable_frequency_analysis: bool = True
    
    # Output settings
    generate_attention_maps: bool = True
    generate_timeline: bool = True
    verbose: bool = False

