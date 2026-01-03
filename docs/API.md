# API Documentation
## Deepfake Detection Agent v2.0

---

## 1. Quick Start

### Installation

```bash
# From PyPI (when published)
pip install deepfake-detector

# From source
git clone https://github.com/your-org/deepfake-detector.git
cd deepfake-detector
pip install -e ".[dev]"
```

### Basic Usage

```python
from deepfake_detector import DeepfakeDetector

# Initialize detector with defaults
detector = DeepfakeDetector()

# Analyze a video
result = detector.analyze("path/to/video.mp4")

# Check verdict
print(f"Verdict: {result.verdict}")        # REAL, DEEPFAKE, or UNCERTAIN
print(f"Confidence: {result.confidence:.1%}")  # e.g., 95.2%
print(f"Explanation: {result.explanation}")
```

---

## 2. Core Classes

### 2.1 DeepfakeDetector

The main entry point for deepfake detection.

```python
class DeepfakeDetector:
    """
    Multi-modal deepfake detection agent.
    
    Aggregates evidence from visual artifacts, temporal analysis,
    physiological signals, and other skills to determine if a
    video is REAL, DEEPFAKE, or UNCERTAIN.
    
    Attributes:
        config (AnalysisConfig): Configuration parameters
        skills (Dict[str, Skill]): Loaded skill modules
    
    Example:
        >>> detector = DeepfakeDetector()
        >>> result = detector.analyze("video.mp4")
        >>> print(result.verdict)
        Verdict.REAL
    """
    
    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
        skills: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the detector.
        
        Args:
            config: Analysis configuration. If None, uses defaults.
            skills: List of skill names to enable. If None, uses all.
        
        Example:
            >>> config = AnalysisConfig(max_frames=100)
            >>> detector = DeepfakeDetector(config=config)
        """
        pass
    
    def analyze(
        self,
        video_path: Union[str, Path],
        callback: Optional[Callable] = None
    ) -> DeepfakeResult:
        """
        Analyze a video for deepfake indicators.
        
        Args:
            video_path: Path to the video file (MP4, AVI, MOV, WebM)
            callback: Optional progress callback(frame_idx, total_frames)
        
        Returns:
            DeepfakeResult containing verdict, confidence, and findings
        
        Raises:
            FileNotFoundError: If video file doesn't exist
            VideoLoadError: If video cannot be read
            AnalysisError: If analysis fails
        
        Example:
            >>> result = detector.analyze("test.mp4")
            >>> if result.verdict == Verdict.DEEPFAKE:
            ...     print(f"Deepfake detected with {result.confidence:.1%} confidence")
        """
        pass
    
    def analyze_batch(
        self,
        video_paths: List[Union[str, Path]],
        max_workers: int = 4,
        progress: bool = True
    ) -> List[DeepfakeResult]:
        """
        Analyze multiple videos in parallel.
        
        Args:
            video_paths: List of paths to video files
            max_workers: Maximum number of parallel workers
            progress: Whether to show progress bar
        
        Returns:
            List of DeepfakeResult objects, one per video
        
        Example:
            >>> videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
            >>> results = detector.analyze_batch(videos, max_workers=4)
        """
        pass
```

### 2.2 DeepfakeResult

The result of a deepfake analysis.

```python
@dataclass
class DeepfakeResult:
    """
    Result of deepfake analysis.
    
    Attributes:
        verdict: The final verdict (REAL, DEEPFAKE, UNCERTAIN)
        confidence: Confidence in the verdict (0.0 to 1.0)
        overall_score: Raw anomaly score (0.0 to 1.0)
        skill_results: Results from each skill module
        findings: List of specific findings/anomalies
        explanation: Human-readable explanation
        processing_time: Time taken for analysis (seconds)
        metadata: Additional metadata about the video
    """
    
    verdict: Verdict
    confidence: float
    overall_score: float
    skill_results: Dict[str, SkillResult]
    findings: List[Finding]
    explanation: str
    processing_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        pass
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        pass
    
    def to_html(self, template: str = "default") -> str:
        """Generate HTML report."""
        pass
    
    @property
    def is_fake(self) -> bool:
        """Returns True if verdict is DEEPFAKE."""
        return self.verdict == Verdict.DEEPFAKE
    
    @property
    def is_real(self) -> bool:
        """Returns True if verdict is REAL."""
        return self.verdict == Verdict.REAL
    
    @property
    def is_uncertain(self) -> bool:
        """Returns True if verdict is UNCERTAIN."""
        return self.verdict == Verdict.UNCERTAIN
```

### 2.3 AnalysisConfig

Configuration for the analysis.

```python
@dataclass
class AnalysisConfig:
    """
    Configuration for deepfake analysis.
    
    Attributes:
        max_frames: Maximum frames to analyze (default: 200)
        sample_rate: Sample every Nth frame (default: 2)
        deepfake_threshold: Score above this = DEEPFAKE (default: 0.35)
        uncertain_threshold: Score below this = REAL (default: 0.25)
        visual_artifacts_weight: Weight for visual artifacts (default: 3.0)
        temporal_weight: Weight for temporal analysis (default: 1.0)
        physiological_weight: Weight for physiological signals (default: 1.0)
        enable_audio: Whether to analyze audio (default: True)
        enable_gpu: Whether to use GPU acceleration (default: False)
    """
    
    max_frames: int = 200
    sample_rate: int = 2
    deepfake_threshold: float = 0.35
    uncertain_threshold: float = 0.25
    visual_artifacts_weight: float = 3.0
    temporal_weight: float = 1.0
    physiological_weight: float = 1.0
    enable_audio: bool = True
    enable_gpu: bool = False
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AnalysisConfig":
        """Load configuration from YAML or JSON file."""
        pass
    
    @classmethod
    def from_env(cls) -> "AnalysisConfig":
        """Load configuration from environment variables."""
        pass
```

### 2.4 Verdict

Possible analysis verdicts.

```python
class Verdict(Enum):
    """
    Analysis verdict.
    
    Values:
        REAL: Video appears to be authentic
        DEEPFAKE: Video appears to be manipulated
        UNCERTAIN: Cannot determine with confidence
    """
    REAL = "REAL"
    DEEPFAKE = "DEEPFAKE"
    UNCERTAIN = "UNCERTAIN"
```

### 2.5 SkillResult

Result from an individual skill module.

```python
@dataclass
class SkillResult:
    """
    Result from a single skill analysis.
    
    Attributes:
        skill_name: Name of the skill (e.g., "visual_artifacts")
        score: Anomaly score (0.0 = normal, 1.0 = highly anomalous)
        confidence: Confidence in the score (0.0 to 1.0)
        findings: Specific findings from this skill
        raw_data: Raw metrics and measurements
    """
    
    skill_name: str
    score: float
    confidence: float
    findings: List[Finding]
    raw_data: Dict[str, Any]
```

### 2.6 Finding

A specific finding or anomaly.

```python
@dataclass
class Finding:
    """
    A specific finding from analysis.
    
    Attributes:
        category: Finding category (e.g., "visual", "temporal")
        type: Specific type (e.g., "over-smoothed-texture")
        severity: Severity level ("low", "medium", "high", "critical")
        description: Human-readable description
        evidence: Supporting evidence/measurements
        frame_range: Optional frame range where finding was detected
    """
    
    category: str
    type: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    evidence: Dict[str, Any]
    frame_range: Optional[Tuple[int, int]] = None
```

---

## 3. Skill Modules

### 3.1 Visual Artifact Analyzer

```python
class VisualArtifactAnalyzer:
    """
    Analyzes visual artifacts that may indicate deepfakes.
    
    Detects:
        - Over-smoothed or over-sharpened textures
        - Facial asymmetry
        - Edge density anomalies
        - Color inconsistencies (face vs neck)
        - Boundary artifacts
        - Static background anomalies
    
    Example:
        >>> analyzer = VisualArtifactAnalyzer(config)
        >>> result = analyzer.analyze(face_crops, full_frames)
        >>> print(result.score)  # 0.0 = normal, 1.0 = anomalous
    """
    
    def analyze(
        self,
        face_crops: List[np.ndarray],
        full_frames: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]]
    ) -> SkillResult:
        """
        Analyze visual artifacts in face crops.
        
        Args:
            face_crops: List of cropped face images
            full_frames: List of full video frames
            bboxes: Bounding boxes for each face
        
        Returns:
            SkillResult with anomaly score and findings
        """
        pass
```

### 3.2 Temporal Analyzer

```python
class TemporalAnalyzer:
    """
    Analyzes temporal consistency across frames.
    
    Detects:
        - Identity drift over time
        - Abnormal blink patterns
        - Inconsistent expressions
        - Unnatural motion
    """
    
    def analyze(
        self,
        face_crops: List[np.ndarray],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> SkillResult:
        """
        Analyze temporal consistency.
        
        Args:
            face_crops: Sequential face images
            embeddings: Optional pre-computed face embeddings
        
        Returns:
            SkillResult with consistency analysis
        """
        pass
```

### 3.3 Physiological Analyzer

```python
class PhysiologicalAnalyzer:
    """
    Analyzes physiological signals (rPPG).
    
    Detects:
        - Absence of blood flow signal
        - Inconsistent heart rate
        - Unnatural skin color variations
    
    Note: Requires sufficient video length (>5 seconds)
    """
    
    def analyze(
        self,
        face_crops: List[np.ndarray],
        fps: float
    ) -> SkillResult:
        """
        Extract and analyze physiological signals.
        
        Args:
            face_crops: Sequential face images
            fps: Video frame rate
        
        Returns:
            SkillResult with physiological analysis
        """
        pass
```

---

## 4. CLI Interface

### 4.1 Commands

```bash
# Basic analysis
deepfake-detector analyze video.mp4

# With output format
deepfake-detector analyze video.mp4 --output report.json
deepfake-detector analyze video.mp4 --format html --output report.html

# Batch processing
deepfake-detector analyze-batch videos/ --workers 4

# Video info
deepfake-detector info video.mp4

# Configuration
deepfake-detector config show
deepfake-detector config validate

# Version
deepfake-detector version
```

### 4.2 Options

```bash
Options:
  --config PATH       Path to config file (YAML/JSON)
  --threshold FLOAT   Detection threshold (default: 0.35)
  --max-frames INT    Maximum frames to analyze (default: 200)
  --format TEXT       Output format: json, html, text (default: text)
  --output PATH       Output file path
  --verbose           Enable verbose output
  --quiet             Suppress all output except verdict
  --version           Show version
  --help              Show help message
```

### 4.3 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success - video is REAL |
| 1 | Success - video is DEEPFAKE |
| 2 | Success - verdict is UNCERTAIN |
| 10 | Error - file not found |
| 11 | Error - invalid video |
| 12 | Error - analysis failed |
| 20 | Error - configuration error |

---

## 5. Configuration

### 5.1 Configuration File (config.yaml)

```yaml
# Deepfake Detector Configuration
version: "2.0"

analysis:
  max_frames: 200
  sample_rate: 2
  enable_audio: true
  enable_gpu: false

thresholds:
  deepfake: 0.35
  uncertain: 0.25

weights:
  visual_artifacts: 3.0
  temporal: 1.0
  physiological: 1.0
  frequency: 1.0

skills:
  enabled:
    - visual_artifacts
    - temporal
    - physiological
    - frequency
  disabled: []

logging:
  level: INFO
  file: null
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

output:
  default_format: text
  include_raw_scores: false
  include_frame_data: false
```

### 5.2 Environment Variables

```bash
# .env file
DEEPFAKE_THRESHOLD=0.35
DEEPFAKE_UNCERTAIN_THRESHOLD=0.25
DEEPFAKE_MAX_FRAMES=200
DEEPFAKE_SAMPLE_RATE=2
DEEPFAKE_ENABLE_GPU=false
DEEPFAKE_LOG_LEVEL=INFO
```

---

## 6. Error Handling

### 6.1 Exception Hierarchy

```python
class DeepfakeError(Exception):
    """Base exception for deepfake detector."""
    pass

class VideoLoadError(DeepfakeError):
    """Error loading video file."""
    pass

class FaceDetectionError(DeepfakeError):
    """Error detecting faces in video."""
    pass

class AnalysisError(DeepfakeError):
    """Error during analysis."""
    pass

class ConfigurationError(DeepfakeError):
    """Error in configuration."""
    pass
```

### 6.2 Error Codes

| Error | Code | Description |
|-------|------|-------------|
| VIDEO_NOT_FOUND | E001 | Video file does not exist |
| VIDEO_INVALID | E002 | Video file is corrupted or unsupported |
| NO_FACES | E003 | No faces detected in video |
| FACE_TOO_SMALL | E004 | Face region too small for analysis |
| INSUFFICIENT_FRAMES | E005 | Not enough frames for temporal analysis |
| AUDIO_MISSING | E006 | Audio track missing (warning only) |
| CONFIG_INVALID | E010 | Invalid configuration value |

---

## 7. Logging

### 7.1 Log Levels

```python
import logging

# Set log level
logging.getLogger("deepfake_detector").setLevel(logging.DEBUG)

# Log format
# 2026-01-03 14:25:30 - deepfake_detector.detector - INFO - Analyzing video.mp4
# 2026-01-03 14:25:32 - deepfake_detector.skills.visual - DEBUG - Texture variance: 45.2
```

### 7.2 Structured Logging

```python
# Enable structured logging
from deepfake_detector import configure_logging

configure_logging(
    level="DEBUG",
    format="json",
    output="analysis.log"
)

# Log output (JSON)
{
    "timestamp": "2026-01-03T14:25:30.123Z",
    "level": "INFO",
    "module": "detector",
    "message": "Analysis complete",
    "data": {
        "video": "test.mp4",
        "verdict": "REAL",
        "confidence": 0.92,
        "processing_time": 15.3
    }
}
```

---

## 8. Performance Tuning

### 8.1 Memory Optimization

```python
# For large videos, use streaming mode
config = AnalysisConfig(
    max_frames=100,      # Limit frames in memory
    sample_rate=3,       # Sample fewer frames
    enable_gpu=True      # Offload to GPU
)
```

### 8.2 Speed Optimization

```python
# Fast mode for quick screening
config = AnalysisConfig(
    max_frames=50,
    sample_rate=5,
    skills=["visual_artifacts"]  # Single skill
)

# Quality mode for thorough analysis
config = AnalysisConfig(
    max_frames=500,
    sample_rate=1,
    skills=None  # All skills
)
```

---

## 9. Examples

### 9.1 Basic Detection

```python
from deepfake_detector import DeepfakeDetector

detector = DeepfakeDetector()
result = detector.analyze("video.mp4")

if result.is_fake:
    print(f"⚠️ DEEPFAKE DETECTED ({result.confidence:.1%} confidence)")
    print(f"Explanation: {result.explanation}")
    for finding in result.findings:
        print(f"  - {finding.severity.upper()}: {finding.description}")
elif result.is_real:
    print(f"✓ Video appears REAL ({result.confidence:.1%} confidence)")
else:
    print(f"❓ UNCERTAIN - manual review recommended")
```

### 9.2 Batch Processing

```python
from deepfake_detector import DeepfakeDetector
from pathlib import Path

detector = DeepfakeDetector()

# Process all videos in a directory
video_dir = Path("videos/")
video_files = list(video_dir.glob("*.mp4"))

results = detector.analyze_batch(video_files, max_workers=4)

for video, result in zip(video_files, results):
    print(f"{video.name}: {result.verdict.value} ({result.confidence:.1%})")
```

### 9.3 Custom Configuration

```python
from deepfake_detector import DeepfakeDetector, AnalysisConfig

# Load config from file
config = AnalysisConfig.from_file("my_config.yaml")

# Or create programmatically
config = AnalysisConfig(
    max_frames=300,
    sample_rate=1,
    deepfake_threshold=0.40,  # More conservative
    visual_artifacts_weight=4.0,
    enable_gpu=True
)

detector = DeepfakeDetector(config=config)
```

### 9.4 Progress Callback

```python
from deepfake_detector import DeepfakeDetector

def progress_handler(frame_idx: int, total_frames: int):
    pct = frame_idx / total_frames * 100
    print(f"\rProcessing: {pct:.1f}%", end="", flush=True)

detector = DeepfakeDetector()
result = detector.analyze("long_video.mp4", callback=progress_handler)
print(f"\nVerdict: {result.verdict}")
```

### 9.5 JSON Report Generation

```python
import json
from deepfake_detector import DeepfakeDetector

detector = DeepfakeDetector()
result = detector.analyze("video.mp4")

# Save as JSON
with open("report.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)

# Or use built-in method
result.to_json_file("report.json")
```

---

## 10. Changelog

### v2.0.0 (January 2026)
- ✨ Added visual artifact analyzer
- ✨ Added bidirectional anomaly detection
- ✨ Added background motion analysis
- ✨ Added synergy scoring
- 🔧 Improved threshold tuning
- 📚 Added comprehensive documentation

### v1.0.0 (December 2025)
- 🎉 Initial release
- Basic face detection and tracking
- Temporal consistency analysis
- Physiological signal extraction

---

*Documentation Version: 2.0*
*Last Updated: January 2026*

