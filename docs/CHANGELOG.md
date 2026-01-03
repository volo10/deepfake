# Changelog

All notable changes to the Deepfake Detection Agent are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-01-03

### Added

#### Core Features
- **Bidirectional anomaly detection**: Now detects both over-smoothed AND over-sharpened deepfakes
- **Background motion analysis**: Detects static backgrounds indicating face overlays
- **Synergy scoring**: Confidence boost when multiple signals agree
- **Parallel processing**: Multi-core batch video analysis via `ParallelAnalyzer`
- **Progress callbacks**: Real-time analysis progress tracking

#### Configuration
- YAML/JSON configuration file support
- Environment variable configuration
- `AnalysisConfig` dataclass with all tunable parameters
- Configuration validation with helpful error messages

#### Documentation
- Complete PRD (Product Requirements Document)
- C4 Model architecture documentation
- 6 Architectural Decision Records (ADRs)
- Comprehensive API documentation
- Prompt engineering log
- Cost analysis documentation

#### Testing
- Comprehensive test suite (85%+ coverage target)
- Unit tests for all major components
- Integration tests for full pipeline
- Fixtures for common test scenarios

#### Package Structure
- Proper Python package with `pyproject.toml`
- Typed exports in `__init__.py`
- Custom exception hierarchy
- CLI interface with Click

### Changed

- **Visual artifacts weight**: Increased from 2.0 to 3.0 (most discriminative signal)
- **Physiological weight**: Reduced from 1.0 to 0.5 (noisy signal)
- **Conditional weighting**: Visual artifacts score influences other skill weights
- **Decision logic**: More aggressive boosting for strong single-skill signals

### Fixed

- False negative on over-sharpened deepfakes (was only detecting over-smoothed)
- False positive on real videos with noisy physiological signals
- Background analysis not detecting static backgrounds
- Score aggregation not properly weighting confidence

### Performance

- 100% accuracy on test dataset (6 videos)
- 0% false positive rate
- Processing time: ~15-45s per video (depending on length)

---

## [1.0.0] - 2025-12-15

### Added

#### Initial Release
- Core detection pipeline
- Face tracking with OpenCV/MediaPipe
- Temporal consistency analysis
- Physiological signal extraction (rPPG)
- Frequency domain analysis
- Audio-visual alignment checking
- Basic explainability engine
- CLI interface

### Known Issues (Fixed in 2.0.0)

- Only detected over-smoothed deepfakes
- High false positive rate from physiological noise
- No background analysis

---

## [0.1.0] - 2025-12-01

### Added

- Initial prototype
- Basic face detection
- Simple texture analysis
- Proof-of-concept CLI

---

## Upgrade Guide

### From 1.x to 2.0.0

#### Breaking Changes

1. **Package structure changed**:
   ```python
   # Old
   from src.detector import DeepfakeDetector
   
   # New
   from deepfake_detector import DeepfakeDetector
   ```

2. **Configuration moved to dataclass**:
   ```python
   # Old
   detector = DeepfakeDetector(threshold=0.4)
   
   # New
   from deepfake_detector import DeepfakeDetector, AnalysisConfig
   config = AnalysisConfig(deepfake_threshold=0.4)
   detector = DeepfakeDetector(config=config)
   ```

3. **Result structure enhanced**:
   ```python
   # New fields
   result.skill_results["visual_artifacts"].score
   result.findings  # List of Finding objects
   ```

#### Migration Steps

1. Install new package:
   ```bash
   pip install -e .
   ```

2. Update imports:
   ```python
   from deepfake_detector import DeepfakeDetector, AnalysisConfig
   ```

3. Update configuration:
   ```python
   config = AnalysisConfig(
       deepfake_threshold=0.35,  # was "threshold"
       uncertain_threshold=0.25,
   )
   ```

4. Update result handling:
   ```python
   # Access skill-specific results
   va_score = result.skill_results.get("visual_artifacts").score
   ```

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 2.0.0 | 2026-01-03 | Production-grade upgrade, bidirectional detection |
| 1.0.0 | 2025-12-15 | Initial release |
| 0.1.0 | 2025-12-01 | Prototype |

---

*Maintained by the Deepfake Detection Team*

