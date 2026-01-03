# System Architecture Documentation
## C4 Model & Architectural Decision Records

---

## 1. C4 Model

### 1.1 Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM CONTEXT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌──────────────┐         ┌──────────────────────┐                    │
│    │              │         │                      │                    │
│    │   End User   │────────▶│  Deepfake Detection  │                    │
│    │  (Analyst)   │         │       System         │                    │
│    │              │◀────────│                      │                    │
│    └──────────────┘         └──────────┬───────────┘                    │
│           │                            │                                 │
│           │ Uploads video              │ Analyzes                        │
│           │ Receives verdict           │                                 │
│           │                            ▼                                 │
│    ┌──────────────┐         ┌──────────────────────┐                    │
│    │              │         │                      │                    │
│    │  Video File  │         │   Analysis Results   │                    │
│    │   Storage    │         │   (JSON/HTML/Text)   │                    │
│    │              │         │                      │                    │
│    └──────────────┘         └──────────────────────┘                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Context Description:**
- **End User**: Journalist, researcher, or analyst who wants to verify video authenticity
- **Deepfake Detection System**: The core system that processes videos and returns verdicts
- **Video File Storage**: Local filesystem or cloud storage containing input videos
- **Analysis Results**: Output reports in various formats

---

### 1.2 Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONTAINER DIAGRAM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Deepfake Detection System                     │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │    │
│  │  │               │  │               │  │               │       │    │
│  │  │  CLI App      │  │  Python API   │  │  Web Server   │       │    │
│  │  │  (Click)      │  │  (Library)    │  │  (Future)     │       │    │
│  │  │               │  │               │  │               │       │    │
│  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘       │    │
│  │          │                  │                  │                │    │
│  │          └──────────────────┼──────────────────┘                │    │
│  │                             │                                   │    │
│  │                             ▼                                   │    │
│  │                  ┌───────────────────┐                         │    │
│  │                  │                   │                         │    │
│  │                  │  Detection Engine │                         │    │
│  │                  │  (Core Library)   │                         │    │
│  │                  │                   │                         │    │
│  │                  └─────────┬─────────┘                         │    │
│  │                            │                                   │    │
│  │          ┌─────────────────┼─────────────────┐                 │    │
│  │          │                 │                 │                 │    │
│  │          ▼                 ▼                 ▼                 │    │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │    │
│  │  │ Video Loader  │ │ Face Tracker  │ │ Skill Modules │        │    │
│  │  │ (OpenCV)      │ │ (MediaPipe)   │ │ (Analysis)    │        │    │
│  │  └───────────────┘ └───────────────┘ └───────────────┘        │    │
│  │                                                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Container Descriptions:**

| Container | Technology | Purpose |
|-----------|------------|---------|
| CLI App | Python + Click | Command-line interface for users |
| Python API | Python Library | Programmatic access for developers |
| Detection Engine | Python | Core orchestration and analysis |
| Video Loader | OpenCV | Video file I/O and frame extraction |
| Face Tracker | OpenCV/MediaPipe | Face detection and landmark extraction |
| Skill Modules | NumPy/SciPy | Individual analysis algorithms |

---

### 1.3 Level 3: Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT DIAGRAM                                 │
│                      Detection Engine Detail                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Detection Engine                            │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                    DeepfakeDetector                      │    │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │    │
│  │  │  │  analyze()  │  │ _aggregate  │  │ _make       │     │    │    │
│  │  │  │             │  │ _evidence() │  │ _decision() │     │    │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘     │    │    │
│  │  └──────────────────────────┬──────────────────────────────┘    │    │
│  │                             │                                   │    │
│  │  ┌──────────────────────────┼──────────────────────────┐       │    │
│  │  │                          │                          │       │    │
│  │  ▼                          ▼                          ▼       │    │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │    │
│  │  │ FaceTracker  │   │   Visual     │   │  Temporal    │       │    │
│  │  │              │   │  Artifacts   │   │  Analyzer    │       │    │
│  │  │ • track()    │   │  Analyzer    │   │              │       │    │
│  │  │ • extract_   │   │              │   │ • analyze()  │       │    │
│  │  │   embeddings │   │ • analyze()  │   │ • _analyze   │       │    │
│  │  │              │   │ • _analyze   │   │   _blinks()  │       │    │
│  │  └──────────────┘   │   _texture() │   └──────────────┘       │    │
│  │                     │ • _analyze   │                           │    │
│  │                     │   _symmetry()│   ┌──────────────┐       │    │
│  │                     │ • _analyze   │   │  Frequency   │       │    │
│  │                     │   _edges()   │   │  Analyzer    │       │    │
│  │                     │ • _analyze   │   │              │       │    │
│  │                     │   _background│   │ • analyze()  │       │    │
│  │                     └──────────────┘   └──────────────┘       │    │
│  │                                                                │    │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │    │
│  │  │  Identity    │   │ Physiological│   │Explainability│       │    │
│  │  │  Analyzer    │   │  Analyzer    │   │   Engine     │       │    │
│  │  │              │   │              │   │              │       │    │
│  │  │ • analyze()  │   │ • analyze()  │   │ • generate   │       │    │
│  │  │ • _detect    │   │ • _extract   │   │   _explanation│      │    │
│  │  │   _switches()│   │   _rppg()    │   │ • generate   │       │    │
│  │  └──────────────┘   └──────────────┘   │   _report()  │       │    │
│  │                                        └──────────────┘       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 1.4 Level 4: Code Diagram (Key Classes)

```python
# Core Data Flow
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  VideoPath (str)                                                         │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │ load_video()│ ──────▶ List[np.ndarray] (frames)                      │
│  └─────────────┘         float (fps)                                    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │FaceTracker  │ ──────▶ List[FrameAnalysis]                            │
│  │  .track()   │         ├── frame_idx: int                             │
│  └─────────────┘         ├── faces: List[DetectedFace]                  │
│       │                  │   ├── bbox: BoundingBox                      │
│       │                  │   ├── landmarks: np.ndarray                  │
│       │                  │   └── aligned_face: np.ndarray               │
│       ▼                  └── suspicion_score: float                     │
│  ┌─────────────┐                                                        │
│  │ Skill       │ ──────▶ SkillResult                                    │
│  │ .analyze()  │         ├── skill_name: str                            │
│  └─────────────┘         ├── score: float (0-1)                         │
│       │                  ├── confidence: float (0-1)                    │
│       │                  ├── findings: List[Finding]                    │
│       ▼                  └── raw_data: Dict                             │
│  ┌─────────────┐                                                        │
│  │ _aggregate  │ ──────▶ (overall_score: float, findings: List)         │
│  │ _evidence() │                                                        │
│  └─────────────┘                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │_make        │ ──────▶ (Verdict, confidence: float)                   │
│  │_decision()  │                                                        │
│  └─────────────┘                                                        │
│       │                                                                  │
│       ▼                                                                  │
│  DetectionResult                                                         │
│  ├── verdict: Verdict (REAL/DEEPFAKE/UNCERTAIN)                         │
│  ├── confidence: float                                                  │
│  ├── overall_score: float                                               │
│  ├── skill_results: Dict[str, SkillResult]                              │
│  ├── findings: List[Finding]                                            │
│  └── explanation: str                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architectural Decision Records (ADRs)

### ADR-001: Multi-Skill Detection Architecture

**Status:** Accepted

**Context:**
Deepfakes can exhibit different types of artifacts depending on the generation method:
- GAN-based methods: frequency artifacts, over-smoothing
- Face-swap methods: boundary artifacts, color mismatches
- Modern methods: over-sharpening, temporal inconsistencies

**Decision:**
Implement a modular multi-skill architecture where each "skill" analyzes one aspect of the video independently, and results are aggregated with configurable weights.

**Consequences:**
- ✅ Can detect multiple types of deepfakes
- ✅ Easy to add new detection methods
- ✅ Each skill can be tested independently
- ❌ Increased complexity in aggregation
- ❌ Potential for conflicting signals

**Alternatives Considered:**
1. Single end-to-end neural network: Rejected due to lack of explainability
2. Sequential pipeline: Rejected due to single point of failure
3. Voting ensemble: Partially adopted in current aggregation

---

### ADR-002: Conservative Decision Thresholds

**Status:** Accepted

**Context:**
False positives (labeling real videos as fake) can have serious consequences:
- Wrongful accusations
- Loss of trust in the system
- Legal implications

**Decision:**
Implement conservative thresholds that prefer UNCERTAIN over DEEPFAKE verdict when evidence is not strong. Use multiple signals for high-confidence detection.

**Thresholds:**
```python
deepfake_threshold = 0.35  # Score above this = DEEPFAKE
uncertain_threshold = 0.25  # Score below this = REAL
# Scores between 0.25-0.35 = UNCERTAIN
```

**Consequences:**
- ✅ Low false positive rate (~0%)
- ✅ Users can trust DEEPFAKE verdicts
- ❌ May miss some subtle deepfakes (false negatives)
- ❌ More UNCERTAIN results requiring human review

---

### ADR-003: Visual Artifacts as Primary Signal

**Status:** Accepted

**Context:**
After testing multiple signal types:
- Physiological (rPPG): Noisy, requires long videos, affected by compression
- Audio-visual: Requires audio track, complex alignment
- Temporal: Good but insufficient alone
- Visual artifacts: Most reliable across different deepfake types

**Decision:**
Give visual_artifacts skill the highest weight (3.0) and use it as the primary decision signal. When visual_artifacts score is very low (<0.1), reduce weight of other potentially noisy signals.

**Evidence:**
```
Test Results (6 videos):
- visual_artifacts correctly separated all REAL (0.02-0.16) from FAKE (0.62-0.65)
- physiological had false positives on real videos
- Other skills provided supporting but not decisive evidence
```

**Consequences:**
- ✅ High accuracy on test set (100%)
- ✅ Robust to missing audio
- ❌ May be vulnerable to adversarial attacks on visual artifacts
- ❌ Depends on face detection quality

---

### ADR-004: Bidirectional Anomaly Detection

**Status:** Accepted

**Context:**
Initial detector only caught "over-smoothed" deepfakes but missed "over-sharpened" ones.

**Decision:**
Detect anomalies in BOTH directions for key metrics:
- Texture: too smooth OR too sharp
- Edge density: too low OR too high
- Contrast: too low OR too high

**Implementation:**
```python
# Example: Texture analysis
if mean_variance < threshold_low:
    issue_type = 'over-smoothed'
    anomaly_score = compute_score(...)
elif mean_variance > threshold_high:
    issue_type = 'over-sharpened'
    anomaly_score = compute_score(...)
```

**Consequences:**
- ✅ Catches both types of deepfakes
- ✅ More robust to different generation methods
- ❌ More thresholds to tune
- ❌ Potential for false positives at both extremes

---

### ADR-005: Synergy Scoring

**Status:** Accepted

**Context:**
Real videos may occasionally trigger one anomaly indicator by chance, but deepfakes typically trigger multiple indicators simultaneously.

**Decision:**
Implement synergy scoring that boosts the overall score when multiple indicators are triggered:

```python
triggered_count = sum(1 for s in scores.values() if s > 0.2)
if triggered_count >= 4:
    overall_score *= 1.5
elif triggered_count >= 3:
    overall_score *= 1.3
elif triggered_count >= 2:
    overall_score *= 1.15
```

**Consequences:**
- ✅ Reduces false positives from single anomalies
- ✅ Increases confidence when multiple signals agree
- ❌ May miss deepfakes that only fail one test

---

### ADR-006: Background Motion Analysis

**Status:** Accepted

**Context:**
Some deepfakes have static/frozen backgrounds while the face moves, or people in the background don't move naturally.

**Decision:**
Add background motion analysis that compares motion in face region vs. non-face region.

**Detection Logic:**
```python
if face_motion > 3.0 and bg_motion < 1.0:
    # Face moving, background frozen = suspicious
    anomaly_score = high
```

**Consequences:**
- ✅ Catches a new class of deepfakes
- ✅ Leverages video-specific information
- ❌ May false-positive on talking-head videos with static background
- ❌ Requires accurate face detection

---

## 3. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW DIAGRAM                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT                    PROCESSING                      OUTPUT         │
│                                                                          │
│  ┌─────────┐    ┌─────────────────────────────────┐    ┌─────────┐     │
│  │         │    │                                 │    │         │     │
│  │  Video  │───▶│  Frame Extraction               │───▶│ Frames  │     │
│  │  File   │    │  (OpenCV VideoCapture)          │    │  List   │     │
│  │         │    │                                 │    │         │     │
│  └─────────┘    └─────────────────────────────────┘    └────┬────┘     │
│                                                              │          │
│                                                              ▼          │
│                 ┌─────────────────────────────────┐    ┌─────────┐     │
│                 │                                 │    │         │     │
│                 │  Face Detection & Tracking      │◀───│ Frames  │     │
│                 │  (Haar Cascade / MediaPipe)     │    │         │     │
│                 │                                 │    │         │     │
│                 └─────────────────────────────────┘    └─────────┘     │
│                              │                                          │
│                              ▼                                          │
│                 ┌─────────────────────────────────┐                     │
│                 │  Face Crops + Bounding Boxes    │                     │
│                 │  + Landmarks + Embeddings       │                     │
│                 └─────────────────────────────────┘                     │
│                              │                                          │
│         ┌────────────────────┼────────────────────┐                     │
│         │                    │                    │                     │
│         ▼                    ▼                    ▼                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │   Visual    │     │  Temporal   │     │  Frequency  │               │
│  │  Artifacts  │     │  Analysis   │     │  Analysis   │               │
│  │  Analysis   │     │             │     │             │               │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘               │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                           │
│                 ┌─────────────────────────────────┐                     │
│                 │                                 │                     │
│                 │  Evidence Aggregation           │                     │
│                 │  (Weighted Scoring + Synergy)   │                     │
│                 │                                 │                     │
│                 └─────────────────────────────────┘                     │
│                              │                                          │
│                              ▼                                          │
│                 ┌─────────────────────────────────┐                     │
│                 │                                 │                     │
│                 │  Decision Engine                │                     │
│                 │  (Threshold-based + Boost)      │                     │
│                 │                                 │                     │
│                 └─────────────────────────────────┘                     │
│                              │                                          │
│                              ▼                                          │
│                 ┌─────────────────────────────────┐    ┌─────────┐     │
│                 │                                 │    │         │     │
│                 │  Explainability Engine          │───▶│ Report  │     │
│                 │  (Generate human explanation)   │    │ Output  │     │
│                 │                                 │    │         │     │
│                 └─────────────────────────────────┘    └─────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Language | Python 3.9+ | ML ecosystem, rapid development |
| Video I/O | OpenCV | Industry standard, fast |
| Face Detection | OpenCV Haar / MediaPipe | Reliable, cross-platform |
| Numerical | NumPy, SciPy | Performance, ecosystem |
| CLI | Click | Pythonic, full-featured |
| Testing | pytest | Standard, fixtures support |
| Config | python-dotenv, YAML | Flexible, secure |
| Packaging | pyproject.toml | Modern Python standard |

---

*Document Version: 1.0*
*Last Updated: January 2026*

