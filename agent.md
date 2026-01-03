# Deepfake Detection Agent

## Objective

Determine whether a video is REAL or DEEPFAKE by aggregating evidence from visual, temporal, physiological, frequency, and audio-visual signals.

The agent must operate conservatively:
- Prefer "UNCERTAIN" over false accusation
- Accumulate evidence across time
- Explain its decision

---

## Primary Inspection Targets

### 1. Eyes & Gaze
- Abnormal blinking rate
- Inconsistent eye openness across frames
- Gaze direction not aligned with head pose
- Unnatural pupil behavior
- Frame-to-frame eye texture instability

---

### 2. Mouth & Lip Sync
- Phoneme–lip mismatch
- Teeth appearing/disappearing between frames
- Jaw motion lagging audio
- Lip boundary artifacts
- Mouth motion disconnected from facial muscles

---

### 3. Facial Skin & Texture
- Over-smoothing of skin
- Loss of pores and fine wrinkles
- Flickering texture across frames
- Mismatch between face and neck texture
- Inconsistent skin tone under stable lighting

---

### 4. Head Pose & Geometry
- Sudden head pose jumps
- Face shape deformation across frames
- Perspective mismatch
- Nose/ear position drift
- Incorrect depth cues

---

### 5. Lighting & Shadows
- Light source inconsistent with environment
- Face lighting not matching background
- Shadow direction changes without motion
- Specular highlights behaving unnaturally

---

### 6. Background Consistency
- Face sharpness vs blurred background mismatch
- Boundary artifacts around hair and ears
- Background motion not aligned with head movement
- Warping or flickering near face edges

---

### 7. Hair & Facial Boundaries
- Hair flickering or vanishing
- Unnatural hairline transitions
- Blending artifacts near ears, glasses, hats
- Alpha-matting errors

---

### 8. Temporal Coherence
- Identity drift over time
- Expression transitions too abrupt
- Frame-level realism but sequence-level failure
- Micro-expression inconsistency

---

### 9. Audio–Visual Alignment
- Speech audio not matching lip motion
- Speaker identity mismatch
- Emotional tone mismatch (angry voice, neutral face)
- Audio compression different from video compression

---

### 10. Frequency & Signal Artifacts
- High-frequency noise anomalies
- Upsampling checkerboard patterns
- DCT coefficient inconsistencies
- GAN fingerprint detection

---

### 11. Physiological Signals
- Heartbeat (rPPG) inconsistency
- No blood flow signal in skin regions
- Unrealistic skin color fluctuation
- Asymmetric physiological patterns

---

### 12. Identity Consistency
- Face embedding instability
- Mismatch with known reference identity
- Face–voice identity mismatch
- Expression not matching identity traits

---

## Decision Strategy

- Aggregate evidence across all skills
- Weight temporal and physiological signals higher
- Apply uncertainty thresholds
- Output:
  - REAL
  - DEEPFAKE
  - UNCERTAIN

---

## Explainability Requirements

- Highlight suspicious regions
- List triggered signals
- Provide confidence score
- Provide time segments of highest suspicion

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEEPFAKE DETECTION AGENT                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Video     │───▶│   Frame     │───▶│    Face     │         │
│  │   Input     │    │  Extractor  │    │   Tracker   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                              │                  │
│         ┌────────────────────────────────────┼──────────────┐   │
│         │                                    │              │   │
│         ▼                                    ▼              ▼   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Temporal   │    │ Physiological│    │  Frequency  │         │
│  │ Consistency │    │   Signals   │    │  Artifacts  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         │           ┌──────┴──────┐           │                 │
│         │           │             │           │                 │
│         ▼           ▼             ▼           ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Audio-Visual│    │  Identity   │    │Explainability│        │
│  │  Alignment  │    │  Reasoning  │    │   Module    │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            ▼                                    │
│                   ┌─────────────────┐                           │
│                   │   Evidence      │                           │
│                   │   Aggregator    │                           │
│                   └─────────────────┘                           │
│                            │                                    │
│                            ▼                                    │
│                   ┌─────────────────┐                           │
│                   │    Decision     │                           │
│                   │     Engine      │                           │
│                   └─────────────────┘                           │
│                            │                                    │
│                            ▼                                    │
│              ┌───────────────────────────┐                      │
│              │  REAL | DEEPFAKE | UNCERTAIN │                   │
│              │  + Confidence + Explanation  │                   │
│              └───────────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Skills Integration

| Skill | Weight | Description |
|-------|--------|-------------|
| Face Tracking | 1.0 | Foundation for all face-based analysis |
| Temporal Consistency | 1.5 | High weight - deepfakes often fail temporally |
| Physiological Signals | 1.5 | High weight - hard to fake rPPG |
| Frequency Artifacts | 1.2 | GAN fingerprints are reliable indicators |
| Audio-Visual Alignment | 1.3 | Lip-sync errors are common |
| Identity Reasoning | 1.1 | Cross-modal identity verification |
| Dataset Generalization | 1.0 | Ensures robust detection |
| Robustness & Security | 1.0 | Adversarial defense |
| Explainability | N/A | Meta-skill for interpretation |

## Usage

```python
from deepfake_detector import DeepfakeDetector

detector = DeepfakeDetector()
result = detector.analyze("suspicious_video.mp4")

print(result.verdict)      # REAL | DEEPFAKE | UNCERTAIN
print(result.confidence)   # 0.0 - 1.0
print(result.explanation)  # Human-readable explanation
print(result.timeline)     # Frame-by-frame suspicion scores
```

