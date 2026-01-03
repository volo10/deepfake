# Temporal Consistency Skill

## Purpose
Analyze frame-to-frame coherence to detect temporal artifacts characteristic of deepfakes, which often exhibit inconsistencies when viewed as sequences rather than individual frames.

## Core Principle
**Deepfakes are often convincing frame-by-frame but fail sequence-level consistency checks.**

---

## Capabilities

### 1. Identity Drift Detection
- Track face embedding stability over time
- Detect gradual identity shifts
- Flag sudden identity changes

### 2. Expression Flow Analysis
- Verify smooth expression transitions
- Detect micro-expression inconsistencies
- Check emotion continuity

### 3. Motion Coherence
- Head pose trajectory smoothness
- Lip motion natural dynamics
- Eye gaze temporal patterns

### 4. Texture Stability
- Skin texture persistence
- Wrinkle/pore consistency
- Hair behavior over time

### 5. Blinking Pattern Analysis
- Natural blink rate (15-20 per minute)
- Blink duration consistency
- Both eyes synchronized

---

## Deepfake Indicators

| Indicator | Description | Weight |
|-----------|-------------|--------|
| Identity drift | Face embedding changes >0.3 over 1 second | 1.5 |
| Unnatural blink rate | <5 or >30 blinks per minute | 1.2 |
| Expression discontinuity | Action unit jumps between frames | 1.3 |
| Texture flicker | High-frequency texture changes | 1.4 |
| Motion jitter | Non-smooth head/face motion | 1.1 |
| Asymmetric blinking | Left/right eye desync | 1.3 |

---

## Implementation Details

### Input
```python
@dataclass
class TemporalInput:
    face_sequence: List[np.ndarray]  # Aligned face crops
    landmarks_sequence: List[np.ndarray]
    embeddings_sequence: List[np.ndarray]
    timestamps: List[float]
    fps: float
```

### Output
```python
@dataclass
class TemporalConsistencyResult:
    identity_stability: float  # 0-1
    expression_smoothness: float  # 0-1
    blink_anomaly_score: float  # 0-1, higher = more suspicious
    texture_stability: float  # 0-1
    motion_smoothness: float  # 0-1
    overall_temporal_score: float  # Weighted combination
    anomaly_frames: List[int]  # Suspicious frame indices
    anomaly_details: List[AnomalyDetail]
```

---

## Analysis Methods

### 1. Embedding Stability Analysis
```python
def analyze_identity_stability(embeddings, window_size=30):
    """
    Compute identity drift over sliding windows.
    Returns stability score (1.0 = perfectly stable).
    """
    drifts = []
    for i in range(len(embeddings) - window_size):
        window = embeddings[i:i+window_size]
        centroid = np.mean(window, axis=0)
        distances = [cosine_distance(e, centroid) for e in window]
        drifts.append(np.max(distances))
    
    max_drift = np.max(drifts)
    stability = 1.0 - min(max_drift / 0.5, 1.0)  # Normalize
    return stability
```

### 2. Blink Detection
```python
def analyze_blinks(landmarks_sequence, fps):
    """
    Extract blink events and analyze patterns.
    """
    ear_values = [eye_aspect_ratio(lm) for lm in landmarks_sequence]
    
    # Detect blink events
    blinks = detect_blink_events(ear_values, threshold=0.2)
    
    # Analyze patterns
    blink_rate = len(blinks) * 60 / (len(ear_values) / fps)
    blink_durations = [b.duration for b in blinks]
    
    anomalies = []
    if blink_rate < 5 or blink_rate > 30:
        anomalies.append("abnormal_blink_rate")
    if np.std(blink_durations) > 0.1:
        anomalies.append("inconsistent_blink_duration")
    
    return BlinkAnalysis(blink_rate, blinks, anomalies)
```

### 3. Expression Flow Analysis
```python
def analyze_expression_flow(action_units_sequence):
    """
    Check for discontinuous expression changes.
    Action Units (AUs) should change smoothly.
    """
    au_velocities = np.diff(action_units_sequence, axis=0)
    au_accelerations = np.diff(au_velocities, axis=0)
    
    # Flag sudden changes (high acceleration)
    threshold = 0.5
    discontinuities = np.where(np.abs(au_accelerations) > threshold)
    
    smoothness = 1.0 - len(discontinuities[0]) / len(au_accelerations)
    return smoothness, discontinuities
```

### 4. Texture Temporal Analysis
```python
def analyze_texture_stability(face_sequence):
    """
    Detect flickering or unstable texture patterns.
    """
    # Extract high-frequency components
    hf_sequence = [extract_high_frequency(face) for face in face_sequence]
    
    # Compute frame-to-frame differences
    diffs = [np.abs(hf_sequence[i+1] - hf_sequence[i]) 
             for i in range(len(hf_sequence)-1)]
    
    # Stable video should have low differences
    mean_diff = np.mean(diffs)
    stability = 1.0 - min(mean_diff / 50.0, 1.0)
    
    return stability
```

---

## Temporal Window Strategies

### Short-term (5-10 frames)
- Micro-expression detection
- Landmark jitter
- Blink events

### Medium-term (30-60 frames / 1-2 seconds)
- Expression transitions
- Identity stability
- Motion patterns

### Long-term (full video)
- Overall identity consistency
- Blink rate statistics
- Global texture stability

---

## Aggregation Strategy

```python
def compute_temporal_score(results):
    weights = {
        'identity_stability': 1.5,
        'blink_anomaly': 1.2,
        'expression_smoothness': 1.3,
        'texture_stability': 1.4,
        'motion_smoothness': 1.1
    }
    
    weighted_sum = sum(
        results[k] * weights[k] for k in weights
    )
    total_weight = sum(weights.values())
    
    return weighted_sum / total_weight
```

---

## Visualization Support

- Timeline graph of anomaly scores
- Frame highlighting for suspicious moments
- Side-by-side comparison of stable vs unstable segments
- Blink pattern visualization

---

## Known Limitations

1. **Compression artifacts**: Heavy compression can mask or mimic temporal issues
2. **Low FPS**: <15 fps reduces temporal analysis reliability
3. **Static videos**: Minimal motion limits temporal signals
4. **High-quality deepfakes**: State-of-art fakes minimize temporal artifacts

## Adaptive Thresholds

Adjust thresholds based on:
- Video quality
- Frame rate
- Motion intensity
- Face resolution

