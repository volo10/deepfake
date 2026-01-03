# Face Tracking Skill

## Purpose
Detect and track faces across video frames, providing the foundational data structure for all downstream analysis skills.

## Capabilities

### 1. Face Detection
- Multi-face detection in each frame
- Bounding box extraction with confidence scores
- Handle partial occlusions
- Detect face orientation (frontal, profile, etc.)

### 2. Landmark Extraction
- 68-point or 478-point facial landmarks
- Eye corners, nose tip, mouth corners
- Jawline contour
- Eyebrow positions

### 3. Face Alignment
- Normalize face orientation
- Consistent eye-line alignment
- Scale normalization for embedding extraction

### 4. Temporal Tracking
- Maintain face identity across frames
- Handle temporary occlusions
- Track multiple faces simultaneously
- Assign persistent IDs to each face

### 5. Region of Interest (ROI) Extraction
- Eye region cropping
- Mouth region cropping
- Forehead region (for rPPG)
- Full face mesh for texture analysis

---

## Deepfake Indicators

| Indicator | Description | Severity |
|-----------|-------------|----------|
| Landmark jitter | Landmarks jumping unnaturally between frames | Medium |
| Face swap boundary | Visible blending line around face | High |
| Tracking loss spikes | Frequent tracking failures in clean video | Medium |
| Asymmetric landmarks | Left-right inconsistency in static pose | Low |
| Scale inconsistency | Face size changes without depth change | High |

---

## Implementation Details

### Input
- Video frames (RGB)
- Frame rate metadata
- Optional: audio track for sync

### Output
```python
@dataclass
class FaceTrackingResult:
    frame_idx: int
    faces: List[DetectedFace]
    
@dataclass
class DetectedFace:
    face_id: int
    bbox: BoundingBox  # (x, y, w, h)
    confidence: float
    landmarks: np.ndarray  # (N, 2) or (N, 3)
    aligned_face: np.ndarray  # 224x224 RGB
    regions: Dict[str, np.ndarray]  # eyes, mouth, forehead
    pose: HeadPose  # pitch, yaw, roll
```

### Dependencies
- MediaPipe Face Mesh
- dlib / face_recognition
- OpenCV
- RetinaFace (optional, for robustness)

---

## Quality Metrics

### Tracking Quality Score
```
Q_track = (1 - detection_failures / total_frames) × smoothness_score
```

### Landmark Stability Score
```
L_stability = 1 / (1 + mean_landmark_velocity)
```

---

## Integration with Other Skills

| Downstream Skill | Data Provided |
|------------------|---------------|
| Temporal Consistency | Face sequence, landmarks over time |
| Physiological Signals | Forehead/cheek ROIs for rPPG |
| Frequency Artifacts | Aligned face crops |
| Audio-Visual Alignment | Mouth region sequence |
| Identity Reasoning | Face embeddings |

---

## Anomaly Detection

### Sudden Landmark Shifts
```python
def detect_landmark_anomaly(landmarks_seq, threshold=5.0):
    """Detect sudden landmark position changes."""
    velocities = np.diff(landmarks_seq, axis=0)
    magnitudes = np.linalg.norm(velocities, axis=-1)
    anomalies = magnitudes > threshold
    return anomalies
```

### Face Swap Boundary Detection
```python
def detect_face_boundary(frame, face_mask):
    """Detect unnatural boundaries around face region."""
    gradient = cv2.Laplacian(frame, cv2.CV_64F)
    boundary_gradient = gradient * face_mask_boundary
    return np.mean(np.abs(boundary_gradient))
```

---

## Failure Modes

1. **Low resolution**: Landmarks become unreliable below 64x64 face size
2. **Extreme poses**: Profile views degrade accuracy
3. **Occlusions**: Hands, objects, other faces
4. **Motion blur**: Fast head movements
5. **Adversarial makeup**: Designed to fool detectors

## Mitigation Strategies

- Use multiple detection backends and ensemble
- Apply temporal smoothing with outlier rejection
- Confidence-weighted landmark aggregation
- Adaptive resolution handling

