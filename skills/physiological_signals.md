# Physiological Signals Skill

## Purpose
Extract and analyze physiological signals (primarily remote photoplethysmography - rPPG) from facial video to detect the absence or inconsistency of biological signals that are extremely difficult to fake.

## Core Principle
**Real human faces exhibit subtle color changes synchronized with heartbeat. Deepfakes typically lack or have inconsistent physiological signals.**

---

## Capabilities

### 1. Remote Photoplethysmography (rPPG)
- Extract pulse signal from facial skin
- Estimate heart rate
- Analyze signal quality and consistency

### 2. Blood Flow Analysis
- Detect micro-color variations in skin
- Map blood perfusion patterns
- Identify symmetric vs asymmetric patterns

### 3. Respiratory Rate Detection
- Chest/shoulder motion analysis
- Subtle color changes from breathing
- Breath rate consistency

### 4. Skin Color Dynamics
- Natural color fluctuations
- Response to lighting changes
- Emotional flushing detection

---

## Deepfake Indicators

| Indicator | Description | Weight |
|-----------|-------------|--------|
| No rPPG signal | Absence of detectable pulse | 1.8 |
| Inconsistent HR | Heart rate varies unrealistically | 1.5 |
| Asymmetric rPPG | Different signals from left/right face | 1.6 |
| Flat skin dynamics | No natural color variation | 1.4 |
| Spatial incoherence | Different body regions show conflicting signals | 1.7 |

---

## Implementation Details

### Input
```python
@dataclass
class PhysiologicalInput:
    face_sequence: List[np.ndarray]  # RGB face crops
    roi_forehead: List[np.ndarray]   # Forehead region sequence
    roi_cheeks: List[Tuple[np.ndarray, np.ndarray]]  # Left/right cheeks
    fps: float
    duration: float
```

### Output
```python
@dataclass
class PhysiologicalResult:
    rppg_signal: np.ndarray  # Extracted pulse signal
    heart_rate: float  # BPM estimate
    hr_confidence: float  # Signal quality
    hr_variability: float  # HRV metric
    signal_present: bool  # Is there a valid signal?
    left_right_correlation: float  # Symmetry check
    spatial_coherence: float  # Cross-region agreement
    anomaly_score: float  # 0-1, higher = more suspicious
    anomaly_reasons: List[str]
```

---

## rPPG Extraction Methods

### Method 1: CHROM (Chrominance-based)
```python
def extract_rppg_chrom(rgb_sequence, fps):
    """
    Chrominance-based rPPG extraction.
    De Haan & Jeanne (2013)
    """
    # Convert to normalized RGB
    rgb_norm = normalize_rgb(rgb_sequence)
    
    # Chrominance signals
    Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
    Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
    
    # Bandpass filter (0.7-4 Hz for 42-240 BPM)
    Xf = bandpass_filter(Xs, 0.7, 4.0, fps)
    Yf = bandpass_filter(Ys, 0.7, 4.0, fps)
    
    # Combine signals
    alpha = np.std(Xf) / np.std(Yf)
    rppg = Xf - alpha * Yf
    
    return rppg
```

### Method 2: POS (Plane Orthogonal to Skin)
```python
def extract_rppg_pos(rgb_sequence, fps):
    """
    Plane-Orthogonal-to-Skin method.
    Wang et al. (2017)
    """
    # Temporal normalization
    rgb_norm = temporal_normalize(rgb_sequence)
    
    # Projection
    S1 = rgb_norm[:, 1] - rgb_norm[:, 2]
    S2 = rgb_norm[:, 1] + rgb_norm[:, 2] - 2 * rgb_norm[:, 0]
    
    # Bandpass and combine
    S1f = bandpass_filter(S1, 0.7, 4.0, fps)
    S2f = bandpass_filter(S2, 0.7, 4.0, fps)
    
    alpha = np.std(S1f) / np.std(S2f)
    rppg = S1f + alpha * S2f
    
    return rppg
```

### Method 3: Deep Learning (PhysNet)
```python
def extract_rppg_deep(face_sequence, model):
    """
    Neural network-based rPPG extraction.
    """
    # Preprocess
    faces_tensor = preprocess_for_physnet(face_sequence)
    
    # Extract with trained model
    with torch.no_grad():
        rppg = model(faces_tensor)
    
    return rppg.numpy()
```

---

## Heart Rate Estimation

```python
def estimate_heart_rate(rppg_signal, fps):
    """
    Estimate heart rate from rPPG signal using FFT.
    """
    # FFT
    fft = np.fft.rfft(rppg_signal)
    freqs = np.fft.rfftfreq(len(rppg_signal), 1/fps)
    
    # Find peak in valid HR range (40-180 BPM)
    valid_mask = (freqs >= 0.67) & (freqs <= 3.0)  # Hz
    valid_fft = np.abs(fft) * valid_mask
    
    peak_idx = np.argmax(valid_fft)
    peak_freq = freqs[peak_idx]
    
    heart_rate = peak_freq * 60  # Convert to BPM
    
    # Confidence based on peak prominence
    confidence = compute_peak_prominence(valid_fft, peak_idx)
    
    return heart_rate, confidence
```

---

## Spatial Coherence Analysis

```python
def analyze_spatial_coherence(roi_signals):
    """
    Check if different face regions show consistent signals.
    """
    regions = ['forehead', 'left_cheek', 'right_cheek', 'nose']
    correlations = []
    
    for i in range(len(regions)):
        for j in range(i+1, len(regions)):
            corr = pearsonr(roi_signals[regions[i]], 
                           roi_signals[regions[j]])[0]
            correlations.append(corr)
    
    mean_correlation = np.mean(correlations)
    
    # Real faces: high correlation (>0.7)
    # Deepfakes: often low correlation (<0.5)
    coherence_score = mean_correlation
    
    return coherence_score, correlations
```

---

## Left-Right Symmetry Check

```python
def check_bilateral_symmetry(left_signal, right_signal):
    """
    Real faces should have symmetric physiological signals.
    """
    # Cross-correlation
    correlation = pearsonr(left_signal, right_signal)[0]
    
    # Phase alignment
    phase_diff = compute_phase_difference(left_signal, right_signal)
    
    # HR should be identical
    hr_left = estimate_heart_rate(left_signal)
    hr_right = estimate_heart_rate(right_signal)
    hr_diff = abs(hr_left - hr_right)
    
    symmetry_score = correlation * (1.0 - min(hr_diff / 10, 1.0))
    
    return symmetry_score, {
        'correlation': correlation,
        'phase_diff': phase_diff,
        'hr_diff': hr_diff
    }
```

---

## Signal Quality Assessment

```python
def assess_signal_quality(rppg_signal, fps):
    """
    Determine if the extracted signal is reliable.
    """
    # SNR estimation
    snr = compute_snr(rppg_signal, fps)
    
    # Spectral quality
    spectral_quality = compute_spectral_quality(rppg_signal, fps)
    
    # Stationarity
    stationarity = check_stationarity(rppg_signal)
    
    quality_score = (snr * 0.4 + spectral_quality * 0.4 + 
                     stationarity * 0.2)
    
    signal_valid = quality_score > 0.5
    
    return signal_valid, quality_score
```

---

## Anomaly Detection Logic

```python
def detect_physiological_anomalies(result):
    """
    Identify deepfake indicators from physiological analysis.
    """
    anomalies = []
    severity = 0.0
    
    # No signal detected
    if not result.signal_present:
        anomalies.append("NO_RPPG_SIGNAL")
        severity += 0.4
    
    # Unrealistic heart rate
    if result.heart_rate < 40 or result.heart_rate > 180:
        anomalies.append("UNREALISTIC_HR")
        severity += 0.2
    
    # Low spatial coherence
    if result.spatial_coherence < 0.5:
        anomalies.append("SPATIAL_INCOHERENCE")
        severity += 0.3
    
    # Asymmetric signals
    if result.left_right_correlation < 0.6:
        anomalies.append("BILATERAL_ASYMMETRY")
        severity += 0.25
    
    # Low signal quality
    if result.hr_confidence < 0.3:
        anomalies.append("LOW_SIGNAL_QUALITY")
        severity += 0.15
    
    return anomalies, min(severity, 1.0)
```

---

## Requirements for Reliable Analysis

1. **Minimum video duration**: 10 seconds (ideally 30+)
2. **Frame rate**: ≥25 fps (30 fps preferred)
3. **Face resolution**: ≥128x128 pixels
4. **Stable lighting**: Avoid flickering lights
5. **Minimal motion**: Excessive movement degrades signal

---

## Limitations

- Poor lighting severely impacts rPPG
- Dark skin tones require adapted algorithms
- Makeup can affect signal extraction
- Very short clips (<10s) are unreliable
- Compression artifacts can destroy subtle signals

