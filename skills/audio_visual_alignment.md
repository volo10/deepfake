# Audio-Visual Alignment Skill

## Purpose
Analyze the synchronization between audio and visual signals to detect lip-sync errors, voice-face mismatches, and audio manipulation indicators that are common in deepfakes.

## Core Principle
**Real videos have tightly coupled audio-visual signals. Deepfakes often exhibit subtle but detectable misalignments.**

---

## Capabilities

### 1. Lip Sync Analysis
- Phoneme-viseme correspondence
- Temporal alignment accuracy
- Mouth shape appropriateness

### 2. Speaker Verification
- Voice-face identity matching
- Speaker embedding comparison
- Cross-modal identity consistency

### 3. Emotion Coherence
- Audio emotion vs facial expression
- Prosody-gesture alignment
- Affect consistency

### 4. Audio Forensics
- Voice synthesis detection
- Audio splicing artifacts
- Compression inconsistencies

---

## Deepfake Indicators

| Indicator | Description | Weight |
|-----------|-------------|--------|
| Lip sync error | >100ms audio-visual offset | 1.5 |
| Voice-face mismatch | Speaker identity inconsistent | 1.4 |
| Emotion incongruence | Angry voice, neutral face | 1.2 |
| Synthetic voice | TTS or voice conversion detected | 1.3 |
| Audio splice | Discontinuities in audio track | 1.1 |

---

## Implementation Details

### Input
```python
@dataclass
class AudioVisualInput:
    video_frames: List[np.ndarray]
    audio_waveform: np.ndarray
    audio_sample_rate: int
    fps: float
    mouth_regions: List[np.ndarray]  # From face tracking
    landmarks: List[np.ndarray]
```

### Output
```python
@dataclass
class AudioVisualResult:
    sync_offset_ms: float  # Estimated AV offset
    sync_confidence: float  # Confidence in offset estimate
    lip_sync_score: float  # 0-1, lip-audio match quality
    voice_face_match: float  # Speaker verification score
    emotion_coherence: float  # Audio-visual emotion match
    voice_authenticity: float  # Real vs synthetic voice
    anomaly_score: float  # Combined assessment
    anomaly_details: List[str]
```

---

## Lip Sync Analysis

### SyncNet-based Analysis
```python
def analyze_lip_sync(mouth_sequence, audio_segment, model):
    """
    Use SyncNet-style model to assess lip-audio synchronization.
    """
    # Extract audio features (MFCC)
    mfcc = librosa.feature.mfcc(
        y=audio_segment, 
        sr=16000, 
        n_mfcc=13
    )
    
    # Preprocess mouth images
    mouth_tensor = preprocess_mouths(mouth_sequence)
    audio_tensor = preprocess_audio(mfcc)
    
    # Get embeddings
    visual_emb = model.encode_visual(mouth_tensor)
    audio_emb = model.encode_audio(audio_tensor)
    
    # Compute similarity
    sync_score = F.cosine_similarity(visual_emb, audio_emb)
    
    return sync_score.mean().item()
```

### Phoneme-Viseme Mapping
```python
def check_phoneme_viseme_correspondence(phonemes, mouth_shapes):
    """
    Verify that spoken phonemes match visual mouth shapes.
    """
    viseme_map = {
        'P': 'closed',  # Bilabial plosives
        'B': 'closed',
        'M': 'closed',
        'F': 'lower_lip_tuck',  # Labiodentals
        'V': 'lower_lip_tuck',
        'TH': 'tongue_visible',  # Dentals
        'S': 'teeth_close',  # Sibilants
        'AH': 'open',  # Vowels
        'OH': 'rounded',
        'EE': 'smile',
        # ... more mappings
    }
    
    matches = 0
    total = len(phonemes)
    
    for phoneme, mouth in zip(phonemes, mouth_shapes):
        expected_viseme = viseme_map.get(phoneme, 'neutral')
        actual_viseme = classify_mouth_shape(mouth)
        
        if expected_viseme == actual_viseme:
            matches += 1
    
    return matches / total if total > 0 else 0.0
```

### Temporal Offset Detection
```python
def detect_av_offset(mouth_sequence, audio_waveform, fps, sr):
    """
    Estimate audio-visual temporal offset using cross-correlation.
    """
    # Extract audio envelope
    audio_envelope = extract_envelope(audio_waveform, sr)
    
    # Extract mouth movement signal (mouth area over time)
    mouth_signal = [compute_mouth_openness(m) for m in mouth_sequence]
    
    # Resample to common rate
    audio_resampled = resample(audio_envelope, len(mouth_signal))
    
    # Cross-correlation
    correlation = np.correlate(mouth_signal, audio_resampled, mode='full')
    
    # Find peak
    peak_idx = np.argmax(correlation)
    center = len(mouth_signal) - 1
    offset_frames = peak_idx - center
    offset_ms = (offset_frames / fps) * 1000
    
    # Confidence based on peak prominence
    confidence = compute_peak_confidence(correlation, peak_idx)
    
    return offset_ms, confidence
```

---

## Speaker Verification

### Voice-Face Matching
```python
def verify_speaker(face_embeddings, voice_embeddings, model):
    """
    Cross-modal speaker verification.
    """
    # Get mean face embedding
    mean_face_emb = np.mean(face_embeddings, axis=0)
    
    # Get voice embedding from audio
    mean_voice_emb = np.mean(voice_embeddings, axis=0)
    
    # Cross-modal verification model
    match_score = model.verify(mean_face_emb, mean_voice_emb)
    
    return match_score
```

### Voice Embedding Extraction
```python
def extract_voice_embedding(audio, model):
    """
    Extract speaker embedding from audio.
    """
    # Preprocess audio
    audio_tensor = preprocess_audio_for_speaker(audio)
    
    # Extract embedding (e.g., using ECAPA-TDNN)
    with torch.no_grad():
        embedding = model(audio_tensor)
    
    return embedding.numpy()
```

---

## Emotion Coherence Analysis

```python
def analyze_emotion_coherence(audio_waveform, face_sequence, 
                               audio_emotion_model, face_emotion_model):
    """
    Check if emotional tone in voice matches facial expression.
    """
    # Audio emotion detection
    audio_features = extract_audio_features(audio_waveform)
    audio_emotions = audio_emotion_model.predict(audio_features)
    
    # Face emotion detection
    face_emotions = []
    for face in face_sequence:
        emotions = face_emotion_model.predict(face)
        face_emotions.append(emotions)
    
    # Aggregate face emotions
    mean_face_emotions = np.mean(face_emotions, axis=0)
    
    # Compare emotion distributions
    emotion_similarity = 1 - jensenshannon(
        audio_emotions, mean_face_emotions
    )
    
    # Check for specific mismatches
    mismatches = detect_emotion_mismatches(audio_emotions, mean_face_emotions)
    
    return emotion_similarity, mismatches
```

---

## Voice Authenticity Detection

### Synthetic Voice Detection
```python
def detect_synthetic_voice(audio_waveform, sr, model):
    """
    Detect if voice is generated by TTS or voice conversion.
    """
    # Extract features
    mel_spec = librosa.feature.melspectrogram(
        y=audio_waveform, sr=sr, n_mels=128
    )
    
    # Additional features
    mfcc = librosa.feature.mfcc(y=audio_waveform, sr=sr, n_mfcc=20)
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio_waveform, sr=sr
    )
    
    features = {
        'mel_spec': mel_spec,
        'mfcc': mfcc,
        'spectral_contrast': spectral_contrast
    }
    
    # Classify
    authenticity_score = model.predict(features)
    
    return authenticity_score
```

### Audio Artifact Detection
```python
def detect_audio_artifacts(audio_waveform, sr):
    """
    Detect splicing, synthesis, or manipulation artifacts.
    """
    artifacts = []
    
    # Spectral discontinuities (splicing)
    splice_points = detect_spectral_discontinuities(audio_waveform, sr)
    if splice_points:
        artifacts.append(('splice', splice_points))
    
    # Unnatural silence patterns
    silence_anomalies = detect_silence_anomalies(audio_waveform, sr)
    if silence_anomalies:
        artifacts.append(('silence', silence_anomalies))
    
    # Phase anomalies
    phase_issues = detect_phase_anomalies(audio_waveform, sr)
    if phase_issues:
        artifacts.append(('phase', phase_issues))
    
    # Vocoder artifacts
    vocoder_score = detect_vocoder_artifacts(audio_waveform, sr)
    if vocoder_score > 0.5:
        artifacts.append(('vocoder', vocoder_score))
    
    return artifacts
```

---

## Combined Analysis Pipeline

```python
def analyze_audio_visual_alignment(video_path, face_tracking_result):
    """
    Complete audio-visual alignment analysis.
    """
    # Load video and audio
    video = cv2.VideoCapture(video_path)
    audio, sr = librosa.load(video_path, sr=16000)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Extract mouth regions from tracking
    mouth_sequence = extract_mouth_regions(face_tracking_result)
    
    # Lip sync analysis
    sync_offset, sync_conf = detect_av_offset(mouth_sequence, audio, fps, sr)
    lip_sync_score = analyze_lip_sync(mouth_sequence, audio, sync_model)
    
    # Speaker verification
    face_embs = extract_face_embeddings(face_tracking_result.faces)
    voice_embs = extract_voice_embedding(audio, speaker_model)
    voice_face_match = verify_speaker(face_embs, voice_embs, cross_modal_model)
    
    # Emotion coherence
    emotion_score, emotion_mismatches = analyze_emotion_coherence(
        audio, face_tracking_result.faces,
        audio_emotion_model, face_emotion_model
    )
    
    # Voice authenticity
    voice_auth = detect_synthetic_voice(audio, sr, voice_auth_model)
    
    # Audio artifacts
    audio_artifacts = detect_audio_artifacts(audio, sr)
    
    # Compile results
    anomalies = []
    anomaly_score = 0.0
    
    if abs(sync_offset) > 100:  # >100ms offset
        anomalies.append(f"Lip sync offset: {sync_offset:.0f}ms")
        anomaly_score += 0.3
    
    if voice_face_match < 0.5:
        anomalies.append("Voice-face identity mismatch")
        anomaly_score += 0.25
    
    if emotion_score < 0.6:
        anomalies.append(f"Emotion incongruence: {emotion_mismatches}")
        anomaly_score += 0.2
    
    if voice_auth < 0.5:
        anomalies.append("Possible synthetic voice")
        anomaly_score += 0.25
    
    if audio_artifacts:
        anomalies.append(f"Audio artifacts: {audio_artifacts}")
        anomaly_score += 0.15
    
    return AudioVisualResult(
        sync_offset_ms=sync_offset,
        sync_confidence=sync_conf,
        lip_sync_score=lip_sync_score,
        voice_face_match=voice_face_match,
        emotion_coherence=emotion_score,
        voice_authenticity=voice_auth,
        anomaly_score=min(anomaly_score, 1.0),
        anomaly_details=anomalies
    )
```

---

## Models Required

| Component | Model | Purpose |
|-----------|-------|---------|
| Lip Sync | SyncNet / Wav2Lip | Lip-audio alignment |
| Speaker ID | ECAPA-TDNN | Voice embedding |
| Face Embedding | ArcFace | Face identity |
| Cross-Modal | VoxCeleb pretrained | Voice-face matching |
| Audio Emotion | Wav2Vec2 fine-tuned | Speech emotion |
| Face Emotion | FER-2013 CNN | Facial emotion |
| Voice Auth | ASVspoof model | Synthetic detection |

---

## Limitations

1. **Background noise**: Degrades audio analysis
2. **Multiple speakers**: Complicates verification
3. **Music/sfx**: Interferes with speech analysis
4. **Low quality audio**: Reduces feature reliability
5. **Silent videos**: Cannot perform AV analysis

