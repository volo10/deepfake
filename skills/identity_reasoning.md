# Identity Reasoning Skill

## Purpose
Perform cross-modal identity verification and consistency checking to detect identity theft, face swaps, and identity instability across time and modalities.

## Core Principle
**Identity should remain consistent across all signals (face, voice, behavior) and across time. Inconsistencies reveal manipulation.**

---

## Capabilities

### 1. Face Identity Tracking
- Extract face embeddings over time
- Detect identity drift or switches
- Compare against known references

### 2. Voice-Face Cross-Modal Matching
- Verify speaker identity matches face
- Detect voice-face swaps
- Cross-modal biometric consistency

### 3. Behavioral Biometrics
- Head movement patterns
- Facial expression tendencies
- Micro-expression consistency

### 4. Reference Verification
- Compare against known identity database
- Public figure verification
- Multi-source identity triangulation

---

## Deepfake Indicators

| Indicator | Description | Weight |
|-----------|-------------|--------|
| Identity drift | Face embedding changes >0.4 over video | 1.5 |
| Identity switch | Sudden embedding jump | 1.6 |
| Voice-face mismatch | Cross-modal inconsistency | 1.4 |
| Reference mismatch | Doesn't match known identity | 1.3 |
| Behavioral anomaly | Uncharacteristic expressions | 1.2 |

---

## Implementation Details

### Input
```python
@dataclass
class IdentityInput:
    face_sequence: List[np.ndarray]  # Aligned face crops
    face_embeddings: List[np.ndarray]  # Pre-computed embeddings
    voice_embedding: np.ndarray  # Speaker embedding
    landmarks_sequence: List[np.ndarray]
    reference_faces: Optional[List[np.ndarray]]  # Known identity refs
    reference_voice: Optional[np.ndarray]  # Known voice ref
    claimed_identity: Optional[str]  # Name if known
```

### Output
```python
@dataclass
class IdentityResult:
    identity_stability: float  # 0-1, consistency over time
    identity_switches: List[int]  # Frame indices of switches
    mean_embedding: np.ndarray  # Average face embedding
    embedding_variance: float  # Embedding spread
    voice_face_consistency: float  # Cross-modal match
    reference_match: Optional[float]  # Match to reference if provided
    behavioral_consistency: float  # Expression/movement patterns
    verified_identity: Optional[str]  # Identified person if matched
    anomaly_score: float
    anomaly_details: List[str]
```

---

## Face Identity Analysis

### Embedding Extraction
```python
def extract_face_embeddings(faces, model):
    """
    Extract face embeddings using ArcFace or similar.
    """
    embeddings = []
    
    for face in faces:
        # Preprocess
        face_tensor = preprocess_face(face)
        
        # Extract embedding
        with torch.no_grad():
            embedding = model(face_tensor)
        
        embeddings.append(embedding.numpy())
    
    return np.array(embeddings)
```

### Identity Stability Analysis
```python
def analyze_identity_stability(embeddings, window_size=30):
    """
    Analyze how stable identity is over time.
    """
    # Compute centroid
    centroid = np.mean(embeddings, axis=0)
    
    # Distance from centroid
    distances = [cosine_distance(e, centroid) for e in embeddings]
    
    # Statistics
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    variance = np.var(distances)
    
    # Stability score (inversely related to distance)
    stability = 1.0 - min(mean_distance / 0.5, 1.0)
    
    # Detect drift over time
    window_centroids = []
    for i in range(0, len(embeddings) - window_size, window_size // 2):
        window = embeddings[i:i + window_size]
        window_centroids.append(np.mean(window, axis=0))
    
    # Check if centroids are drifting
    if len(window_centroids) > 1:
        drift_distances = [
            cosine_distance(window_centroids[i], window_centroids[i+1])
            for i in range(len(window_centroids) - 1)
        ]
        drift = np.max(drift_distances)
    else:
        drift = 0.0
    
    return {
        'stability': stability,
        'mean_distance': mean_distance,
        'max_distance': max_distance,
        'variance': variance,
        'drift': drift
    }
```

### Identity Switch Detection
```python
def detect_identity_switches(embeddings, threshold=0.4):
    """
    Detect sudden identity changes (face swaps).
    """
    switches = []
    
    for i in range(1, len(embeddings)):
        distance = cosine_distance(embeddings[i], embeddings[i-1])
        
        if distance > threshold:
            switches.append({
                'frame': i,
                'distance': distance,
                'severity': 'high' if distance > 0.6 else 'medium'
            })
    
    return switches
```

---

## Cross-Modal Identity Verification

### Voice-Face Matching
```python
def verify_voice_face_match(face_embedding, voice_embedding, model):
    """
    Verify that voice and face belong to same person.
    """
    # Project to common space
    face_proj = model.project_face(face_embedding)
    voice_proj = model.project_voice(voice_embedding)
    
    # Compute similarity
    similarity = cosine_similarity(face_proj, voice_proj)
    
    # Threshold-based decision
    match_score = similarity
    
    return match_score
```

### Multi-Modal Identity Score
```python
def compute_multimodal_identity_score(
    face_embeddings, voice_embedding, 
    face_voice_model, face_model, voice_model
):
    """
    Aggregate identity signals from multiple modalities.
    """
    # Face identity consistency
    face_stability = analyze_identity_stability(face_embeddings)
    
    # Voice-face cross-modal
    mean_face = np.mean(face_embeddings, axis=0)
    voice_face_match = verify_voice_face_match(
        mean_face, voice_embedding, face_voice_model
    )
    
    # Combined score
    scores = {
        'face_stability': face_stability['stability'],
        'voice_face_match': voice_face_match,
        'combined': 0.6 * face_stability['stability'] + 0.4 * voice_face_match
    }
    
    return scores
```

---

## Reference Verification

### Known Identity Matching
```python
def match_against_reference(embeddings, reference_embeddings, threshold=0.3):
    """
    Compare detected face against known reference images.
    """
    # Mean embedding from video
    video_embedding = np.mean(embeddings, axis=0)
    
    # Mean reference embedding
    ref_embedding = np.mean(reference_embeddings, axis=0)
    
    # Distance
    distance = cosine_distance(video_embedding, ref_embedding)
    
    # Match score (inverse of distance)
    match_score = 1.0 - min(distance / 0.6, 1.0)
    
    # Per-reference matching
    ref_distances = [
        cosine_distance(video_embedding, ref)
        for ref in reference_embeddings
    ]
    
    return {
        'match_score': match_score,
        'distance': distance,
        'ref_distances': ref_distances,
        'is_match': distance < threshold
    }
```

### Public Figure Database Search
```python
def search_public_figures(embedding, database, top_k=5):
    """
    Search database of known public figures.
    """
    distances = []
    
    for name, ref_embedding in database.items():
        dist = cosine_distance(embedding, ref_embedding)
        distances.append((name, dist))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    # Return top matches
    top_matches = distances[:top_k]
    
    # Best match
    best_name, best_dist = top_matches[0]
    
    return {
        'top_matches': top_matches,
        'best_match': best_name if best_dist < 0.3 else None,
        'best_distance': best_dist,
        'confidence': 1.0 - best_dist if best_dist < 0.3 else 0.0
    }
```

---

## Behavioral Biometrics

### Expression Pattern Analysis
```python
def analyze_expression_patterns(landmarks_sequence, au_sequence):
    """
    Analyze facial expression patterns for identity verification.
    """
    # Extract expression features over time
    expression_features = []
    
    for landmarks, aus in zip(landmarks_sequence, au_sequence):
        features = {
            'mouth_asymmetry': compute_mouth_asymmetry(landmarks),
            'eye_asymmetry': compute_eye_asymmetry(landmarks),
            'dominant_au': np.argmax(aus),
            'au_intensities': aus
        }
        expression_features.append(features)
    
    # Compute behavioral signature
    behavioral_signature = {
        'mean_asymmetries': np.mean([f['mouth_asymmetry'] for f in expression_features]),
        'au_distribution': np.mean([f['au_intensities'] for f in expression_features], axis=0),
        'expression_variability': compute_expression_variability(expression_features)
    }
    
    return behavioral_signature
```

### Micro-Expression Consistency
```python
def check_micro_expression_consistency(au_sequence):
    """
    Check that micro-expressions are consistent with claimed identity.
    """
    # Detect micro-expressions
    micro_expressions = detect_micro_expressions(au_sequence)
    
    # Check for inconsistencies
    # Real people have consistent micro-expression patterns
    # Deepfakes may have inconsistent or missing micro-expressions
    
    consistency_score = compute_micro_expression_consistency(micro_expressions)
    
    return consistency_score
```

---

## Combined Identity Analysis

```python
def analyze_identity(input_data: IdentityInput):
    """
    Complete identity analysis pipeline.
    """
    # Face identity stability
    stability_result = analyze_identity_stability(input_data.face_embeddings)
    
    # Detect switches
    switches = detect_identity_switches(input_data.face_embeddings)
    
    # Cross-modal verification
    voice_face_score = verify_voice_face_match(
        np.mean(input_data.face_embeddings, axis=0),
        input_data.voice_embedding,
        voice_face_model
    )
    
    # Reference matching
    reference_match = None
    if input_data.reference_faces is not None:
        ref_embeddings = extract_face_embeddings(
            input_data.reference_faces, face_model
        )
        reference_match = match_against_reference(
            input_data.face_embeddings, ref_embeddings
        )
    
    # Behavioral analysis
    behavioral = analyze_expression_patterns(
        input_data.landmarks_sequence,
        extract_action_units(input_data.face_sequence)
    )
    
    # Compile anomalies
    anomalies = []
    anomaly_score = 0.0
    
    if stability_result['stability'] < 0.7:
        anomalies.append(f"Identity instability: {stability_result['stability']:.2f}")
        anomaly_score += 0.3
    
    if len(switches) > 0:
        anomalies.append(f"Identity switches detected at frames: {[s['frame'] for s in switches]}")
        anomaly_score += 0.4
    
    if voice_face_score < 0.5:
        anomalies.append("Voice-face identity mismatch")
        anomaly_score += 0.25
    
    if reference_match and not reference_match['is_match']:
        anomalies.append(f"Does not match claimed identity (dist: {reference_match['distance']:.2f})")
        anomaly_score += 0.3
    
    return IdentityResult(
        identity_stability=stability_result['stability'],
        identity_switches=[s['frame'] for s in switches],
        mean_embedding=np.mean(input_data.face_embeddings, axis=0),
        embedding_variance=stability_result['variance'],
        voice_face_consistency=voice_face_score,
        reference_match=reference_match['match_score'] if reference_match else None,
        behavioral_consistency=behavioral.get('consistency', 0.8),
        verified_identity=None,  # Would be filled by database search
        anomaly_score=min(anomaly_score, 1.0),
        anomaly_details=anomalies
    )
```

---

## Visualization

```python
def visualize_identity_analysis(result, embeddings, save_path):
    """
    Visualize identity analysis results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Embedding distances over time
    centroid = result.mean_embedding
    distances = [cosine_distance(e, centroid) for e in embeddings]
    axes[0, 0].plot(distances)
    for switch in result.identity_switches:
        axes[0, 0].axvline(x=switch, color='red', linestyle='--')
    axes[0, 0].set_title('Identity Distance from Centroid')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Cosine Distance')
    
    # t-SNE of embeddings
    if len(embeddings) > 10:
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1))
        emb_2d = tsne.fit_transform(embeddings)
        axes[0, 1].scatter(emb_2d[:, 0], emb_2d[:, 1], 
                          c=range(len(embeddings)), cmap='viridis')
        axes[0, 1].set_title('Embedding Space (t-SNE)')
    
    # Score summary
    scores = {
        'Identity Stability': result.identity_stability,
        'Voice-Face Match': result.voice_face_consistency,
        'Reference Match': result.reference_match or 0,
        'Behavioral': result.behavioral_consistency
    }
    axes[1, 0].barh(list(scores.keys()), list(scores.values()))
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_title('Identity Scores')
    
    # Anomaly summary
    axes[1, 1].text(0.1, 0.9, f"Anomaly Score: {result.anomaly_score:.2f}",
                    fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    for i, anomaly in enumerate(result.anomaly_details):
        axes[1, 1].text(0.1, 0.7 - i*0.15, f"• {anomaly}",
                       fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Detected Anomalies')
    
    plt.tight_layout()
    plt.savefig(save_path)
```

---

## Limitations

1. **Twins/lookalikes**: May produce false positives
2. **Age changes**: Reference images must be recent
3. **Disguises**: Makeup, glasses can affect matching
4. **Cross-ethnic bias**: Some models perform unequally
5. **Low quality**: Poor video reduces accuracy

