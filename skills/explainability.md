# Explainability Skill

## Purpose
Provide human-interpretable explanations for detection decisions, enabling trust, debugging, and actionable insights.

## Core Principle
**A black-box "FAKE" verdict is not useful. Users need to understand WHY the system flagged content.**

---

## Capabilities

### 1. Visual Explanations
- Attention heatmaps showing suspicious regions
- Artifact highlighting on frames
- Temporal anomaly visualization

### 2. Textual Explanations
- Natural language descriptions of findings
- Confidence breakdowns by category
- Evidence summaries

### 3. Evidence Aggregation
- List of triggered detectors
- Severity ranking
- Timeline of suspicion scores

---

## Implementation

### Attention Visualization
```python
def generate_attention_map(model, image):
    """
    Generate Grad-CAM attention map.
    """
    model.eval()
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    score = output[0, 1]  # Fake class score
    
    # Backward pass
    score.backward()
    
    # Get gradients and activations
    gradients = model.get_gradients()
    activations = model.get_activations()
    
    # Weight activations by gradients
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1)
    cam = F.relu(cam)
    
    # Normalize and resize
    cam = cam / cam.max()
    cam = F.interpolate(cam, image.shape[-2:])
    
    return cam.squeeze().numpy()
```

### Generate Text Explanation
```python
def generate_explanation(results):
    """
    Create human-readable explanation.
    """
    explanation = []
    
    # Verdict
    if results.verdict == 'DEEPFAKE':
        explanation.append(f"⚠️ This video is likely a DEEPFAKE (confidence: {results.confidence:.1%})")
    elif results.verdict == 'REAL':
        explanation.append(f"✓ This video appears AUTHENTIC (confidence: {results.confidence:.1%})")
    else:
        explanation.append(f"❓ Analysis is UNCERTAIN (confidence: {results.confidence:.1%})")
    
    explanation.append("\n**Key Findings:**")
    
    # List evidence
    for finding in results.findings:
        severity_icon = '🔴' if finding.severity > 0.7 else '🟡' if finding.severity > 0.4 else '🟢'
        explanation.append(f"  {severity_icon} {finding.description}")
    
    # Time segments
    if results.suspicious_segments:
        explanation.append("\n**Most Suspicious Segments:**")
        for seg in results.suspicious_segments[:3]:
            explanation.append(f"  • {seg.start:.1f}s - {seg.end:.1f}s: {seg.reason}")
    
    return '\n'.join(explanation)
```

### Evidence Timeline
```python
def create_evidence_timeline(frame_scores, findings, fps):
    """
    Create timeline visualization of suspicion.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Suspicion score over time
    times = np.arange(len(frame_scores)) / fps
    axes[0].plot(times, frame_scores, 'b-', linewidth=1)
    axes[0].fill_between(times, frame_scores, alpha=0.3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    axes[0].set_ylabel('Suspicion Score')
    axes[0].set_title('Detection Timeline')
    axes[0].legend()
    
    # Finding markers
    for finding in findings:
        if hasattr(finding, 'frame_idx'):
            t = finding.frame_idx / fps
            axes[0].axvline(x=t, color='orange', alpha=0.5)
    
    # Category breakdown over time
    categories = ['face', 'temporal', 'audio', 'frequency']
    for cat in categories:
        if cat in findings:
            axes[1].plot(times, findings[cat], label=cat)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Category Scores')
    axes[1].legend()
    
    plt.tight_layout()
    return fig
```

---

## Explanation Report

```python
@dataclass
class ExplanationReport:
    verdict: str
    confidence: float
    summary: str
    detailed_findings: List[Finding]
    attention_maps: Dict[int, np.ndarray]  # frame_idx -> heatmap
    timeline_plot: matplotlib.figure.Figure
    suspicious_segments: List[TimeSegment]
    
    def to_html(self):
        """Generate HTML report."""
        ...
    
    def to_json(self):
        """Generate JSON report."""
        ...
```

---

## User Interface Elements

### Confidence Breakdown
```
Overall Confidence: 87%
├── Face Analysis:      92% suspicious
├── Temporal Analysis:  85% suspicious
├── Audio-Visual:       78% suspicious
├── Frequency Analysis: 71% suspicious
└── Physiological:      INSUFFICIENT DATA
```

### Suspicious Region Markers
- Red overlay on detected artifacts
- Bounding boxes around problem areas
- Frame-by-frame annotation

---

## Best Practices

1. **Be specific**: "Lip sync offset of 150ms at 2.3s" not "audio issues"
2. **Show uncertainty**: Clearly indicate when analysis is inconclusive
3. **Provide context**: Compare to baseline expectations
4. **Avoid jargon**: Use plain language for non-experts
5. **Support verification**: Allow users to inspect specific frames

