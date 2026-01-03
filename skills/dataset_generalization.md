# Dataset Generalization Skill

## Purpose
Ensure the detector performs well across diverse datasets, manipulation methods, and real-world conditions by implementing domain adaptation and robust feature learning.

## Core Principle
**A detector that only works on one dataset is useless in production. True detection requires generalization across unknown manipulation methods.**

---

## Capabilities

### 1. Cross-Dataset Evaluation
- Test on multiple benchmark datasets
- Measure generalization gap
- Identify dataset-specific biases

### 2. Domain Adaptation
- Adapt to new manipulation types
- Handle distribution shift
- Few-shot adaptation to new domains

### 3. Robust Feature Learning
- Learn manipulation-agnostic features
- Avoid dataset-specific shortcuts
- Focus on fundamental artifacts

### 4. Calibration
- Reliable confidence scores
- Cross-domain calibration
- Uncertainty quantification

---

## Key Datasets

| Dataset | Size | Manipulation Types | Challenge |
|---------|------|-------------------|-----------|
| FaceForensics++ | 1000 videos × 5 methods | DeepFakes, Face2Face, FaceSwap, NeuralTextures | Compression levels |
| Celeb-DF | 590 real, 5639 fake | High-quality deepfakes | Visual quality |
| DFDC | 100K+ clips | Various methods | Scale & diversity |
| DeeperForensics | 60K videos | DF-VAE | Real-world perturbations |
| WildDeepfake | 7314 sequences | Internet-collected | In-the-wild conditions |

---

## Implementation

### Cross-Dataset Evaluation Protocol
```python
def evaluate_cross_dataset(model, datasets):
    """
    Evaluate model on multiple datasets without retraining.
    """
    results = {}
    
    for dataset_name, dataset in datasets.items():
        # Evaluate
        predictions = []
        labels = []
        
        for video, label in dataset:
            pred = model.predict(video)
            predictions.append(pred)
            labels.append(label)
        
        # Metrics
        metrics = compute_metrics(predictions, labels)
        results[dataset_name] = {
            'accuracy': metrics['accuracy'],
            'auc': metrics['auc'],
            'eer': metrics['eer'],
            'ap': metrics['average_precision']
        }
    
    # Generalization gap
    train_dataset = model.training_dataset
    train_perf = results[train_dataset]['auc']
    
    generalization_gaps = {
        name: train_perf - result['auc']
        for name, result in results.items()
        if name != train_dataset
    }
    
    return results, generalization_gaps
```

### Domain Shift Detection
```python
def detect_domain_shift(train_features, test_features):
    """
    Detect if test data comes from different distribution.
    """
    # Maximum Mean Discrepancy
    mmd = compute_mmd(train_features, test_features)
    
    # Feature statistics comparison
    train_mean = np.mean(train_features, axis=0)
    test_mean = np.mean(test_features, axis=0)
    mean_shift = np.linalg.norm(train_mean - test_mean)
    
    train_cov = np.cov(train_features.T)
    test_cov = np.cov(test_features.T)
    cov_shift = np.linalg.norm(train_cov - test_cov, 'fro')
    
    # Domain shift score
    shift_detected = mmd > 0.1 or mean_shift > 1.0
    
    return {
        'mmd': mmd,
        'mean_shift': mean_shift,
        'cov_shift': cov_shift,
        'shift_detected': shift_detected
    }
```

---

## Domain Adaptation Methods

### Method 1: Adversarial Domain Adaptation
```python
class DomainAdaptiveDetector(nn.Module):
    """
    Detector with domain adversarial training.
    """
    def __init__(self, feature_dim, num_classes=2):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Task classifier (real vs fake)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Source vs target
        )
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        
        # Task prediction
        class_output = self.classifier(features)
        
        # Domain prediction (with gradient reversal)
        reversed_features = GradientReversal.apply(features, alpha)
        domain_output = self.domain_discriminator(reversed_features)
        
        return class_output, domain_output
```

### Method 2: Feature Alignment
```python
def align_features(source_features, target_features):
    """
    Align feature distributions across domains.
    """
    # Compute domain statistics
    source_mean = torch.mean(source_features, dim=0)
    source_std = torch.std(source_features, dim=0)
    
    target_mean = torch.mean(target_features, dim=0)
    target_std = torch.std(target_features, dim=0)
    
    # Align target to source distribution
    aligned_target = (target_features - target_mean) / (target_std + 1e-8)
    aligned_target = aligned_target * source_std + source_mean
    
    return aligned_target
```

### Method 3: Meta-Learning for Generalization
```python
class MAMLDetector:
    """
    Model-Agnostic Meta-Learning for few-shot adaptation.
    """
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    
    def adapt(self, support_set, num_steps=5):
        """
        Quickly adapt to new domain with few examples.
        """
        # Clone model
        adapted_model = copy.deepcopy(self.model)
        
        # Inner loop adaptation
        for _ in range(num_steps):
            loss = compute_loss(adapted_model, support_set)
            grads = torch.autograd.grad(loss, adapted_model.parameters())
            
            # Update adapted model
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data -= self.inner_lr * grad
        
        return adapted_model
```

---

## Robust Feature Learning

### Manipulation-Agnostic Features
```python
def learn_manipulation_agnostic_features(model, multi_dataset_loader):
    """
    Learn features that generalize across manipulation types.
    """
    # Contrastive learning across manipulations
    criterion = SupConLoss()
    
    for batch in multi_dataset_loader:
        images, labels, manipulation_types = batch
        
        # Extract features
        features = model.encode(images)
        
        # Contrastive loss: same class (real/fake) should be similar
        # regardless of manipulation type
        loss = criterion(features, labels)
        
        # Penalize manipulation-specific features
        manipulation_classifier = ManipulationClassifier(features)
        manipulation_loss = F.cross_entropy(manipulation_classifier, manipulation_types)
        
        # Total loss: minimize task loss, maximize confusion on manipulation type
        total_loss = loss - 0.1 * manipulation_loss  # Gradient reversal effect
        
        total_loss.backward()
        optimizer.step()
```

### Attention to Fundamental Artifacts
```python
def extract_fundamental_artifacts(face_crop):
    """
    Extract features that are fundamental to all deepfakes.
    """
    artifacts = {}
    
    # Blending boundary (universal)
    artifacts['boundary'] = detect_blending_boundary(face_crop)
    
    # Frequency anomalies (GAN-universal)
    artifacts['frequency'] = analyze_frequency_spectrum(face_crop)
    
    # Temporal inconsistency (sequence-based)
    # artifacts['temporal'] = ... (requires sequence)
    
    # Compression mismatch
    artifacts['compression'] = detect_compression_mismatch(face_crop)
    
    return artifacts
```

---

## Calibration

### Temperature Scaling
```python
def calibrate_model(model, calibration_set):
    """
    Calibrate model confidence using temperature scaling.
    """
    model.eval()
    
    logits_list = []
    labels_list = []
    
    # Collect logits
    for images, labels in calibration_set:
        with torch.no_grad():
            logits = model(images)
        logits_list.append(logits)
        labels_list.append(labels)
    
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    # Find optimal temperature
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    def eval():
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    return temperature.item()
```

### Uncertainty Quantification
```python
def quantify_uncertainty(model, image, num_samples=30):
    """
    Estimate prediction uncertainty using MC Dropout.
    """
    model.train()  # Enable dropout
    
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            pred = model(image)
        predictions.append(pred)
    
    predictions = torch.stack(predictions)
    
    # Mean prediction
    mean_pred = predictions.mean(dim=0)
    
    # Uncertainty (variance)
    uncertainty = predictions.var(dim=0)
    
    # Entropy-based uncertainty
    probs = F.softmax(mean_pred, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    
    return {
        'prediction': mean_pred,
        'uncertainty': uncertainty,
        'entropy': entropy,
        'is_confident': entropy < 0.5
    }
```

---

## Evaluation Metrics

### Standard Metrics
```python
def compute_metrics(predictions, labels):
    """
    Compute standard detection metrics.
    """
    # Binary predictions
    binary_preds = (predictions > 0.5).astype(int)
    
    # Accuracy
    accuracy = accuracy_score(labels, binary_preds)
    
    # AUC-ROC
    auc = roc_auc_score(labels, predictions)
    
    # Average Precision
    ap = average_precision_score(labels, predictions)
    
    # Equal Error Rate
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'ap': ap,
        'eer': eer
    }
```

### Generalization Metrics
```python
def compute_generalization_metrics(results_dict):
    """
    Compute metrics specifically for generalization.
    """
    aucs = [r['auc'] for r in results_dict.values()]
    
    metrics = {
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'min_auc': np.min(aucs),
        'max_auc': np.max(aucs),
        'generalization_gap': np.max(aucs) - np.min(aucs)
    }
    
    return metrics
```

---

## Best Practices

1. **Train on diverse data**: Mix multiple datasets during training
2. **Use data augmentation**: Compression, blur, noise, color jitter
3. **Avoid shortcuts**: Ensure model doesn't learn dataset-specific biases
4. **Regular cross-evaluation**: Test on held-out datasets
5. **Ensemble methods**: Combine multiple detectors
6. **Continuous updating**: Retrain on newly discovered manipulation types

---

## Known Challenges

1. **New manipulation methods**: Zero-shot detection is hard
2. **Adversarial attacks**: Crafted inputs to fool detectors
3. **Quality variations**: Compression, resolution differences
4. **Bias propagation**: Inheriting biases from training data
5. **Real-time constraints**: Generalization vs speed tradeoff

