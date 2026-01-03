# Frequency Artifacts Skill

## Purpose
Detect artifacts in the frequency domain that reveal synthetic image generation, including GAN fingerprints, upsampling patterns, and compression inconsistencies.

## Core Principle
**Neural networks leave distinctive fingerprints in the frequency spectrum that are invisible to humans but detectable through spectral analysis.**

---

## Capabilities

### 1. GAN Fingerprint Detection
- Detect periodic artifacts from transposed convolutions
- Identify generator-specific spectral patterns
- Cross-GAN fingerprint matching

### 2. Upsampling Artifact Detection
- Checkerboard patterns from deconvolution
- Nearest-neighbor upsampling artifacts
- Bilinear interpolation traces

### 3. Compression Forensics
- Double compression detection
- Inconsistent quantization tables
- Re-encoding artifacts

### 4. Noise Pattern Analysis
- Sensor noise consistency
- PRNU (Photo Response Non-Uniformity) analysis
- Synthetic noise detection

---

## Deepfake Indicators

| Indicator | Description | Weight |
|-----------|-------------|--------|
| GAN fingerprint | Characteristic spectral peaks | 1.5 |
| Checkerboard pattern | Upsampling artifacts in DCT | 1.4 |
| Noise inconsistency | Face noise differs from background | 1.3 |
| Double compression | Multiple encoding passes | 1.2 |
| Missing high-freq | Unnatural spectral rolloff | 1.3 |

---

## Implementation Details

### Input
```python
@dataclass
class FrequencyInput:
    frames: List[np.ndarray]  # Full frames
    face_crops: List[np.ndarray]  # Aligned face crops
    background_crops: List[np.ndarray]  # Non-face regions
```

### Output
```python
@dataclass
class FrequencyResult:
    gan_fingerprint_score: float  # 0-1, likelihood of GAN
    checkerboard_score: float  # Upsampling artifact severity
    noise_consistency: float  # Face vs background noise match
    compression_anomaly: float  # Double compression indicator
    spectral_naturalness: float  # Overall spectral plausibility
    detected_artifacts: List[ArtifactDetail]
    anomaly_score: float  # Combined score
```

---

## GAN Fingerprint Detection

### Spectral Analysis
```python
def detect_gan_fingerprint(face_crop):
    """
    Analyze frequency spectrum for GAN-specific patterns.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    
    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log(1 + np.abs(f_shift))
    
    # Analyze for periodic peaks (GAN artifact)
    peaks = find_spectral_peaks(magnitude)
    
    # Check for characteristic GAN patterns
    gan_score = analyze_peak_pattern(peaks)
    
    return gan_score, magnitude
```

### Azimuthal Average Analysis
```python
def azimuthal_average(spectrum):
    """
    Compute radially averaged power spectrum.
    Real images have smooth decay; GANs show artifacts.
    """
    center = np.array(spectrum.shape) // 2
    y, x = np.ogrid[:spectrum.shape[0], :spectrum.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # Bin by radius
    radial_mean = ndimage.mean(spectrum, r, 
                               index=np.arange(0, r.max()))
    
    return radial_mean
```

### GAN Fingerprint Classifier
```python
def classify_gan_fingerprint(spectrum, model):
    """
    Use trained classifier to detect GAN fingerprints.
    """
    # Extract features
    azimuthal = azimuthal_average(spectrum)
    high_freq_energy = compute_hf_energy(spectrum)
    peak_features = extract_peak_features(spectrum)
    
    features = np.concatenate([
        azimuthal, 
        [high_freq_energy], 
        peak_features
    ])
    
    # Classify
    gan_probability = model.predict_proba([features])[0, 1]
    
    return gan_probability
```

---

## Checkerboard Artifact Detection

```python
def detect_checkerboard(image):
    """
    Detect checkerboard patterns from transposed convolutions.
    """
    # High-pass filter to isolate artifacts
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    high_pass = cv2.filter2D(image, -1, kernel)
    
    # FFT of high-pass result
    fft = np.fft.fft2(high_pass)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # Look for periodic peaks at specific frequencies
    # Checkerboard = peaks at Nyquist frequency
    h, w = magnitude.shape
    corners = [
        magnitude[0:h//4, 0:w//4],
        magnitude[0:h//4, 3*w//4:],
        magnitude[3*h//4:, 0:w//4],
        magnitude[3*h//4:, 3*w//4:]
    ]
    
    corner_energy = sum(np.sum(c) for c in corners)
    center_energy = np.sum(magnitude[h//4:3*h//4, w//4:3*w//4])
    
    checkerboard_score = corner_energy / (center_energy + 1e-8)
    
    return checkerboard_score
```

---

## Noise Consistency Analysis

```python
def analyze_noise_consistency(face_region, background_region):
    """
    Compare noise patterns between face and background.
    Inconsistency indicates manipulation.
    """
    # Extract noise using denoising difference
    face_denoised = cv2.fastNlMeansDenoisingColored(face_region)
    bg_denoised = cv2.fastNlMeansDenoisingColored(background_region)
    
    face_noise = face_region.astype(float) - face_denoised.astype(float)
    bg_noise = background_region.astype(float) - bg_denoised.astype(float)
    
    # Compare noise statistics
    face_noise_std = np.std(face_noise)
    bg_noise_std = np.std(bg_noise)
    
    # Compare noise spectra
    face_spectrum = compute_noise_spectrum(face_noise)
    bg_spectrum = compute_noise_spectrum(bg_noise)
    
    spectral_similarity = compute_spectral_similarity(
        face_spectrum, bg_spectrum
    )
    
    # Compute consistency score
    std_ratio = min(face_noise_std, bg_noise_std) / \
                max(face_noise_std, bg_noise_std)
    
    consistency = (std_ratio + spectral_similarity) / 2
    
    return consistency
```

---

## DCT Analysis for Compression Forensics

```python
def analyze_dct_coefficients(image):
    """
    Analyze DCT coefficients for compression artifacts.
    """
    # Convert to YCbCr
    ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel = ycbcr[:, :, 0]
    
    # Block-wise DCT (8x8 blocks like JPEG)
    h, w = y_channel.shape
    dct_coeffs = []
    
    for i in range(0, h-7, 8):
        for j in range(0, w-7, 8):
            block = y_channel[i:i+8, j:j+8].astype(float)
            dct_block = cv2.dct(block)
            dct_coeffs.append(dct_block.flatten())
    
    dct_coeffs = np.array(dct_coeffs)
    
    # Analyze coefficient distribution
    # Double compression creates distinctive patterns
    return analyze_dct_distribution(dct_coeffs)
```

### Double Compression Detection
```python
def detect_double_compression(dct_coeffs):
    """
    Detect signs of double JPEG compression.
    """
    # First digit distribution (Benford's Law variation)
    first_digits = extract_first_digits(dct_coeffs)
    
    # In single compression: smooth distribution
    # In double compression: periodic artifacts
    periodicity = compute_periodicity(first_digits)
    
    # Blocking artifact analysis
    blocking_score = compute_blocking_artifacts(dct_coeffs)
    
    double_compression_score = (periodicity + blocking_score) / 2
    
    return double_compression_score
```

---

## High-Frequency Energy Analysis

```python
def analyze_high_frequency_content(image):
    """
    GANs often have unnatural high-frequency characteristics.
    """
    # FFT
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    center = (h // 2, w // 2)
    
    # Create frequency bands
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Low, mid, high frequency masks
    low_mask = r < min(h, w) * 0.1
    mid_mask = (r >= min(h, w) * 0.1) & (r < min(h, w) * 0.3)
    high_mask = r >= min(h, w) * 0.3
    
    # Energy in each band
    low_energy = np.sum(magnitude * low_mask)
    mid_energy = np.sum(magnitude * mid_mask)
    high_energy = np.sum(magnitude * high_mask)
    
    total_energy = low_energy + mid_energy + high_energy
    
    # Natural images have specific energy distribution
    # GANs often lack high-frequency detail or have artifacts
    hf_ratio = high_energy / total_energy
    
    return {
        'low': low_energy / total_energy,
        'mid': mid_energy / total_energy,
        'high': hf_ratio,
        'naturalness': compute_naturalness(hf_ratio)
    }
```

---

## Combined Frequency Analysis

```python
def analyze_frequency_artifacts(face_crop, background, full_frame):
    """
    Comprehensive frequency domain analysis.
    """
    results = {}
    
    # GAN fingerprint
    results['gan_fingerprint'] = detect_gan_fingerprint(face_crop)
    
    # Checkerboard artifacts
    results['checkerboard'] = detect_checkerboard(face_crop)
    
    # Noise consistency
    results['noise_consistency'] = analyze_noise_consistency(
        face_crop, background
    )
    
    # Compression analysis
    results['compression'] = analyze_dct_coefficients(full_frame)
    
    # High-frequency analysis
    results['hf_analysis'] = analyze_high_frequency_content(face_crop)
    
    # Compute overall anomaly score
    anomaly_score = compute_frequency_anomaly_score(results)
    
    return FrequencyResult(**results, anomaly_score=anomaly_score)
```

---

## Visualization

```python
def visualize_frequency_analysis(result, save_path):
    """
    Create visualization of frequency analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original spectrum
    axes[0, 0].imshow(np.log(1 + result.magnitude), cmap='viridis')
    axes[0, 0].set_title('Frequency Spectrum')
    
    # Azimuthal average
    axes[0, 1].plot(result.azimuthal_average)
    axes[0, 1].set_title('Radial Power Spectrum')
    
    # Checkerboard visualization
    axes[0, 2].imshow(result.checkerboard_map, cmap='hot')
    axes[0, 2].set_title('Checkerboard Artifacts')
    
    # Noise pattern
    axes[1, 0].imshow(result.noise_pattern, cmap='gray')
    axes[1, 0].set_title('Extracted Noise')
    
    # DCT coefficients
    axes[1, 1].bar(range(64), result.mean_dct_coeffs)
    axes[1, 1].set_title('Mean DCT Coefficients')
    
    # Score summary
    axes[1, 2].barh(['GAN', 'Checker', 'Noise', 'Compress'],
                    [result.gan_fingerprint_score,
                     result.checkerboard_score,
                     1 - result.noise_consistency,
                     result.compression_anomaly])
    axes[1, 2].set_title('Anomaly Scores')
    
    plt.tight_layout()
    plt.savefig(save_path)
```

---

## Known Limitations

1. **Compression**: Heavy compression destroys subtle artifacts
2. **Post-processing**: Blur, noise addition can hide fingerprints
3. **New GANs**: May have unknown fingerprints
4. **Resolution**: Low resolution reduces artifact visibility

