"""
Frequency Artifact Analyzer

Detects artifacts in the frequency domain that reveal synthetic image
generation, including GAN fingerprints and upsampling patterns.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from scipy import ndimage

from ..models import Finding, Severity, SkillResult

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """
    Analyze frequency domain artifacts characteristic of deepfakes.
    
    Detects:
    - GAN fingerprints (periodic spectral patterns)
    - Checkerboard artifacts from upsampling
    - Noise inconsistencies
    - Compression anomalies
    """
    
    def __init__(self):
        pass
        
    def analyze(self, face_crops: List[np.ndarray],
                full_frames: Optional[List[np.ndarray]] = None) -> SkillResult:
        """
        Analyze frequency domain artifacts.
        
        Args:
            face_crops: Aligned face crops
            full_frames: Optional full video frames for context
            
        Returns:
            SkillResult with frequency analysis
        """
        findings = []
        
        if len(face_crops) == 0:
            return SkillResult(
                skill_name="frequency",
                score=0.5,
                confidence=0.1,
                findings=[],
                raw_data={}
            )
        
        # Sample faces for analysis (don't need all frames)
        sample_indices = np.linspace(0, len(face_crops) - 1, 
                                     min(30, len(face_crops)), dtype=int)
        sampled_faces = [face_crops[i] for i in sample_indices]
        
        # GAN fingerprint detection
        gan_score, gan_details = self._detect_gan_fingerprint(sampled_faces)
        if gan_score > 0.4:
            findings.append(Finding(
                category="frequency",
                description=f"Potential GAN fingerprint detected (score: {gan_score:.2f})",
                severity=Severity.HIGH if gan_score > 0.7 else Severity.MEDIUM,
                confidence=gan_score,
                evidence=gan_details
            ))
        
        # Checkerboard artifact detection
        checker_score = self._detect_checkerboard(sampled_faces)
        if checker_score > 0.3:
            findings.append(Finding(
                category="frequency",
                description=f"Checkerboard artifacts detected (score: {checker_score:.2f})",
                severity=Severity.MEDIUM,
                confidence=checker_score
            ))
        
        # High-frequency analysis
        hf_result = self._analyze_high_frequency(sampled_faces)
        if hf_result["anomaly_score"] > 0.4:
            findings.append(Finding(
                category="frequency",
                description=f"Unnatural high-frequency characteristics",
                severity=Severity.MEDIUM,
                confidence=hf_result["anomaly_score"]
            ))
        
        # Noise consistency (if full frames available)
        noise_score = 1.0
        if full_frames and len(full_frames) > 0:
            noise_score = self._analyze_noise_consistency(
                sampled_faces, 
                [full_frames[i] for i in sample_indices]
            )
            if noise_score < 0.5:
                findings.append(Finding(
                    category="frequency",
                    description=f"Noise inconsistency between face and background",
                    severity=Severity.MEDIUM,
                    confidence=1.0 - noise_score
                ))
        
        # Overall anomaly score
        anomaly_score = max(
            gan_score * 0.4,
            checker_score * 0.25,
            hf_result["anomaly_score"] * 0.2,
            (1.0 - noise_score) * 0.15
        )
        
        # Aggregate as weighted sum
        anomaly_score = (
            gan_score * 0.4 +
            checker_score * 0.25 +
            hf_result["anomaly_score"] * 0.2 +
            (1.0 - noise_score) * 0.15
        )
        
        return SkillResult(
            skill_name="frequency",
            score=anomaly_score,
            confidence=0.7,
            findings=findings,
            raw_data={
                "gan_score": gan_score,
                "checkerboard_score": checker_score,
                "hf_analysis": hf_result,
                "noise_consistency": noise_score
            }
        )
    
    def _detect_gan_fingerprint(self, faces: List[np.ndarray]) -> tuple:
        """
        Detect GAN-specific spectral patterns.
        
        Returns:
            Tuple of (score, details)
        """
        spectral_features = []
        
        for face in faces:
            # Convert to grayscale
            if len(face.shape) == 3:
                gray = np.mean(face, axis=2)
            else:
                gray = face
            
            # 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log(1 + np.abs(f_shift))
            
            # Compute azimuthal average (radial power spectrum)
            azimuthal = self._azimuthal_average(magnitude)
            spectral_features.append(azimuthal)
            
            # Detect periodic peaks (GAN artifact)
            peaks = self._find_spectral_peaks(magnitude)
        
        if not spectral_features:
            return 0.0, {}
        
        # Average spectral features
        mean_spectrum = np.mean(spectral_features, axis=0)
        
        # Analyze for GAN patterns
        # GANs often show: 1) periodic peaks, 2) unusual rolloff
        
        # Check for periodic patterns
        periodicity = self._compute_periodicity(mean_spectrum)
        
        # Check spectral slope (natural images have specific decay)
        slope_anomaly = self._check_spectral_slope(mean_spectrum)
        
        gan_score = (periodicity * 0.6 + slope_anomaly * 0.4)
        
        return gan_score, {
            "periodicity": periodicity,
            "slope_anomaly": slope_anomaly,
            "mean_spectrum_sample": mean_spectrum[:50].tolist() if len(mean_spectrum) > 50 else mean_spectrum.tolist()
        }
    
    def _azimuthal_average(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute radially averaged power spectrum."""
        h, w = spectrum.shape
        center = (h // 2, w // 2)
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        max_r = min(center[0], center[1])
        radial_mean = ndimage.mean(spectrum, r, index=np.arange(0, max_r))
        
        return radial_mean
    
    def _find_spectral_peaks(self, magnitude: np.ndarray) -> List[tuple]:
        """Find significant peaks in spectrum."""
        # Simple peak detection
        threshold = np.mean(magnitude) + 2 * np.std(magnitude)
        peaks = np.where(magnitude > threshold)
        
        return list(zip(peaks[0], peaks[1]))
    
    def _compute_periodicity(self, spectrum: np.ndarray) -> float:
        """Detect periodic patterns in radial spectrum."""
        if len(spectrum) < 10:
            return 0.0
        
        # Remove DC component
        spectrum = spectrum[1:]
        
        # Compute autocorrelation
        autocorr = np.correlate(spectrum, spectrum, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation (indicates periodicity)
        if len(autocorr) < 5:
            return 0.0
        
        # Look for secondary peaks
        secondary_peaks = []
        for i in range(2, len(autocorr) - 2):
            if (autocorr[i] > autocorr[i-1] and 
                autocorr[i] > autocorr[i+1] and
                autocorr[i] > 0.3):
                secondary_peaks.append(autocorr[i])
        
        if secondary_peaks:
            return min(np.max(secondary_peaks), 1.0)
        return 0.0
    
    def _check_spectral_slope(self, spectrum: np.ndarray) -> float:
        """
        Check if spectral decay matches natural images.
        
        Natural images typically have 1/f^β spectrum where β ≈ 2.
        GANs may deviate from this.
        """
        if len(spectrum) < 10:
            return 0.0
        
        # Log-log fit
        freqs = np.arange(1, len(spectrum) + 1)
        log_freqs = np.log(freqs)
        log_spectrum = np.log(spectrum + 1e-10)
        
        # Linear regression for slope
        try:
            slope, intercept = np.polyfit(log_freqs, log_spectrum, 1)
        except:
            return 0.0
        
        # Natural images: slope ≈ -2
        # Deviation from this indicates anomaly
        expected_slope = -2.0
        deviation = abs(slope - expected_slope)
        
        # Normalize to 0-1 score
        anomaly = min(deviation / 2.0, 1.0)
        
        return anomaly
    
    def _detect_checkerboard(self, faces: List[np.ndarray]) -> float:
        """
        Detect checkerboard patterns from transposed convolutions.
        """
        checker_scores = []
        
        for face in faces:
            if len(face.shape) == 3:
                gray = np.mean(face, axis=2)
            else:
                gray = face.astype(float)
            
            # High-pass filter
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
            high_pass = ndimage.convolve(gray, kernel)
            
            # FFT of high-pass result
            fft = np.fft.fft2(high_pass)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Check corners (checkerboard = high energy at Nyquist)
            h, w = magnitude.shape
            corner_size = h // 8
            
            corners = [
                magnitude[:corner_size, :corner_size],
                magnitude[:corner_size, -corner_size:],
                magnitude[-corner_size:, :corner_size],
                magnitude[-corner_size:, -corner_size:]
            ]
            
            corner_energy = sum(np.sum(c) for c in corners)
            center_h, center_w = h // 4, w // 4
            center = magnitude[center_h:-center_h, center_w:-center_w]
            center_energy = np.sum(center)
            
            if center_energy > 0:
                ratio = corner_energy / center_energy
                checker_scores.append(min(ratio / 0.5, 1.0))
            else:
                checker_scores.append(0.0)
        
        if checker_scores:
            return np.mean(checker_scores)
        return 0.0
    
    def _analyze_high_frequency(self, faces: List[np.ndarray]) -> Dict:
        """Analyze high-frequency content characteristics."""
        hf_energies = []
        total_energies = []
        
        for face in faces:
            if len(face.shape) == 3:
                gray = np.mean(face, axis=2)
            else:
                gray = face
            
            # FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            h, w = magnitude.shape
            center = (h // 2, w // 2)
            
            # Create frequency masks
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            
            # High frequency = outer 30% of spectrum
            high_mask = r >= min(h, w) * 0.35
            
            hf_energy = np.sum(magnitude * high_mask)
            total_energy = np.sum(magnitude)
            
            hf_energies.append(hf_energy)
            total_energies.append(total_energy)
        
        if not total_energies or sum(total_energies) == 0:
            return {"hf_ratio": 0.0, "anomaly_score": 0.0}
        
        mean_hf_ratio = sum(hf_energies) / sum(total_energies)
        
        # Natural face images typically have hf_ratio around 0.1-0.3
        # Too low = over-smoothed (common in deepfakes)
        # Too high = artifacts or noise
        
        if mean_hf_ratio < 0.05:
            anomaly_score = 0.6  # Over-smoothed
        elif mean_hf_ratio > 0.4:
            anomaly_score = 0.5  # Unusual high-frequency
        else:
            anomaly_score = 0.0
        
        return {
            "hf_ratio": mean_hf_ratio,
            "anomaly_score": anomaly_score
        }
    
    def _analyze_noise_consistency(self, face_crops: List[np.ndarray],
                                   full_frames: List[np.ndarray]) -> float:
        """
        Compare noise patterns between face and background.
        """
        consistencies = []
        
        for face, frame in zip(face_crops, full_frames):
            if frame.size == 0 or face.size == 0:
                continue
            
            # Extract background region (avoid face area)
            h, w = frame.shape[:2]
            
            # Simple background: top and bottom strips
            bg_top = frame[:h//4, :]
            bg_bottom = frame[3*h//4:, :]
            
            if bg_top.size == 0 or bg_bottom.size == 0:
                continue
            
            background = np.vstack([bg_top, bg_bottom])
            
            # Extract noise via high-pass filtering
            face_noise = self._extract_noise(face)
            bg_noise = self._extract_noise(background)
            
            # Compare noise statistics
            face_noise_std = np.std(face_noise)
            bg_noise_std = np.std(bg_noise)
            
            if max(face_noise_std, bg_noise_std) > 0:
                ratio = min(face_noise_std, bg_noise_std) / max(face_noise_std, bg_noise_std)
                consistencies.append(ratio)
        
        if consistencies:
            return np.mean(consistencies)
        return 1.0
    
    def _extract_noise(self, image: np.ndarray) -> np.ndarray:
        """Extract noise component from image."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        # Simple noise extraction: high-pass filter
        # In practice, use more sophisticated denoising
        blurred = ndimage.gaussian_filter(gray, sigma=2)
        noise = gray - blurred
        
        return noise

