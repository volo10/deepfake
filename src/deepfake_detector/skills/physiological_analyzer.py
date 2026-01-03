"""
Physiological Signal Analyzer

Extracts and analyzes physiological signals (rPPG) from facial video
to detect the absence or inconsistency of biological signals.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

from ..models import Finding, Severity, SkillResult

logger = logging.getLogger(__name__)


class PhysiologicalAnalyzer:
    """
    Extract and analyze remote photoplethysmography (rPPG) signals.
    
    Real faces exhibit subtle color changes synchronized with heartbeat.
    Deepfakes typically lack or have inconsistent physiological signals.
    """
    
    def __init__(self):
        self.min_hr = 40   # Minimum expected heart rate (BPM)
        self.max_hr = 180  # Maximum expected heart rate (BPM)
        
    def analyze(self, face_sequence: List[np.ndarray],
                fps: float,
                forehead_regions: Optional[List[np.ndarray]] = None,
                cheek_regions: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
               ) -> SkillResult:
        """
        Analyze physiological signals in face video.
        
        Args:
            face_sequence: Aligned face crops over time
            fps: Video frame rate
            forehead_regions: Optional forehead ROIs
            cheek_regions: Optional (left_cheek, right_cheek) ROI pairs
            
        Returns:
            SkillResult with physiological analysis
        """
        findings = []
        
        # Need sufficient frames for rPPG analysis
        min_frames = int(fps * 5)  # At least 5 seconds
        if len(face_sequence) < min_frames:
            return SkillResult(
                skill_name="physiological",
                score=0.5,  # Uncertain
                confidence=0.2,
                findings=[Finding(
                    category="physiological",
                    description="Insufficient video length for rPPG analysis",
                    severity=Severity.LOW,
                    confidence=1.0
                )],
                raw_data={"reason": "insufficient_frames"}
            )
        
        # Extract rPPG signal using CHROM method
        rppg_signal = self._extract_rppg_chrom(face_sequence, fps)
        
        # Estimate heart rate
        hr, hr_confidence = self._estimate_heart_rate(rppg_signal, fps)
        
        # Assess signal quality
        signal_present, signal_quality = self._assess_signal_quality(rppg_signal, fps)
        
        # Analyze spatial coherence if regions available
        spatial_coherence = 1.0
        if forehead_regions or cheek_regions:
            spatial_coherence = self._analyze_spatial_coherence(
                face_sequence, forehead_regions, cheek_regions, fps
            )
        
        # Analyze left-right symmetry
        lr_correlation = self._analyze_bilateral_symmetry(face_sequence, fps)
        
        # Generate findings
        anomaly_score = 0.0
        
        if not signal_present:
            findings.append(Finding(
                category="physiological",
                description="No detectable rPPG signal",
                severity=Severity.HIGH,
                confidence=0.8
            ))
            anomaly_score += 0.4
        
        if signal_present and (hr < self.min_hr or hr > self.max_hr):
            findings.append(Finding(
                category="physiological",
                description=f"Unrealistic heart rate estimate: {hr:.0f} BPM",
                severity=Severity.MEDIUM,
                confidence=hr_confidence
            ))
            anomaly_score += 0.2
        
        if spatial_coherence < 0.5:
            findings.append(Finding(
                category="physiological",
                description=f"Spatial incoherence in physiological signals ({spatial_coherence:.2f})",
                severity=Severity.HIGH,
                confidence=0.7
            ))
            anomaly_score += 0.3
        
        if lr_correlation < 0.6:
            findings.append(Finding(
                category="physiological",
                description=f"Bilateral asymmetry in physiological signals ({lr_correlation:.2f})",
                severity=Severity.MEDIUM,
                confidence=0.6
            ))
            anomaly_score += 0.25
        
        if signal_quality < 0.3:
            findings.append(Finding(
                category="physiological",
                description="Low quality physiological signal",
                severity=Severity.LOW,
                confidence=0.5
            ))
            anomaly_score += 0.15
        
        anomaly_score = min(anomaly_score, 1.0)
        
        # Overall confidence based on video quality
        confidence = min(signal_quality + 0.3, 0.9)
        
        return SkillResult(
            skill_name="physiological",
            score=anomaly_score,
            confidence=confidence,
            findings=findings,
            raw_data={
                "heart_rate": hr,
                "hr_confidence": hr_confidence,
                "signal_present": signal_present,
                "signal_quality": signal_quality,
                "spatial_coherence": spatial_coherence,
                "lr_correlation": lr_correlation,
                "rppg_signal": rppg_signal.tolist() if len(rppg_signal) < 1000 else rppg_signal[:1000].tolist()
            }
        )
    
    def _extract_rppg_chrom(self, face_sequence: List[np.ndarray], 
                           fps: float) -> np.ndarray:
        """
        Extract rPPG signal using CHROM method.
        
        De Haan & Jeanne (2013) - Chrominance-based method.
        """
        # Extract mean RGB values from skin regions
        rgb_signals = []
        
        for face in face_sequence:
            # Use central face region (avoid eyes, hair)
            h, w = face.shape[:2]
            roi = face[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
            
            if roi.size == 0:
                rgb_signals.append([0, 0, 0])
                continue
            
            # Mean RGB
            mean_rgb = np.mean(roi, axis=(0, 1))
            rgb_signals.append(mean_rgb)
        
        rgb_signals = np.array(rgb_signals)
        
        if len(rgb_signals) == 0 or np.all(rgb_signals == 0):
            return np.zeros(len(face_sequence))
        
        # Normalize RGB signals
        rgb_norm = rgb_signals / (np.mean(rgb_signals, axis=0) + 1e-8)
        
        # CHROM projection
        Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
        Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
        
        # Bandpass filter (0.7-4 Hz for 42-240 BPM)
        Xf = self._bandpass_filter(Xs, 0.7, 4.0, fps)
        Yf = self._bandpass_filter(Ys, 0.7, 4.0, fps)
        
        # Combine signals
        std_x = np.std(Xf)
        std_y = np.std(Yf)
        
        if std_y > 1e-8:
            alpha = std_x / std_y
        else:
            alpha = 1.0
        
        rppg = Xf - alpha * Yf
        
        return rppg
    
    def _bandpass_filter(self, sig: np.ndarray, low: float, high: float, 
                        fs: float) -> np.ndarray:
        """Apply bandpass filter to signal."""
        if len(sig) < 10:
            return sig
        
        nyq = fs / 2
        low_norm = low / nyq
        high_norm = high / nyq
        
        # Ensure valid frequency range
        low_norm = max(0.01, min(low_norm, 0.99))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
        
        try:
            b, a = signal.butter(2, [low_norm, high_norm], btype='band')
            filtered = signal.filtfilt(b, a, sig)
        except ValueError:
            filtered = sig
        
        return filtered
    
    def _estimate_heart_rate(self, rppg_signal: np.ndarray, 
                            fps: float) -> Tuple[float, float]:
        """
        Estimate heart rate from rPPG signal using FFT.
        
        Returns:
            Tuple of (heart_rate_bpm, confidence)
        """
        if len(rppg_signal) < 30:
            return 0.0, 0.0
        
        # FFT
        fft_result = rfft(rppg_signal)
        freqs = rfftfreq(len(rppg_signal), 1/fps)
        magnitude = np.abs(fft_result)
        
        # Find peak in valid HR range (40-180 BPM = 0.67-3.0 Hz)
        valid_mask = (freqs >= 0.67) & (freqs <= 3.0)
        
        if not np.any(valid_mask):
            return 0.0, 0.0
        
        valid_magnitude = magnitude.copy()
        valid_magnitude[~valid_mask] = 0
        
        peak_idx = np.argmax(valid_magnitude)
        peak_freq = freqs[peak_idx]
        
        heart_rate = peak_freq * 60  # Convert to BPM
        
        # Confidence based on peak prominence
        peak_value = magnitude[peak_idx]
        mean_value = np.mean(magnitude[valid_mask])
        
        if mean_value > 0:
            snr = peak_value / mean_value
            confidence = min(snr / 5.0, 1.0)
        else:
            confidence = 0.0
        
        return heart_rate, confidence
    
    def _assess_signal_quality(self, rppg_signal: np.ndarray, 
                               fps: float) -> Tuple[bool, float]:
        """
        Assess quality of the extracted rPPG signal.
        
        Returns:
            Tuple of (signal_present, quality_score)
        """
        if len(rppg_signal) < 30:
            return False, 0.0
        
        # Check signal variance
        signal_std = np.std(rppg_signal)
        if signal_std < 1e-6:
            return False, 0.0
        
        # SNR estimation
        hr, hr_conf = self._estimate_heart_rate(rppg_signal, fps)
        
        # Spectral quality - look for clear peak
        fft_result = rfft(rppg_signal)
        freqs = rfftfreq(len(rppg_signal), 1/fps)
        magnitude = np.abs(fft_result)
        
        valid_mask = (freqs >= 0.67) & (freqs <= 3.0)
        valid_magnitude = magnitude[valid_mask]
        
        if len(valid_magnitude) == 0:
            return False, 0.0
        
        peak_idx = np.argmax(valid_magnitude)
        peak_value = valid_magnitude[peak_idx]
        
        # Compute spectral flatness (closer to 0 = clear peak)
        geometric_mean = np.exp(np.mean(np.log(valid_magnitude + 1e-10)))
        arithmetic_mean = np.mean(valid_magnitude)
        
        if arithmetic_mean > 0:
            spectral_flatness = geometric_mean / arithmetic_mean
        else:
            spectral_flatness = 1.0
        
        # Quality score
        quality = 1.0 - spectral_flatness
        quality = quality * hr_conf
        
        # Signal is present if quality is above threshold
        signal_present = quality > 0.2
        
        return signal_present, quality
    
    def _analyze_spatial_coherence(self, 
                                   face_sequence: List[np.ndarray],
                                   forehead_regions: Optional[List[np.ndarray]],
                                   cheek_regions: Optional[List[Tuple[np.ndarray, np.ndarray]]],
                                   fps: float) -> float:
        """
        Analyze spatial coherence of rPPG signals across face regions.
        """
        signals = {}
        
        # Extract from forehead
        if forehead_regions and len(forehead_regions) > 30:
            forehead_rgb = [np.mean(r, axis=(0, 1)) if r.size > 0 else [0, 0, 0]
                          for r in forehead_regions]
            signals['forehead'] = self._rgb_to_rppg(np.array(forehead_rgb), fps)
        
        # Extract from cheeks
        if cheek_regions and len(cheek_regions) > 30:
            left_rgb = [np.mean(lr, axis=(0, 1)) if lr.size > 0 else [0, 0, 0]
                       for lr, _ in cheek_regions]
            right_rgb = [np.mean(rr, axis=(0, 1)) if rr.size > 0 else [0, 0, 0]
                        for _, rr in cheek_regions]
            signals['left_cheek'] = self._rgb_to_rppg(np.array(left_rgb), fps)
            signals['right_cheek'] = self._rgb_to_rppg(np.array(right_rgb), fps)
        
        # Extract from full face
        face_rgb = []
        for face in face_sequence:
            h, w = face.shape[:2]
            roi = face[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
            if roi.size > 0:
                face_rgb.append(np.mean(roi, axis=(0, 1)))
            else:
                face_rgb.append([0, 0, 0])
        signals['face'] = self._rgb_to_rppg(np.array(face_rgb), fps)
        
        # Compute pairwise correlations
        if len(signals) < 2:
            return 1.0  # Can't assess coherence
        
        correlations = []
        signal_names = list(signals.keys())
        
        for i in range(len(signal_names)):
            for j in range(i + 1, len(signal_names)):
                s1 = signals[signal_names[i]]
                s2 = signals[signal_names[j]]
                
                if len(s1) == len(s2) and len(s1) > 10:
                    try:
                        corr = np.corrcoef(s1, s2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                    except:
                        pass
        
        if correlations:
            return np.mean(correlations)
        return 1.0
    
    def _analyze_bilateral_symmetry(self, face_sequence: List[np.ndarray],
                                    fps: float) -> float:
        """
        Analyze left-right symmetry of physiological signals.
        """
        left_signals = []
        right_signals = []
        
        for face in face_sequence:
            h, w = face.shape[:2]
            
            # Left half (excluding center)
            left_roi = face[int(h*0.2):int(h*0.8), :int(w*0.45)]
            # Right half
            right_roi = face[int(h*0.2):int(h*0.8), int(w*0.55):]
            
            if left_roi.size > 0 and right_roi.size > 0:
                left_signals.append(np.mean(left_roi, axis=(0, 1)))
                right_signals.append(np.mean(right_roi, axis=(0, 1)))
        
        if len(left_signals) < 30:
            return 1.0  # Can't assess
        
        left_rppg = self._rgb_to_rppg(np.array(left_signals), fps)
        right_rppg = self._rgb_to_rppg(np.array(right_signals), fps)
        
        try:
            correlation = np.corrcoef(left_rppg, right_rppg)[0, 1]
            if np.isnan(correlation):
                return 1.0
            return max(0, correlation)
        except:
            return 1.0
    
    def _rgb_to_rppg(self, rgb_signals: np.ndarray, fps: float) -> np.ndarray:
        """Convert RGB signal to rPPG using CHROM."""
        if len(rgb_signals) == 0:
            return np.array([])
        
        # Normalize
        mean_rgb = np.mean(rgb_signals, axis=0)
        if np.any(mean_rgb == 0):
            return np.zeros(len(rgb_signals))
        
        rgb_norm = rgb_signals / mean_rgb
        
        # CHROM projection
        Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
        Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
        
        Xf = self._bandpass_filter(Xs, 0.7, 4.0, fps)
        Yf = self._bandpass_filter(Ys, 0.7, 4.0, fps)
        
        std_y = np.std(Yf)
        alpha = np.std(Xf) / std_y if std_y > 1e-8 else 1.0
        
        return Xf - alpha * Yf

