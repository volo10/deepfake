"""
Signal Analysis Module.

Analyzes rPPG signals for heart rate estimation and quality assessment.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.fft import rfft, rfftfreq

from .rppg import RPPGExtractor


class SignalAnalyzer:
    """
    Analyze physiological signal quality and characteristics.
    """

    def __init__(self, min_hr: float = 40, max_hr: float = 180):
        """
        Initialize signal analyzer.

        Args:
            min_hr: Minimum expected heart rate (BPM)
            max_hr: Maximum expected heart rate (BPM)
        """
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.rppg_extractor = RPPGExtractor()

    def estimate_heart_rate(
        self,
        rppg_signal: np.ndarray,
        fps: float
    ) -> Tuple[float, float]:
        """
        Estimate heart rate from rPPG signal using FFT.

        Args:
            rppg_signal: rPPG signal array
            fps: Video frame rate

        Returns:
            Tuple of (heart_rate_bpm, confidence)
        """
        if len(rppg_signal) < 30:
            return 0.0, 0.0

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
        heart_rate = peak_freq * 60

        # Confidence based on peak prominence
        peak_value = magnitude[peak_idx]
        mean_value = np.mean(magnitude[valid_mask])

        if mean_value > 0:
            snr = peak_value / mean_value
            confidence = min(snr / 5.0, 1.0)
        else:
            confidence = 0.0

        return heart_rate, confidence

    def assess_signal_quality(
        self,
        rppg_signal: np.ndarray,
        fps: float
    ) -> Tuple[bool, float]:
        """
        Assess quality of the extracted rPPG signal.

        Args:
            rppg_signal: rPPG signal array
            fps: Video frame rate

        Returns:
            Tuple of (signal_present, quality_score)
        """
        if len(rppg_signal) < 30:
            return False, 0.0

        if np.std(rppg_signal) < 1e-6:
            return False, 0.0

        hr, hr_conf = self.estimate_heart_rate(rppg_signal, fps)

        # Spectral quality analysis
        fft_result = rfft(rppg_signal)
        freqs = rfftfreq(len(rppg_signal), 1/fps)
        magnitude = np.abs(fft_result)

        valid_mask = (freqs >= 0.67) & (freqs <= 3.0)
        valid_magnitude = magnitude[valid_mask]

        if len(valid_magnitude) == 0:
            return False, 0.0

        # Spectral flatness (closer to 0 = clear peak)
        geometric_mean = np.exp(np.mean(np.log(valid_magnitude + 1e-10)))
        arithmetic_mean = np.mean(valid_magnitude)

        spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 1.0
        quality = (1.0 - spectral_flatness) * hr_conf
        signal_present = quality > 0.2

        return signal_present, quality

    def analyze_spatial_coherence(
        self,
        face_sequence: List[np.ndarray],
        forehead_regions: Optional[List[np.ndarray]],
        cheek_regions: Optional[List[Tuple[np.ndarray, np.ndarray]]],
        fps: float
    ) -> float:
        """
        Analyze spatial coherence of rPPG signals across face regions.

        Args:
            face_sequence: List of face images
            forehead_regions: Optional forehead ROIs
            cheek_regions: Optional (left, right) cheek ROI pairs
            fps: Video frame rate

        Returns:
            Coherence score (0-1)
        """
        signals = {}

        # Extract from forehead
        if forehead_regions and len(forehead_regions) > 30:
            forehead_rgb = [
                np.mean(r, axis=(0, 1)) if r.size > 0 else [0, 0, 0]
                for r in forehead_regions
            ]
            signals['forehead'] = self.rppg_extractor.rgb_to_rppg(
                np.array(forehead_rgb), fps
            )

        # Extract from cheeks
        if cheek_regions and len(cheek_regions) > 30:
            left_rgb = [
                np.mean(lr, axis=(0, 1)) if lr.size > 0 else [0, 0, 0]
                for lr, _ in cheek_regions
            ]
            right_rgb = [
                np.mean(rr, axis=(0, 1)) if rr.size > 0 else [0, 0, 0]
                for _, rr in cheek_regions
            ]
            signals['left_cheek'] = self.rppg_extractor.rgb_to_rppg(
                np.array(left_rgb), fps
            )
            signals['right_cheek'] = self.rppg_extractor.rgb_to_rppg(
                np.array(right_rgb), fps
            )

        # Extract from full face
        face_rgb = []
        for face in face_sequence:
            h, w = face.shape[:2]
            roi = face[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
            face_rgb.append(np.mean(roi, axis=(0, 1)) if roi.size > 0 else [0, 0, 0])
        signals['face'] = self.rppg_extractor.rgb_to_rppg(np.array(face_rgb), fps)

        # Compute pairwise correlations
        if len(signals) < 2:
            return 1.0

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
                    except Exception:
                        pass

        return np.mean(correlations) if correlations else 1.0

    def analyze_bilateral_symmetry(
        self,
        face_sequence: List[np.ndarray],
        fps: float
    ) -> float:
        """
        Analyze left-right symmetry of physiological signals.

        Args:
            face_sequence: List of face images
            fps: Video frame rate

        Returns:
            Correlation score between left and right sides
        """
        left_signals = []
        right_signals = []

        for face in face_sequence:
            h, w = face.shape[:2]
            left_roi = face[int(h*0.2):int(h*0.8), :int(w*0.45)]
            right_roi = face[int(h*0.2):int(h*0.8), int(w*0.55):]

            if left_roi.size > 0 and right_roi.size > 0:
                left_signals.append(np.mean(left_roi, axis=(0, 1)))
                right_signals.append(np.mean(right_roi, axis=(0, 1)))

        if len(left_signals) < 30:
            return 1.0

        left_rppg = self.rppg_extractor.rgb_to_rppg(np.array(left_signals), fps)
        right_rppg = self.rppg_extractor.rgb_to_rppg(np.array(right_signals), fps)

        try:
            correlation = np.corrcoef(left_rppg, right_rppg)[0, 1]
            return max(0, correlation) if not np.isnan(correlation) else 1.0
        except Exception:
            return 1.0
