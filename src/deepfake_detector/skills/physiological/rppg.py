"""
rPPG Signal Extraction Module.

Implements CHROM method for remote photoplethysmography extraction.
"""

from typing import List
import numpy as np
from scipy import signal


class RPPGExtractor:
    """
    Extract rPPG signals from facial video.

    Uses the CHROM method (De Haan & Jeanne, 2013) for
    robust chrominance-based signal extraction.
    """

    def extract_chrom(
        self,
        face_sequence: List[np.ndarray],
        fps: float
    ) -> np.ndarray:
        """
        Extract rPPG signal using CHROM method.

        Args:
            face_sequence: List of aligned face crops
            fps: Video frame rate

        Returns:
            rPPG signal array
        """
        # Extract mean RGB values from skin regions
        rgb_signals = []

        for face in face_sequence:
            h, w = face.shape[:2]
            roi = face[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

            if roi.size == 0:
                rgb_signals.append([0, 0, 0])
                continue

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
        Xf = self.bandpass_filter(Xs, 0.7, 4.0, fps)
        Yf = self.bandpass_filter(Ys, 0.7, 4.0, fps)

        # Combine signals
        std_x = np.std(Xf)
        std_y = np.std(Yf)

        alpha = std_x / std_y if std_y > 1e-8 else 1.0
        rppg = Xf - alpha * Yf

        return rppg

    def bandpass_filter(
        self,
        sig: np.ndarray,
        low: float,
        high: float,
        fs: float
    ) -> np.ndarray:
        """
        Apply bandpass filter to signal.

        Args:
            sig: Input signal
            low: Low cutoff frequency (Hz)
            high: High cutoff frequency (Hz)
            fs: Sampling frequency

        Returns:
            Filtered signal
        """
        if len(sig) < 10:
            return sig

        nyq = fs / 2
        low_norm = max(0.01, min(low / nyq, 0.99))
        high_norm = max(low_norm + 0.01, min(high / nyq, 0.99))

        try:
            b, a = signal.butter(2, [low_norm, high_norm], btype='band')
            filtered = signal.filtfilt(b, a, sig)
        except ValueError:
            filtered = sig

        return filtered

    def rgb_to_rppg(self, rgb_signals: np.ndarray, fps: float) -> np.ndarray:
        """
        Convert RGB signal to rPPG using CHROM.

        Args:
            rgb_signals: RGB signal array (N x 3)
            fps: Video frame rate

        Returns:
            rPPG signal
        """
        if len(rgb_signals) == 0:
            return np.array([])

        mean_rgb = np.mean(rgb_signals, axis=0)
        if np.any(mean_rgb == 0):
            return np.zeros(len(rgb_signals))

        rgb_norm = rgb_signals / mean_rgb

        # CHROM projection
        Xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
        Ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]

        Xf = self.bandpass_filter(Xs, 0.7, 4.0, fps)
        Yf = self.bandpass_filter(Ys, 0.7, 4.0, fps)

        std_y = np.std(Yf)
        alpha = np.std(Xf) / std_y if std_y > 1e-8 else 1.0

        return Xf - alpha * Yf
