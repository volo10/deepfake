"""
GAN Fingerprint Detection Module.

Detects GAN-specific spectral patterns in images.
"""

from typing import List, Dict, Tuple
import numpy as np
from scipy import ndimage


class GANDetector:
    """
    Detect GAN fingerprints in facial images.

    GANs often leave characteristic spectral patterns:
    - Periodic peaks in frequency domain
    - Unusual spectral rolloff
    """

    def detect(self, faces: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Detect GAN-specific spectral patterns.

        Args:
            faces: List of face images

        Returns:
            Tuple of (score, details)
        """
        spectral_features = []

        for face in faces:
            if len(face.shape) == 3:
                gray = np.mean(face, axis=2)
            else:
                gray = face

            # 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log(1 + np.abs(f_shift))

            # Compute azimuthal average
            azimuthal = self._azimuthal_average(magnitude)
            spectral_features.append(azimuthal)

        if not spectral_features:
            return 0.0, {}

        mean_spectrum = np.mean(spectral_features, axis=0)

        # Analyze for GAN patterns
        periodicity = self._compute_periodicity(mean_spectrum)
        slope_anomaly = self._check_spectral_slope(mean_spectrum)

        gan_score = periodicity * 0.6 + slope_anomaly * 0.4

        return gan_score, {
            "periodicity": periodicity,
            "slope_anomaly": slope_anomaly,
            "mean_spectrum_sample": (
                mean_spectrum[:50].tolist()
                if len(mean_spectrum) > 50
                else mean_spectrum.tolist()
            )
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

    def _compute_periodicity(self, spectrum: np.ndarray) -> float:
        """Detect periodic patterns in radial spectrum."""
        if len(spectrum) < 10:
            return 0.0

        spectrum = spectrum[1:]  # Remove DC component

        # Compute autocorrelation
        autocorr = np.correlate(spectrum, spectrum, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        if len(autocorr) < 5:
            return 0.0

        # Find secondary peaks
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

        Natural images have 1/f^β spectrum where β ≈ 2.
        """
        if len(spectrum) < 10:
            return 0.0

        freqs = np.arange(1, len(spectrum) + 1)
        log_freqs = np.log(freqs)
        log_spectrum = np.log(spectrum + 1e-10)

        try:
            slope, _ = np.polyfit(log_freqs, log_spectrum, 1)
        except Exception:
            return 0.0

        expected_slope = -2.0
        deviation = abs(slope - expected_slope)

        return min(deviation / 2.0, 1.0)
