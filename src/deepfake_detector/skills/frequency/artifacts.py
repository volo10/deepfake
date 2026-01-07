"""
Artifact Detection Module.

Detects various frequency-domain artifacts in images.
"""

from typing import List, Dict
import numpy as np
from scipy import ndimage


class ArtifactDetector:
    """
    Detect frequency-domain artifacts characteristic of deepfakes.

    Detects:
    - Checkerboard artifacts from upsampling
    - High-frequency anomalies
    - Noise inconsistencies
    """

    def detect_checkerboard(self, faces: List[np.ndarray]) -> float:
        """
        Detect checkerboard patterns from transposed convolutions.

        Args:
            faces: List of face images

        Returns:
            Checkerboard artifact score (0-1)
        """
        checker_scores = []

        for face in faces:
            if len(face.shape) == 3:
                gray = np.mean(face, axis=2)
            else:
                gray = face.astype(float)

            # High-pass filter
            kernel = np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ])
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

        return np.mean(checker_scores) if checker_scores else 0.0

    def analyze_high_frequency(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze high-frequency content characteristics.

        Args:
            faces: List of face images

        Returns:
            Dict with HF analysis results
        """
        hf_energies = []
        total_energies = []

        for face in faces:
            if len(face.shape) == 3:
                gray = np.mean(face, axis=2)
            else:
                gray = face

            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            h, w = magnitude.shape
            center = (h // 2, w // 2)

            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

            high_mask = r >= min(h, w) * 0.35

            hf_energies.append(np.sum(magnitude * high_mask))
            total_energies.append(np.sum(magnitude))

        if not total_energies or sum(total_energies) == 0:
            return {"hf_ratio": 0.0, "anomaly_score": 0.0}

        mean_hf_ratio = sum(hf_energies) / sum(total_energies)

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

    def analyze_noise_consistency(
        self,
        face_crops: List[np.ndarray],
        full_frames: List[np.ndarray]
    ) -> float:
        """
        Compare noise patterns between face and background.

        Args:
            face_crops: List of face images
            full_frames: List of full frame images

        Returns:
            Consistency score (0-1)
        """
        consistencies = []

        for face, frame in zip(face_crops, full_frames):
            if frame.size == 0 or face.size == 0:
                continue

            h, w = frame.shape[:2]

            bg_top = frame[:h//4, :]
            bg_bottom = frame[3*h//4:, :]

            if bg_top.size == 0 or bg_bottom.size == 0:
                continue

            background = np.vstack([bg_top, bg_bottom])

            face_noise = self._extract_noise(face)
            bg_noise = self._extract_noise(background)

            face_noise_std = np.std(face_noise)
            bg_noise_std = np.std(bg_noise)

            if max(face_noise_std, bg_noise_std) > 0:
                ratio = (
                    min(face_noise_std, bg_noise_std) /
                    max(face_noise_std, bg_noise_std)
                )
                consistencies.append(ratio)

        return np.mean(consistencies) if consistencies else 1.0

    def _extract_noise(self, image: np.ndarray) -> np.ndarray:
        """Extract noise component from image."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)

        blurred = ndimage.gaussian_filter(gray, sigma=2)
        return gray - blurred
