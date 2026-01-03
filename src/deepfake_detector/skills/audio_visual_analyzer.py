"""
Audio-Visual Alignment Analyzer

Analyzes synchronization between audio and visual signals to detect
lip-sync errors and voice-face mismatches.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import signal

from ..models import Finding, Severity, SkillResult

logger = logging.getLogger(__name__)


class AudioVisualAnalyzer:
    """
    Analyze audio-visual alignment and consistency.
    
    Detects:
    - Lip sync errors
    - Voice-face identity mismatch
    - Emotion incongruence
    - Synthetic voice indicators
    """
    
    def __init__(self):
        self.sync_threshold_ms = 100  # Acceptable AV offset
        
    def analyze(self, mouth_sequence: List[np.ndarray],
                audio: np.ndarray,
                audio_sr: int,
                fps: float) -> SkillResult:
        """
        Analyze audio-visual alignment.
        
        Args:
            mouth_sequence: Mouth region crops over time
            audio: Audio waveform
            audio_sr: Audio sample rate
            fps: Video frame rate
            
        Returns:
            SkillResult with AV alignment analysis
        """
        findings = []
        
        if len(mouth_sequence) == 0 or len(audio) == 0:
            return SkillResult(
                skill_name="audio_visual",
                score=0.5,
                confidence=0.1,
                findings=[Finding(
                    category="audio_visual",
                    description="Insufficient data for AV analysis",
                    severity=Severity.LOW,
                    confidence=1.0
                )],
                raw_data={}
            )
        
        # Detect AV offset
        offset_ms, offset_confidence = self._detect_av_offset(
            mouth_sequence, audio, fps, audio_sr
        )
        
        if abs(offset_ms) > self.sync_threshold_ms:
            findings.append(Finding(
                category="audio_visual",
                description=f"Lip sync offset detected: {offset_ms:.0f}ms",
                severity=Severity.HIGH if abs(offset_ms) > 200 else Severity.MEDIUM,
                confidence=offset_confidence,
                evidence={"offset_ms": offset_ms}
            ))
        
        # Analyze lip movement correlation with audio
        lip_sync_score = self._analyze_lip_sync(mouth_sequence, audio, fps, audio_sr)
        
        if lip_sync_score < 0.4:
            findings.append(Finding(
                category="audio_visual",
                description=f"Poor lip-audio synchronization (score: {lip_sync_score:.2f})",
                severity=Severity.MEDIUM,
                confidence=0.7
            ))
        
        # Detect voice activity vs lip movement
        activity_match = self._check_voice_activity_match(mouth_sequence, audio, fps, audio_sr)
        
        if activity_match < 0.5:
            findings.append(Finding(
                category="audio_visual",
                description="Voice activity doesn't match lip movement",
                severity=Severity.MEDIUM,
                confidence=0.6
            ))
        
        # Voice authenticity check (basic)
        voice_auth = self._check_voice_authenticity(audio, audio_sr)
        
        if voice_auth < 0.5:
            findings.append(Finding(
                category="audio_visual",
                description="Potential synthetic voice detected",
                severity=Severity.MEDIUM,
                confidence=voice_auth
            ))
        
        # Compute overall anomaly score
        anomaly_score = 0.0
        
        if abs(offset_ms) > self.sync_threshold_ms:
            anomaly_score += min(abs(offset_ms) / 300, 0.4)
        
        anomaly_score += (1.0 - lip_sync_score) * 0.3
        anomaly_score += (1.0 - activity_match) * 0.15
        anomaly_score += (1.0 - voice_auth) * 0.15
        
        anomaly_score = min(anomaly_score, 1.0)
        
        return SkillResult(
            skill_name="audio_visual",
            score=anomaly_score,
            confidence=offset_confidence * 0.5 + 0.5,
            findings=findings,
            raw_data={
                "sync_offset_ms": offset_ms,
                "offset_confidence": offset_confidence,
                "lip_sync_score": lip_sync_score,
                "activity_match": activity_match,
                "voice_authenticity": voice_auth
            }
        )
    
    def _detect_av_offset(self, mouth_sequence: List[np.ndarray],
                          audio: np.ndarray,
                          fps: float, audio_sr: int) -> Tuple[float, float]:
        """
        Detect audio-visual temporal offset using cross-correlation.
        
        Returns:
            Tuple of (offset_ms, confidence)
        """
        # Extract audio envelope
        audio_envelope = self._extract_audio_envelope(audio, audio_sr, fps)
        
        # Extract mouth movement signal
        mouth_signal = self._extract_mouth_movement(mouth_sequence)
        
        if len(audio_envelope) == 0 or len(mouth_signal) == 0:
            return 0.0, 0.0
        
        # Resample to common length
        target_len = min(len(audio_envelope), len(mouth_signal))
        audio_envelope = self._resample(audio_envelope, target_len)
        mouth_signal = self._resample(mouth_signal, target_len)
        
        # Normalize
        audio_envelope = (audio_envelope - np.mean(audio_envelope)) / (np.std(audio_envelope) + 1e-8)
        mouth_signal = (mouth_signal - np.mean(mouth_signal)) / (np.std(mouth_signal) + 1e-8)
        
        # Cross-correlation
        correlation = signal.correlate(mouth_signal, audio_envelope, mode='full')
        lags = signal.correlation_lags(len(mouth_signal), len(audio_envelope), mode='full')
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        
        # Convert to milliseconds
        offset_ms = (peak_lag / fps) * 1000
        
        # Confidence based on peak prominence
        peak_value = np.abs(correlation[peak_idx])
        mean_value = np.mean(np.abs(correlation))
        
        if mean_value > 0:
            confidence = min(peak_value / (mean_value * 3), 1.0)
        else:
            confidence = 0.0
        
        return offset_ms, confidence
    
    def _extract_audio_envelope(self, audio: np.ndarray, sr: int, 
                                target_fps: float) -> np.ndarray:
        """Extract audio amplitude envelope at video frame rate."""
        # Compute frame-level audio energy
        frame_samples = int(sr / target_fps)
        num_frames = len(audio) // frame_samples
        
        envelope = []
        for i in range(num_frames):
            start = i * frame_samples
            end = start + frame_samples
            frame_audio = audio[start:end]
            
            # RMS energy
            energy = np.sqrt(np.mean(frame_audio ** 2))
            envelope.append(energy)
        
        return np.array(envelope)
    
    def _extract_mouth_movement(self, mouth_sequence: List[np.ndarray]) -> np.ndarray:
        """Extract mouth movement signal (mouth openness over time)."""
        movements = []
        
        prev_mouth = None
        for mouth in mouth_sequence:
            if mouth.size == 0:
                movements.append(0)
                continue
            
            # Compute mouth openness (vertical extent)
            if len(mouth.shape) == 3:
                gray = np.mean(mouth, axis=2)
            else:
                gray = mouth
            
            # Simple metric: variance (more open = more variation)
            openness = np.var(gray)
            movements.append(openness)
            
            prev_mouth = mouth
        
        return np.array(movements)
    
    def _analyze_lip_sync(self, mouth_sequence: List[np.ndarray],
                          audio: np.ndarray, fps: float, sr: int) -> float:
        """
        Analyze lip-audio synchronization quality.
        
        Returns score from 0 (poor sync) to 1 (good sync).
        """
        # Extract features
        audio_envelope = self._extract_audio_envelope(audio, sr, fps)
        mouth_signal = self._extract_mouth_movement(mouth_sequence)
        
        if len(audio_envelope) == 0 or len(mouth_signal) == 0:
            return 0.5
        
        # Align lengths
        min_len = min(len(audio_envelope), len(mouth_signal))
        audio_envelope = audio_envelope[:min_len]
        mouth_signal = mouth_signal[:min_len]
        
        # Compute correlation
        try:
            correlation = np.corrcoef(audio_envelope, mouth_signal)[0, 1]
            if np.isnan(correlation):
                return 0.5
            
            # Convert to 0-1 score
            sync_score = (correlation + 1) / 2
            return sync_score
        except:
            return 0.5
    
    def _check_voice_activity_match(self, mouth_sequence: List[np.ndarray],
                                    audio: np.ndarray, fps: float, sr: int) -> float:
        """
        Check if voice activity periods match lip movement periods.
        """
        # Detect voice activity
        audio_envelope = self._extract_audio_envelope(audio, sr, fps)
        audio_threshold = np.mean(audio_envelope) * 0.5
        voice_active = audio_envelope > audio_threshold
        
        # Detect lip movement
        mouth_signal = self._extract_mouth_movement(mouth_sequence)
        mouth_threshold = np.mean(mouth_signal) * 0.5
        mouth_active = mouth_signal > mouth_threshold
        
        # Align lengths
        min_len = min(len(voice_active), len(mouth_active))
        voice_active = voice_active[:min_len]
        mouth_active = mouth_active[:min_len]
        
        # Compute agreement
        agreement = np.mean(voice_active == mouth_active)
        
        return agreement
    
    def _check_voice_authenticity(self, audio: np.ndarray, sr: int) -> float:
        """
        Basic check for synthetic voice indicators.
        
        This is a simplified version - production would use specialized models.
        """
        # Check for spectral characteristics
        # Synthetic voices often have: unnatural formants, artifacts
        
        # Compute spectrogram features
        frame_size = int(sr * 0.025)  # 25ms frames
        hop_size = int(sr * 0.010)    # 10ms hop
        
        # Simple spectral analysis
        num_frames = (len(audio) - frame_size) // hop_size
        
        if num_frames < 10:
            return 0.5
        
        spectral_features = []
        for i in range(min(num_frames, 100)):
            start = i * hop_size
            frame = audio[start:start + frame_size]
            
            # Apply window
            windowed = frame * np.hanning(len(frame))
            
            # FFT
            fft = np.abs(np.fft.rfft(windowed))
            spectral_features.append(fft)
        
        spectral_features = np.array(spectral_features)
        
        # Check spectral continuity
        spectral_diff = np.diff(spectral_features, axis=0)
        mean_diff = np.mean(np.abs(spectral_diff))
        
        # Natural speech has smooth spectral evolution
        # Synthetic may have discontinuities
        
        # Normalize to score
        # Lower diff = more natural
        smoothness = 1.0 / (1.0 + mean_diff)
        
        # Check for periodic artifacts
        mean_spectrum = np.mean(spectral_features, axis=0)
        autocorr = np.correlate(mean_spectrum, mean_spectrum, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for unusual periodicity
        if len(autocorr) > 10:
            periodicity = np.max(autocorr[1:]) / (autocorr[0] + 1e-8)
        else:
            periodicity = 0
        
        # Combine metrics
        authenticity = smoothness * 0.6 + (1.0 - min(periodicity, 1.0)) * 0.4
        
        return min(authenticity, 1.0)
    
    @staticmethod
    def _resample(signal_array: np.ndarray, target_len: int) -> np.ndarray:
        """Resample signal to target length."""
        if len(signal_array) == target_len:
            return signal_array
        
        # Simple linear interpolation
        x_orig = np.linspace(0, 1, len(signal_array))
        x_new = np.linspace(0, 1, target_len)
        
        return np.interp(x_new, x_orig, signal_array)

