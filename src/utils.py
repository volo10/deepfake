"""
Utility functions for the deepfake detection agent.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


def load_video(video_path: str, 
               max_frames: int = 300,
               sample_rate: int = 1) -> Tuple[List[np.ndarray], float]:
    """
    Load video frames from file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        sample_rate: Sample every Nth frame
        
    Returns:
        Tuple of (frames list, fps)
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV not available. Install with: pip install opencv-python")
        return [], 0.0
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return [], 0.0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_idx = 0
    
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    
    logger.info(f"Loaded {len(frames)} frames from {video_path} at {fps:.1f} fps")
    
    return frames, fps


def extract_audio(video_path: str, 
                  target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to video file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio waveform, sample rate) or (None, 0) if failed
    """
    try:
        import subprocess
        import tempfile
        import os
    except ImportError:
        return None, 0
    
    # Try using moviepy first
    try:
        from moviepy.editor import VideoFileClip
        
        clip = VideoFileClip(video_path)
        
        if clip.audio is None:
            logger.warning("No audio track found in video")
            clip.close()
            return None, 0
        
        # Extract audio
        audio_array = clip.audio.to_soundarray(fps=target_sr)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        clip.close()
        
        return audio_array.astype(np.float32), target_sr
        
    except ImportError:
        logger.warning("moviepy not available, trying ffmpeg directly")
    
    # Fallback to ffmpeg via subprocess
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(target_sr), '-ac', '1',
            tmp_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"ffmpeg failed: {result.stderr}")
            return None, 0
        
        # Load the wav file
        try:
            import scipy.io.wavfile as wav
            sr, audio = wav.read(tmp_path)
            audio = audio.astype(np.float32) / 32768.0  # Normalize int16
        except ImportError:
            # Try with wave module
            import wave
            with wave.open(tmp_path, 'rb') as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return audio, sr
        
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None, 0


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


def compute_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    return 1.0 - compute_cosine_similarity(a, b)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame to [0, 1] range."""
    return frame.astype(np.float32) / 255.0


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to target size."""
    try:
        import cv2
        return cv2.resize(frame, size)
    except ImportError:
        # Fallback to simple nearest neighbor
        h, w = frame.shape[:2]
        target_w, target_h = size
        
        x_ratio = w / target_w
        y_ratio = h / target_h
        
        result = np.zeros((target_h, target_w, *frame.shape[2:]), dtype=frame.dtype)
        
        for y in range(target_h):
            for x in range(target_w):
                src_x = int(x * x_ratio)
                src_y = int(y * y_ratio)
                result[y, x] = frame[src_y, src_x]
        
        return result


def create_video_writer(output_path: str, fps: float, 
                        frame_size: Tuple[int, int]):
    """Create video writer for output."""
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    except ImportError:
        logger.error("OpenCV not available for video writing")
        return None


class ProgressBar:
    """Simple progress bar for long operations."""
    
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.current = 0
        self.desc = desc
        
    def update(self, n: int = 1):
        self.current += n
        self._display()
    
    def _display(self):
        percent = (self.current / self.total) * 100
        bar_len = 30
        filled = int(bar_len * self.current / self.total)
        bar = '=' * filled + '-' * (bar_len - filled)
        print(f'\r{self.desc}: [{bar}] {percent:.1f}% ({self.current}/{self.total})', end='')
        
        if self.current >= self.total:
            print()  # New line at completion
    
    def close(self):
        if self.current < self.total:
            print()


def setup_logging(level: int = logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_video_info(video_path: str) -> dict:
    """Get video metadata without loading all frames."""
    try:
        import cv2
    except ImportError:
        return {}
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {}
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    if info['fps'] > 0:
        info['duration'] = info['total_frames'] / info['fps']
    else:
        info['duration'] = 0
    
    cap.release()
    
    return info

