"""
Base utilities for visual artifact analysis.

Provides common imports, type definitions, and helper functions.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

# Type aliases
FaceSequence = List[np.ndarray]
FrameSequence = List[np.ndarray]
BoundingBox = Tuple[int, int, int, int]
BoxList = List[Optional[BoundingBox]]
AnalysisResult = Dict[str, float]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def is_cv2_available() -> bool:
    """Check if OpenCV is available."""
    return CV2_AVAILABLE
