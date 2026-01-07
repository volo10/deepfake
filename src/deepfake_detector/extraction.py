"""
Data Extraction Module.

Provides utilities for extracting data from video processing results.
"""

from typing import List, Optional
import numpy as np
import cv2

from .models import VideoMetadata


def load_video_metadata(video_path: str) -> VideoMetadata:
    """
    Extract video metadata from file.

    Args:
        video_path: Path to the video file

    Returns:
        VideoMetadata with duration, fps, dimensions, etc.
    """
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Check for audio (basic check)
    has_audio = True  # Assume true, will be verified during extraction

    cap.release()

    return VideoMetadata(
        path=video_path,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        has_audio=has_audio
    )


def extract_face_sequence(face_results: List) -> List[np.ndarray]:
    """
    Extract aligned face crops from tracking results.

    Args:
        face_results: List of face tracking results per frame

    Returns:
        List of aligned face images
    """
    faces = []
    for frame_result in face_results:
        if frame_result.faces:
            # Use the primary face (highest confidence or largest)
            primary_face = max(frame_result.faces, key=lambda f: f.confidence)
            if primary_face.aligned_face is not None:
                faces.append(primary_face.aligned_face)
    return faces


def extract_landmarks_sequence(face_results: List) -> List[np.ndarray]:
    """
    Extract landmark sequences from tracking results.

    Args:
        face_results: List of face tracking results per frame

    Returns:
        List of landmark arrays
    """
    landmarks = []
    for frame_result in face_results:
        if frame_result.faces:
            primary_face = max(frame_result.faces, key=lambda f: f.confidence)
            landmarks.append(primary_face.landmarks)
    return landmarks


def extract_mouth_regions(face_results: List) -> List[np.ndarray]:
    """
    Extract mouth region crops from tracking results.

    Args:
        face_results: List of face tracking results per frame

    Returns:
        List of mouth region images
    """
    mouths = []
    for frame_result in face_results:
        if frame_result.faces:
            primary_face = max(frame_result.faces, key=lambda f: f.confidence)
            if "mouth" in primary_face.regions:
                mouths.append(primary_face.regions["mouth"])
    return mouths


def extract_face_boxes(face_results: List) -> List[Optional[tuple]]:
    """
    Extract face bounding boxes from tracking results.

    Args:
        face_results: List of face tracking results per frame

    Returns:
        List of (x, y, w, h) tuples or None for frames without faces
    """
    boxes = []
    for frame_result in face_results:
        if frame_result.faces:
            primary_face = max(frame_result.faces, key=lambda f: f.confidence)
            bbox = primary_face.bbox
            boxes.append((bbox.x, bbox.y, bbox.width, bbox.height))
        else:
            boxes.append(None)
    return boxes
