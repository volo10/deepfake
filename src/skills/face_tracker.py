"""
Face Tracking Skill

Detects and tracks faces across video frames, providing the foundational
data structure for all downstream analysis skills.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

from ..models import (
    DetectedFace, BoundingBox, HeadPose, FrameAnalysis, 
    Finding, Severity, SkillResult
)

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Face detection and tracking across video frames.
    
    Supports multiple backends:
    - MediaPipe Face Mesh (default, 478 landmarks)
    - dlib (68 landmarks)
    - OpenCV Haar Cascades (fallback)
    """
    
    def __init__(self, backend: str = "opencv", 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize face tracker.
        
        Args:
            backend: Detection backend ("mediapipe", "dlib", "opencv")
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.backend = backend
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self._init_detector()
        self.face_id_counter = 0
        self.tracked_faces: Dict[int, np.ndarray] = {}  # face_id -> last_embedding
        
    def _init_detector(self):
        """Initialize the face detection backend."""
        # Default to OpenCV which is most reliable
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.backend = "opencv"
        logger.info("Using OpenCV Haar Cascade backend")
    
    def track(self, frames: List[np.ndarray]) -> List[FrameAnalysis]:
        """
        Track faces across all frames.
        
        Args:
            frames: List of video frames (RGB)
            
        Returns:
            List of FrameAnalysis objects with detected faces
        """
        results = []
        
        for idx, frame in enumerate(frames):
            frame_result = self._process_frame(frame, idx)
            results.append(frame_result)
            
        # Post-process: assign consistent IDs, smooth landmarks
        results = self._smooth_tracking(results)
        
        return results
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameAnalysis:
        """Process a single frame for face detection."""
        if self.backend == "mediapipe":
            return self._process_mediapipe(frame, frame_idx)
        elif self.backend == "dlib":
            return self._process_dlib(frame, frame_idx)
        else:
            return self._process_opencv(frame, frame_idx)
    
    def _process_mediapipe(self, frame: np.ndarray, frame_idx: int) -> FrameAnalysis:
        """Process frame using MediaPipe."""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.detector.process(rgb_frame)
        
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks as numpy array
                landmarks = np.array([
                    [lm.x * w, lm.y * h, lm.z * w]
                    for lm in face_landmarks.landmark
                ])
                
                # Compute bounding box from landmarks
                x_min = int(np.min(landmarks[:, 0]))
                y_min = int(np.min(landmarks[:, 1]))
                x_max = int(np.max(landmarks[:, 0]))
                y_max = int(np.max(landmarks[:, 1]))
                
                # Add margin
                margin = int(0.1 * max(x_max - x_min, y_max - y_min))
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                bbox = BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)
                
                # Extract aligned face
                aligned = self._align_face(frame, landmarks)
                
                # Estimate head pose
                pose = self._estimate_pose(landmarks, w, h)
                
                # Extract ROIs
                regions = self._extract_regions(frame, landmarks)
                
                face = DetectedFace(
                    face_id=self._get_face_id(landmarks),
                    bbox=bbox,
                    confidence=0.9,  # MediaPipe doesn't provide confidence per face
                    landmarks=landmarks[:, :2],  # Keep only x, y
                    aligned_face=aligned,
                    pose=pose,
                    regions=regions
                )
                faces.append(face)
        
        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp=frame_idx / 30.0,  # Assume 30fps, will be corrected later
            faces=faces
        )
    
    def _process_dlib(self, frame: np.ndarray, frame_idx: int) -> FrameAnalysis:
        """Process frame using dlib."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector(gray)
        
        faces = []
        for det in detections:
            shape = self.predictor(gray, det)
            landmarks = np.array([
                [shape.part(i).x, shape.part(i).y]
                for i in range(68)
            ])
            
            bbox = BoundingBox(det.left(), det.top(), det.width(), det.height())
            aligned = self._align_face(frame, landmarks)
            
            face = DetectedFace(
                face_id=self._get_face_id(landmarks),
                bbox=bbox,
                confidence=0.8,
                landmarks=landmarks,
                aligned_face=aligned
            )
            faces.append(face)
        
        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp=frame_idx / 30.0,
            faces=faces
        )
    
    def _process_opencv(self, frame: np.ndarray, frame_idx: int) -> FrameAnalysis:
        """Process frame using OpenCV Haar Cascades (fallback)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detections:
            bbox = BoundingBox(x, y, w, h)
            face_crop = frame[y:y+h, x:x+w]
            aligned = cv2.resize(face_crop, (224, 224))
            
            face = DetectedFace(
                face_id=self.face_id_counter,
                bbox=bbox,
                confidence=0.7,
                landmarks=np.array([]),  # No landmarks from Haar
                aligned_face=aligned
            )
            self.face_id_counter += 1
            faces.append(face)
        
        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp=frame_idx / 30.0,
            faces=faces
        )
    
    def _align_face(self, frame: np.ndarray, landmarks: np.ndarray, 
                    size: int = 224) -> np.ndarray:
        """
        Align face using eye landmarks.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks
            size: Output size
            
        Returns:
            Aligned face crop
        """
        if len(landmarks) >= 468:  # MediaPipe
            left_eye = landmarks[33][:2]
            right_eye = landmarks[263][:2]
        elif len(landmarks) >= 68:  # dlib
            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
        else:
            # No landmarks, return simple crop
            h, w = frame.shape[:2]
            return cv2.resize(frame, (size, size))
        
        # Calculate angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Calculate center and scale
        eye_center = ((left_eye[0] + right_eye[0]) / 2,
                      (left_eye[1] + right_eye[1]) / 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        
        # Rotate
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        
        # Crop around face
        x, y = int(eye_center[0]), int(eye_center[1])
        crop_size = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 2.5)
        
        x1 = max(0, x - crop_size)
        y1 = max(0, y - crop_size)
        x2 = min(frame.shape[1], x + crop_size)
        y2 = min(frame.shape[0], y + crop_size)
        
        face_crop = rotated[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return cv2.resize(frame, (size, size))
        
        return cv2.resize(face_crop, (size, size))
    
    def _estimate_pose(self, landmarks: np.ndarray, 
                       img_w: int, img_h: int) -> HeadPose:
        """Estimate head pose from 3D landmarks."""
        if landmarks.shape[1] < 3:
            return HeadPose(0, 0, 0)
        
        # Use nose tip and face contour points
        nose = landmarks[1]
        left = landmarks[234]
        right = landmarks[454]
        top = landmarks[10]
        bottom = landmarks[152]
        
        # Simple pose estimation
        yaw = np.arctan2(nose[0] - (left[0] + right[0])/2, 
                         np.linalg.norm(right[:2] - left[:2])) * 180 / np.pi
        pitch = np.arctan2(nose[1] - (top[1] + bottom[1])/2,
                          np.linalg.norm(top[:2] - bottom[:2])) * 180 / np.pi
        roll = np.arctan2(right[1] - left[1], right[0] - left[0]) * 180 / np.pi
        
        return HeadPose(pitch=pitch, yaw=yaw, roll=roll)
    
    def _extract_regions(self, frame: np.ndarray, 
                        landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract face region crops (eyes, mouth, forehead)."""
        regions = {}
        h, w = frame.shape[:2]
        
        if len(landmarks) >= 468:  # MediaPipe
            # Eye regions
            left_eye_pts = landmarks[[33, 133, 160, 144, 153, 154, 155, 157]]
            right_eye_pts = landmarks[[263, 362, 387, 373, 380, 381, 382, 384]]
            
            regions["left_eye"] = self._crop_region(frame, left_eye_pts)
            regions["right_eye"] = self._crop_region(frame, right_eye_pts)
            
            # Mouth region
            mouth_pts = landmarks[list(range(61, 68)) + list(range(78, 96))]
            regions["mouth"] = self._crop_region(frame, mouth_pts)
            
            # Forehead region (above eyebrows)
            forehead_y = int(landmarks[10][1])
            forehead_x = int(landmarks[10][0])
            fw = int(np.linalg.norm(landmarks[234][:2] - landmarks[454][:2]) * 0.6)
            fh = int(fw * 0.4)
            
            fy1 = max(0, forehead_y - fh)
            fy2 = forehead_y
            fx1 = max(0, forehead_x - fw//2)
            fx2 = min(w, forehead_x + fw//2)
            
            if fy2 > fy1 and fx2 > fx1:
                regions["forehead"] = frame[fy1:fy2, fx1:fx2]
        
        return regions
    
    def _crop_region(self, frame: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Crop a region defined by points."""
        x_min = max(0, int(np.min(points[:, 0])) - 5)
        y_min = max(0, int(np.min(points[:, 1])) - 5)
        x_max = min(frame.shape[1], int(np.max(points[:, 0])) + 5)
        y_max = min(frame.shape[0], int(np.max(points[:, 1])) + 5)
        
        return frame[y_min:y_max, x_min:x_max]
    
    def _get_face_id(self, landmarks: np.ndarray) -> int:
        """Get or assign face ID based on landmarks similarity."""
        # Simple approach: assign incrementing IDs
        # More sophisticated: track by embedding similarity
        face_id = self.face_id_counter
        self.face_id_counter += 1
        return face_id
    
    def _smooth_tracking(self, results: List[FrameAnalysis]) -> List[FrameAnalysis]:
        """Apply temporal smoothing to tracking results."""
        # TODO: Implement Kalman filtering or similar
        return results
    
    def extract_embeddings(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract face embeddings for identity analysis.
        
        Args:
            faces: List of aligned face crops
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for face in faces:
            # Simple embedding: flatten normalized face
            # In production, use ArcFace, FaceNet, etc.
            resized = cv2.resize(face, (112, 112))
            normalized = resized.astype(np.float32) / 255.0
            embedding = normalized.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return embeddings
    
    def detect_anomalies(self, results: List[FrameAnalysis]) -> List[Finding]:
        """Detect tracking-level anomalies."""
        findings = []
        
        # Check for landmark jitter
        if len(results) > 1:
            jitter_scores = []
            for i in range(1, len(results)):
                if results[i].faces and results[i-1].faces:
                    curr_lm = results[i].faces[0].landmarks
                    prev_lm = results[i-1].faces[0].landmarks
                    
                    if len(curr_lm) > 0 and len(prev_lm) > 0:
                        velocity = np.linalg.norm(curr_lm - prev_lm, axis=1)
                        jitter_scores.append(np.mean(velocity))
            
            if jitter_scores:
                mean_jitter = np.mean(jitter_scores)
                max_jitter = np.max(jitter_scores)
                
                if max_jitter > 10:  # Threshold
                    findings.append(Finding(
                        category="face_tracking",
                        description=f"High landmark jitter detected (max: {max_jitter:.1f}px)",
                        severity=Severity.MEDIUM,
                        confidence=min(max_jitter / 20, 1.0)
                    ))
        
        return findings

