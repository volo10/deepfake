"""
Visual Artifact Analyzer

Detects visual artifacts that indicate deepfake manipulation:
- Texture over-smoothing
- Facial asymmetry anomalies
- Face-background color inconsistencies
- Edge density abnormalities
- Temporal stability anomalies
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..models import Finding, Severity, SkillResult

logger = logging.getLogger(__name__)


class VisualArtifactAnalyzer:
    """
    Analyze visual artifacts characteristic of deepfakes.
    
    Detects:
    - Over-smoothed skin texture (common in GAN-generated faces)
    - Facial symmetry anomalies (unnatural asymmetry)
    - Face-neck/background color mismatches
    - Low edge density (lack of natural detail)
    - Unnaturally stable frames (too little natural variation)
    """
    
    def __init__(self):
        # Thresholds based on empirical analysis across multiple deepfake types
        # Real videos typically fall within a "natural" range
        # Deepfakes can fail EITHER direction: too smooth OR too sharp/processed
        
        # Texture variance: Real avg ~83, range 22-153
        # FAKE1: 15 (too smooth), FAKE2: 297 (too sharp)
        self.texture_variance_low = 20.0   # Below = over-smoothed
        self.texture_variance_high = 200.0  # Above = over-processed
        
        # Edge density: Real avg ~0.05, range 0.03-0.07
        # FAKE1: 0.01 (too low), FAKE2: 0.11 (too high)
        self.edge_density_low = 0.025   # Below = lack of detail
        self.edge_density_high = 0.09   # Above = over-sharpened
        
        # Symmetry: Real avg ~0.88
        self.symmetry_threshold = 0.82  # Below = asymmetry issue  
        
        # Face-neck color diff: Real avg ~31, FAKE1: 61, FAKE2: 116
        self.color_diff_threshold = 50.0  # Above = color mismatch
        
        # Boundary sharpness: Real avg ~4.4, FAKE2: 8.7
        self.boundary_sharpness_threshold = 7.0  # Above = artificial boundary
        
        # Local contrast: Real avg ~5.9, FAKE2: 8.95
        self.local_contrast_high = 7.5  # Above = over-processed
        
        # Saturation variance: Real avg ~740, FAKE1: 2900, FAKE2: 1738
        self.saturation_variance_high = 1400  # Above = color manipulation
        
        self.frame_stability_threshold = 2.0  # Below = too stable
        
    def analyze(self, face_sequence: List[np.ndarray],
                full_frames: Optional[List[np.ndarray]] = None,
                face_boxes: Optional[List[Tuple[int, int, int, int]]] = None
               ) -> SkillResult:
        """
        Analyze visual artifacts in face sequence.
        
        Args:
            face_sequence: Aligned face crops
            full_frames: Optional full video frames
            face_boxes: Optional face bounding boxes (x, y, w, h)
        """
        if not CV2_AVAILABLE:
            return SkillResult(
                skill_name="visual_artifacts",
                score=0.5,
                confidence=0.1,
                findings=[],
                raw_data={"error": "OpenCV not available"}
            )
        
        findings = []
        scores = {}
        
        if len(face_sequence) < 5:
            return SkillResult(
                skill_name="visual_artifacts",
                score=0.5,
                confidence=0.1,
                findings=[],
                raw_data={}
            )
        
        # 1. Texture analysis (detect both over-smooth AND over-sharp)
        texture_result = self._analyze_texture(face_sequence)
        scores['texture'] = texture_result['anomaly_score']
        if texture_result['anomaly_score'] > 0.3:
            issue_type = texture_result.get('issue_type', 'abnormal')
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Texture anomaly: {issue_type} (variance: {texture_result['mean_variance']:.1f})",
                severity=Severity.HIGH if texture_result['anomaly_score'] > 0.6 else Severity.MEDIUM,
                confidence=texture_result['anomaly_score'],
                evidence=texture_result
            ))
        
        # 2. Facial symmetry analysis
        symmetry_result = self._analyze_facial_symmetry(face_sequence)
        scores['symmetry'] = symmetry_result['anomaly_score']
        if symmetry_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Facial symmetry anomaly (symmetry: {symmetry_result['mean_symmetry']:.2f})",
                severity=Severity.HIGH if symmetry_result['anomaly_score'] > 0.5 else Severity.MEDIUM,
                confidence=symmetry_result['anomaly_score'],
                evidence=symmetry_result
            ))
        
        # 3. Edge density analysis (detect both too low AND too high)
        edge_result = self._analyze_edge_density(face_sequence)
        scores['edge_density'] = edge_result['anomaly_score']
        if edge_result['anomaly_score'] > 0.3:
            issue_type = edge_result.get('issue_type', 'abnormal')
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Edge density anomaly: {issue_type} ({edge_result['mean_density']:.4f})",
                severity=Severity.MEDIUM,
                confidence=edge_result['anomaly_score'],
                evidence=edge_result
            ))
        
        # 3b. Boundary sharpness analysis (new)
        boundary_result = self._analyze_boundary_sharpness(face_sequence, full_frames, face_boxes)
        scores['boundary'] = boundary_result['anomaly_score']
        if boundary_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Unnatural face boundary sharpness ({boundary_result['mean_sharpness']:.2f})",
                severity=Severity.HIGH if boundary_result['anomaly_score'] > 0.6 else Severity.MEDIUM,
                confidence=boundary_result['anomaly_score'],
                evidence=boundary_result
            ))
        
        # 3c. Local contrast analysis (new)
        contrast_result = self._analyze_local_contrast(face_sequence)
        scores['local_contrast'] = contrast_result['anomaly_score']
        if contrast_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Abnormal local contrast ({contrast_result['mean_contrast']:.2f})",
                severity=Severity.MEDIUM,
                confidence=contrast_result['anomaly_score'],
                evidence=contrast_result
            ))
        
        # 3d. Saturation variance analysis (new)
        saturation_result = self._analyze_saturation(face_sequence)
        scores['saturation'] = saturation_result['anomaly_score']
        if saturation_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Abnormal color saturation variance ({saturation_result['mean_variance']:.1f})",
                severity=Severity.MEDIUM,
                confidence=saturation_result['anomaly_score'],
                evidence=saturation_result
            ))
        
        # 4. Face-neck color consistency (if full frames available)
        if full_frames and face_boxes and len(face_boxes) > 0:
            # Filter out None boxes
            valid_pairs = [(f, b) for f, b in zip(full_frames, face_boxes) if b is not None]
            if valid_pairs:
                valid_frames, valid_boxes = zip(*valid_pairs)
                color_result = self._analyze_color_consistency(list(valid_frames), list(valid_boxes))
                scores['color_consistency'] = color_result['anomaly_score']
                if color_result['anomaly_score'] > 0.2:
                    findings.append(Finding(
                        category="visual_artifacts",
                        description=f"Face-neck color mismatch ({color_result['mean_diff']:.1f})",
                        severity=Severity.HIGH if color_result['anomaly_score'] > 0.5 else Severity.MEDIUM,
                        confidence=color_result['anomaly_score'],
                        evidence=color_result
                    ))
        
        # 5. Temporal stability (too stable = unnatural)
        stability_result = self._analyze_temporal_stability(face_sequence)
        scores['temporal_stability'] = stability_result['anomaly_score']
        if stability_result['anomaly_score'] > 0.3:
            findings.append(Finding(
                category="visual_artifacts",
                description=f"Unnaturally stable frames (variation: {stability_result['mean_diff']:.2f})",
                severity=Severity.LOW,
                confidence=stability_result['anomaly_score'],
                evidence=stability_result
            ))
        
        # 6. Background motion analysis (static background while face moves)
        if full_frames and face_boxes and len(full_frames) > 10:
            bg_result = self._analyze_background_motion(full_frames, face_boxes)
            scores['background_motion'] = bg_result['anomaly_score']
            if bg_result['anomaly_score'] > 0.3:
                findings.append(Finding(
                    category="visual_artifacts",
                    description=f"Static/frozen background detected ({bg_result['bg_motion']:.2f} vs face {bg_result['face_motion']:.2f})",
                    severity=Severity.HIGH if bg_result['anomaly_score'] > 0.6 else Severity.MEDIUM,
                    confidence=bg_result['anomaly_score'],
                    evidence=bg_result
                ))
        
        # Compute overall score with weights
        weights = {
            'texture': 1.5,           # Texture anomalies (both smooth and sharp)
            'symmetry': 1.8,          # Facial symmetry
            'edge_density': 1.4,      # Edge anomalies (both low and high)
            'boundary': 1.8,          # Face boundary sharpness (important!)
            'local_contrast': 1.3,    # Over-processing indicator
            'saturation': 1.2,        # Color manipulation indicator
            'color_consistency': 1.5, # Face-neck color mismatch
            'background_motion': 1.7, # Static background detection
            'temporal_stability': 0.6 # Lower weight - less reliable
        }
        
        weighted_sum = sum(scores.get(k, 0) * weights[k] for k in weights if k in scores)
        total_weight = sum(weights[k] for k in weights if k in scores)
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Boost score if multiple indicators are triggered (synergy effect)
        # This helps distinguish deepfakes which typically fail multiple checks
        triggered_count = sum(1 for s in scores.values() if s > 0.2)
        if triggered_count >= 4:
            overall_score = min(overall_score * 1.5, 1.0)
        elif triggered_count >= 3:
            overall_score = min(overall_score * 1.3, 1.0)
        elif triggered_count >= 2:
            overall_score = min(overall_score * 1.15, 1.0)
        
        return SkillResult(
            skill_name="visual_artifacts",
            score=overall_score,
            confidence=min(len(face_sequence) / 50, 1.0),
            findings=findings,
            raw_data={
                'scores': scores,
                'texture_result': texture_result,
                'symmetry_result': symmetry_result,
                'edge_result': edge_result
            }
        )
    
    def _analyze_texture(self, faces: List[np.ndarray]) -> Dict:
        """
        Detect texture anomalies - both over-smoothed AND over-sharpened.
        
        Real faces have natural texture from pores, wrinkles, etc.
        Deepfakes can be unnaturally smooth (GAN artifacts) OR over-processed/sharpened.
        """
        variances = []
        
        for face in faces:
            if face.size == 0:
                continue
                
            # Convert to grayscale
            if len(face.shape) == 3:
                gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            else:
                gray = face
            
            # Focus on skin regions (center of face, avoiding eyes/mouth)
            h, w = gray.shape
            skin_region = gray[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
            
            if skin_region.size == 0:
                continue
            
            # Laplacian variance measures texture detail
            laplacian = cv2.Laplacian(skin_region, cv2.CV_64F)
            variance = np.var(laplacian)
            variances.append(variance)
        
        if not variances:
            return {'anomaly_score': 0.0, 'mean_variance': 0.0, 'issue_type': 'none'}
        
        mean_variance = np.mean(variances)
        
        # Detect BOTH extremes
        anomaly_score = 0.0
        issue_type = 'normal'
        
        if mean_variance < self.texture_variance_low:
            # Too smooth (over-smoothed deepfake)
            ratio = mean_variance / self.texture_variance_low
            anomaly_score = min((1.0 - ratio) * 1.2, 1.0)
            issue_type = 'over-smoothed'
        elif mean_variance > self.texture_variance_high:
            # Too sharp (over-processed deepfake)
            excess = (mean_variance - self.texture_variance_high) / self.texture_variance_high
            anomaly_score = min(excess * 1.5, 1.0)
            issue_type = 'over-sharpened'
        
        return {
            'anomaly_score': anomaly_score,
            'mean_variance': mean_variance,
            'std_variance': np.std(variances),
            'min_variance': np.min(variances),
            'issue_type': issue_type
        }
    
    def _analyze_facial_symmetry(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze facial symmetry anomalies.
        
        Real faces have high but not perfect symmetry.
        Some deepfakes have unusual asymmetry patterns.
        """
        symmetry_scores = []
        
        for face in faces:
            if face.size == 0:
                continue
            
            if len(face.shape) == 3:
                gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            else:
                gray = face
            
            h, w = gray.shape
            
            # Split face into left and right halves
            mid = w // 2
            left_half = gray[:, :mid]
            right_half = gray[:, mid:mid + left_half.shape[1]]
            
            if left_half.shape[1] == 0 or right_half.shape[1] == 0:
                continue
            
            # Flip right half for comparison
            right_flipped = cv2.flip(right_half, 1)
            
            # Ensure same size
            min_w = min(left_half.shape[1], right_flipped.shape[1])
            if min_w < 10:
                continue
            
            left_half = left_half[:, :min_w]
            right_flipped = right_flipped[:, :min_w]
            
            # Compute symmetry as 1 - normalized difference
            diff = np.abs(left_half.astype(float) - right_flipped.astype(float))
            symmetry = 1.0 - np.mean(diff) / 255.0
            symmetry_scores.append(symmetry)
        
        if not symmetry_scores:
            return {'anomaly_score': 0.0, 'mean_symmetry': 1.0}
        
        mean_symmetry = np.mean(symmetry_scores)
        
        # Low symmetry is suspicious
        # Normal range: 0.85-0.95
        # Suspicious: < 0.82
        if mean_symmetry < self.symmetry_threshold:
            # Scale: the further below threshold, the higher the score
            gap = self.symmetry_threshold - mean_symmetry
            anomaly_score = min(gap / 0.12, 1.0)  # Gap of 0.12 = score 1.0
        else:
            anomaly_score = 0.0
        
        return {
            'anomaly_score': anomaly_score,
            'mean_symmetry': mean_symmetry,
            'std_symmetry': np.std(symmetry_scores),
            'min_symmetry': np.min(symmetry_scores)
        }
    
    def _analyze_edge_density(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze edge density in faces - detect both too low AND too high.
        
        Real faces have natural edges from features, wrinkles, etc.
        Deepfakes may have unnaturally low edge density (smoothed) OR 
        unnaturally high edge density (over-sharpened).
        """
        densities = []
        
        for face in faces:
            if face.size == 0:
                continue
            
            if len(face.shape) == 3:
                gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            else:
                gray = face
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Edge density = proportion of edge pixels
            density = np.mean(edges) / 255.0
            densities.append(density)
        
        if not densities:
            return {'anomaly_score': 0.0, 'mean_density': 0.0, 'issue_type': 'none'}
        
        mean_density = np.mean(densities)
        
        # Detect BOTH extremes
        anomaly_score = 0.0
        issue_type = 'normal'
        
        if mean_density < self.edge_density_low:
            # Too few edges (over-smoothed)
            gap = self.edge_density_low - mean_density
            anomaly_score = min(gap / 0.015, 1.0)
            issue_type = 'too few edges (smoothed)'
        elif mean_density > self.edge_density_high:
            # Too many edges (over-sharpened)
            excess = mean_density - self.edge_density_high
            anomaly_score = min(excess / 0.03, 1.0)
            issue_type = 'too many edges (sharpened)'
        
        return {
            'anomaly_score': anomaly_score,
            'mean_density': mean_density,
            'std_density': np.std(densities),
            'issue_type': issue_type
        }
    
    def _analyze_color_consistency(self, frames: List[np.ndarray],
                                   boxes: List[Tuple[int, int, int, int]]) -> Dict:
        """
        Check color consistency between face and neck/background.
        
        Deepfakes often have color mismatches at boundaries.
        """
        color_diffs = []
        
        for frame, box in zip(frames, boxes):
            if frame.size == 0 or box is None:
                continue
            
            x, y, w, h = box
            
            # Get face region
            face_region = frame[y:y+h, x:x+w]
            
            # Get neck region (below face)
            neck_y_start = y + h
            neck_y_end = min(neck_y_start + h//3, frame.shape[0])
            neck_x_start = max(0, x + w//4)
            neck_x_end = min(x + 3*w//4, frame.shape[1])
            
            if neck_y_end <= neck_y_start or neck_x_end <= neck_x_start:
                continue
            
            neck_region = frame[neck_y_start:neck_y_end, neck_x_start:neck_x_end]
            
            if face_region.size == 0 or neck_region.size == 0:
                continue
            
            # Compare mean colors
            face_color = np.mean(face_region, axis=(0, 1))
            neck_color = np.mean(neck_region, axis=(0, 1))
            
            color_diff = np.linalg.norm(face_color - neck_color)
            color_diffs.append(color_diff)
        
        if not color_diffs:
            return {'anomaly_score': 0.0, 'mean_diff': 0.0}
        
        mean_diff = np.mean(color_diffs)
        
        # High color difference is suspicious
        # Real videos typically have diff of 30-45, fake can be >55
        if mean_diff > self.color_diff_threshold:
            gap = mean_diff - self.color_diff_threshold
            anomaly_score = min(gap / 25.0, 1.0)  # Gap of 25 = score 1.0
        else:
            anomaly_score = 0.0
        
        return {
            'anomaly_score': anomaly_score,
            'mean_diff': mean_diff,
            'max_diff': np.max(color_diffs) if color_diffs else 0.0
        }
    
    def _analyze_boundary_sharpness(self, faces: List[np.ndarray],
                                      frames: Optional[List[np.ndarray]],
                                      boxes: Optional[List]) -> Dict:
        """
        Analyze face boundary sharpness.
        
        Deepfakes often have unnaturally sharp or artificial boundaries
        where the face meets the background.
        """
        if not frames or not boxes:
            return {'anomaly_score': 0.0, 'mean_sharpness': 0.0}
        
        sharpness_values = []
        
        for frame, box in zip(frames, boxes):
            if frame is None or box is None or frame.size == 0:
                continue
            
            x, y, w, h = box
            
            # Create face mask
            face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            face_mask[y:y+h, x:x+w] = 255
            
            # Get boundary region
            boundary = cv2.Canny(face_mask, 100, 200)
            dilated = cv2.dilate(boundary, np.ones((5, 5), np.uint8))
            
            if np.sum(dilated) == 0:
                continue
            
            # Compute Laplacian on the frame
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            laplacian = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)
            
            # Get sharpness at boundary
            boundary_sharpness = np.mean(np.abs(laplacian[dilated > 0]))
            sharpness_values.append(boundary_sharpness)
        
        if not sharpness_values:
            return {'anomaly_score': 0.0, 'mean_sharpness': 0.0}
        
        mean_sharpness = np.mean(sharpness_values)
        
        # High boundary sharpness is suspicious (artificial boundary)
        # Real videos: ~2.6-5.8, threshold: 7.0
        # Extreme values like 30+ are very suspicious
        if mean_sharpness > self.boundary_sharpness_threshold:
            excess = mean_sharpness - self.boundary_sharpness_threshold
            # More aggressive scaling for high values
            anomaly_score = min(excess / 5.0, 1.0)
            # Extra boost for extremely high values
            if mean_sharpness > 15.0:
                anomaly_score = min(anomaly_score + 0.3, 1.0)
            if mean_sharpness > 25.0:
                anomaly_score = min(anomaly_score + 0.2, 1.0)
        else:
            anomaly_score = 0.0
        
        return {
            'anomaly_score': anomaly_score,
            'mean_sharpness': mean_sharpness,
            'max_sharpness': np.max(sharpness_values) if sharpness_values else 0.0
        }
    
    def _analyze_local_contrast(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze local contrast in faces.
        
        Over-processed deepfakes often have unnaturally high local contrast.
        """
        contrast_values = []
        
        for face in faces:
            if face.size == 0:
                continue
            
            if len(face.shape) == 3:
                gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            else:
                gray = face
            
            # Compute local standard deviation
            from scipy import ndimage
            local_std = ndimage.generic_filter(gray.astype(float), np.std, size=5)
            local_contrast = np.mean(local_std)
            contrast_values.append(local_contrast)
        
        if not contrast_values:
            return {'anomaly_score': 0.0, 'mean_contrast': 0.0}
        
        mean_contrast = np.mean(contrast_values)
        
        # High local contrast is suspicious
        if mean_contrast > self.local_contrast_high:
            excess = mean_contrast - self.local_contrast_high
            anomaly_score = min(excess / 2.5, 1.0)
        else:
            anomaly_score = 0.0
        
        return {
            'anomaly_score': anomaly_score,
            'mean_contrast': mean_contrast,
            'std_contrast': np.std(contrast_values)
        }
    
    def _analyze_saturation(self, faces: List[np.ndarray]) -> Dict:
        """
        Analyze color saturation variance.
        
        Deepfakes often have abnormal saturation patterns due to
        color manipulation during face swapping.
        """
        sat_variances = []
        
        for face in faces:
            if face.size == 0 or len(face.shape) != 3:
                continue
            
            # Convert to HSV
            hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            
            sat_variance = np.var(saturation)
            sat_variances.append(sat_variance)
        
        if not sat_variances:
            return {'anomaly_score': 0.0, 'mean_variance': 0.0}
        
        mean_variance = np.mean(sat_variances)
        
        # High saturation variance is suspicious
        if mean_variance > self.saturation_variance_high:
            excess = mean_variance - self.saturation_variance_high
            anomaly_score = min(excess / 1000, 1.0)
        else:
            anomaly_score = 0.0
        
        return {
            'anomaly_score': anomaly_score,
            'mean_variance': mean_variance,
            'max_variance': np.max(sat_variances) if sat_variances else 0.0
        }
    
    def _analyze_background_motion(self, frames: List[np.ndarray],
                                     boxes: List) -> Dict:
        """
        Analyze background motion consistency.
        
        In real videos, background should have natural motion relative to face.
        Deepfakes often have static/frozen backgrounds while the face moves,
        or inconsistent motion between face and background.
        """
        if len(frames) < 10:
            return {'anomaly_score': 0.0, 'bg_motion': 0.0, 'face_motion': 0.0}
        
        face_motions = []
        bg_motions = []
        
        prev_frame = None
        prev_box = None
        
        for i, (frame, box) in enumerate(zip(frames, boxes)):
            if frame is None or box is None or prev_frame is None or prev_box is None:
                prev_frame = frame
                prev_box = box
                continue
            
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
                prev_gray = prev_frame
            
            x, y, w, h = box
            px, py, pw, ph = prev_box
            
            # Face region motion
            face_curr = gray[y:y+h, x:x+w]
            face_prev = prev_gray[py:py+ph, px:px+pw]
            
            if face_curr.size > 0 and face_prev.size > 0:
                # Resize to same size for comparison
                target_h = min(face_curr.shape[0], face_prev.shape[0])
                target_w = min(face_curr.shape[1], face_prev.shape[1])
                if target_h > 10 and target_w > 10:
                    face_curr_resized = cv2.resize(face_curr, (target_w, target_h))
                    face_prev_resized = cv2.resize(face_prev, (target_w, target_h))
                    face_diff = np.mean(np.abs(face_curr_resized.astype(float) - face_prev_resized.astype(float)))
                    face_motions.append(face_diff)
            
            # Background region motion (exclude face area)
            # Create mask for non-face regions
            mask = np.ones(gray.shape, dtype=bool)
            mask[y:y+h, x:x+w] = False
            
            # Also exclude face region from previous frame
            prev_mask = np.ones(prev_gray.shape, dtype=bool)
            prev_mask[py:py+ph, px:px+pw] = False
            
            # Compute background motion
            combined_mask = mask & prev_mask
            if np.sum(combined_mask) > 100:
                bg_diff = np.mean(np.abs(gray[combined_mask].astype(float) - prev_gray[combined_mask].astype(float)))
                bg_motions.append(bg_diff)
            
            prev_frame = frame
            prev_box = box
        
        if not face_motions or not bg_motions:
            return {'anomaly_score': 0.0, 'bg_motion': 0.0, 'face_motion': 0.0}
        
        mean_face_motion = np.mean(face_motions)
        mean_bg_motion = np.mean(bg_motions)
        
        # Anomaly: face moves but background is static
        # Real videos: bg_motion and face_motion should be correlated
        # Fake videos: face moves, background frozen OR inconsistent
        
        anomaly_score = 0.0
        
        # If face has significant motion but background is very static
        if mean_face_motion > 3.0 and mean_bg_motion < 1.0:
            # Very suspicious - face moving, background frozen
            anomaly_score = min((mean_face_motion - mean_bg_motion) / 10.0, 1.0)
        elif mean_face_motion > 2.0 and mean_bg_motion < mean_face_motion * 0.2:
            # Background motion is much less than face motion
            ratio = mean_bg_motion / (mean_face_motion + 1e-8)
            anomaly_score = min((1.0 - ratio) * 0.7, 0.8)
        
        return {
            'anomaly_score': anomaly_score,
            'face_motion': mean_face_motion,
            'bg_motion': mean_bg_motion,
            'motion_ratio': mean_bg_motion / (mean_face_motion + 1e-8)
        }
    
    def _analyze_temporal_stability(self, faces: List[np.ndarray]) -> Dict:
        """
        Check if frames are unnaturally stable.
        
        Real video has natural micro-movements and variations.
        Some deepfakes are too stable frame-to-frame.
        """
        if len(faces) < 2:
            return {'anomaly_score': 0.0, 'mean_diff': 0.0}
        
        frame_diffs = []
        
        for i in range(1, len(faces)):
            if faces[i].size == 0 or faces[i-1].size == 0:
                continue
            
            prev = faces[i-1]
            curr = faces[i]
            
            # Ensure same size
            if prev.shape != curr.shape:
                continue
            
            if len(prev.shape) == 3:
                prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
            else:
                prev_gray = prev
                curr_gray = curr
            
            diff = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
            frame_diffs.append(diff)
        
        if not frame_diffs:
            return {'anomaly_score': 0.0, 'mean_diff': 0.0}
        
        mean_diff = np.mean(frame_diffs)
        
        # Very low frame difference is suspicious (too stable)
        if mean_diff < self.frame_stability_threshold:
            anomaly_score = min((self.frame_stability_threshold - mean_diff) / 1.5, 0.5)
        else:
            anomaly_score = 0.0
        
        return {
            'anomaly_score': anomaly_score,
            'mean_diff': mean_diff,
            'std_diff': np.std(frame_diffs)
        }

