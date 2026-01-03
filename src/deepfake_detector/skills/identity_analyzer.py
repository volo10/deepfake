"""
Identity Reasoning Analyzer

Performs cross-modal identity verification and consistency checking
to detect identity theft, face swaps, and identity instability.
"""

import logging
from typing import List, Dict, Optional
import numpy as np

from ..models import Finding, Severity, SkillResult

logger = logging.getLogger(__name__)


class IdentityAnalyzer:
    """
    Analyze identity consistency across time and modalities.
    
    Detects:
    - Identity drift over time
    - Sudden identity switches (face swaps)
    - Reference identity mismatch
    - Embedding instability
    """
    
    def __init__(self):
        self.identity_threshold = 0.4  # Cosine distance threshold
        self.switch_threshold = 0.5   # Threshold for identity switch
        
    def analyze(self, embeddings_sequence: List[np.ndarray],
                reference_faces: Optional[List[np.ndarray]] = None,
                reference_embeddings: Optional[List[np.ndarray]] = None
               ) -> SkillResult:
        """
        Analyze identity consistency.
        
        Args:
            embeddings_sequence: Face embeddings over time
            reference_faces: Optional reference face images
            reference_embeddings: Optional pre-computed reference embeddings
            
        Returns:
            SkillResult with identity analysis
        """
        findings = []
        
        if len(embeddings_sequence) < 2:
            return SkillResult(
                skill_name="identity",
                score=0.5,
                confidence=0.1,
                findings=[],
                raw_data={}
            )
        
        embeddings_array = np.array(embeddings_sequence)
        
        # Analyze identity stability
        stability_result = self._analyze_identity_stability(embeddings_array)
        
        if stability_result["stability"] < 0.6:
            findings.append(Finding(
                category="identity",
                description=f"Identity instability detected (stability: {stability_result['stability']:.2f})",
                severity=Severity.MEDIUM,
                confidence=1.0 - stability_result["stability"],
                evidence=stability_result
            ))
        
        if stability_result["drift"] > 0.3:
            findings.append(Finding(
                category="identity",
                description=f"Identity drift detected over video ({stability_result['drift']:.2f})",
                severity=Severity.HIGH if stability_result["drift"] > 0.5 else Severity.MEDIUM,
                confidence=min(stability_result["drift"], 1.0)
            ))
        
        # Detect identity switches
        switches = self._detect_identity_switches(embeddings_array)
        
        if switches:
            findings.append(Finding(
                category="identity",
                description=f"Identity switch(es) detected at frames: {[s['frame'] for s in switches]}",
                severity=Severity.CRITICAL,
                confidence=0.9,
                evidence={"switches": switches}
            ))
        
        # Reference matching
        reference_match = None
        if reference_embeddings or reference_faces:
            if reference_embeddings is None and reference_faces:
                # Would need to compute embeddings from faces
                # For now, skip
                pass
            
            if reference_embeddings:
                reference_match = self._match_against_reference(
                    embeddings_array, reference_embeddings
                )
                
                if not reference_match["is_match"]:
                    findings.append(Finding(
                        category="identity",
                        description=f"Does not match reference identity (distance: {reference_match['distance']:.2f})",
                        severity=Severity.HIGH,
                        confidence=1.0 - reference_match["match_score"]
                    ))
        
        # Compute overall anomaly score
        anomaly_score = 0.0
        
        # Stability contribution
        anomaly_score += (1.0 - stability_result["stability"]) * 0.3
        
        # Drift contribution
        anomaly_score += min(stability_result["drift"] / 0.5, 1.0) * 0.3
        
        # Switches contribution
        if switches:
            anomaly_score += min(len(switches) * 0.2, 0.4)
        
        # Reference match contribution
        if reference_match and not reference_match["is_match"]:
            anomaly_score += 0.3
        
        anomaly_score = min(anomaly_score, 1.0)
        
        # Confidence based on data quality
        confidence = min(len(embeddings_sequence) / 50, 1.0) * 0.8
        
        return SkillResult(
            skill_name="identity",
            score=anomaly_score,
            confidence=confidence,
            findings=findings,
            raw_data={
                "stability_result": stability_result,
                "switches": switches,
                "reference_match": reference_match,
                "num_embeddings": len(embeddings_sequence)
            }
        )
    
    def _analyze_identity_stability(self, embeddings: np.ndarray) -> Dict:
        """Analyze embedding stability over time."""
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Compute distances from centroid
        distances = np.array([self._cosine_distance(e, centroid) for e in embeddings])
        
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        variance = np.var(distances)
        
        # Stability score (inversely related to distance)
        stability = 1.0 - min(mean_distance / 0.5, 1.0)
        
        # Compute drift (change over time windows)
        window_size = max(10, len(embeddings) // 5)
        drift = 0.0
        
        if len(embeddings) >= window_size * 2:
            window_centroids = []
            for i in range(0, len(embeddings) - window_size + 1, window_size // 2):
                window = embeddings[i:i + window_size]
                window_centroids.append(np.mean(window, axis=0))
            
            if len(window_centroids) > 1:
                drift_distances = [
                    self._cosine_distance(window_centroids[i], window_centroids[i+1])
                    for i in range(len(window_centroids) - 1)
                ]
                drift = np.max(drift_distances)
        
        return {
            "stability": stability,
            "drift": drift,
            "mean_distance": mean_distance,
            "max_distance": max_distance,
            "variance": variance
        }
    
    def _detect_identity_switches(self, embeddings: np.ndarray) -> List[Dict]:
        """Detect sudden identity changes."""
        switches = []
        
        for i in range(1, len(embeddings)):
            distance = self._cosine_distance(embeddings[i], embeddings[i-1])
            
            if distance > self.switch_threshold:
                # Also check against overall centroid
                centroid = np.mean(embeddings, axis=0)
                dist_to_centroid = self._cosine_distance(embeddings[i], centroid)
                
                switches.append({
                    "frame": i,
                    "distance": float(distance),
                    "dist_to_centroid": float(dist_to_centroid),
                    "severity": "high" if distance > 0.7 else "medium"
                })
        
        return switches
    
    def _match_against_reference(self, embeddings: np.ndarray,
                                  reference_embeddings: List[np.ndarray]) -> Dict:
        """Match detected identity against reference."""
        # Mean embedding from video
        video_embedding = np.mean(embeddings, axis=0)
        
        # Mean reference embedding
        ref_embedding = np.mean(reference_embeddings, axis=0)
        
        # Compute distance
        distance = self._cosine_distance(video_embedding, ref_embedding)
        
        # Match score (inverse of distance)
        match_score = 1.0 - min(distance / 0.6, 1.0)
        
        # Per-frame matching
        frame_distances = [
            self._cosine_distance(emb, ref_embedding) 
            for emb in embeddings
        ]
        
        return {
            "is_match": distance < self.identity_threshold,
            "distance": float(distance),
            "match_score": match_score,
            "frame_distances": frame_distances[:100] if len(frame_distances) > 100 else frame_distances
        }
    
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        similarity = dot / (norm_a * norm_b)
        return 1.0 - similarity

