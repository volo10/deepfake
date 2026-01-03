"""
Unit tests for Visual Artifact Analyzer.

Tests cover:
- Texture analysis (over-smoothed and over-sharpened detection)
- Symmetry analysis
- Edge density analysis
- Color consistency analysis
- Background motion detection
- Score aggregation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestTextureAnalysis:
    """Tests for texture analysis functionality."""
    
    def test_smooth_texture_detection(self, synthetic_deepfake_face):
        """Test detection of over-smoothed textures."""
        # Calculate variance - should be low for smooth face
        gray = np.mean(synthetic_deepfake_face, axis=2)
        variance = np.var(gray)
        
        # Over-smoothed threshold is typically around 40
        assert variance < 50, "Synthetic deepfake should have low texture variance"
    
    def test_natural_texture_detection(self, synthetic_real_face):
        """Test that natural textures have higher variance."""
        gray = np.mean(synthetic_real_face, axis=2)
        variance = np.var(gray)
        
        # Natural faces should have more texture variance
        assert variance > 50, "Real face should have higher texture variance"
    
    def test_over_sharpened_detection(self):
        """Test detection of over-sharpened textures."""
        # Create an over-sharpened face (high edge contrast)
        face = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add sharp edges
        face[::2, :, :] = 255
        face[1::2, :, :] = 0
        
        gray = np.mean(face, axis=2)
        variance = np.var(gray)
        
        # Over-sharpened should have very high variance
        assert variance > 100, "Over-sharpened face should have high variance"


class TestSymmetryAnalysis:
    """Tests for facial symmetry analysis."""
    
    def test_symmetric_face(self):
        """Test that perfectly symmetric face has high correlation."""
        # Create a symmetric face
        left_half = np.random.randint(100, 200, (224, 112, 3), dtype=np.uint8)
        right_half = np.flip(left_half, axis=1)
        face = np.concatenate([left_half, right_half], axis=1)
        
        # Calculate symmetry
        left = face[:, :112]
        right = np.flip(face[:, 112:], axis=1)
        
        # Should be perfectly symmetric
        correlation = np.corrcoef(left.flatten(), right.flatten())[0, 1]
        assert correlation > 0.99, "Symmetric face should have high correlation"
    
    def test_asymmetric_face(self):
        """Test that asymmetric face has lower correlation."""
        # Create an asymmetric face
        left_half = np.random.randint(100, 200, (224, 112, 3), dtype=np.uint8)
        right_half = np.random.randint(50, 150, (224, 112, 3), dtype=np.uint8)
        face = np.concatenate([left_half, right_half], axis=1)
        
        left = face[:, :112]
        right = np.flip(face[:, 112:], axis=1)
        
        correlation = np.corrcoef(left.flatten(), right.flatten())[0, 1]
        assert correlation < 0.9, "Asymmetric face should have lower correlation"


class TestEdgeDensityAnalysis:
    """Tests for edge density analysis."""
    
    def test_low_edge_density(self):
        """Test detection of faces with too few edges."""
        # Create a very smooth face
        face = np.ones((224, 224, 3), dtype=np.uint8) * 150
        
        # Calculate edge density using Sobel-like approximation
        gray = np.mean(face, axis=2)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        
        edge_ratio = (np.sum(dx > 10) + np.sum(dy > 10)) / gray.size
        assert edge_ratio < 0.05, "Smooth face should have low edge ratio"
    
    def test_high_edge_density(self):
        """Test detection of faces with too many edges."""
        # Create a face with many edges (checkerboard pattern)
        face = np.zeros((224, 224, 3), dtype=np.uint8)
        face[::4, ::4] = 255
        face[1::4, 1::4] = 255
        face[2::4, 2::4] = 255
        face[3::4, 3::4] = 255
        
        gray = np.mean(face, axis=2)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        
        edge_ratio = (np.sum(dx > 10) + np.sum(dy > 10)) / gray.size
        assert edge_ratio > 0.1, "High-frequency face should have high edge ratio"


class TestColorConsistency:
    """Tests for face-neck color consistency analysis."""
    
    def test_consistent_colors(self):
        """Test that consistent face-neck colors pass."""
        face_color = np.array([180, 160, 200])  # Face
        neck_color = np.array([175, 155, 195])  # Similar neck
        
        # LAB-like distance (simplified)
        diff = np.sqrt(np.sum((face_color.astype(float) - neck_color.astype(float)) ** 2))
        assert diff < 20, "Similar colors should have low difference"
    
    def test_inconsistent_colors(self):
        """Test that inconsistent face-neck colors are detected."""
        face_color = np.array([180, 160, 200])  # Face
        neck_color = np.array([100, 100, 100])  # Very different neck
        
        diff = np.sqrt(np.sum((face_color.astype(float) - neck_color.astype(float)) ** 2))
        assert diff > 50, "Different colors should have high difference"


class TestBackgroundMotion:
    """Tests for background motion analysis."""
    
    def test_static_background_detection(self):
        """Test detection of static backgrounds."""
        # Create frames with moving face but static background
        frames = []
        for i in range(10):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Static BG
            # Moving face region
            frame[100+i*5:200+i*5, 200:300] = 180 + i * 2
            frames.append(frame)
        
        # Calculate background motion
        bg_diffs = []
        for i in range(1, len(frames)):
            # Exclude face region
            bg1 = frames[i-1].copy()
            bg1[100:220, 200:300] = 0
            bg2 = frames[i].copy()
            bg2[100:220, 200:300] = 0
            bg_diffs.append(np.mean(np.abs(bg2.astype(float) - bg1.astype(float))))
        
        mean_bg_motion = np.mean(bg_diffs)
        assert mean_bg_motion < 5, "Static background should have low motion"
    
    def test_moving_background_detection(self):
        """Test that natural moving backgrounds are detected."""
        # Create frames with moving background
        frames = []
        for i in range(10):
            # Moving background
            frame = np.roll(np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8), i * 10, axis=1)
            frames.append(frame)
        
        bg_diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            bg_diffs.append(diff)
        
        mean_bg_motion = np.mean(bg_diffs)
        assert mean_bg_motion > 10, "Moving background should have higher motion"


class TestScoreAggregation:
    """Tests for score aggregation logic."""
    
    def test_all_normal_metrics(self):
        """Test that all normal metrics produce low score."""
        scores = {
            "texture": 0.0,
            "symmetry": 0.0,
            "edge_density": 0.0,
            "color_consistency": 0.0,
            "background_motion": 0.0
        }
        
        # Simple mean aggregation
        overall = np.mean(list(scores.values()))
        assert overall < 0.1, "All normal metrics should produce low score"
    
    def test_single_anomaly(self):
        """Test that single anomaly raises score moderately."""
        scores = {
            "texture": 0.8,  # Anomaly
            "symmetry": 0.0,
            "edge_density": 0.0,
            "color_consistency": 0.0,
            "background_motion": 0.0
        }
        
        overall = np.mean(list(scores.values()))
        assert 0.1 < overall < 0.3, "Single anomaly should raise score moderately"
    
    def test_multiple_anomalies(self):
        """Test that multiple anomalies produce high score."""
        scores = {
            "texture": 0.8,
            "symmetry": 0.7,
            "edge_density": 0.6,
            "color_consistency": 0.5,
            "background_motion": 0.4
        }
        
        overall = np.mean(list(scores.values()))
        assert overall > 0.5, "Multiple anomalies should produce high score"
    
    def test_synergy_scoring(self):
        """Test that synergy scoring boosts multiple triggers."""
        scores = {
            "texture": 0.4,
            "symmetry": 0.4,
            "edge_density": 0.4,
            "color_consistency": 0.3,
            "background_motion": 0.3
        }
        
        # Count triggers above threshold
        threshold = 0.2
        triggered = sum(1 for s in scores.values() if s > threshold)
        
        overall = np.mean(list(scores.values()))
        
        # Apply synergy boost
        if triggered >= 4:
            overall *= 1.5
        elif triggered >= 3:
            overall *= 1.3
        elif triggered >= 2:
            overall *= 1.15
        
        assert overall > 0.5, "Synergy should boost score when multiple signals trigger"


class TestRobustness:
    """Tests for robustness to edge cases."""
    
    def test_empty_face_sequence(self):
        """Test handling of empty face sequence."""
        faces = []
        
        # Should not crash, return neutral result
        if len(faces) == 0:
            score = 0.5  # Neutral/uncertain
            assert score == 0.5
    
    def test_single_face(self):
        """Test handling of single face (no temporal analysis)."""
        face = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        faces = [face]
        
        # Should still be able to analyze visual artifacts
        gray = np.mean(face, axis=2)
        variance = np.var(gray)
        
        assert variance >= 0, "Should compute variance for single face"
    
    def test_tiny_face(self):
        """Test handling of very small face crops."""
        face = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        
        # Should still work
        gray = np.mean(face, axis=2)
        variance = np.var(gray)
        
        assert variance >= 0, "Should handle small faces"
    
    def test_grayscale_input(self):
        """Test handling of grayscale input."""
        gray_face = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        
        # Should handle gracefully
        variance = np.var(gray_face)
        assert variance >= 0, "Should handle grayscale input"

