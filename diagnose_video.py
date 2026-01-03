#!/usr/bin/env python3
"""
Deep diagnostic analysis of a video to understand deepfake signals.
"""

import sys
import os
import cv2
import numpy as np
from scipy import ndimage
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def analyze_video_deeply(video_path):
    """Perform deep diagnostic analysis."""
    print(f"\n{'='*60}")
    print(f"DEEP DIAGNOSTIC: {os.path.basename(video_path)}")
    print('='*60)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    
    # Face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Collect metrics
    metrics = defaultdict(list)
    prev_frame = None
    prev_face = None
    
    frame_idx = 0
    sample_rate = 2
    
    while cap.isOpened() and frame_idx < 300:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # 1. Face boundary sharpness (blending artifacts)
            face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            face_mask[y:y+h, x:x+w] = 255
            boundary = cv2.Canny(face_mask, 100, 200)
            dilated = cv2.dilate(boundary, np.ones((5,5), np.uint8))
            boundary_region = cv2.bitwise_and(gray, gray, mask=dilated)
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            boundary_sharpness = np.mean(np.abs(laplacian[dilated > 0])) if np.sum(dilated) > 0 else 0
            metrics['boundary_sharpness'].append(boundary_sharpness)
            
            # 2. Texture analysis - skin smoothness
            face_laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
            texture_variance = np.var(face_laplacian)
            metrics['texture_variance'].append(texture_variance)
            
            # 3. Color consistency - face vs neck/background
            if y + h + 30 < frame.shape[0]:
                neck_region = frame[y+h:y+h+30, x:x+w]
                face_mean_color = np.mean(face_crop, axis=(0,1))
                neck_mean_color = np.mean(neck_region, axis=(0,1))
                color_diff = np.linalg.norm(face_mean_color - neck_mean_color)
                metrics['face_neck_color_diff'].append(color_diff)
            
            # 4. Temporal consistency - face position stability
            if prev_face is not None:
                px, py, pw, ph = prev_face
                position_jump = np.sqrt((x - px)**2 + (y - py)**2)
                size_change = abs(w*h - pw*ph) / (pw*ph + 1)
                metrics['position_jump'].append(position_jump)
                metrics['size_change'].append(size_change)
            prev_face = (x, y, w, h)
            
            # 5. Frequency analysis - GAN artifacts
            face_fft = np.fft.fft2(face_gray)
            fft_shift = np.fft.fftshift(face_fft)
            magnitude = np.log(1 + np.abs(fft_shift))
            
            # Check for periodic artifacts (GAN fingerprint)
            fh, fw = magnitude.shape
            center = (fh//2, fw//2)
            
            # High frequency energy ratio
            Y, X = np.ogrid[:fh, :fw]
            dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
            hf_mask = dist > min(fh, fw) * 0.3
            hf_energy = np.sum(magnitude[hf_mask])
            total_energy = np.sum(magnitude)
            hf_ratio = hf_energy / (total_energy + 1e-8)
            metrics['hf_ratio'].append(hf_ratio)
            
            # 6. Blurriness detection
            blur_score = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            metrics['blur_score'].append(blur_score)
            
            # 7. Edge consistency
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.mean(edges) / 255
            metrics['edge_density'].append(edge_density)
            
            # 8. Symmetry analysis
            face_left = face_gray[:, :face_gray.shape[1]//2]
            face_right = face_gray[:, face_gray.shape[1]//2:]
            face_right_flipped = cv2.flip(face_right, 1)
            min_w = min(face_left.shape[1], face_right_flipped.shape[1])
            if min_w > 10:
                symmetry = 1 - np.mean(np.abs(face_left[:, :min_w].astype(float) - 
                                              face_right_flipped[:, :min_w].astype(float))) / 255
                metrics['symmetry'].append(symmetry)
        
        # 9. Frame-to-frame difference
        if prev_frame is not None:
            frame_diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
            metrics['frame_diff'].append(frame_diff)
        
        prev_frame = gray.copy()
        frame_idx += 1
    
    cap.release()
    
    # Print analysis
    print("\n📊 DETAILED METRICS:")
    print("-" * 40)
    
    for metric, values in metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            max_val = np.max(values)
            min_val = np.min(values)
            print(f"{metric}:")
            print(f"  mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    return metrics


def compare_videos():
    """Compare metrics between real and fake videos."""
    videos = [
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/ssstwitter.com_1710426862820.mp4", "REAL"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/ssstwitter.com_1722408787274.mp4", "REAL"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/The_mann_in_202601031346_3qkij.mp4", "FAKE")
    ]
    
    all_metrics = {}
    
    for video_path, label in videos:
        if os.path.exists(video_path):
            metrics = analyze_video_deeply(video_path)
            all_metrics[os.path.basename(video_path)] = (label, metrics)
    
    # Compare
    print("\n" + "="*60)
    print("📈 COMPARISON: REAL vs FAKE")
    print("="*60)
    
    metric_names = set()
    for _, (_, metrics) in all_metrics.items():
        metric_names.update(metrics.keys())
    
    for metric in sorted(metric_names):
        print(f"\n{metric}:")
        for video_name, (label, metrics) in all_metrics.items():
            if metric in metrics and metrics[metric]:
                mean_val = np.mean(metrics[metric])
                print(f"  [{label}] {video_name[:30]}: {mean_val:.4f}")


if __name__ == "__main__":
    compare_videos()

