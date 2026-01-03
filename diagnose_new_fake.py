#!/usr/bin/env python3
"""
Deep diagnostic comparing the two deepfake videos and the real videos.
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
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    metrics = defaultdict(list)
    prev_frame = None
    prev_face = None
    prev_face_gray = None
    
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
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            
            # 1. Texture variance (Laplacian)
            face_laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
            texture_variance = np.var(face_laplacian)
            metrics['texture_variance'].append(texture_variance)
            
            # 2. Symmetry
            mid = face_gray.shape[1] // 2
            left_half = face_gray[:, :mid]
            right_half = face_gray[:, mid:mid + left_half.shape[1]]
            if left_half.shape[1] > 0 and right_half.shape[1] > 0:
                right_flipped = cv2.flip(right_half, 1)
                min_w = min(left_half.shape[1], right_flipped.shape[1])
                if min_w > 10:
                    symmetry = 1 - np.mean(np.abs(left_half[:, :min_w].astype(float) - 
                                                  right_flipped[:, :min_w].astype(float))) / 255
                    metrics['symmetry'].append(symmetry)
            
            # 3. Edge density
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.mean(edges) / 255
            metrics['edge_density'].append(edge_density)
            
            # 4. Face-neck color difference
            if y + h + 30 < frame.shape[0]:
                neck_region = frame[y+h:y+h+30, x:x+w]
                face_mean_color = np.mean(face_crop, axis=(0,1))
                neck_mean_color = np.mean(neck_region, axis=(0,1))
                color_diff = np.linalg.norm(face_mean_color - neck_mean_color)
                metrics['face_neck_color_diff'].append(color_diff)
            
            # 5. Blur score
            blur_score = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            metrics['blur_score'].append(blur_score)
            
            # 6. High frequency ratio
            face_fft = np.fft.fft2(face_gray)
            fft_shift = np.fft.fftshift(face_fft)
            magnitude = np.log(1 + np.abs(fft_shift))
            fh, fw = magnitude.shape
            center = (fh//2, fw//2)
            Y, X = np.ogrid[:fh, :fw]
            dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
            hf_mask = dist > min(fh, fw) * 0.3
            hf_energy = np.sum(magnitude[hf_mask])
            total_energy = np.sum(magnitude)
            hf_ratio = hf_energy / (total_energy + 1e-8)
            metrics['hf_ratio'].append(hf_ratio)
            
            # 7. Local contrast (new)
            local_std = ndimage.generic_filter(face_gray.astype(float), np.std, size=5)
            local_contrast = np.mean(local_std)
            metrics['local_contrast'].append(local_contrast)
            
            # 8. Gradient magnitude (new)
            gx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            metrics['gradient_magnitude'].append(np.mean(gradient_mag))
            
            # 9. Color variance per channel (new)
            if len(face_crop.shape) == 3:
                for i, channel in enumerate(['b', 'g', 'r']):
                    metrics[f'color_var_{channel}'].append(np.var(face_crop[:,:,i]))
            
            # 10. Temporal face difference (new - important for deepfakes)
            if prev_face_gray is not None and prev_face_gray.shape == face_gray.shape:
                face_diff = np.mean(np.abs(face_gray.astype(float) - prev_face_gray.astype(float)))
                metrics['temporal_face_diff'].append(face_diff)
            
            # 11. Skin color uniformity (new)
            hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            # Skin typically has H in range 0-50, S 20-255
            h_var = np.var(hsv[:,:,0])
            s_var = np.var(hsv[:,:,1])
            metrics['hue_variance'].append(h_var)
            metrics['saturation_variance'].append(s_var)
            
            # 12. Boundary sharpness
            face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            face_mask[y:y+h, x:x+w] = 255
            boundary = cv2.Canny(face_mask, 100, 200)
            dilated = cv2.dilate(boundary, np.ones((5,5), np.uint8))
            laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            boundary_sharpness = np.mean(np.abs(laplacian[dilated > 0])) if np.sum(dilated) > 0 else 0
            metrics['boundary_sharpness'].append(boundary_sharpness)
            
            prev_face_gray = face_gray.copy()
            prev_face = (x, y, w, h)
        
        # Frame difference
        if prev_frame is not None:
            frame_diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
            metrics['frame_diff'].append(frame_diff)
        
        prev_frame = gray.copy()
        frame_idx += 1
    
    cap.release()
    
    # Print analysis
    print("\n📊 DETAILED METRICS:")
    print("-" * 50)
    
    for metric, values in sorted(metrics.items()):
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric:25s}: mean={mean_val:8.4f}, std={std_val:8.4f}")
    
    return metrics


def compare_all_videos():
    """Compare all videos including the new fake."""
    videos = [
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/ssstwitter.com_1710426862820.mp4", "REAL"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/ssstwitter.com_1722408787274.mp4", "REAL"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/Ben_Volovelsky_Summer2026_Video.mp4", "REAL"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/The_mann_in_202601031346_3qkij.mp4", "FAKE1"),
        ("/Users/bvolovelsky/Desktop/LLM/DEEPFAKE/The_man_in_202601031425_w8qo6.mp4", "FAKE2"),
    ]
    
    all_metrics = {}
    
    for video_path, label in videos:
        if os.path.exists(video_path):
            metrics = analyze_video_deeply(video_path)
            all_metrics[os.path.basename(video_path)[:25]] = (label, metrics)
    
    # Compare key metrics
    print("\n" + "="*70)
    print("📈 COMPARISON: REAL vs FAKE")
    print("="*70)
    
    # Collect all metric names
    metric_names = set()
    for _, (_, metrics) in all_metrics.items():
        metric_names.update(metrics.keys())
    
    # Key metrics to focus on
    key_metrics = ['texture_variance', 'symmetry', 'edge_density', 'local_contrast', 
                   'gradient_magnitude', 'temporal_face_diff', 'hue_variance', 
                   'saturation_variance', 'face_neck_color_diff', 'boundary_sharpness']
    
    for metric in key_metrics:
        if metric in metric_names:
            print(f"\n{metric}:")
            real_vals = []
            fake_vals = []
            for video_name, (label, metrics) in all_metrics.items():
                if metric in metrics and metrics[metric]:
                    mean_val = np.mean(metrics[metric])
                    marker = "  " if "REAL" in label else "**"
                    print(f"  {marker}[{label:5s}] {video_name}: {mean_val:.4f}")
                    if "REAL" in label:
                        real_vals.append(mean_val)
                    else:
                        fake_vals.append(mean_val)
            
            if real_vals and fake_vals:
                real_mean = np.mean(real_vals)
                fake_mean = np.mean(fake_vals)
                diff_pct = ((fake_mean - real_mean) / real_mean) * 100 if real_mean != 0 else 0
                print(f"  → REAL avg: {real_mean:.4f}, FAKE avg: {fake_mean:.4f} ({diff_pct:+.1f}%)")


if __name__ == "__main__":
    compare_all_videos()

