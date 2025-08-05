#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Video Filtreleme - OpenCV Video Efektleri ve Filtreler
========================================================

Bu modül video filtreleme ve efekt tekniklerini kapsar:
- Real-time blur ve keskinleştirme
- Renk düzeltmeleri ve LUT uygulamaları
- Video stabilizasyon (temel)
- Histogram eşitleme ve kontrast ayarları
- Custom video filtreleri ve efektleri

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from collections import deque
import math

class VideoFilter:
    """Video filtre sınıfı"""
    
    def __init__(self):
        self.filter_history = deque(maxlen=10)
        self.stabilization_data = deque(maxlen=30)
        
    def apply_blur_filter(self, frame, blur_type='gaussian', intensity=5):
        """Blur filtresi uygula"""
        if blur_type == 'gaussian':
            return cv2.GaussianBlur(frame, (intensity*2+1, intensity*2+1), 0)
        elif blur_type == 'motion':
            # Motion blur kernel
            kernel_size = intensity * 2 + 1
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1.0 / kernel_size
            return cv2.filter2D(frame, -1, kernel)
        elif blur_type == 'median':
            return cv2.medianBlur(frame, intensity*2+1)
        elif blur_type == 'bilateral':
            return cv2.bilateralFilter(frame, intensity*2+1, 80, 80)
        return frame
    
    def apply_sharpen_filter(self, frame, intensity=1.0):
        """Keskinleştirme filtresi"""
        # Unsharp masking
        blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
        sharpened = cv2.addWeighted(frame, 1.0 + intensity, blurred, -intensity, 0)
        return sharpened
    
    def apply_color_filter(self, frame, filter_type='normal'):
        """Renk filtresi uygula"""
        if filter_type == 'warm':
            # Sıcak tonlar (sarı/turuncu artırma)
            warm_filter = np.array([[[1.2, 1.1, 0.9]]], dtype=np.float32)
            return np.clip(frame * warm_filter, 0, 255).astype(np.uint8)
        
        elif filter_type == 'cool':
            # Soğuk tonlar (mavi artırma)
            cool_filter = np.array([[[0.9, 1.0, 1.2]]], dtype=np.float32)
            return np.clip(frame * cool_filter, 0, 255).astype(np.uint8)
        
        elif filter_type == 'vintage':
            # Vintage efekti
            # Sepia tone matrix
            sepia_filter = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])
            return cv2.transform(frame, sepia_filter)
        
        elif filter_type == 'high_contrast':
            # Yüksek kontrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return frame

def ornek_1_temel_filtreler():
    """
    Örnek 1: Temel video filtreleri
    """
    print("\n🎯 Örnek 1: Temel Video Filtreleri")
    print("=" * 40)
    
    # Webcam veya test video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    filter_processor = VideoFilter()
    
    # Filter parametreleri
    current_filter = 'normal'
    blur_intensity = 3
    sharpen_intensity = 1.0
    
    print("🎨 Filtre Kontrolleri:")
    print("1-5: Blur filtreleri (Gaussian, Motion, Median, Bilateral, None)")
    print("s: Sharpen toggle")
    print("w/c/v/h: Renk filtreleri (Warm, Cool, Vintage, High Contrast)")
    print("n: Normal")
    print("+/-: Intensity ayarla")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # Frame kopyası
        processed_frame = frame.copy()
        filter_info = []
        
        # Blur filtreleri
        if current_filter == 'gaussian':
            processed_frame = filter_processor.apply_blur_filter(
                processed_frame, 'gaussian', blur_intensity)
            filter_info.append(f"Gaussian Blur ({blur_intensity})")
            
        elif current_filter == 'motion':
            processed_frame = filter_processor.apply_blur_filter(
                processed_frame, 'motion', blur_intensity)
            filter_info.append(f"Motion Blur ({blur_intensity})")
            
        elif current_filter == 'median':
            processed_frame = filter_processor.apply_blur_filter(
                processed_frame, 'median', blur_intensity)
            filter_info.append(f"Median Blur ({blur_intensity})")
            
        elif current_filter == 'bilateral':
            processed_frame = filter_processor.apply_blur_filter(
                processed_frame, 'bilateral', blur_intensity)
            filter_info.append(f"Bilateral Filter ({blur_intensity})")
            
        elif current_filter == 'sharpen':
            processed_frame = filter_processor.apply_sharpen_filter(
                processed_frame, sharpen_intensity)
            filter_info.append(f"Sharpen ({sharpen_intensity:.1f})")
        
        # Renk filtreleri
        elif current_filter in ['warm', 'cool', 'vintage', 'high_contrast']:
            processed_frame = filter_processor.apply_color_filter(
                processed_frame, current_filter)
            filter_info.append(f"Color: {current_filter.title()}")
        
        else:
            filter_info.append("Normal")
        
        # Bilgi ekle
        y_pos = 30
        for info in filter_info:
            cv2.putText(processed_frame, info, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
        
        # Kontrol bilgisi
        cv2.putText(processed_frame, "1-5:Blur s:Sharp w/c/v/h:Color n:Normal +/-:Intensity", 
                   (10, processed_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Video Filtreleri', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            current_filter = 'gaussian'
        elif key == ord('2'):
            current_filter = 'motion'
        elif key == ord('3'):
            current_filter = 'median'
        elif key == ord('4'):
            current_filter = 'bilateral'
        elif key == ord('5'):
            current_filter = 'normal'
        elif key == ord('s'):
            current_filter = 'sharpen'
        elif key == ord('w'):
            current_filter = 'warm'
        elif key == ord('c'):
            current_filter = 'cool'
        elif key == ord('v'):
            current_filter = 'vintage'
        elif key == ord('h'):
            current_filter = 'high_contrast'
        elif key == ord('n'):
            current_filter = 'normal'
        elif key == ord('+') or key == ord('='):
            if current_filter in ['gaussian', 'motion', 'median', 'bilateral']:
                blur_intensity = min(10, blur_intensity + 1)
            elif current_filter == 'sharpen':
                sharpen_intensity = min(3.0, sharpen_intensity + 0.2)
        elif key == ord('-'):
            if current_filter in ['gaussian', 'motion', 'median', 'bilateral']:
                blur_intensity = max(1, blur_intensity - 1)
            elif current_filter == 'sharpen':
                sharpen_intensity = max(0.2, sharpen_intensity - 0.2)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_lut_uygulamalari():
    """
    Örnek 2: Look-Up Table (LUT) uygulamaları
    """
    print("\n🎯 Örnek 2: LUT Uygulamaları")
    print("=" * 30)
    
    def create_gamma_lut(gamma):
        """Gamma düzeltme LUT'u oluştur"""
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)])
        return lut.astype(np.uint8)
    
    def create_contrast_lut(contrast, brightness):
        """Kontrast ve parlaklık LUT'u"""
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            new_val = contrast * i + brightness
            lut[i] = np.clip(new_val, 0, 255)
        return lut
    
    def create_custom_lut(lut_type):
        """Özel LUT'lar"""
        lut = np.arange(256, dtype=np.uint8)
        
        if lut_type == 'invert':
            lut = 255 - lut
        elif lut_type == 'posterize':
            # Posterize efekti (renk seviyelerini azalt)
            levels = 8
            lut = (lut // (256 // levels)) * (255 // (levels - 1))
        elif lut_type == 'threshold':
            # Binary threshold
            lut[lut < 128] = 0
            lut[lut >= 128] = 255
        elif lut_type == 'solarize':
            # Solarize efekti
            lut = np.where(lut < 128, lut, 255 - lut)
        
        return lut
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # LUT parametreleri
    gamma = 1.0
    contrast = 1.0
    brightness = 0
    lut_type = 'normal'
    
    print("🎨 LUT Kontrolleri:")
    print("g: Gamma ayarla")
    print("c: Kontrast ayarla") 
    print("b: Parlaklık ayarla")
    print("1-4: Özel LUT'lar (Invert, Posterize, Threshold, Solarize)")
    print("r: Reset")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        processed_frame = frame.copy()
        
        # LUT uygulama
        if lut_type == 'gamma':
            lut = create_gamma_lut(gamma)
            processed_frame = cv2.LUT(processed_frame, lut)
            info_text = f"Gamma: {gamma:.2f}"
            
        elif lut_type == 'contrast':
            lut = create_contrast_lut(contrast, brightness)
            processed_frame = cv2.LUT(processed_frame, lut)
            info_text = f"Contrast: {contrast:.2f}, Brightness: {brightness}"
            
        elif lut_type in ['invert', 'posterize', 'threshold', 'solarize']:
            lut = create_custom_lut(lut_type)
            processed_frame = cv2.LUT(processed_frame, lut)
            info_text = f"LUT: {lut_type.title()}"
            
        else:
            info_text = "Normal"
        
        # Bilgi göster
        cv2.putText(processed_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Kontrol bilgisi
        cv2.putText(processed_frame, "g:Gamma c:Contrast b:Brightness 1-4:Special r:Reset", 
                   (10, processed_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('LUT Uygulamaları', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('g'):
            lut_type = 'gamma'
        elif key == ord('c'):
            lut_type = 'contrast'
        elif key == ord('b'):
            lut_type = 'contrast'  # Aynı LUT'u kullan
        elif key == ord('1'):
            lut_type = 'invert'
        elif key == ord('2'):
            lut_type = 'posterize'
        elif key == ord('3'):
            lut_type = 'threshold'
        elif key == ord('4'):
            lut_type = 'solarize'
        elif key == ord('r'):
            lut_type = 'normal'
            gamma = 1.0
            contrast = 1.0
            brightness = 0
        elif key == ord('+') or key == ord('='):
            if lut_type == 'gamma':
                gamma = min(3.0, gamma + 0.1)
            elif lut_type == 'contrast':
                contrast = min(3.0, contrast + 0.1)
        elif key == ord('-'):
            if lut_type == 'gamma':
                gamma = max(0.1, gamma - 0.1)
            elif lut_type == 'contrast':
                contrast = max(0.1, contrast - 0.1)
        elif key == ord('i'):  # Brightness increase
            brightness = min(100, brightness + 10)
        elif key == ord('d'):  # Brightness decrease
            brightness = max(-100, brightness - 10)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_histogram_esliteme():
    """
    Örnek 3: Real-time histogram eşitleme
    """
    print("\n🎯 Örnek 3: Histogram Eşitleme")
    print("=" * 35)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    equalization_mode = 'normal'
    
    print("📊 Histogram Eşitleme Modları:")
    print("1: Normal")
    print("2: Global Histogram Eşitleme")
    print("3: CLAHE (Adaptive)")
    print("4: HSV V-channel Eşitleme")
    print("5: LAB L-channel Eşitleme")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        processed_frame = frame.copy()
        
        if equalization_mode == 'global':
            # Global histogram eşitleme (gri tonlama)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            processed_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            info_text = "Global Histogram Eşitleme"
            
        elif equalization_mode == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            equalized = clahe.apply(gray)
            processed_frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            info_text = "CLAHE"
            
        elif equalization_mode == 'hsv':
            # HSV V-channel eşitleme
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            processed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            info_text = "HSV V-channel Eşitleme"
            
        elif equalization_mode == 'lab':
            # LAB L-channel eşitleme
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            processed_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            info_text = "LAB L-channel Eşitleme"
            
        else:
            info_text = "Normal"
        
        # Split screen gösterimi (orijinal | işlenmiş)
        h, w = frame.shape[:2]
        combined = np.zeros((h, w, 3), dtype=np.uint8)
        combined[:, :w//2] = frame[:, :w//2]
        combined[:, w//2:] = processed_frame[:, w//2:]
        
        # Ayırıcı çizgi
        cv2.line(combined, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        
        # Başlıklar
        cv2.putText(combined, "Orijinal", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, info_text, (w//2 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Kontrol bilgisi
        cv2.putText(combined, "1-5: Eşitleme modları", 
                   (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Histogram Eşitleme', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            equalization_mode = 'normal'
        elif key == ord('2'):
            equalization_mode = 'global'
        elif key == ord('3'):
            equalization_mode = 'clahe'
        elif key == ord('4'):
            equalization_mode = 'hsv'
        elif key == ord('5'):
            equalization_mode = 'lab'
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_4_video_stabilizasyon():
    """
    Örnek 4: Temel video stabilizasyon
    """
    print("\n🎯 Örnek 4: Video Stabilizasyon")
    print("=" * 35)
    
    class SimpleStabilizer:
        def __init__(self, smoothing_window=30):
            self.prev_gray = None
            self.transforms = []
            self.smoothed_transforms = []
            self.smoothing_window = smoothing_window
            
        def stabilize_frame(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.prev_gray is None:
                self.prev_gray = gray
                return frame
            
            # Feature detection
            prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, 
                                               maxCorners=200,
                                               qualityLevel=0.01,
                                               minDistance=30,
                                               blockSize=3)
            
            if prev_pts is None:
                self.prev_gray = gray
                return frame
            
            # Optical flow
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
            
            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            
            if len(prev_pts) < 10:
                self.prev_gray = gray
                return frame
            
            # Transformation matrix
            transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
            
            if transform is None:
                self.prev_gray = gray
                return frame
            
            # Extract transformation parameters
            dx = transform[0, 2]
            dy = transform[1, 2]
            da = np.arctan2(transform[1, 0], transform[0, 0])
            
            self.transforms.append([dx, dy, da])
            
            # Smooth transformations
            if len(self.transforms) >= self.smoothing_window:
                # Moving average
                recent_transforms = np.array(self.transforms[-self.smoothing_window:])
                smooth_transform = np.mean(recent_transforms, axis=0)
                self.smoothed_transforms.append(smooth_transform)
                
                # Calculate corrected transformation
                diff_transform = np.array(self.transforms[-1]) - smooth_transform
                
                # Create correction matrix
                correction_transform = np.array([
                    [np.cos(-diff_transform[2]), -np.sin(-diff_transform[2]), -diff_transform[0]],
                    [np.sin(-diff_transform[2]), np.cos(-diff_transform[2]), -diff_transform[1]]
                ], dtype=np.float32)
                
                # Apply correction
                h, w = frame.shape[:2]
                stabilized = cv2.warpAffine(frame, correction_transform, (w, h))
                
                self.prev_gray = gray
                return stabilized
            
            self.prev_gray = gray
            return frame
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    stabilizer = SimpleStabilizer()
    stabilization_enabled = True
    
    print("📹 Video Stabilizasyon")
    print("s: Stabilizasyon on/off")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        if stabilization_enabled:
            processed_frame = stabilizer.stabilize_frame(frame)
            status_text = "Stabilizasyon: ON"
            status_color = (0, 255, 0)
        else:
            processed_frame = frame.copy()
            status_text = "Stabilizasyon: OFF"
            status_color = (0, 0, 255)
        
        # Durum bilgisi
        cv2.putText(processed_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(processed_frame, f"Transforms: {len(stabilizer.transforms)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Video Stabilizasyon', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            stabilization_enabled = not stabilization_enabled
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_5_ozel_efektler():
    """
    Örnek 5: Özel video efektleri
    """
    print("\n🎯 Örnek 5: Özel Video Efektleri")
    print("=" * 35)
    
    def apply_vignette(frame, intensity=0.3):
        """Vignette efekti"""
        h, w = frame.shape[:2]
        
        # Merkez ve yarıçap
        center_x, center_y = w // 2, h // 2
        max_radius = min(center_x, center_y)
        
        # Vignette mask oluştur
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Radial gradient
        mask = 1.0 - (dist_from_center / max_radius) * intensity
        mask = np.clip(mask, 0, 1)
        
        # 3 kanala genişlet
        mask = np.dstack([mask] * 3)
        
        return (frame * mask).astype(np.uint8)
    
    def apply_film_grain(frame, intensity=0.1):
        """Film grain efekti"""
        noise = np.random.normal(0, intensity * 255, frame.shape)
        noisy_frame = frame.astype(np.float32) + noise
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
    def apply_edge_glow(frame, glow_intensity=2.0):
        """Edge glow efekti"""
        # Kenarları bul
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Kenarları blur'la (glow efekti)
        glowing_edges = cv2.GaussianBlur(edges, (15, 15), 0)
        
        # Renkli yap
        glowing_edges_colored = cv2.cvtColor(glowing_edges, cv2.COLOR_GRAY2BGR)
        
        # Orijinal frame ile birleştir
        result = cv2.addWeighted(frame, 1.0, glowing_edges_colored, glow_intensity, 0)
        return result
    
    def apply_color_shift(frame, shift_type='rgb'):
        """Renk kayması efekti"""
        h, w = frame.shape[:2]
        
        if shift_type == 'rgb':
            # RGB kanallarını kaydır
            shifted = np.zeros_like(frame)
            shifted[:, :-5, 0] = frame[:, 5:, 0]  # R kanalını sola kaydır
            shifted[:, :, 1] = frame[:, :, 1]      # G kanalını olduğu gibi bırak
            shifted[:, 5:, 2] = frame[:, :-5, 2]   # B kanalını sağa kaydır
            return shifted
        
        elif shift_type == 'chromatic':
            # Chromatic aberration
            shifted = frame.copy()
            # Kırmızı kanalı büyüt
            red_channel = frame[:, :, 2]
            red_resized = cv2.resize(red_channel, (w+4, h+4))
            shifted[2:-2, 2:-2, 2] = red_resized[2:-2, 2:-2]
            return shifted
        
        return frame
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    effect_type = 'normal'
    
    print("🎨 Özel Efektler:")
    print("1: Normal")
    print("2: Vignette")
    print("3: Film Grain")
    print("4: Edge Glow")
    print("5: RGB Shift")
    print("6: Chromatic Aberration")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        processed_frame = frame.copy()
        
        if effect_type == 'vignette':
            processed_frame = apply_vignette(processed_frame, 0.5)
            info_text = "Vignette"
        elif effect_type == 'grain':
            processed_frame = apply_film_grain(processed_frame, 0.1)
            info_text = "Film Grain"
        elif effect_type == 'glow':
            processed_frame = apply_edge_glow(processed_frame, 1.5)
            info_text = "Edge Glow"
        elif effect_type == 'rgb_shift':
            processed_frame = apply_color_shift(processed_frame, 'rgb')
            info_text = "RGB Shift"
        elif effect_type == 'chromatic':
            processed_frame = apply_color_shift(processed_frame, 'chromatic')
            info_text = "Chromatic Aberration"
        else:
            info_text = "Normal"
        
        # Bilgi göster
        cv2.putText(processed_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Özel Efektler', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key >= ord('1') and key <= ord('6'):
            effect_map = {
                ord('1'): 'normal',
                ord('2'): 'vignette', 
                ord('3'): 'grain',
                ord('4'): 'glow',
                ord('5'): 'rgb_shift',
                ord('6'): 'chromatic'
            }
            effect_type = effect_map[key]
    
    cap.release()
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🎨 OpenCV Video Filtreleme Demo")
        print("="*50)
        print("1. 🎭 Temel Video Filtreleri")
        print("2. 🎨 LUT Uygulamaları")
        print("3. 📊 Histogram Eşitleme")
        print("4. 📹 Video Stabilizasyon")
        print("5. ✨ Özel Efektler")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-5): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_temel_filtreler()
            elif secim == "2":
                ornek_2_lut_uygulamalari()
            elif secim == "3":
                ornek_3_histogram_esliteme()
            elif secim == "4":
                ornek_4_video_stabilizasyon()
            elif secim == "5":
                ornek_5_ozel_efektler()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-5 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🎨 OpenCV Video Filtreleme")
    print("Bu modül real-time video filtreleme ve efekt tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (önerilen)")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Real-time filtreleme CPU yoğun olabilir
# 2. LUT'lar hızlı renk düzeltmeleri için idealdir
# 3. Video stabilizasyon basit feature tracking kullanır
# 4. Efektler yaratıcı video düzenleme için kullanışlıdır
# 5. Webcam kullanımı daha interaktif deneyim sağlar