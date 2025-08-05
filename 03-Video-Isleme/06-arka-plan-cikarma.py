#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎭 Arka Plan Çıkarma - OpenCV Background Subtraction
=================================================

Bu modül arka plan çıkarma tekniklerini kapsar:
- MOG ve MOG2 background subtraction
- KNN-based background modeling
- GMG algorithm
- Adaptive background learning
- Shadow detection ve removal
- Foreground object detection

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from collections import deque

class BackgroundSubtractor:
    """Arka plan çıkarma sınıfı"""
    
    def __init__(self, method='MOG2'):
        self.method = method
        self.subtractor = None
        self.learning_rate = 0.01
        self.shadow_detection = True
        self.create_subtractor()
        
    def create_subtractor(self):
        """Background subtractor oluştur"""
        if self.method == 'MOG2':
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=self.shadow_detection)
        elif self.method == 'KNN':
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                detectShadows=self.shadow_detection)
        elif self.method == 'MOG':
            # MOG legacy (eski OpenCV versiyonlarında)
            try:
                self.subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
            except:
                print("⚠️ MOG mevcut değil, MOG2 kullanılıyor")
                self.subtractor = cv2.createBackgroundSubtractorMOG2()
        elif self.method == 'GMG':
            # GMG legacy
            try:
                self.subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
            except:
                print("⚠️ GMG mevcut değil, MOG2 kullanılıyor")
                self.subtractor = cv2.createBackgroundSubtractorMOG2()
        else:
            self.subtractor = cv2.createBackgroundSubtractorMOG2()
    
    def apply(self, frame):
        """Arka plan çıkarma uygula"""
        if self.subtractor is None:
            return None
        
        # Foreground mask
        fg_mask = self.subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Background model
        background = self.subtractor.getBackgroundImage()
        
        return fg_mask, background
    
    def set_learning_rate(self, rate):
        """Learning rate ayarla"""
        self.learning_rate = max(0.0, min(1.0, rate))
    
    def reset(self):
        """Subtractor'ı sıfırla"""
        self.create_subtractor()

def ornek_1_temel_background_subtraction():
    """
    Örnek 1: Temel arka plan çıkarma
    """
    print("\n🎯 Örnek 1: Temel Background Subtraction")
    print("=" * 45)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
        print("📹 Test videosu kullanılıyor")
    else:
        print("📷 Webcam kullanılıyor - hareket edin!")
    
    subtractor = BackgroundSubtractor('MOG2')
    
    print("\n🎮 Kontroller:")
    print("1-4: Algorithm değiştir (MOG2, KNN, MOG, GMG)")
    print("s: Shadow detection on/off")
    print("+/-: Learning rate ayarla")
    print("r: Reset background model")
    print("ESC: Çıkış")
    
    methods = ['MOG2', 'KNN', 'MOG', 'GMG']
    current_method_idx = 0
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        frame_count += 1
        
        # Background subtraction uygula
        result = subtractor.apply(frame)
        if result is None:
            continue
            
        fg_mask, background = result
        
        # Foreground objeler (morfolojik işlemler)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask_clean = cv2.morphologyEx(fg_mask_clean, cv2.MORPH_CLOSE, kernel)
        
        # Foreground'u orijinal frame'de göster
        foreground = cv2.bitwise_and(frame, frame, mask=fg_mask_clean)
        
        # 4 panel görünüm oluştur
        h, w = frame.shape[:2]
        
        # Panelleri resize et
        panel_size = (w//2, h//2)
        
        original = cv2.resize(frame, panel_size)
        fg_display = cv2.resize(cv2.cvtColor(fg_mask_clean, cv2.COLOR_GRAY2BGR), panel_size)
        foreground_display = cv2.resize(foreground, panel_size)
        
        if background is not None:
            background_display = cv2.resize(background, panel_size)
        else:
            background_display = np.zeros((panel_size[1], panel_size[0], 3), dtype=np.uint8)
        
        # Panelleri birleştir
        top_row = np.hstack((original, background_display))
        bottom_row = np.hstack((fg_display, foreground_display))
        combined = np.vstack((top_row, bottom_row))
        
        # Panel başlıkları
        cv2.putText(combined, "Original", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Background", (w//2 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Foreground Mask", (10, h//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Foreground Objects", (w//2 + 10, h//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Bilgi paneli
        info_y = h - 80
        cv2.putText(combined, f"Method: {subtractor.method}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, f"Learning Rate: {subtractor.learning_rate:.3f}", (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, f"Shadow Detection: {'ON' if subtractor.shadow_detection else 'OFF'}", (10, info_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(combined, f"Frame: {frame_count}", (w//2 + 10, info_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Background Subtraction', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key >= ord('1') and key <= ord('4'):
            idx = key - ord('1')
            if idx < len(methods):
                current_method_idx = idx
                new_method = methods[idx]
                subtractor.method = new_method
                subtractor.create_subtractor()
                print(f"🔄 Method: {new_method}")
        elif key == ord('s'):
            subtractor.shadow_detection = not subtractor.shadow_detection
            subtractor.create_subtractor()
            print(f"👥 Shadow Detection: {'ON' if subtractor.shadow_detection else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            new_rate = min(1.0, subtractor.learning_rate + 0.005)
            subtractor.set_learning_rate(new_rate)
            print(f"📈 Learning Rate: {subtractor.learning_rate:.3f}")
        elif key == ord('-'):
            new_rate = max(0.0, subtractor.learning_rate - 0.005)
            subtractor.set_learning_rate(new_rate)
            print(f"📉 Learning Rate: {subtractor.learning_rate:.3f}")
        elif key == ord('r'):
            subtractor.reset()
            frame_count = 0
            print("🔄 Background model reset")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_adaptive_background_learning():
    """
    Örnek 2: Adaptive background learning
    """
    print("\n🎯 Örnek 2: Adaptive Background Learning")
    print("=" * 45)
    
    class AdaptiveBackgroundModel:
        def __init__(self, alpha=0.01):
            self.background = None
            self.alpha = alpha  # Learning rate
            self.initialized = False
            
        def update(self, frame):
            """Background model güncelle"""
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if not self.initialized:
                self.background = gray.astype(np.float32)
                self.initialized = True
                return np.zeros_like(gray, dtype=np.uint8)
            
            # Running average ile background güncelle
            cv2.accumulateWeighted(gray, self.background, self.alpha)
            
            # Foreground mask
            diff = cv2.absdiff(gray, self.background.astype(np.uint8))
            _, fg_mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
            
            return fg_mask
        
        def get_background(self):
            """Background image'ı al"""
            if self.background is not None:
                return self.background.astype(np.uint8)
            return None
        
        def reset(self):
            """Model'i sıfırla"""
            self.background = None
            self.initialized = False
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # Models
    opencv_subtractor = BackgroundSubtractor('MOG2')
    adaptive_model = AdaptiveBackgroundModel(alpha=0.01)
    
    print("📊 Karşılaştırma: OpenCV MOG2 vs Custom Adaptive")
    print("a: Adaptive learning rate ayarla")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # OpenCV MOG2
        opencv_result = opencv_subtractor.apply(frame)
        if opencv_result:
            opencv_fg, opencv_bg = opencv_result
        else:
            opencv_fg = np.zeros(frame.shape[:2], dtype=np.uint8)
            opencv_bg = None
        
        # Custom adaptive
        adaptive_fg = adaptive_model.update(frame)
        adaptive_bg = adaptive_model.get_background()
        
        # Görselleştirme
        h, w = frame.shape[:2]
        
        # 3x2 grid layout
        display_size = (w//3, h//2)
        
        # Paneller
        original = cv2.resize(frame, display_size)
        
        opencv_fg_display = cv2.resize(cv2.cvtColor(opencv_fg, cv2.COLOR_GRAY2BGR), display_size)
        adaptive_fg_display = cv2.resize(cv2.cvtColor(adaptive_fg, cv2.COLOR_GRAY2BGR), display_size)
        
        if opencv_bg is not None:
            opencv_bg_display = cv2.resize(opencv_bg, display_size)
        else:
            opencv_bg_display = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
        
        if adaptive_bg is not None:
            adaptive_bg_display = cv2.resize(cv2.cvtColor(adaptive_bg, cv2.COLOR_GRAY2BGR), display_size)
        else:
            adaptive_bg_display = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
        
        # Fark analizi
        if adaptive_bg is not None and opencv_bg is not None:
            bg_diff = cv2.absdiff(cv2.cvtColor(opencv_bg, cv2.COLOR_BGR2GRAY), adaptive_bg)
            bg_diff_display = cv2.resize(cv2.cvtColor(bg_diff, cv2.COLOR_GRAY2BGR), display_size)
        else:
            bg_diff_display = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
        
        # Layout oluştur
        top_row = np.hstack((original, opencv_bg_display, adaptive_bg_display))
        bottom_row = np.hstack((bg_diff_display, opencv_fg_display, adaptive_fg_display))
        combined = np.vstack((top_row, bottom_row))
        
        # Başlıklar
        cv2.putText(combined, "Original", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "OpenCV BG", (w//3 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "Adaptive BG", (2*w//3 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "BG Difference", (10, h//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "OpenCV FG", (w//3 + 10, h//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "Adaptive FG", (2*w//3 + 10, h//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bilgiler
        info_y = h - 30
        cv2.putText(combined, f"Adaptive Alpha: {adaptive_model.alpha:.3f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Adaptive vs OpenCV Background Learning', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('a'):
            try:
                new_alpha = float(input(f"Yeni alpha değeri (mevcut: {adaptive_model.alpha:.3f}): "))
                adaptive_model.alpha = max(0.001, min(1.0, new_alpha))
                print(f"📈 Alpha: {adaptive_model.alpha:.3f}")
            except:
                print("Geçersiz değer!")
        elif key == ord('r'):
            adaptive_model.reset()
            opencv_subtractor.reset()
            print("🔄 Models reset")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_shadow_detection():
    """
    Örnek 3: Gölge tespiti ve kaldırma
    """
    print("\n🎯 Örnek 3: Shadow Detection & Removal")
    print("=" * 40)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # MOG2 with shadow detection
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    print("👥 Gölge tespiti aktif - iyi ışıklandırma gerekli")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # Background subtraction (shadows included)
        fg_mask = subtractor.apply(frame)
        
        # Shadow değerleri genellikle 127 (gri) olarak işaretlenir
        # Foreground: 255 (beyaz)
        # Background: 0 (siyah)  
        # Shadow: 127 (gri)
        
        # Sadece foreground (gölge olmayan)
        fg_only = np.where(fg_mask == 255, 255, 0).astype(np.uint8)
        
        # Sadece gölgeler
        shadow_only = np.where(fg_mask == 127, 255, 0).astype(np.uint8)
        
        # Gölge + foreground
        fg_with_shadow = np.where(fg_mask >= 127, 255, 0).astype(np.uint8)
        
        # Renkli görselleştirme
        colored_mask = np.zeros((fg_mask.shape[0], fg_mask.shape[1], 3), dtype=np.uint8)
        colored_mask[fg_mask == 255] = [0, 255, 0]  # Foreground -> Yeşil
        colored_mask[fg_mask == 127] = [0, 255, 255]  # Shadow -> Sarı
        
        # 2x3 grid layout
        h, w = frame.shape[:2]
        display_size = (w//3, h//2)
        
        original = cv2.resize(frame, display_size)
        fg_raw = cv2.resize(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), display_size)
        colored = cv2.resize(colored_mask, display_size)
        fg_clean = cv2.resize(cv2.cvtColor(fg_only, cv2.COLOR_GRAY2BGR), display_size)
        shadows = cv2.resize(cv2.cvtColor(shadow_only, cv2.COLOR_GRAY2BGR), display_size)
        fg_shadow = cv2.resize(cv2.cvtColor(fg_with_shadow, cv2.COLOR_GRAY2BGR), display_size)
        
        # Layout
        top_row = np.hstack((original, fg_raw, colored))
        bottom_row = np.hstack((fg_clean, shadows, fg_shadow))
        combined = np.vstack((top_row, bottom_row))
        
        # Başlıklar
        titles = ["Original", "Raw Mask", "Colored (G:FG, Y:Shadow)",
                 "Foreground Only", "Shadows Only", "FG + Shadows"]
        positions = [(10, 25), (w//3 + 10, 25), (2*w//3 + 10, 25),
                    (10, h//2 + 25), (w//3 + 10, h//2 + 25), (2*w//3 + 10, h//2 + 25)]
        
        for title, pos in zip(titles, positions):
            cv2.putText(combined, title, pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # İstatistikler
        fg_pixels = np.sum(fg_mask == 255)
        shadow_pixels = np.sum(fg_mask == 127)
        total_motion = fg_pixels + shadow_pixels
        
        info_y = h - 30
        cv2.putText(combined, f"FG: {fg_pixels} | Shadow: {shadow_pixels} | Total: {total_motion}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if total_motion > 0:
            shadow_ratio = shadow_pixels / total_motion * 100
            cv2.putText(combined, f"Shadow Ratio: {shadow_ratio:.1f}%", 
                       (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('Shadow Detection', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_4_foreground_object_detection():
    """
    Örnek 4: Foreground object detection ve analiz
    """
    print("\n🎯 Örnek 4: Foreground Object Detection")
    print("=" * 45)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    # Object tracking
    object_id = 0
    min_area = 500
    
    print("🔍 Foreground nesnelerini tespit ediyor...")
    print("m: Min area ayarla, ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        # Background subtraction
        fg_mask = subtractor.apply(frame)
        
        # Gölgeleri kaldır
        fg_mask = np.where(fg_mask == 255, 255, 0).astype(np.uint8)
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Contour detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Object detection ve analiz
        display_frame = frame.copy()
        objects_detected = 0
        total_area = 0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area > min_area:
                objects_detected += 1
                total_area += area
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Contour çiz
                cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
                
                # Bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Merkez nokta
                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Object bilgileri
                cv2.putText(display_frame, f"#{i+1}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(display_frame, f"Area: {int(area)}", (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                cv2.putText(display_frame, f"AR: {aspect_ratio:.2f}", (x, y + h + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Extent (contour area / bounding box area)
                extent = area / (w * h) if (w * h) > 0 else 0
                cv2.putText(display_frame, f"Ext: {extent:.2f}", (x, y + h + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Genel bilgiler
        cv2.putText(display_frame, f"Objects: {objects_detected}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Total Area: {int(total_area)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Min Area: {min_area}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Coverage percentage
        frame_area = frame.shape[0] * frame.shape[1]
        coverage = (total_area / frame_area) * 100 if frame_area > 0 else 0
        cv2.putText(display_frame, f"Coverage: {coverage:.1f}%", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Foreground mask göster (küçük pencere)
        fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        fg_small = cv2.resize(fg_colored, (160, 120))
        
        h_frame, w_frame = display_frame.shape[:2]
        display_frame[h_frame-130:h_frame-10, w_frame-170:w_frame-10] = fg_small
        cv2.rectangle(display_frame, (w_frame-170, h_frame-130), (w_frame-10, h_frame-10), (255, 255, 255), 2)
        cv2.putText(display_frame, "FG Mask", (w_frame-165, h_frame-135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Foreground Object Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('m'):
            try:
                new_area = int(input(f"Yeni minimum area (mevcut: {min_area}): "))
                min_area = max(100, new_area)
                print(f"📏 Min Area: {min_area}")
            except:
                print("Geçersiz değer!")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_5_multi_algorithm_comparison():
    """
    Örnek 5: Çoklu algoritma karşılaştırması
    """
    print("\n🎯 Örnek 5: Multi-Algorithm Comparison")
    print("=" * 45)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # Background subtractors
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    knn = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    
    # Custom simple method
    class SimpleBS:
        def __init__(self):
            self.background = None
            self.alpha = 0.01
        
        def apply(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.background is None:
                self.background = gray.astype(np.float32)
                return np.zeros_like(gray)
            
            # Running average
            cv2.accumulateWeighted(gray, self.background, self.alpha)
            
            # Difference
            diff = cv2.absdiff(gray, self.background.astype(np.uint8))
            _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            return mask
    
    simple_bs = SimpleBS()
    
    # Performance metrics
    processing_times = {'MOG2': [], 'KNN': [], 'Simple': []}
    
    print("⚡ Algoritma karşılaştırması - performans ve kalite")
    print("ESC: Çıkış")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        frame_count += 1
        
        # MOG2
        start = time.time()
        mog2_mask = mog2.apply(frame)
        mog2_time = (time.time() - start) * 1000
        processing_times['MOG2'].append(mog2_time)
        
        # KNN
        start = time.time()
        knn_mask = knn.apply(frame)
        knn_time = (time.time() - start) * 1000
        processing_times['KNN'].append(knn_time)
        
        # Simple
        start = time.time()
        simple_mask = simple_bs.apply(frame)
        simple_time = (time.time() - start) * 1000
        processing_times['Simple'].append(simple_time)
        
        # Limit history
        for key in processing_times:
            if len(processing_times[key]) > 100:
                processing_times[key] = processing_times[key][-100:]
        
        # 2x2 grid layout
        h, w = frame.shape[:2]
        display_size = (w//2, h//2)
        
        original = cv2.resize(frame, display_size)
        mog2_display = cv2.resize(cv2.cvtColor(mog2_mask, cv2.COLOR_GRAY2BGR), display_size)
        knn_display = cv2.resize(cv2.cvtColor(knn_mask, cv2.COLOR_GRAY2BGR), display_size)
        simple_display = cv2.resize(cv2.cvtColor(simple_mask, cv2.COLOR_GRAY2BGR), display_size)
        
        # Layout
        top_row = np.hstack((original, mog2_display))
        bottom_row = np.hstack((knn_display, simple_display))
        combined = np.vstack((top_row, bottom_row))
        
        # Başlıklar ve performans bilgileri
        cv2.putText(combined, "Original", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, f"MOG2 ({mog2_time:.1f}ms)", (w//2 + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, f"KNN ({knn_time:.1f}ms)", (10, h//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, f"Simple ({simple_time:.1f}ms)", (w//2 + 10, h//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Ortalama performans
        info_y = h - 60  # info_y'i burada tanımla
        
        if frame_count > 10:
            avg_times = {k: np.mean(v[-50:]) for k, v in processing_times.items()}
            
            cv2.putText(combined, "Average Processing Times:", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            for i, (alg, avg_time) in enumerate(avg_times.items()):
                cv2.putText(combined, f"{alg}: {avg_time:.1f}ms", (10, info_y + 20 + i*15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # En hızlı algoritma
            fastest = min(avg_times, key=avg_times.get)
            cv2.putText(combined, f"Fastest: {fastest}", (w//2 + 10, info_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(combined, f"Frame: {frame_count}", (w//2 + 10, info_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Algorithm Comparison', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final performance report
    if processing_times['MOG2']:
        print("\n📊 Final Performance Report:")
        print("-" * 30)
        for alg, times in processing_times.items():
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                print(f"{alg:8}: {avg_time:6.1f}ms ± {std_time:4.1f}ms")

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🎭 OpenCV Background Subtraction Demo")
        print("="*50)
        print("1. 🎯 Temel Background Subtraction")
        print("2. 🧠 Adaptive Background Learning")
        print("3. 👥 Shadow Detection & Removal")
        print("4. 🔍 Foreground Object Detection")
        print("5. ⚡ Multi-Algorithm Comparison")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-5): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_temel_background_subtraction()
            elif secim == "2":
                ornek_2_adaptive_background_learning()
            elif secim == "3":
                ornek_3_shadow_detection()
            elif secim == "4":
                ornek_4_foreground_object_detection()
            elif secim == "5":
                ornek_5_multi_algorithm_comparison()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-5 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🎭 OpenCV Background Subtraction")
    print("Bu modül arka plan çıkarma ve foreground analiz tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (önerilen, hareket için)")
    print("\n🎯 Not: En iyi sonuçlar için sabit kamera ve değişken sahne kullanın.")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. MOG2 çoğu durumda en iyi performansı verir
# 2. KNN yavaş değişen arka planlar için iyidir
# 3. Shadow detection iyi aydınlatma gerektirir
# 4. Learning rate arka plan değişim hızına göre ayarlanmalı
# 5. Foreground detection için morfolojik işlemler önemlidir