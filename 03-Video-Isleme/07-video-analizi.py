#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 Video Analizi - OpenCV Video Analysis & Statistics
===================================================

Bu modül video analiz ve istatistik tekniklerini kapsar:
- Video kalite metrikleri (PSNR, SSIM, MSE)
- Hareket analizi ve istatistikleri
- Sahne değişikliği tespiti
- Video özetleme teknikleri
- Performans analizi ve profiling
- Frame rate ve temporal analiz

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import math
from skimage.metrics import structural_similarity as ssim

class VideoAnalyzer:
    """Video analiz sınıfı"""
    
    def __init__(self, history_size=100):
        self.frame_history = deque(maxlen=history_size)
        self.motion_history = deque(maxlen=history_size)
        self.quality_history = deque(maxlen=history_size)
        self.scene_changes = []
        self.frame_timestamps = deque(maxlen=history_size)
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'motion_frames': 0,
            'scene_changes': 0,
            'avg_quality': 0,
            'processing_times': deque(maxlen=100)
        }
    
    def add_frame(self, frame, timestamp=None):
        """Frame ekle ve temel analizi yap"""
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_history.append(frame)
        self.frame_timestamps.append(timestamp)
        self.frame_count += 1
        self.stats['total_frames'] += 1
    
    def calculate_motion(self, threshold=1000):
        """Hareket miktarını hesapla"""
        if len(self.frame_history) < 2:
            return 0, None
        
        current = cv2.cvtColor(self.frame_history[-1], cv2.COLOR_BGR2GRAY)
        previous = cv2.cvtColor(self.frame_history[-2], cv2.COLOR_BGR2GRAY)
        
        # Frame difference
        diff = cv2.absdiff(current, previous)
        motion_pixels = cv2.countNonZero(cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1])
        
        self.motion_history.append(motion_pixels)
        
        if motion_pixels > threshold:
            self.stats['motion_frames'] += 1
        
        return motion_pixels, diff
    
    def detect_scene_change(self, threshold=0.3):
        """Sahne değişikliği tespit et"""
        if len(self.frame_history) < 2:
            return False, 0
        
        current = cv2.cvtColor(self.frame_history[-1], cv2.COLOR_BGR2GRAY)
        previous = cv2.cvtColor(self.frame_history[-2], cv2.COLOR_BGR2GRAY)
        
        # Histogram karşılaştırma
        hist1 = cv2.calcHist([current], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([previous], [0], None, [256], [0, 256])
        
        # Bhattacharyya distance
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        scene_change = correlation < threshold
        if scene_change:
            self.scene_changes.append(self.frame_count)
            self.stats['scene_changes'] += 1
        
        return scene_change, correlation
    
    def calculate_quality_metrics(self, reference_frame=None):
        """Video kalite metriklerini hesapla"""
        if len(self.frame_history) < 2:
            return {}
        
        current = self.frame_history[-1]
        if reference_frame is None:
            reference_frame = self.frame_history[-2]
        
        # Convert to grayscale for some metrics
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        
        metrics = {}
        
        # MSE (Mean Squared Error)
        mse = np.mean((current_gray.astype(np.float32) - ref_gray.astype(np.float32)) ** 2)
        metrics['MSE'] = mse
        
        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        metrics['PSNR'] = psnr
        
        # SSIM (Structural Similarity Index)
        ssim_value = ssim(ref_gray, current_gray)
        metrics['SSIM'] = ssim_value
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(current_gray, cv2.CV_64F)
        sharpness = laplacian.var()
        metrics['Sharpness'] = sharpness
        
        # Brightness
        brightness = np.mean(current_gray)
        metrics['Brightness'] = brightness
        
        # Contrast (standard deviation)
        contrast = np.std(current_gray)
        metrics['Contrast'] = contrast
        
        self.quality_history.append(metrics)
        return metrics
    
    def get_statistics(self):
        """Genel istatistikleri al"""
        stats = self.stats.copy()
        
        # Frame rate calculation
        if len(self.frame_timestamps) > 1:
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if time_span > 0:
                fps = (len(self.frame_timestamps) - 1) / time_span
            else:
                fps = 0
        else:
            fps = 0
        
        stats['fps'] = fps
        
        # Motion statistics
        if self.motion_history:
            stats['avg_motion'] = np.mean(self.motion_history)
            stats['max_motion'] = np.max(self.motion_history)
            stats['motion_variance'] = np.var(self.motion_history)
        
        # Quality statistics
        if self.quality_history:
            recent_quality = list(self.quality_history)[-10:]  # Son 10 frame
            if recent_quality:
                stats['avg_psnr'] = np.mean([q.get('PSNR', 0) for q in recent_quality if q.get('PSNR', 0) != float('inf')])
                stats['avg_ssim'] = np.mean([q.get('SSIM', 0) for q in recent_quality])
                stats['avg_sharpness'] = np.mean([q.get('Sharpness', 0) for q in recent_quality])
        
        return stats

def ornek_1_video_quality_analysis():
    """
    Örnek 1: Video kalite analizi
    """
    print("\n🎯 Örnek 1: Video Kalite Analizi")
    print("=" * 35)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
        print("📹 Test videosu kullanılıyor")
    else:
        print("📷 Webcam kullanılıyor")
    
    analyzer = VideoAnalyzer()
    
    # Reference frame için
    reference_frame = None
    use_reference = False
    
    print("\n🎮 Kontroller:")
    print("r: Reference frame ayarla")
    print("t: Toggle reference/previous frame")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        start_time = time.time()
        
        # Frame'i analyzer'a ekle
        analyzer.add_frame(frame, start_time)
        
        # Kalite metriklerini hesapla
        ref_frame = reference_frame if use_reference else None
        quality_metrics = analyzer.calculate_quality_metrics(ref_frame)
        
        processing_time = (time.time() - start_time) * 1000
        analyzer.stats['processing_times'].append(processing_time)
        
        # Görselleştirme
        display_frame = frame.copy()
        
        # Kalite metrikleri göster
        y_offset = 30
        if quality_metrics:
            for metric_name, value in quality_metrics.items():
                if value != float('inf'):
                    if metric_name == 'PSNR':
                        color = (0, 255, 0) if value > 30 else (0, 255, 255) if value > 20 else (0, 0, 255)
                    elif metric_name == 'SSIM':
                        color = (0, 255, 0) if value > 0.8 else (0, 255, 255) if value > 0.6 else (0, 0, 255)
                    else:
                        color = (255, 255, 255)
                    
                    text = f"{metric_name}: {value:.2f}"
                    cv2.putText(display_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 30
        
        # Analiz modunu göster
        mode_text = "Reference Mode" if use_reference else "Previous Frame Mode"
        cv2.putText(display_frame, mode_text, (10, display_frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if reference_frame is not None:
            cv2.putText(display_frame, "Reference Set", (10, display_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Processing time
        cv2.putText(display_frame, f"Process: {processing_time:.1f}ms", 
                   (display_frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame count
        cv2.putText(display_frame, f"Frame: {analyzer.frame_count}", 
                   (display_frame.shape[1] - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Video Quality Analysis', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            reference_frame = frame.copy()
            print("📍 Reference frame ayarlandı")
        elif key == ord('t'):
            use_reference = not use_reference
            mode = "Reference" if use_reference else "Previous Frame"
            print(f"🔄 Mod: {mode}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    stats = analyzer.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Total Frames: {stats['total_frames']}")
    print(f"   Average FPS: {stats.get('fps', 0):.1f}")
    if 'avg_psnr' in stats:
        print(f"   Average PSNR: {stats['avg_psnr']:.2f}")
    if 'avg_ssim' in stats:
        print(f"   Average SSIM: {stats['avg_ssim']:.3f}")

def ornek_2_motion_analysis():
    """
    Örnek 2: Hareket analizi ve istatistikleri
    """
    print("\n🎯 Örnek 2: Hareket Analizi")
    print("=" * 30)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    analyzer = VideoAnalyzer()
    
    # Optical flow için
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    feature_params = dict(maxCorners=100,
                         qualityLevel=0.3,
                         minDistance=7,
                         blockSize=7)
    
    # İlk frame için
    ret, old_frame = cap.read()
    if not ret:
        print("❌ Video okunamadı!")
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Motion vectors için
    motion_vectors = deque(maxlen=50)
    speed_history = deque(maxlen=100)
    direction_history = deque(maxlen=100)
    
    print("🏃 Hareket analizinde - kameranın önünde hareket edin!")
    print("ESC: Çıkış, r: Reset tracking points")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else old_gray
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                continue
            break
        
        analyzer.add_frame(frame)
        
        # Basic motion detection
        motion_pixels, motion_diff = analyzer.calculate_motion()
        
        # Optical flow analysis
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        display_frame = frame.copy()
        
        if p0 is not None and len(p0) > 0:
            # Optical flow hesapla
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            if p1 is not None:
                # Good points seç
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                if len(good_new) > 0:
                    # Hareket vektörleri çiz ve analiz et
                    speeds = []
                    directions = []
                    
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        
                        # Hareket vektörü
                        dx = a - c
                        dy = b - d
                        speed = np.sqrt(dx*dx + dy*dy)
                        direction = np.degrees(np.arctan2(dy, dx))
                        
                        speeds.append(speed)
                        directions.append(direction)
                        
                        # Vektör çiz
                        cv2.arrowedLine(display_frame, (c, d), (a, b), (0, 255, 0), 2)
                        cv2.circle(display_frame, (a, b), 3, (0, 255, 0), -1)
                    
                    if speeds:
                        avg_speed = np.mean(speeds)
                        avg_direction = np.mean(directions)
                        
                        speed_history.append(avg_speed)
                        direction_history.append(avg_direction)
                        
                        motion_vectors.append({
                            'speed': avg_speed,
                            'direction': avg_direction,
                            'points': len(good_new)
                        })
                
                # Points güncelle
                p0 = good_new.reshape(-1, 1, 2)
        
        # Motion statistics göster
        y_offset = 30
        cv2.putText(display_frame, f"Motion Pixels: {motion_pixels}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        if speed_history:
            avg_speed = np.mean(list(speed_history)[-10:])  # Son 10 frame
            cv2.putText(display_frame, f"Avg Speed: {avg_speed:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        if direction_history:
            recent_directions = list(direction_history)[-10:]
            avg_direction = np.mean(recent_directions)
            cv2.putText(display_frame, f"Avg Direction: {avg_direction:.0f}°", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        cv2.putText(display_frame, f"Tracking Points: {len(p0) if p0 is not None else 0}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Motion history visualization (sağ üst köşe)
        if len(analyzer.motion_history) > 1:
            motion_data = list(analyzer.motion_history)
            max_motion = max(motion_data) if motion_data else 1
            
            # Mini grafik çiz
            graph_w, graph_h = 200, 100
            graph = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)
            
            for i in range(1, min(len(motion_data), graph_w)):
                y1 = int(graph_h - (motion_data[i-1] / max_motion) * graph_h)
                y2 = int(graph_h - (motion_data[i] / max_motion) * graph_h)
                cv2.line(graph, (i-1, y1), (i, y2), (0, 255, 0), 1)
            
            # Graph'ı frame'e yerleştir
            display_frame[10:10+graph_h, display_frame.shape[1]-graph_w-10:display_frame.shape[1]-10] = graph
            cv2.rectangle(display_frame, 
                         (display_frame.shape[1]-graph_w-10, 10),
                         (display_frame.shape[1]-10, 10+graph_h), (255, 255, 255), 1)
            cv2.putText(display_frame, "Motion History", 
                       (display_frame.shape[1]-graph_w-5, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Motion Analysis', display_frame)
        
        # Frame güncelle
        old_gray = frame_gray.copy()
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            # Reset tracking points
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            print("🔄 Tracking points reset")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_scene_change_detection():
    """
    Örnek 3: Sahne değişikliği tespiti
    """
    print("\n🎯 Örnek 3: Sahne Değişikliği Tespiti")
    print("=" * 40)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    analyzer = VideoAnalyzer()
    
    # Scene change detection parametreleri
    scene_threshold = 0.3
    
    # Histogram için
    hist_size = 256
    hist_range = [0, 256]
    
    print("🎬 Sahne değişikliği tespiti - kamerayı farklı yönlere çevirin!")
    print("t: Threshold ayarla, ESC: Çıkış")
    
    last_scene_change = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        analyzer.add_frame(frame)
        
        # Sahne değişikliği tespit et
        scene_change, correlation = analyzer.detect_scene_change(scene_threshold)
        
        if scene_change:
            last_scene_change = analyzer.frame_count
        
        # Görselleştirme
        display_frame = frame.copy()
        
        # Sahne değişikliği durumu
        if analyzer.frame_count - last_scene_change < 30:  # Son 30 frame içinde
            status_color = (0, 0, 255)
            status_text = "SAHNE DEĞİŞİKLİĞİ!"
        else:
            status_color = (0, 255, 0)
            status_text = "Sahne Stabil"
        
        cv2.putText(display_frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 3)
        
        # İstatistikler
        cv2.putText(display_frame, f"Correlation: {correlation:.3f}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Threshold: {scene_threshold:.3f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Total Changes: {analyzer.stats['scene_changes']}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {analyzer.frame_count}", (10, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Histogram göster (sağ taraf)
        if len(analyzer.frame_history) >= 2:
            current_gray = cv2.cvtColor(analyzer.frame_history[-1], cv2.COLOR_BGR2GRAY)
            previous_gray = cv2.cvtColor(analyzer.frame_history[-2], cv2.COLOR_BGR2GRAY)
            
            # Histogramları hesapla
            hist_current = cv2.calcHist([current_gray], [0], None, [hist_size], hist_range)
            hist_previous = cv2.calcHist([previous_gray], [0], None, [hist_size], hist_range)
            
            # Normalize
            cv2.normalize(hist_current, hist_current, 0, 100, cv2.NORM_MINMAX)
            cv2.normalize(hist_previous, hist_previous, 0, 100, cv2.NORM_MINMAX)
            
            # Histogram çiz
            hist_w = 256
            hist_h = 150
            hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
            
            for i in range(hist_size):
                # Current frame histogramı (yeşil)
                cv2.line(hist_img, 
                        (i, hist_h), 
                        (i, hist_h - int(hist_current[i])), 
                        (0, 255, 0), 1)
                # Previous frame histogramı (mavi)
                cv2.line(hist_img, 
                        (i, hist_h), 
                        (i, hist_h - int(hist_previous[i])), 
                        (255, 0, 0), 1)
            
            # Histogram'ı frame'e yerleştir
            display_frame[10:10+hist_h, display_frame.shape[1]-hist_w-10:display_frame.shape[1]-10] = hist_img
            cv2.rectangle(display_frame, 
                         (display_frame.shape[1]-hist_w-10, 10),
                         (display_frame.shape[1]-10, 10+hist_h), (255, 255, 255), 1)
            cv2.putText(display_frame, "Histogram", 
                       (display_frame.shape[1]-hist_w-5, 175),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, "G:Current B:Previous", 
                       (display_frame.shape[1]-hist_w-5, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Scene change timeline (alt kısım)
        timeline_h = 30
        timeline_y = display_frame.shape[0] - timeline_h - 10
        timeline_w = display_frame.shape[1] - 20
        
        cv2.rectangle(display_frame, (10, timeline_y), (10 + timeline_w, timeline_y + timeline_h), (50, 50, 50), -1)
        
        # Son sahne değişikliklerini göster
        for change_frame in analyzer.scene_changes[-20:]:  # Son 20 değişiklik
            if analyzer.frame_count > 0:
                pos_x = int(10 + (change_frame / analyzer.frame_count) * timeline_w)
                cv2.line(display_frame, (pos_x, timeline_y), (pos_x, timeline_y + timeline_h), (0, 0, 255), 2)
        
        # Current position
        current_pos = int(10 + timeline_w * 0.8)  # Son %20'lik kısımda
        cv2.line(display_frame, (current_pos, timeline_y), (current_pos, timeline_y + timeline_h), (255, 255, 255), 2)
        
        cv2.putText(display_frame, "Scene Changes Timeline", (10, timeline_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Scene Change Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('t'):
            try:
                new_threshold = float(input(f"Yeni threshold (mevcut: {scene_threshold:.3f}): "))
                scene_threshold = max(0.1, min(0.9, new_threshold))
                print(f"📊 Threshold: {scene_threshold:.3f}")
            except:
                print("Geçersiz değer!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print(f"\n📊 Scene Change Report:")
    print(f"   Total Frames: {analyzer.frame_count}")
    print(f"   Scene Changes: {analyzer.stats['scene_changes']}")
    if analyzer.frame_count > 0:
        change_rate = analyzer.stats['scene_changes'] / analyzer.frame_count * 100
        print(f"   Change Rate: {change_rate:.1f}%")

def ornek_4_performance_profiling():
    """
    Örnek 4: Performance profiling ve analiz
    """
    print("\n🎯 Örnek 4: Performance Profiling")
    print("=" * 35)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # Performance metrics
    performance_data = {
        'frame_read': deque(maxlen=100),
        'processing': deque(maxlen=100),
        'display': deque(maxlen=100),
        'total': deque(maxlen=100),
        'memory_usage': deque(maxlen=100)
    }
    
    # Test operations
    operations = ['blur', 'edge', 'motion', 'histogram']
    current_operation = 0
    
    print("⚡ Performance profiling - farklı işlemler test ediliyor")
    print("o: Operation değiştir, ESC: Çıkış")
    
    frame_count = 0
    
    while True:
        # Frame read time
        read_start = time.time()
        ret, frame = cap.read()
        read_time = (time.time() - read_start) * 1000
        
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        performance_data['frame_read'].append(read_time)
        
        # Processing time
        process_start = time.time()
        
        # Test farklı işlemler
        operation = operations[current_operation]
        processed_frame = frame.copy()
        
        if operation == 'blur':
            processed_frame = cv2.GaussianBlur(processed_frame, (15, 15), 0)
        elif operation == 'edge':
            gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif operation == 'motion':
            if frame_count > 0:
                # Fake motion detection
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, gray)  # Dummy operation
        elif operation == 'histogram':
            # Histogram equalization
            hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            processed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        process_time = (time.time() - process_start) * 1000
        performance_data['processing'].append(process_time)
        
        # Display preparation time
        display_start = time.time()
        
        display_frame = processed_frame.copy()
        
        # Performance bilgilerini ekle
        y_offset = 30
        cv2.putText(display_frame, f"Operation: {operation.upper()}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 40
        
        # Real-time metrics
        if performance_data['frame_read']:
            cv2.putText(display_frame, f"Frame Read: {read_time:.1f}ms", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        if performance_data['processing']:
            cv2.putText(display_frame, f"Processing: {process_time:.1f}ms", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Average metrics (son 30 frame)
        if len(performance_data['processing']) > 30:
            recent_process = list(performance_data['processing'])[-30:]
            recent_read = list(performance_data['frame_read'])[-30:]
            
            avg_process = np.mean(recent_process)
            avg_read = np.mean(recent_read)
            total_avg = avg_process + avg_read
            
            cv2.putText(display_frame, f"Avg Total: {total_avg:.1f}ms", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30
            
            # FPS estimate
            fps_estimate = 1000 / total_avg if total_avg > 0 else 0
            cv2.putText(display_frame, f"Est. FPS: {fps_estimate:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Performance graph (sağ taraf)
        if len(performance_data['processing']) > 1:
            graph_w, graph_h = 200, 150
            graph = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)
            
            # Processing times graph
            process_data = list(performance_data['processing'])
            max_time = max(process_data) if process_data else 1
            
            for i in range(1, min(len(process_data), graph_w)):
                y1 = int(graph_h - (process_data[i-1] / max_time) * graph_h)
                y2 = int(graph_h - (process_data[i] / max_time) * graph_h)
                cv2.line(graph, (i-1, y1), (i, y2), (0, 255, 0), 1)
            
            # Average line
            if len(process_data) > 10:
                avg_line = int(graph_h - (np.mean(process_data[-30:]) / max_time) * graph_h)
                cv2.line(graph, (0, avg_line), (graph_w, avg_line), (255, 255, 0), 1)
            
            # Graph'ı frame'e yerleştir
            display_frame[10:10+graph_h, display_frame.shape[1]-graph_w-10:display_frame.shape[1]-10] = graph
            cv2.rectangle(display_frame, 
                         (display_frame.shape[1]-graph_w-10, 10),
                         (display_frame.shape[1]-10, 10+graph_h), (255, 255, 255), 1)
            cv2.putText(display_frame, "Processing Time", 
                       (display_frame.shape[1]-graph_w-5, 175),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Max: {max_time:.1f}ms", 
                       (display_frame.shape[1]-graph_w-5, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        display_time = (time.time() - display_start) * 1000
        performance_data['display'].append(display_time)
        
        total_time = read_time + process_time + display_time
        performance_data['total'].append(total_time)
        
        cv2.imshow('Performance Profiling', display_frame)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('o'):
            current_operation = (current_operation + 1) % len(operations)
            print(f"🔄 Operation: {operations[current_operation]}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final performance report
    print(f"\n⚡ Performance Report:")
    print("-" * 30)
    
    for metric_name, data in performance_data.items():
        if data and metric_name != 'memory_usage':
            avg_time = np.mean(data)
            std_time = np.std(data)
            max_time = np.max(data)
            min_time = np.min(data)
            
            print(f"{metric_name:12}: {avg_time:6.2f}ms ± {std_time:5.2f}ms (min: {min_time:5.2f}, max: {max_time:5.2f})")
    
    # Operation comparison
    print(f"\nOperation tested: {operations[current_operation]}")

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("📊 OpenCV Video Analizi Demo")
        print("="*50)
        print("1. 🎯 Video Kalite Analizi (PSNR, SSIM, MSE)")
        print("2. 🏃 Hareket Analizi ve İstatistikleri")
        print("3. 🎬 Sahne Değişikliği Tespiti")
        print("4. ⚡ Performance Profiling")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-4): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_video_quality_analysis()
            elif secim == "2":
                ornek_2_motion_analysis()
            elif secim == "3":
                ornek_3_scene_change_detection()
            elif secim == "4":
                ornek_4_performance_profiling()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("📊 OpenCV Video Analizi")
    print("Bu modül video analiz ve istatistik tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - scikit-image (pip install scikit-image)")
    print("   - Matplotlib (pip install matplotlib)")
    print("   - Webcam (önerilen)")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. PSNR değerleri 30+ iyi, 20-30 orta, <20 kötü kalite
# 2. SSIM değerleri 1'e yakın daha iyi benzerlik
# 3. Sahne değişikliği threshold değeri içeriğe göre ayarlanmalı
# 4. Performance profiling optimize edilecek alanları gösterir
# 5. Real-time analiz için hafif algoritmalar tercih edilmeli