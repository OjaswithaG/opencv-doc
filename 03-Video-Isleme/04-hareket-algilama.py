#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏃 Hareket Algılama - OpenCV Motion Detection
===========================================

Bu modül hareket algılama tekniklerini kapsar:
- Frame differencing ve temporal filtering
- Background subtraction yöntemleri
- Optical flow algoritmaları
- Motion vectors ve analizi
- Hareket tabanlı olay tespiti

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from collections import deque
import math

class MotionDetector:
    """Hareket algılama sınıfı"""
    
    def __init__(self, history_size=10):
        self.frame_history = deque(maxlen=history_size)
        self.motion_history = deque(maxlen=50)
        self.background_subtractor = None
        self.last_motion_time = 0
        self.motion_threshold = 1000  # Minimum hareket alanı
        
    def add_frame(self, frame):
        """Frame geçmişine ekle"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_history.append(gray)
    
    def detect_motion_simple(self, current_frame, threshold=30):
        """Basit frame differencing ile hareket algılama"""
        if len(self.frame_history) < 2:
            return None, 0
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_gray = self.frame_history[-1]
        
        # Frame farkı
        diff = cv2.absdiff(previous_gray, current_gray)
        
        # Threshold uygula
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Morfolojik işlemler (gürültü azaltma)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSING, kernel)
        
        # Hareket alanını hesapla
        motion_area = cv2.countNonZero(thresh)
        
        return thresh, motion_area
    
    def detect_motion_three_frame(self, current_frame):
        """Üç frame differencing ile gelişmiş hareket algılama"""
        if len(self.frame_history) < 3:
            return None, 0
        
        # Son üç frame'i al
        frame1 = self.frame_history[-3]
        frame2 = self.frame_history[-2] 
        frame3 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # İki fark hesapla
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)
        
        # İki farkın AND'i (daha kararlı hareket tespiti)
        motion_mask = cv2.bitwise_and(diff1, diff2)
        
        # Threshold
        _, thresh = cv2.threshold(motion_mask, 30, 255, cv2.THRESH_BINARY)
        
        # Morfolojik işlemler
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSING, kernel)
        
        motion_area = cv2.countNonZero(thresh)
        
        return thresh, motion_area

def ornek_1_basit_hareket_algilama():
    """
    Örnek 1: Basit hareket algılama
    """
    print("\n🎯 Örnek 1: Basit Hareket Algılama")
    print("=" * 40)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
        print("📹 Test videosu kullanılıyor")
    else:
        print("📷 Webcam kullanılıyor - önünde hareket edin!")
    
    detector = MotionDetector()
    
    # Parametreler
    motion_threshold = 30
    sensitivity = 1000
    
    print("\n🎮 Kontroller:")
    print("t: Threshold ayarla")
    print("s: Sensitivity ayarla")
    print("ESC: Çıkış")
    
    motion_detected = False
    last_motion_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        detector.add_frame(frame)
        
        # Hareket algılama
        motion_mask, motion_area = detector.detect_motion_simple(frame, motion_threshold)
        
        if motion_mask is not None:
            # Hareket durumu kontrol et
            if motion_area > sensitivity:
                motion_detected = True
                last_motion_time = time.time()
            else:
                # 2 saniye sonra hareket yok
                if time.time() - last_motion_time > 2.0:
                    motion_detected = False
            
            # Görselleştirme
            display_frame = frame.copy()
            
            # Hareket durumu
            status_text = "HAREKET TESPİT EDİLDİ!" if motion_detected else "Hareket Yok"
            status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
            
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, f"Hareket Alanı: {motion_area}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Threshold: {motion_threshold}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Sensitivity: {sensitivity}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Hareket maskesi (küçük pencere)
            motion_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            motion_small = cv2.resize(motion_display, (160, 120))
            
            # Ana frame'in köşesine yerleştir
            h, w = display_frame.shape[:2]
            display_frame[h-130:h-10, w-170:w-10] = motion_small
            
            # Çerçeve çiz
            cv2.rectangle(display_frame, (w-170, h-130), (w-10, h-10), (255, 255, 255), 2)
            cv2.putText(display_frame, "Motion Mask", (w-165, h-135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Basit Hareket Algılama', display_frame)
        else:
            cv2.imshow('Basit Hareket Algılama', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('t'):
            new_threshold = input(f"Yeni threshold (mevcut: {motion_threshold}): ")
            try:
                motion_threshold = max(1, min(255, int(new_threshold)))
            except:
                print("Geçersiz değer!")
        elif key == ord('s'):
            new_sensitivity = input(f"Yeni sensitivity (mevcut: {sensitivity}): ")
            try:
                sensitivity = max(100, int(new_sensitivity))
            except:
                print("Geçersiz değer!")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_optical_flow():
    """
    Örnek 2: Optical Flow ile hareket vektörleri
    """
    print("\n🎯 Örnek 2: Optical Flow")
    print("=" * 30)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # Lucas-Kanade optical flow parametreleri
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Feature detection parametreleri
    feature_params = dict(maxCorners=100,
                         qualityLevel=0.3,
                         minDistance=7,
                         blockSize=7)
    
    # Renkler (track izleri için)
    colors = np.random.randint(0, 255, (100, 3))
    
    # İlk frame ve özellikler
    ret, old_frame = cap.read()
    if not ret:
        print("❌ Video okunamadı!")
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Track izleri için maske
    mask = np.zeros_like(old_frame)
    
    flow_method = 'lk'  # 'lk' or 'dense'
    
    print("🎮 Kontroller:")
    print("l: Lucas-Kanade Flow")
    print("d: Dense Flow")
    print("r: Reset tracks")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Reset için
                old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else old_gray
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                mask = np.zeros_like(frame) if ret else mask
                continue
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if flow_method == 'lk':
            # Lucas-Kanade Optical Flow
            if p0 is not None:
                # Optical flow hesapla
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
                # İyi noktaları seç
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    # Track çizgileri çiz
                    for i, (tr, to) in enumerate(zip(good_new, good_old)):
                        a, b = tr.ravel().astype(int)
                        c, d = to.ravel().astype(int)
                        mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
                        frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)
                    
                    # Sonucu göster
                    img = cv2.add(frame, mask)
                    
                    # Bilgi ekle
                    cv2.putText(img, f"Lucas-Kanade Flow", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Points: {len(good_new)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow('Optical Flow', img)
                    
                    # Noktaları güncelle
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    cv2.imshow('Optical Flow', frame)
            else:
                cv2.imshow('Optical Flow', frame)
                
        elif flow_method == 'dense':
            # Dense Optical Flow (Farneback)
            flow = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, None, None)
            if flow is not None and len(flow) > 0:
                # Flow'u HSV'ye çevir
                hsv = np.zeros_like(frame)
                hsv[..., 1] = 255
                
                # Magnitude ve angle hesapla
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                
                # BGR'ye çevir
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                # Orijinal frame ile birleştir
                result = cv2.addWeighted(frame, 0.7, bgr, 0.3, 0)
                
                cv2.putText(result, "Dense Optical Flow", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Optical Flow', result)
            else:
                # Farneback algoritması kullan
                flow = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, None, None, 
                                               pyr_scale=0.5, levels=3, winsize=15,
                                               iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                
                cv2.imshow('Optical Flow', frame)
        
        # Frame'i güncelle
        old_gray = frame_gray.copy()
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('l'):
            flow_method = 'lk'
            # Reset
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)
        elif key == ord('d'):
            flow_method = 'dense'
        elif key == ord('r'):
            # Reset tracks
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_hareket_analizi():
    """
    Örnek 3: Hareket analizi ve istatistikleri
    """
    print("\n🎯 Örnek 3: Hareket Analizi")
    print("=" * 35)
    
    class MotionAnalyzer:
        def __init__(self):
            self.motion_history = deque(maxlen=100)
            self.direction_history = deque(maxlen=50)
            self.speed_history = deque(maxlen=50)
            
        def analyze_motion(self, motion_mask, frame_time=1/30):
            """Hareket analizi yap"""
            # Contour bulma
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # En büyük contour'u al
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area < 500:  # Çok küçük hareketleri filtrele
                return None
            
            # Hareket merkezi
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                return None
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Önceki pozisyonla karşılaştır
            current_pos = (cx, cy)
            
            if self.motion_history:
                prev_pos = self.motion_history[-1]['center']
                
                # Hareket vektörü
                dx = cx - prev_pos[0]
                dy = cy - prev_pos[1]
                
                # Hız (pixel/frame)
                speed = math.sqrt(dx*dx + dy*dy) / frame_time
                
                # Yön (derece)
                direction = math.degrees(math.atan2(dy, dx))
                if direction < 0:
                    direction += 360
                
                self.speed_history.append(speed)
                self.direction_history.append(direction)
            else:
                speed = 0
                direction = 0
            
            motion_data = {
                'center': current_pos,
                'area': area,
                'bbox': (x, y, w, h),
                'speed': speed,
                'direction': direction,
                'contour': largest_contour
            }
            
            self.motion_history.append(motion_data)
            return motion_data
        
        def get_statistics(self):
            """Hareket istatistikleri"""
            if not self.motion_history:
                return {}
            
            areas = [m['area'] for m in self.motion_history]
            speeds = list(self.speed_history)
            directions = list(self.direction_history)
            
            stats = {
                'avg_area': np.mean(areas) if areas else 0,
                'max_area': np.max(areas) if areas else 0,
                'avg_speed': np.mean(speeds) if speeds else 0,
                'max_speed': np.max(speeds) if speeds else 0,
                'dominant_direction': np.mean(directions) if directions else 0,
                'motion_count': len(self.motion_history)
            }
            
            return stats
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    detector = MotionDetector()
    analyzer = MotionAnalyzer()
    
    print("📊 Hareket analizi başlatıldı")
    print("ESC: Çıkış, r: Reset istatistikler")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        detector.add_frame(frame)
        
        # Hareket algılama
        motion_mask, motion_area = detector.detect_motion_simple(frame, 30)
        
        if motion_mask is not None:
            # Hareket analizi
            motion_data = analyzer.analyze_motion(motion_mask)
            
            # Görselleştirme
            display_frame = frame.copy()
            
            if motion_data:
                # Hareket merkezi
                center = motion_data['center']
                cv2.circle(display_frame, center, 10, (0, 255, 0), -1)
                
                # Bounding box
                x, y, w, h = motion_data['bbox']
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Hareket vektörü (son 5 pozisyon)
                if len(analyzer.motion_history) > 1:
                    recent_positions = [m['center'] for m in list(analyzer.motion_history)[-5:]]
                    for i in range(len(recent_positions)-1):
                        cv2.line(display_frame, recent_positions[i], recent_positions[i+1], 
                                (0, 255, 255), 2)
                
                # Anlık bilgiler
                cv2.putText(display_frame, f"Hız: {motion_data['speed']:.1f} px/s", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Yön: {motion_data['direction']:.0f}°", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Alan: {motion_data['area']:.0f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # İstatistikler
            stats = analyzer.get_statistics()
            y_offset = 120
            cv2.putText(display_frame, "İSTATİSTİKLER:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 25
            
            for key, value in stats.items():
                text = f"{key}: {value:.1f}"
                cv2.putText(display_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
            
            cv2.imshow('Hareket Analizi', display_frame)
        else:
            cv2.imshow('Hareket Analizi', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            # İstatistikleri sıfırla
            analyzer = MotionAnalyzer()
            detector = MotionDetector()
            print("📊 İstatistikler sıfırlandı")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_4_hareket_tabanlı_olay_tespiti():
    """
    Örnek 4: Hareket tabanlı olay tespiti
    """
    print("\n🎯 Örnek 4: Hareket Tabanlı Olay Tespiti")
    print("=" * 45)
    
    class EventDetector:
        def __init__(self):
            self.motion_threshold = 2000
            self.idle_timeout = 5.0  # saniye
            self.last_motion_time = 0
            self.event_count = 0
            self.recording = False
            self.recorded_frames = []
            
        def detect_events(self, motion_area, frame):
            """Olay tespiti"""
            current_time = time.time()
            events = []
            
            if motion_area > self.motion_threshold:
                if current_time - self.last_motion_time > self.idle_timeout:
                    # Yeni hareket olayı
                    events.append('MOTION_START')
                    self.event_count += 1
                    self.start_recording()
                
                self.last_motion_time = current_time
                
            else:
                # Hareket yok
                if self.recording and current_time - self.last_motion_time > 2.0:
                    events.append('MOTION_END')
                    self.stop_recording()
            
            # Kayıt sırasında frame'leri sakla
            if self.recording:
                self.recorded_frames.append(frame.copy())
                
                # Çok fazla frame biriktirme
                if len(self.recorded_frames) > 150:  # ~5 saniye
                    self.recorded_frames = self.recorded_frames[-100:]
            
            return events
        
        def start_recording(self):
            """Kayıt başlat"""
            self.recording = True
            self.recorded_frames = []
            print(f"🎬 Olay kaydı başladı (#{self.event_count})")
        
        def stop_recording(self):
            """Kayıt durdur ve kaydet"""
            if self.recording and self.recorded_frames:
                self.save_event_video()
            self.recording = False
            self.recorded_frames = []
            print("⏹️ Olay kaydı durduruldu")
        
        def save_event_video(self):
            """Olay videosunu kaydet"""
            if not self.recorded_frames:
                return
            
            filename = f"event_{self.event_count}_{int(time.time())}.avi"
            h, w = self.recorded_frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, 15.0, (w, h))
            
            for frame in self.recorded_frames:
                out.write(frame)
            
            out.release()
            print(f"💾 Olay videosu kaydedildi: {filename}")
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    detector = MotionDetector()
    event_detector = EventDetector()
    
    print("🚨 Olay tespiti başlatıldı")
    print("Hareket algılandığında otomatik kayıt başlar")
    print("ESC: Çıkış, t: Threshold ayarla")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        detector.add_frame(frame)
        
        # Hareket algılama
        motion_mask, motion_area = detector.detect_motion_simple(frame, 30)
        
        if motion_mask is not None:
            # Olay tespiti
            events = event_detector.detect_events(motion_area, frame)
            
            # Görselleştirme
            display_frame = frame.copy()
            
            # Durum bilgileri
            status_color = (0, 0, 255) if event_detector.recording else (0, 255, 0)
            status_text = "KAYIT YAPILIYOR" if event_detector.recording else "BEKLEMEde"
            
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_frame, f"Hareket Alanı: {motion_area}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Threshold: {event_detector.motion_threshold}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Olay Sayısı: {event_detector.event_count}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if event_detector.recording:
                cv2.putText(display_frame, f"Frame: {len(event_detector.recorded_frames)}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Son idle zamanı
            idle_time = time.time() - event_detector.last_motion_time
            cv2.putText(display_frame, f"İdil Süre: {idle_time:.1f}s", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Kayıt çerçevesi
            if event_detector.recording:
                cv2.rectangle(display_frame, (5, 5), 
                             (display_frame.shape[1]-5, display_frame.shape[0]-5), 
                             (0, 0, 255), 3)
            
            # Olayları göster
            for i, event in enumerate(events):
                cv2.putText(display_frame, f"OLAY: {event}", (10, 220 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Olay Tespiti', display_frame)
        else:
            cv2.imshow('Olay Tespiti', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            # Kayıt devam ediyorsa durdur
            if event_detector.recording:
                event_detector.stop_recording()
            break
        elif key == ord('t'):
            new_threshold = input(f"Yeni threshold (mevcut: {event_detector.motion_threshold}): ")
            try:
                event_detector.motion_threshold = max(100, int(new_threshold))
            except:
                print("Geçersiz değer!")
    
    cap.release()
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🏃 OpenCV Hareket Algılama Demo")
        print("="*50)
        print("1. 🔍 Basit Hareket Algılama")
        print("2. 🌊 Optical Flow")
        print("3. 📊 Hareket Analizi")
        print("4. 🚨 Olay Tespiti")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-4): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_basit_hareket_algilama()
            elif secim == "2":
                ornek_2_optical_flow()
            elif secim == "3":
                ornek_3_hareket_analizi()
            elif secim == "4":
                ornek_4_hareket_tabanlı_olay_tespiti()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🏃 OpenCV Hareket Algılama")
    print("Bu modül video hareket algılama ve analiz tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (önerilen)")
    print("\n🎬 Not: Olay tespiti örneği video dosyaları oluşturur.")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Hareket algılama gürültüye duyarlıdır
# 2. Threshold değerleri ortama göre ayarlanmalı
# 3. Optical flow hesaplama yoğun işlemdir
# 4. Olay tespiti disk alanı kullanır
# 5. Webcam kullanımı daha gerçekçi sonuçlar verir