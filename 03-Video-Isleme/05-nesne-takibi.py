#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Nesne Takibi - OpenCV Object Tracking
======================================

Bu modül nesne takip algoritmalarını kapsar:
- Template matching ve correlation tracking
- Mean Shift ve CamShift algoritmaları
- Kalman Filter ile prediction
- Modern OpenCV tracker'ları (CSRT, KCF, etc.)
- Multi-object tracking

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from collections import deque
import math

class ObjectTracker:
    """Nesne takip sınıfı"""
    
    def __init__(self, tracker_type='CSRT'):
        self.tracker_type = tracker_type
        self.tracker = None
        self.is_initialized = False
        self.tracking_box = None
        self.tracking_history = deque(maxlen=50)
        
    def create_tracker(self, tracker_type):
        """Tracker oluştur"""
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif tracker_type == 'BOOSTING':
            return cv2.TrackerBoosting_create()
        elif tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        else:
            return cv2.TrackerCSRT_create()
    
    def initialize(self, frame, bbox):
        """Tracker'ı başlat"""
        self.tracker = self.create_tracker(self.tracker_type)
        self.is_initialized = self.tracker.init(frame, bbox)
        self.tracking_box = bbox
        if self.is_initialized:
            center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
            self.tracking_history.append(center)
        return self.is_initialized
    
    def update(self, frame):
        """Tracker'ı güncelle"""
        if not self.is_initialized or self.tracker is None:
            return False, None
        
        success, bbox = self.tracker.update(frame)
        if success:
            self.tracking_box = bbox
            center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
            self.tracking_history.append(center)
        
        return success, bbox
    
    def reset(self):
        """Tracker'ı sıfırla"""
        self.tracker = None
        self.is_initialized = False
        self.tracking_box = None
        self.tracking_history.clear()

class MultiObjectTracker:
    """Çoklu nesne takip sınıfı"""
    
    def __init__(self, tracker_type='CSRT'):
        self.trackers = []
        self.tracker_type = tracker_type
        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), 
                       (255,0,255), (0,255,255), (128,0,128), (255,165,0)]
    
    def add_tracker(self, frame, bbox):
        """Yeni tracker ekle"""
        tracker = ObjectTracker(self.tracker_type)
        if tracker.initialize(frame, bbox):
            self.trackers.append(tracker)
            return True
        return False
    
    def update_all(self, frame):
        """Tüm tracker'ları güncelle"""
        results = []
        active_trackers = []
        
        for tracker in self.trackers:
            success, bbox = tracker.update(frame)
            if success:
                results.append(bbox)
                active_trackers.append(tracker)
            else:
                results.append(None)
        
        # Başarısız tracker'ları kaldır
        self.trackers = active_trackers
        return results
    
    def clear_all(self):
        """Tüm tracker'ları temizle"""
        self.trackers.clear()

def ornek_1_temel_tracking():
    """
    Örnek 1: Temel nesne takibi
    """
    print("\n🎯 Örnek 1: Temel Nesne Takibi")
    print("=" * 35)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
        print("📹 Test videosu kullanılıyor")
    else:
        print("📷 Webcam kullanılıyor")
    
    tracker = ObjectTracker('CSRT')
    selecting = False
    selection_box = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selection_box
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            selection_box = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            selection_box[2] = abs(x - selection_box[0])
            selection_box[3] = abs(y - selection_box[1])
            selection_box[0] = min(x, selection_box[0]) if x < selection_box[0] else selection_box[0]
            selection_box[1] = min(y, selection_box[1]) if y < selection_box[1] else selection_box[1]
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            if selection_box[2] > 10 and selection_box[3] > 10:
                # Tracker'ı başlat
                frame = param
                if tracker.initialize(frame, tuple(selection_box)):
                    print(f"✅ Tracker başlatıldı: {tracker.tracker_type}")
                else:
                    print("❌ Tracker başlatılamadı!")
            selection_box = None
    
    cv2.namedWindow('Nesne Takibi')
    
    print("\n🎮 Kontroller:")
    print("🖱️ Sol tık + sürükle: Nesne seç")
    print("r: Tracker sıfırla")
    print("1-6: Tracker türü değiştir")
    print("ESC: Çıkış")
    
    tracker_types = ['CSRT', 'KCF', 'MOSSE', 'MIL', 'BOOSTING', 'GOTURN']
    current_tracker_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        display_frame = frame.copy()
        
        # Mouse callback için frame'i gönder
        cv2.setMouseCallback('Nesne Takibi', mouse_callback, frame)
        
        # Tracker güncellemesi
        if tracker.is_initialized:
            success, bbox = tracker.update(frame)
            
            if success:
                # Bounding box çiz
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Merkez nokta
                center = (x + w//2, y + h//2)
                cv2.circle(display_frame, center, 5, (0, 255, 0), -1)
                
                # Takip geçmişi
                if len(tracker.tracking_history) > 1:
                    points = list(tracker.tracking_history)
                    for i in range(len(points)-1):
                        cv2.line(display_frame, points[i], points[i+1], (255, 0, 0), 2)
                
                # Bilgi metni
                cv2.putText(display_frame, f"Tracker: {tracker.tracker_type}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Takip Aktif", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, f"Tracker: {tracker.tracker_type}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Takip Kayboldu!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, f"Tracker: {tracker_types[current_tracker_idx]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Nesne seçin (sürükleyerek)", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Seçim kutusu çiz
        if selecting and selection_box:
            x, y, w, h = selection_box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Mevcut tracker türleri
        y_offset = display_frame.shape[0] - 120
        cv2.putText(display_frame, "Tracker Türleri:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, t_type in enumerate(tracker_types):
            color = (0, 255, 0) if i == current_tracker_idx else (255, 255, 255)
            cv2.putText(display_frame, f"{i+1}: {t_type}", (10, y_offset + 20 + i*15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow('Nesne Takibi', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            tracker.reset()
            print("🔄 Tracker sıfırlandı")
        elif key >= ord('1') and key <= ord('6'):
            idx = key - ord('1')
            if idx < len(tracker_types):
                current_tracker_idx = idx
                new_type = tracker_types[idx]
                if tracker.is_initialized:
                    # Mevcut tracker'ı yeni türle değiştir
                    bbox = tracker.tracking_box
                    tracker.reset()
                    tracker.tracker_type = new_type
                    tracker.initialize(frame, bbox)
                else:
                    tracker.tracker_type = new_type
                print(f"🔄 Tracker türü: {new_type}")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_meanshift_tracking():
    """
    Örnek 2: Mean Shift tracking
    """
    print("\n🎯 Örnek 2: Mean Shift Tracking")
    print("=" * 35)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # Mean shift parametreleri
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
    # Seçim değişkenleri
    selecting = False
    selection_box = None
    track_window = None
    roi_hist = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selection_box, track_window, roi_hist
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            selection_box = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            selection_box[2] = abs(x - selection_box[0])
            selection_box[3] = abs(y - selection_box[1])
            selection_box[0] = min(x, selection_box[0]) if x < selection_box[0] else selection_box[0]
            selection_box[1] = min(y, selection_box[1]) if y < selection_box[1] else selection_box[1]
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            frame = param
            if selection_box and selection_box[2] > 10 and selection_box[3] > 10:
                # ROI seç ve histogram hesapla
                x, y, w, h = selection_box
                track_window = (x, y, w, h)
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                print("✅ Mean Shift tracker başlatıldı")
            selection_box = None
    
    cv2.namedWindow('Mean Shift Tracking')
    
    print("🖱️ Sol tık + sürükle: ROI seç")
    print("r: Reset, ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        display_frame = frame.copy()
        cv2.setMouseCallback('Mean Shift Tracking', mouse_callback, frame)
        
        # Mean shift tracking
        if track_window and roi_hist is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            # Mean shift uygula
            ret, track_window = cv2.meanShift(dst, track_window, term_criteria)
            
            # Sonucu göster
            x, y, w, h = track_window
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Back projection'ı göster (küçük pencere)
            dst_colored = cv2.applyColorMap(dst, cv2.COLORMAP_JET)
            dst_small = cv2.resize(dst_colored, (160, 120))
            
            frame_h, frame_w = display_frame.shape[:2]
            display_frame[10:130, frame_w-170:frame_w-10] = dst_small
            cv2.rectangle(display_frame, (frame_w-170, 10), (frame_w-10, 130), (255, 255, 255), 2)
            cv2.putText(display_frame, "Back Projection", (frame_w-165, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(display_frame, "Mean Shift Active", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "ROI seçin (sürükleyerek)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Seçim kutusu
        if selecting and selection_box:
            x, y, w, h = selection_box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        cv2.imshow('Mean Shift Tracking', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            track_window = None
            roi_hist = None
            print("🔄 Mean Shift sıfırlandı")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_camshift_tracking():
    """
    Örnek 3: CamShift tracking (adaptive size)
    """
    print("\n🎯 Örnek 3: CamShift Tracking")
    print("=" * 30)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # CamShift parametreleri
    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
    # Tracking değişkenleri
    selecting = False
    selection_box = None
    track_window = None
    roi_hist = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selection_box, track_window, roi_hist
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            selection_box = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            selection_box[2] = abs(x - selection_box[0])
            selection_box[3] = abs(y - selection_box[1])
            selection_box[0] = min(x, selection_box[0]) if x < selection_box[0] else selection_box[0]
            selection_box[1] = min(y, selection_box[1]) if y < selection_box[1] else selection_box[1]
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            frame = param
            if selection_box and selection_box[2] > 10 and selection_box[3] > 10:
                x, y, w, h = selection_box
                track_window = (x, y, w, h)
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                print("✅ CamShift tracker başlatıldı")
            selection_box = None
    
    cv2.namedWindow('CamShift Tracking')
    
    print("🖱️ Sol tık + sürükle: ROI seç")
    print("r: Reset, ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        display_frame = frame.copy()
        cv2.setMouseCallback('CamShift Tracking', mouse_callback, frame)
        
        # CamShift tracking
        if track_window and roi_hist is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            # CamShift uygula
            ret, track_window = cv2.CamShift(dst, track_window, term_criteria)
            
            # Elips çiz (CamShift yönelim bilgisi de verir)
            if ret[1][0] > 0 and ret[1][1] > 0:
                # Elips parametreleri
                center = (int(ret[0][0]), int(ret[0][1]))
                axes = (int(ret[1][0]/2), int(ret[1][1]/2))
                angle = ret[2]
                
                # Elips çiz
                cv2.ellipse(display_frame, center, axes, angle, 0, 360, (0, 255, 0), 2)
                
                # Merkez nokta
                cv2.circle(display_frame, center, 5, (0, 255, 0), -1)
                
                # Yönelim çizgisi
                end_x = int(center[0] + axes[0] * math.cos(math.radians(angle)))
                end_y = int(center[1] + axes[1] * math.sin(math.radians(angle)))
                cv2.line(display_frame, center, (end_x, end_y), (255, 0, 0), 2)
                
                # Bilgi metni
                cv2.putText(display_frame, f"Açı: {angle:.1f}°", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Boyut: {axes[0]}x{axes[1]}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Bounding box de çiz
            x, y, w, h = track_window
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
            
            cv2.putText(display_frame, "CamShift Active", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Adaptive Size & Rotation", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "ROI seçin (sürükleyerek)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Seçim kutusu
        if selecting and selection_box:
            x, y, w, h = selection_box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        cv2.imshow('CamShift Tracking', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            track_window = None
            roi_hist = None
            print("🔄 CamShift sıfırlandı")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_4_multi_object_tracking():
    """
    Örnek 4: Çoklu nesne takibi
    """
    print("\n🎯 Örnek 4: Çoklu Nesne Takibi")
    print("=" * 35)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    multi_tracker = MultiObjectTracker('CSRT')
    
    # Seçim değişkenleri
    selecting = False
    selection_box = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selection_box
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            selection_box = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            selection_box[2] = abs(x - selection_box[0])
            selection_box[3] = abs(y - selection_box[1])
            selection_box[0] = min(x, selection_box[0]) if x < selection_box[0] else selection_box[0]
            selection_box[1] = min(y, selection_box[1]) if y < selection_box[1] else selection_box[1]
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            frame = param
            if selection_box and selection_box[2] > 10 and selection_box[3] > 10:
                # Yeni tracker ekle
                bbox = tuple(selection_box)
                if multi_tracker.add_tracker(frame, bbox):
                    print(f"✅ Tracker #{len(multi_tracker.trackers)} eklendi")
                else:
                    print("❌ Tracker eklenemedi!")
            selection_box = None
    
    cv2.namedWindow('Multi Object Tracking')
    
    print("\n🎮 Kontroller:")
    print("🖱️ Sol tık + sürükle: Yeni nesne ekle")
    print("c: Tüm tracker'ları temizle")
    print("1-3: Tracker türü değiştir")
    print("ESC: Çıkış")
    
    tracker_types = ['CSRT', 'KCF', 'MOSSE']
    current_tracker_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        display_frame = frame.copy()
        cv2.setMouseCallback('Multi Object Tracking', mouse_callback, frame)
        
        # Tüm tracker'ları güncelle
        results = multi_tracker.update_all(frame)
        
        # Sonuçları çiz
        for i, bbox in enumerate(results):
            if bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                color = multi_tracker.colors[i % len(multi_tracker.colors)]
                
                # Bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Tracker ID
                cv2.putText(display_frame, f"#{i+1}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Merkez nokta
                center = (x + w//2, y + h//2)
                cv2.circle(display_frame, center, 3, color, -1)
                
                # Takip geçmişi
                if i < len(multi_tracker.trackers):
                    tracker = multi_tracker.trackers[i]
                    if len(tracker.tracking_history) > 1:
                        points = list(tracker.tracking_history)
                        for j in range(len(points)-1):
                            cv2.line(display_frame, points[j], points[j+1], color, 1)
        
        # Bilgi paneli
        cv2.putText(display_frame, f"Aktif Tracker: {len(multi_tracker.trackers)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Tür: {tracker_types[current_tracker_idx]}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(multi_tracker.trackers) == 0:
            cv2.putText(display_frame, "Nesne seçin (sürükleyerek)", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Seçim kutusu
        if selecting and selection_box:
            x, y, w, h = selection_box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Tracker türleri
        y_offset = display_frame.shape[0] - 80
        for i, t_type in enumerate(tracker_types):
            color = (0, 255, 0) if i == current_tracker_idx else (255, 255, 255)
            cv2.putText(display_frame, f"{i+1}: {t_type}", (10, y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow('Multi Object Tracking', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):
            multi_tracker.clear_all()
            print("🗑️ Tüm tracker'lar temizlendi")
        elif key >= ord('1') and key <= ord('3'):
            idx = key - ord('1')
            if idx < len(tracker_types):
                current_tracker_idx = idx
                multi_tracker.tracker_type = tracker_types[idx]
                print(f"🔄 Yeni tracker türü: {tracker_types[idx]}")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_5_kalman_filter_tracking():
    """
    Örnek 5: Kalman Filter ile predictive tracking
    """
    print("\n🎯 Örnek 5: Kalman Filter Tracking")
    print("=" * 40)
    
    class KalmanTracker:
        def __init__(self):
            # Kalman Filter (4 durum: x, y, vx, vy)
            self.kalman = cv2.KalmanFilter(4, 2)
            self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                     [0, 1, 0, 0]], np.float32)
            self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                    [0, 1, 0, 1],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32)
            
            # Gürültü matrisleri
            self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
            self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
            self.kalman.errorCovPost = 0.1 * np.eye(4, dtype=np.float32)
            
            self.initialized = False
            self.predictions = deque(maxlen=30)
            self.measurements = deque(maxlen=30)
            
        def initialize(self, x, y):
            """Kalman filter'ı başlat"""
            self.kalman.statePre = np.array([x, y, 0, 0], dtype=np.float32)
            self.initialized = True
            
        def predict(self):
            """Tahmin yap"""
            if not self.initialized:
                return None
            prediction = self.kalman.predict()
            pred_point = (int(prediction[0]), int(prediction[1]))
            self.predictions.append(pred_point)
            return pred_point
            
        def update(self, x, y):
            """Ölçüm ile güncelle"""
            if not self.initialized:
                return
            measurement = np.array([[x], [y]], dtype=np.float32)
            self.kalman.correct(measurement)
            self.measurements.append((x, y))
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    # Tracker ve Kalman filter
    opencv_tracker = ObjectTracker('CSRT')
    kalman_tracker = KalmanTracker()
    
    # Seçim değişkenleri
    selecting = False
    selection_box = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selection_box
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            selection_box = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            selection_box[2] = abs(x - selection_box[0])
            selection_box[3] = abs(y - selection_box[1])
            selection_box[0] = min(x, selection_box[0]) if x < selection_box[0] else selection_box[0]
            selection_box[1] = min(y, selection_box[1]) if y < selection_box[1] else selection_box[1]
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            frame = param
            if selection_box and selection_box[2] > 10 and selection_box[3] > 10:
                bbox = tuple(selection_box)
                if opencv_tracker.initialize(frame, bbox):
                    # Kalman filter'ı da başlat
                    center_x = selection_box[0] + selection_box[2]//2
                    center_y = selection_box[1] + selection_box[3]//2
                    kalman_tracker.initialize(center_x, center_y)
                    print("✅ Kalman Filter tracking başlatıldı")
            selection_box = None
    
    cv2.namedWindow('Kalman Filter Tracking')
    
    print("🖱️ Sol tık + sürükle: Nesne seç")
    print("r: Reset, ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        
        display_frame = frame.copy()
        cv2.setMouseCallback('Kalman Filter Tracking', mouse_callback, frame)
        
        # Kalman prediction
        predicted_point = kalman_tracker.predict()
        
        # OpenCV tracker güncelleme
        if opencv_tracker.is_initialized:
            success, bbox = opencv_tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                center = (x + w//2, y + h//2)
                
                # Kalman filter'ı gerçek ölçümle güncelle
                kalman_tracker.update(center[0], center[1])
                
                # OpenCV tracker sonucu (yeşil)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display_frame, center, 5, (0, 255, 0), -1)
                cv2.putText(display_frame, "Gerçek", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Kalman prediction (mavi)
                if predicted_point:
                    cv2.circle(display_frame, predicted_point, 8, (255, 0, 0), 2)
                    cv2.putText(display_frame, "Tahmin", (predicted_point[0]+10, predicted_point[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Prediction ile gerçek arasındaki fark
                    distance = math.sqrt((center[0] - predicted_point[0])**2 + 
                                       (center[1] - predicted_point[1])**2)
                    cv2.line(display_frame, center, predicted_point, (255, 255, 0), 1)
                    
                    cv2.putText(display_frame, f"Hata: {distance:.1f}px", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Geçmiş tahminler (mavi çizgi)
                if len(kalman_tracker.predictions) > 1:
                    points = list(kalman_tracker.predictions)
                    for i in range(len(points)-1):
                        cv2.line(display_frame, points[i], points[i+1], (255, 0, 0), 1)
                
                # Gerçek ölçümler (yeşil çizgi)
                if len(kalman_tracker.measurements) > 1:
                    points = list(kalman_tracker.measurements)
                    for i in range(len(points)-1):
                        cv2.line(display_frame, points[i], points[i+1], (0, 255, 0), 2)
                
                cv2.putText(display_frame, "Kalman Filter Tracking", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Mavi: Tahmin, Yeşil: Gerçek", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Predictions: {len(kalman_tracker.predictions)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            else:
                # Tracking kaybedildi, sadece prediction göster
                if predicted_point:
                    cv2.circle(display_frame, predicted_point, 8, (0, 0, 255), 2)
                    cv2.putText(display_frame, "Sadece Tahmin", (predicted_point[0]+10, predicted_point[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.putText(display_frame, "Tracking Kayboldu - Kalman Aktif", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Nesne seçin (sürükleyerek)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Seçim kutusu
        if selecting and selection_box:
            x, y, w, h = selection_box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        cv2.imshow('Kalman Filter Tracking', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            opencv_tracker.reset()
            kalman_tracker = KalmanTracker()
            print("🔄 Trackers sıfırlandı")
    
    cap.release()
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🎯 OpenCV Nesne Takibi Demo")
        print("="*50)
        print("1. 🎯 Temel Nesne Takibi (CSRT, KCF, MOSSE)")
        print("2. 📍 Mean Shift Tracking")
        print("3. 🔄 CamShift Tracking (Adaptive)")
        print("4. 🎯🎯 Çoklu Nesne Takibi")
        print("5. 🔮 Kalman Filter Tracking")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-5): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_temel_tracking()
            elif secim == "2":
                ornek_2_meanshift_tracking()
            elif secim == "3":
                ornek_3_camshift_tracking()
            elif secim == "4":
                ornek_4_multi_object_tracking()
            elif secim == "5":
                ornek_5_kalman_filter_tracking()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-5 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🎯 OpenCV Nesne Takibi")
    print("Bu modül çeşitli nesne takip algoritmalarını öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (önerilen)")
    print("\n📝 Not: Bazı tracker'lar (GOTURN) ek model dosyaları gerektirebilir.")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. CSRT ve KCF en güvenilir modern tracker'lardır
# 2. Mean Shift renk tabanlı takip yapar
# 3. CamShift boyut ve yönelim değişikliklerini takip eder
# 4. Kalman Filter tahminli takip yapar
# 5. Multi-tracking çok nesne için uygundur