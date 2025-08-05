#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Alıştırma 3: İleri Video İşleme ve Nesne Takibi
===============================================

Bu alıştırmada çoklu nesne tespiti, takibi ve trajectory analizi
yapan ileri seviye bir video işleme sistemi geliştireceksiniz.

GÖREV: Aşağıdaki TODO kısımlarını tamamlayın!

Yazan: [ADINIZI BURAYA YAZIN]
Tarih: 2024
"""

import cv2
import numpy as np
import time
import json
from collections import defaultdict, deque
import math

class MultiObjectTracker:
    """Multi-object tracking ve analiz sistemi"""
    
    def __init__(self):
        # Video capture
        self.cap = None
        self.is_running = False
        
        # TODO 1: Color detection ranges tanımlayın
        # HSV formatında renk aralıkları (lower, upper)
        self.color_ranges = {
            'red': {'lower': None, 'upper': None},     # BURAYA HSV değerleri yazın
            'blue': {'lower': None, 'upper': None},    # BURAYA HSV değerleri yazın  
            'green': {'lower': None, 'upper': None},   # BURAYA HSV değerleri yazın
            'yellow': {'lower': None, 'upper': None}   # BURAYA HSV değerleri yazın
        }
        
        # TODO 2: Detection parameters
        self.min_object_area = 500       # Minimum object area
        self.max_object_area = 50000     # Maximum object area
        self.min_solidity = 0.3          # Minimum solidity ratio
        
        # TODO 3: Tracking parameters
        self.max_tracking_distance = 50  # Maximum distance for ID assignment
        self.max_disappeared_frames = 10 # Frames before removing object
        
        # Object tracking data
        self.next_object_id = 0
        self.tracked_objects = {}        # object_id: ObjectInfo
        self.disappeared_objects = {}    # object_id: frame_count
        
        # TODO 4: Trajectory management
        self.trajectories = None  # BURAYA trajectory storage yazın (defaultdict + deque)
        self.max_trajectory_length = 100
        
        # Performance tracking
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.start_time = time.time()
        
        # UI Windows
        self.show_hsv_controls = True
        self.show_trajectories = True
        self.show_statistics = True
        
        # Statistics
        self.detection_stats = {
            'total_objects_detected': 0,
            'active_tracks': 0,
            'lost_tracks': 0
        }
    
    def initialize_color_ranges(self):
        """
        TODO 5: Default HSV color ranges initialize et
        
        HSV format: Hue (0-179), Saturation (0-255), Value (0-255)
        """
        # TODO 5a: Kırmızı renk aralığı
        # İpucu: Kırmızı HSV'de 0-10 ve 170-179 aralığında
        # self.color_ranges['red'] = {'lower': (0, 50, 50), 'upper': (10, 255, 255)}
        
        # TODO 5b: Mavi renk aralığı
        # İpucu: Mavi HSV'de yaklaşık 100-130 aralığında
        # self.color_ranges['blue'] = ?
        
        # TODO 5c: Yeşil renk aralığı  
        # İpucu: Yeşil HSV'de yaklaşık 40-80 aralığında
        # self.color_ranges['green'] = ?
        
        # TODO 5d: Sarı renk aralığı
        # İpucu: Sarı HSV'de yaklaşık 20-30 aralığında
        # self.color_ranges['yellow'] = ?
        
        print("✅ Color ranges initialized")
    
    def create_hsv_trackbars(self):
        """
        TODO 6: HSV ayarlama için trackbar interface oluştur
        """
        # TODO 6a: HSV control window oluştur
        window_name = "HSV Controls"
        # cv2.namedWindow ile pencere oluşturun
        # BURAYA KOD YAZIN
        
        # TODO 6b: Her renk için trackbar'lar oluştur
        # Her renk için H_min, H_max, S_min, S_max, V_min, V_max trackbar'ları
        # cv2.createTrackbar kullanın
        # BURAYA KOD YAZIN
        
        print("✅ HSV trackbars created")
    
    def detect_colored_objects(self, frame):
        """
        TODO 7: Renkli nesneleri tespit et
        
        Args:
            frame: BGR frame
            
        Returns:
            list: Detected objects with color, contour, centroid
        """
        detected_objects = []
        
        # TODO 7a: Frame'i HSV'ye çevir
        hsv = None  # BURAYA cv2.cvtColor yazın
        
        # TODO 7b: Her renk için detection yap
        for color_name, color_range in self.color_ranges.items():
            if color_range['lower'] is None:
                continue
                
            # TODO 7c: Color mask oluştur
            # cv2.inRange kullanın
            mask = None  # BURAYA KOD YAZIN
            
            # TODO 7d: Morphological operations (noise reduction)
            # Önce opening, sonra closing uygulayın
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # BURAYA morphologyEx işlemleri yazın
            
            # TODO 7e: Contour detection
            contours, _ = None, None  # BURAYA cv2.findContours yazın
            
            # TODO 7f: Valid objects filtrele
            for contour in contours or []:
                if self.is_valid_object(contour):
                    centroid = self.calculate_centroid(contour)
                    detected_objects.append({
                        'color': color_name,
                        'contour': contour,
                        'centroid': centroid,
                        'area': cv2.contourArea(contour)
                    })
        
        return detected_objects
    
    def is_valid_object(self, contour):
        """
        TODO 8: Object validation
        
        Args:
            contour: OpenCV contour
            
        Returns:
            bool: True if valid object
        """
        # TODO 8a: Area kontrolü
        area = cv2.contourArea(contour)
        if area < self.min_object_area or area > self.max_object_area:
            return False
        
        # TODO 8b: Solidity kontrolü (convex hull'a göre)
        # Solidity = area / convex_hull_area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        
        solidity = area / hull_area
        if solidity < self.min_solidity:
            return False
        
        # TODO 8c: Aspect ratio kontrolü (opsiyonel)
        # Çok uzun/ince nesneleri filtreleyin
        # BURAYA KOD YAZIN (opsiyonel)
        
        return True
    
    def calculate_centroid(self, contour):
        """
        TODO 9: Contour'un merkez noktasını hesapla
        
        Args:
            contour: OpenCV contour
            
        Returns:
            tuple: (cx, cy) center coordinates
        """
        # TODO 9: Moments kullanarak centroid hesapla
        # cv2.moments ve moment hesapları
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return (0, 0)
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def update_tracking(self, detected_objects):
        """
        TODO 10: Object tracking güncelleme
        
        Args:
            detected_objects: List of detected objects this frame
        """
        # TODO 10a: Eğer tracked object yoksa, tüm detectionları register et
        if len(self.tracked_objects) == 0:
            for obj in detected_objects:
                self.register_object(obj)
            return
        
        # TODO 10b: Distance matrix hesapla
        # Her tracked object ile detected object arasındaki mesafe
        input_centroids = [obj['centroid'] for obj in detected_objects]
        object_centroids = list(self.tracked_objects.values())
        
        if len(input_centroids) == 0:
            # No detections - mark all as disappeared
            # BURAYA disappeared handling yazın
            return
        
        # TODO 10c: Distance matrix oluştur
        distance_matrix = self.compute_distance_matrix(object_centroids, input_centroids)
        
        # TODO 10d: Hungarian algorithm veya simple assignment
        # Basit assignment için minimum distance kullanabilirsiniz
        assignments = self.assign_objects(distance_matrix)
        
        # TODO 10e: Assignments'ları işle
        self.process_assignments(assignments, detected_objects)
    
    def compute_distance_matrix(self, object_centroids, input_centroids):
        """
        TODO 11: Distance matrix hesaplama
        
        Args:
            object_centroids: List of tracked object centroids
            input_centroids: List of detected object centroids
            
        Returns:
            np.array: Distance matrix
        """
        # TODO 11: Euclidean distance matrix
        rows = len(object_centroids)
        cols = len(input_centroids)
        
        distance_matrix = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                # Euclidean distance hesapla
                # BURAYA distance hesaplama yazın
                dx = object_centroids[i]['centroid'][0] - input_centroids[j][0]
                dy = object_centroids[i]['centroid'][1] - input_centroids[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def assign_objects(self, distance_matrix):
        """
        TODO 12: Simple object assignment
        
        Args:
            distance_matrix: Distance matrix
            
        Returns:
            dict: Assignment mapping
        """
        # TODO 12: Basit greedy assignment
        # Her tracked object için en yakın detection'ı bul
        assignments = {}
        used_cols = set()
        
        for row in range(distance_matrix.shape[0]):
            min_col = None
            min_distance = float('inf')
            
            for col in range(distance_matrix.shape[1]):
                if col in used_cols:
                    continue
                
                if distance_matrix[row, col] < min_distance:
                    min_distance = distance_matrix[row, col]
                    min_col = col
            
            # Distance threshold kontrolü
            if min_col is not None and min_distance < self.max_tracking_distance:
                assignments[row] = min_col
                used_cols.add(min_col)
        
        return assignments
    
    def process_assignments(self, assignments, detected_objects):
        """
        TODO 13: Assignment sonuçlarını işle
        
        Args:
            assignments: Object assignments
            detected_objects: Detected objects this frame
        """
        # TODO 13a: Assigned objects'i güncelle
        tracked_ids = list(self.tracked_objects.keys())
        
        for row, col in assignments.items():
            object_id = tracked_ids[row]
            detected_obj = detected_objects[col]
            
            # Update object info
            self.tracked_objects[object_id] = detected_obj
            
            # Reset disappeared counter
            if object_id in self.disappeared_objects:
                del self.disappeared_objects[object_id]
            
            # Update trajectory
            # BURAYA trajectory update yazın
        
        # TODO 13b: Unassigned detections'ı register et
        assigned_cols = set(assignments.values())
        for col, detected_obj in enumerate(detected_objects):
            if col not in assigned_cols:
                self.register_object(detected_obj)
        
        # TODO 13c: Unassigned tracked objects'i disappeared olarak işaretle
        assigned_rows = set(assignments.keys())
        for row, object_id in enumerate(tracked_ids):
            if row not in assigned_rows:
                self.disappeared_objects[object_id] = \
                    self.disappeared_objects.get(object_id, 0) + 1
                
                # Remove if disappeared too long
                if self.disappeared_objects[object_id] > self.max_disappeared_frames:
                    self.deregister_object(object_id)
    
    def register_object(self, detected_obj):
        """
        TODO 14: Yeni object register et
        
        Args:
            detected_obj: Detected object info
        """
        # TODO 14a: New object ID assign et
        object_id = self.next_object_id
        self.tracked_objects[object_id] = detected_obj
        self.next_object_id += 1
        
        # TODO 14b: Trajectory başlat
        # BURAYA trajectory initialization yazın
        
        # TODO 14c: Statistics güncelle
        self.detection_stats['total_objects_detected'] += 1
        
        print(f"🆕 Yeni nesne kaydedildi: ID-{object_id} ({detected_obj['color']})")
    
    def deregister_object(self, object_id):
        """
        TODO 15: Object'i tracking'den çıkar
        
        Args:
            object_id: Object ID to remove
        """
        # TODO 15a: Tracking listesinden çıkar
        if object_id in self.tracked_objects:
            color = self.tracked_objects[object_id]['color']
            del self.tracked_objects[object_id]
            print(f"❌ Nesne kaybedildi: ID-{object_id} ({color})")
        
        # TODO 15b: Disappeared listesinden çıkar
        if object_id in self.disappeared_objects:
            del self.disappeared_objects[object_id]
        
        # TODO 15c: Statistics güncelle
        self.detection_stats['lost_tracks'] += 1
    
    def calculate_movement_stats(self, trajectory):
        """
        TODO 16: Movement statistics hesapla
        
        Args:
            trajectory: List of (x, y) points
            
        Returns:
            dict: Movement statistics
        """
        if len(trajectory) < 2:
            return {'speed': 0, 'distance': 0, 'direction': 0}
        
        # TODO 16a: Total distance hesapla
        total_distance = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        # TODO 16b: Average speed hesapla (pixel/frame)
        time_frames = len(trajectory) - 1
        avg_speed = total_distance / time_frames if time_frames > 0 else 0
        
        # TODO 16c: Overall direction hesapla
        start_point = trajectory[0]
        end_point = trajectory[-1]
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        direction = math.degrees(math.atan2(dy, dx))
        
        return {
            'speed': avg_speed,
            'distance': total_distance,
            'direction': direction
        }
    
    def draw_tracking_visualization(self, frame):
        """
        TODO 17: Tracking sonuçlarını görselleştir
        
        Args:
            frame: Frame to draw on
            
        Returns:
            np.array: Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # TODO 17a: Tracked objects'i çiz
        for object_id, obj_info in self.tracked_objects.items():
            centroid = obj_info['centroid']
            color = obj_info['color']
            contour = obj_info['contour']
            
            # Object contour çiz
            color_bgr = self.get_color_bgr(color)
            cv2.drawContours(vis_frame, [contour], -1, color_bgr, 2)
            
            # Centroid çiz
            cv2.circle(vis_frame, centroid, 5, color_bgr, -1)
            
            # Object ID yazın
            cv2.putText(vis_frame, f"ID-{object_id}", 
                       (centroid[0] - 20, centroid[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        
        # TODO 17b: Trajectories çiz (eğer aktifse)
        if self.show_trajectories:
            # BURAYA trajectory çizim yazın
            pass
        
        # TODO 17c: Statistics overlay
        if self.show_statistics:
            self.draw_statistics_overlay(vis_frame)
        
        return vis_frame
    
    def get_color_bgr(self, color_name):
        """Color name'den BGR değeri"""
        color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255)
        }
        return color_map.get(color_name, (255, 255, 255))
    
    def draw_statistics_overlay(self, frame):
        """
        TODO 18: Statistics overlay çiz
        
        Args:
            frame: Frame to draw on
        """
        # TODO 18a: Current tracking statistics
        active_tracks = len(self.tracked_objects)
        total_detected = self.detection_stats['total_objects_detected']
        lost_tracks = self.detection_stats['lost_tracks']
        
        # TODO 18b: FPS calculation
        current_fps = len(self.fps_history) / (time.time() - self.fps_history[0]) if len(self.fps_history) > 1 else 0
        
        # TODO 18c: Text overlay
        stats_text = [
            f"Active Tracks: {active_tracks}",
            f"Total Detected: {total_detected}", 
            f"Lost Tracks: {lost_tracks}",
            f"FPS: {current_fps:.1f}",
            f"Frame: {self.frame_count}"
        ]
        
        # Draw background
        overlay_height = len(stats_text) * 25 + 20
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, overlay_height), (255, 255, 255), 2)
        
        # Draw text
        for i, text in enumerate(stats_text):
            y = 30 + i * 25
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def handle_keyboard_input(self, key):
        """
        TODO 19: Keyboard input handling
        
        Args:
            key: Pressed key code
        """
        # TODO 19a: HSV controls toggle (H key)
        if key == ord('h') or key == ord('H'):
            # BURAYA HSV window toggle yazın
            pass
        
        # TODO 19b: Trajectory toggle (T key)  
        elif key == ord('t') or key == ord('T'):
            # BURAYA trajectory toggle yazın
            pass
        
        # TODO 19c: Statistics toggle (S key)
        elif key == ord('s') or key == ord('S'):
            # BURAYA statistics toggle yazın
            pass
        
        # TODO 19d: Save configuration (C key)
        elif key == ord('c') or key == ord('C'):
            # BURAYA config save yazın
            pass
        
        # TODO 19e: Reset tracking (R key)
        elif key == ord('r') or key == ord('R'):
            self.reset_tracking()
    
    def reset_tracking(self):
        """
        TODO 20: Tracking sistemini reset et
        """
        # TODO 20a: Tüm tracking data'yı temizle
        self.tracked_objects.clear()
        self.disappeared_objects.clear()
        # BURAYA trajectory clearing yazın
        
        # TODO 20b: Statistics reset
        self.detection_stats = {
            'total_objects_detected': 0,
            'active_tracks': 0,
            'lost_tracks': 0
        }
        
        # TODO 20c: ID counter reset
        self.next_object_id = 0
        
        print("🔄 Tracking sistemi reset edildi")
    
    def run(self):
        """
        TODO 21: Ana çalıştırma fonksiyonu
        """
        print("🎯 Multi-Object Tracking Sistemi")
        print("=" * 40)
        
        # TODO 21a: Webcam başlat
        self.cap = None  # BURAYA cv2.VideoCapture(0) yazın
        
        if not self.cap or not self.cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
        
        # TODO 21b: Initialize systems
        self.initialize_color_ranges()
        self.create_hsv_trackbars()
        
        # TODO 21c: Trajectory storage initialize
        # self.trajectories = defaultdict(lambda: deque(maxlen=self.max_trajectory_length))
        # BURAYA KOD YAZIN
        
        print("📷 Kamera başlatıldı")
        print("🎨 HSV kontrolleri hazır")
        print("\n🎮 Kontroller:")
        print("   H: HSV controls toggle")
        print("   T: Trajectory görünüm")
        print("   S: Statistics toggle")
        print("   C: Configuration kaydet")
        print("   R: Reset tracking")
        print("   Q: Çıkış")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # TODO 21d: Frame oku
                ret, frame = False, None  # BURAYA self.cap.read() yazın
                
                if not ret:
                    print("❌ Frame okunamadı!")
                    break
                
                # TODO 21e: Object detection
                detected_objects = self.detect_colored_objects(frame)
                
                # TODO 21f: Tracking update
                self.update_tracking(detected_objects)
                
                # TODO 21g: Visualization
                vis_frame = self.draw_tracking_visualization(frame)
                
                # TODO 21h: Display
                cv2.imshow('Multi-Object Tracking', vis_frame)
                
                # TODO 21i: FPS tracking
                self.fps_history.append(time.time())
                self.frame_count += 1
                
                # TODO 21j: Keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                else:
                    self.handle_keyboard_input(key)
        
        except KeyboardInterrupt:
            print("\n⚠️ Kullanıcı tarafından durduruldu")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        TODO 22: Temizlik işlemleri
        """
        print("🔄 Sistem kapatılıyor...")
        
        # TODO 22a: Session statistics yazdır
        session_duration = time.time() - self.start_time
        print(f"📊 Session Özeti:")
        print(f"   Süre: {session_duration:.1f} saniye")
        print(f"   Toplam frame: {self.frame_count}")
        print(f"   Tespit edilen nesne: {self.detection_stats['total_objects_detected']}")
        print(f"   Kaybedilen track: {self.detection_stats['lost_tracks']}")
        
        # TODO 22b: Webcam ve windows kapat
        # BURAYA cleanup kodu yazın
        
        self.is_running = False

def main():
    """Ana fonksiyon"""
    # TODO 23: MultiObjectTracker oluştur ve çalıştır
    tracker = MultiObjectTracker()
    tracker.run()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. TODO kısımlarını sırayla tamamlayın
# 2. HSV color ranges'i doğru ayarlayın
# 3. Object validation kriterlerini optimize edin
# 4. Tracking algoritmasını test edin
# 5. Trajectory visualization'ı implement edin

# 🎯 TEST ÖNERİLERİ:
# - Farklı renklerde objeler deneyin
# - Çoklu nesne senaryolarını test edin
# - Occlusion durumlarını kontrol edin
# - Uzun süre çalıştırın (memory test)

# 🚀 BONUS ÖZELLIKLER:
# - Kalman filter integration
# - Behavioral pattern analysis
# - Heatmap generation
# - Export functionality