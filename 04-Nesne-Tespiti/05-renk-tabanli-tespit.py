#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Renk Tabanlı Tespit - OpenCV Color-based Detection
===================================================

Bu modül renk tabanlı nesne tespit yöntemlerini kapsar:
- HSV Color Space ile renk filtreleme
- Multi-color object detection
- Color range calibration
- Color-based object tracking
- Dominant color analysis

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import time
from collections import defaultdict, deque
import math

class ColorDetector:
    """Renk tabanlı tespit sınıfı"""
    
    def __init__(self):
        # Önceden tanımlanmış renk aralıkları (HSV)
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'red2': ([170, 50, 50], [180, 255, 255]),  # Kırmızının diğer yarısı
            'green': ([40, 50, 50], [80, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'yellow': ([20, 50, 50], [40, 255, 255]),
            'orange': ([10, 50, 50], [20, 255, 255]),
            'purple': ([130, 50, 50], [170, 255, 255]),
            'cyan': ([80, 50, 50], [100, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'black': ([0, 0, 0], [180, 255, 50])
        }
        
        self.custom_ranges = {}
        self.detection_history = deque(maxlen=30)
        
    def detect_color_objects(self, frame, color_name, min_area=500):
        """Belirli renkteki nesneleri tespit et"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Renk maskesi oluştur
        if color_name in self.color_ranges:
            lower, upper = self.color_ranges[color_name]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Kırmızı için özel durum (HSV'de kırmızı iki aralıkta)
            if color_name == 'red':
                lower2, upper2 = self.color_ranges['red2']
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask, mask2)
                
        elif color_name in self.custom_ranges:
            lower, upper = self.custom_ranges[color_name]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        else:
            return [], np.zeros_like(frame[:,:,0])
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Contour bulma
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Nesneleri filtrele
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                
                objects.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'center': center,
                    'area': area,
                    'color': color_name
                })
        
        return objects, mask
    
    def detect_multiple_colors(self, frame, colors, min_area=500):
        """Birden fazla rengi aynı anda tespit et"""
        all_objects = []
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for color in colors:
            objects, mask = self.detect_color_objects(frame, color, min_area)
            all_objects.extend(objects)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return all_objects, combined_mask
    
    def calibrate_color_range(self, frame, roi):
        """ROI'dan renk aralığı kalibre et"""
        x, y, w, h = roi
        roi_area = frame[y:y+h, x:x+w]
        
        if roi_area.size == 0:
            return None
        
        hsv_roi = cv2.cvtColor(roi_area, cv2.COLOR_BGR2HSV)
        
        # İstatistiksel analiz
        h_channel = hsv_roi[:,:,0]
        s_channel = hsv_roi[:,:,1]
        v_channel = hsv_roi[:,:,2]
        
        # Ortalama ve standart sapma
        h_mean, h_std = np.mean(h_channel), np.std(h_channel)
        s_mean, s_std = np.mean(s_channel), np.std(s_channel)
        v_mean, v_std = np.mean(v_channel), np.std(v_channel)
        
        # Aralık hesaplama (mean ± 2*std)
        h_range = [max(0, h_mean - 2*h_std), min(179, h_mean + 2*h_std)]
        s_range = [max(0, s_mean - 2*s_std), min(255, s_mean + 2*s_std)]
        v_range = [max(0, v_mean - 2*v_std), min(255, v_mean + 2*v_std)]
        
        lower = [int(h_range[0]), int(s_range[0]), int(v_range[0])]
        upper = [int(h_range[1]), int(s_range[1]), int(v_range[1])]
        
        return (lower, upper)
    
    def get_dominant_colors(self, frame, k=5):
        """Frame'deki dominant renkleri bul"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (150, 150))
        data = small_frame.reshape((-1, 3))
        data = np.float32(data)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        
        # Count pixels in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors = centers[sorted_indices]
        percentages = counts[sorted_indices] / len(data) * 100
        
        return dominant_colors, percentages

class ColorTracker:
    """Renk tabanlı nesne takip sınıfı"""
    
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_distance = 50
        
    def update(self, detected_objects):
        """Tespit edilen nesneleri trackle"""
        if not detected_objects:
            return self.tracked_objects
        
        # Eğer hiç tracked object yoksa hepsini ekle
        if not self.tracked_objects:
            for obj in detected_objects:
                self.tracked_objects[self.next_id] = {
                    'center': obj['center'],
                    'color': obj['color'],
                    'bbox': obj['bbox'],
                    'area': obj['area'],
                    'history': deque([obj['center']], maxlen=20)
                }
                self.next_id += 1
            return self.tracked_objects
        
        # Mevcut objelerle eşleştir
        used_detections = set()
        updated_objects = {}
        
        for track_id, tracked_obj in self.tracked_objects.items():
            best_match = None
            min_distance = float('inf')
            best_idx = -1
            
            for i, detected_obj in enumerate(detected_objects):
                if i in used_detections:
                    continue
                
                # Distance calculation
                dx = tracked_obj['center'][0] - detected_obj['center'][0]
                dy = tracked_obj['center'][1] - detected_obj['center'][1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Color match check
                if (tracked_obj['color'] == detected_obj['color'] and 
                    distance < min_distance and distance < self.max_distance):
                    min_distance = distance
                    best_match = detected_obj
                    best_idx = i
            
            if best_match:
                # Update existing track
                tracked_obj['center'] = best_match['center']
                tracked_obj['bbox'] = best_match['bbox']
                tracked_obj['area'] = best_match['area']
                tracked_obj['history'].append(best_match['center'])
                updated_objects[track_id] = tracked_obj
                used_detections.add(best_idx)
        
        # Add new objects
        for i, detected_obj in enumerate(detected_objects):
            if i not in used_detections:
                updated_objects[self.next_id] = {
                    'center': detected_obj['center'],
                    'color': detected_obj['color'],
                    'bbox': detected_obj['bbox'],
                    'area': detected_obj['area'],
                    'history': deque([detected_obj['center']], maxlen=20)
                }
                self.next_id += 1
        
        self.tracked_objects = updated_objects
        return self.tracked_objects

def ornek_1_color_detection_calibration():
    """
    Örnek 1: Renk tespiti ve kalibrasyon
    """
    print("\n🎯 Örnek 1: Renk Tespiti ve Kalibrasyon")
    print("=" * 45)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = ColorDetector()
    
    # Color selection
    available_colors = list(detector.color_ranges.keys())
    available_colors.remove('red2')  # Remove duplicate red
    current_color_idx = 0
    current_color = available_colors[current_color_idx]
    
    # Calibration mode
    calibration_mode = False
    calibrating = False
    selection_start = None
    selection_roi = None
    
    min_area = 500
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal calibrating, selection_start, selection_roi
        
        if calibration_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                calibrating = True
                selection_start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and calibrating:
                selection_roi = [selection_start[0], selection_start[1], 
                               x - selection_start[0], y - selection_start[1]]
            elif event == cv2.EVENT_LBUTTONUP:
                calibrating = False
                if selection_roi and abs(selection_roi[2]) > 20 and abs(selection_roi[3]) > 20:
                    return True
        return False
    
    cv2.namedWindow('Color Detection')
    cv2.setMouseCallback('Color Detection', mouse_callback)
    
    print("🎨 Renk tespiti")
    print("Kontroller:")
    print("  1-9: Renk değiştir")
    print("  c: Kalibrasyon modu")
    print("  +/-: Min area")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        if calibration_mode:
            # Calibration mode
            cv2.putText(display_frame, "KALIBRASYON MODU", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Renk alanını seçin", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Selection rectangle
            if selection_roi:
                x, y, w, h = selection_roi
                if w < 0:
                    x, w = x + w, -w
                if h < 0:
                    y, h = y + h, -h
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
                # Calibrate if selection finished
                if mouse_callback(cv2.EVENT_LBUTTONUP, 0, 0, 0, None):
                    color_range = detector.calibrate_color_range(frame, (x, y, w, h))
                    if color_range:
                        custom_name = f"custom_{current_color}"
                        detector.custom_ranges[custom_name] = color_range
                        print(f"✅ {custom_name} kalibre edildi: {color_range}")
                        current_color = custom_name
                    selection_roi = None
        
        else:
            # Detection mode
            start_time = time.time()
            
            # Color detection
            objects, mask = detector.detect_color_objects(frame, current_color, min_area)
            
            detection_time = (time.time() - start_time) * 1000
            
            # Draw detected objects
            for i, obj in enumerate(objects):
                contour = obj['contour']
                bbox = obj['bbox']
                center = obj['center']
                area = obj['area']
                
                # Object drawing
                cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
                
                x, y, w, h = bbox
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.circle(display_frame, center, 5, (0, 0, 255), -1)
                
                # Label
                cv2.putText(display_frame, f"{current_color} #{i+1}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Area: {int(area)}", (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Info panel
            cv2.putText(display_frame, f"Color: {current_color}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Objects: {len(objects)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Min Area: {min_area}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Time: {detection_time:.1f}ms", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Color mask preview
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_small = cv2.resize(mask_colored, (160, 120))
            
            h, w = display_frame.shape[:2]
            display_frame[h-130:h-10, w-170:w-10] = mask_small
            cv2.rectangle(display_frame, (w-170, h-130), (w-10, h-10), (255, 255, 255), 2)
            cv2.putText(display_frame, "Color Mask", (w-165, h-135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Available colors list
        colors_x = display_frame.shape[1] - 150
        cv2.putText(display_frame, "COLORS:", (colors_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        for i, color in enumerate(available_colors[:9]):
            color_text = f"{i+1}: {color}"
            text_color = (0, 255, 0) if color == current_color else (255, 255, 255)
            cv2.putText(display_frame, color_text, (colors_x, 55 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        cv2.imshow('Color Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key >= ord('1') and key <= ord('9'):
            idx = key - ord('1')
            if idx < len(available_colors):
                current_color = available_colors[idx]
                print(f"🔄 Color: {current_color}")
        elif key == ord('c'):
            calibration_mode = not calibration_mode
            mode = "CALIBRATION" if calibration_mode else "DETECTION"
            print(f"🔄 Mode: {mode}")
        elif key == ord('+') or key == ord('='):
            min_area = min(2000, min_area + 100)
        elif key == ord('-'):
            min_area = max(100, min_area - 100)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_multi_color_tracking():
    """
    Örnek 2: Çoklu renk takibi
    """
    print("\n🎯 Örnek 2: Çoklu Renk Takibi")
    print("=" * 35)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = ColorDetector()
    tracker = ColorTracker()
    
    # Track multiple colors
    target_colors = ['red', 'green', 'blue', 'yellow']
    
    # Color to BGR mapping for visualization
    color_bgr = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'orange': (0, 165, 255),
        'purple': (128, 0, 128)
    }
    
    print("🌈 Multi-color tracking - farklı renkli objeler gösterin!")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Detect all target colors
        all_objects, combined_mask = detector.detect_multiple_colors(frame, target_colors, 300)
        
        # Update tracker
        tracked_objects = tracker.update(all_objects)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Draw tracked objects
        for track_id, obj in tracked_objects.items():
            center = obj['center']
            bbox = obj['bbox']
            color_name = obj['color']
            history = obj['history']
            
            # Get display color
            display_color = color_bgr.get(color_name, (255, 255, 255))
            
            # Bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), display_color, 2)
            
            # Center point
            cv2.circle(frame, center, 8, display_color, -1)
            cv2.circle(frame, center, 8, (255, 255, 255), 2)
            
            # Track ID and color
            cv2.putText(frame, f"ID:{track_id}", (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)
            cv2.putText(frame, color_name, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)
            
            # Trail (history)
            if len(history) > 1:
                points = list(history)
                for i in range(len(points) - 1):
                    # Fade effect
                    alpha = (i + 1) / len(points)
                    thickness = max(1, int(alpha * 3))
                    cv2.line(frame, points[i], points[i + 1], display_color, thickness)
        
        # Statistics panel
        color_counts = defaultdict(int)
        for obj in tracked_objects.values():
            color_counts[obj['color']] += 1
        
        # Info panel
        cv2.putText(frame, f"Tracked Objects: {len(tracked_objects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Processing: {processing_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Color statistics
        y_offset = 90
        cv2.putText(frame, "COLOR COUNTS:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        for color_name, count in color_counts.items():
            y_offset += 25
            display_color = color_bgr.get(color_name, (255, 255, 255))
            cv2.putText(frame, f"{color_name}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)
        
        # Target colors legend
        legend_x = frame.shape[1] - 120
        cv2.putText(frame, "TARGETS:", (legend_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        for i, color_name in enumerate(target_colors):
            display_color = color_bgr.get(color_name, (255, 255, 255))
            cv2.putText(frame, color_name, (legend_x, 55 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)
        
        cv2.imshow('Multi-Color Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_dominant_color_analysis():
    """
    Örnek 3: Dominant renk analizi
    """
    print("\n🎯 Örnek 3: Dominant Renk Analizi")
    print("=" * 35)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = ColorDetector()
    
    # Analysis parameters
    k_clusters = 5
    update_interval = 10  # frames
    frame_count = 0
    
    # Color analysis results
    dominant_colors = []
    percentages = []
    
    print("🎨 Dominant color analysis")
    print("k: Cluster sayısı değiştir, ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Periodic analysis (expensive operation)
        if frame_count % update_interval == 0:
            start_time = time.time()
            dominant_colors, percentages = detector.get_dominant_colors(frame, k_clusters)
            analysis_time = (time.time() - start_time) * 1000
        
        # Display frame
        display_frame = frame.copy()
        
        # Draw dominant colors panel
        if len(dominant_colors) > 0:
            panel_width = 200
            panel_height = 300
            panel_x = frame.shape[1] - panel_width - 10
            panel_y = 10
            
            # Background
            cv2.rectangle(display_frame, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), (50, 50, 50), -1)
            cv2.rectangle(display_frame, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
            
            # Title
            cv2.putText(display_frame, "DOMINANT COLORS", (panel_x + 10, panel_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Color bars
            bar_height = 30
            bar_spacing = 5
            
            for i, (color, percentage) in enumerate(zip(dominant_colors, percentages)):
                y = panel_y + 40 + i * (bar_height + bar_spacing)
                
                # Color swatch
                color_bgr = tuple(map(int, color))
                cv2.rectangle(display_frame, (panel_x + 10, y), 
                             (panel_x + 50, y + bar_height), color_bgr, -1)
                cv2.rectangle(display_frame, (panel_x + 10, y), 
                             (panel_x + 50, y + bar_height), (255, 255, 255), 1)
                
                # Percentage bar
                bar_width = int((percentage / 100) * 130)
                cv2.rectangle(display_frame, (panel_x + 60, y + 5), 
                             (panel_x + 60 + bar_width, y + bar_height - 5), (0, 255, 0), -1)
                cv2.rectangle(display_frame, (panel_x + 60, y + 5), 
                             (panel_x + 190, y + bar_height - 5), (255, 255, 255), 1)
                
                # Percentage text
                cv2.putText(display_frame, f"{percentage:.1f}%", 
                           (panel_x + 65, y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # RGB values
                cv2.putText(display_frame, f"RGB: {color[2]},{color[1]},{color[0]}", 
                           (panel_x + 10, y + bar_height + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Info panel
        cv2.putText(display_frame, f"K-Clusters: {k_clusters}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Update Interval: {update_interval}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if 'analysis_time' in locals():
            cv2.putText(display_frame, f"Analysis Time: {analysis_time:.1f}ms", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progress indicator
        progress = frame_count % update_interval
        progress_width = int((progress / update_interval) * 200)
        cv2.rectangle(display_frame, (10, 110), (10 + progress_width, 125), (255, 255, 0), -1)
        cv2.rectangle(display_frame, (10, 110), (210, 125), (255, 255, 255), 2)
        cv2.putText(display_frame, "Analysis Progress", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Dominant Color Analysis', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('k'):
            try:
                new_k = int(input(f"Yeni K değeri (mevcut: {k_clusters}): "))
                k_clusters = max(2, min(10, new_k))
                print(f"🔄 K-clusters: {k_clusters}")
            except:
                print("Geçersiz değer!")
    
    cap.release()
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🎨 OpenCV Renk Tabanlı Tespit Demo")
        print("="*50)
        print("1. 🎨 Renk Tespiti ve Kalibrasyon")
        print("2. 🌈 Çoklu Renk Takibi")
        print("3. 📊 Dominant Renk Analizi")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-3): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_color_detection_calibration()
            elif secim == "2":
                ornek_2_multi_color_tracking()
            elif secim == "3":
                ornek_3_dominant_color_analysis()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-3 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🎨 OpenCV Renk Tabanlı Tespit")
    print("Bu modül renk tabanlı nesne tespit tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (önerilen)")
    print("\n📝 Notlar:")
    print("   - HSV renk uzayı daha güvenilir")
    print("   - İyi aydınlatma kritik")
    print("   - Renk kalibrasyonu doğruluğu artırır")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. HSV renk uzayı RGB'den daha kararlı
# 2. Kırmızı renk HSV'de iki aralıkta bulunur
# 3. Morfolojik işlemler gürültü azaltır
# 4. K-means clustering dominant renk analizi için ideal
# 5. Color tracking centroid-based distance kullanır