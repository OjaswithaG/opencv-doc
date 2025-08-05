#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Özellik Tabanlı Tespit - OpenCV Feature-based Detection
========================================================

Bu modül özellik tabanlı nesne tespit yöntemlerini kapsar:
- SIFT (Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF)
- Feature Matching ve Homography
- Contour-based Detection

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from collections import deque
import math

class FeatureDetector:
    """Özellik tabanlı tespit sınıfı"""
    
    def __init__(self):
        self.detectors = {}
        self.matchers = {}
        self.create_detectors()
        
    def create_detectors(self):
        """Farklı özellik detector'larını oluştur"""
        try:
            # SIFT detector
            self.detectors['SIFT'] = cv2.SIFT_create()
            print("✅ SIFT detector oluşturuldu")
        except Exception as e:
            print(f"⚠️ SIFT detector oluşturulamadı: {e}")
        
        try:
            # ORB detector
            self.detectors['ORB'] = cv2.ORB_create(nfeatures=1000)
            print("✅ ORB detector oluşturuldu")
        except Exception as e:
            print(f"⚠️ ORB detector oluşturulamadı: {e}")
        
        try:
            # AKAZE detector
            self.detectors['AKAZE'] = cv2.AKAZE_create()
            print("✅ AKAZE detector oluşturuldu")
        except Exception as e:
            print(f"⚠️ AKAZE detector oluşturulamadı: {e}")
        
        try:
            # BRISK detector
            self.detectors['BRISK'] = cv2.BRISK_create()
            print("✅ BRISK detector oluşturuldu")
        except Exception as e:
            print(f"⚠️ BRISK detector oluşturulamadı: {e}")
        
        # Feature matchers
        self.matchers['BF'] = cv2.BFMatcher()
        self.matchers['FLANN'] = cv2.FlannBasedMatcher()
        
    def detect_features(self, image, detector_type='ORB'):
        """Özellikleri tespit et"""
        if detector_type not in self.detectors:
            return None, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        detector = self.detectors[detector_type]
        
        # Keypoints ve descriptors hesapla
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, matcher_type='BF', detector_type='ORB'):
        """Özellikleri eşleştir"""
        if desc1 is None or desc2 is None:
            return []
        
        # Matcher seç
        if matcher_type == 'BF':
            if detector_type in ['SIFT', 'SURF']:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:  # FLANN
            if detector_type in ['SIFT', 'SURF']:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            else:
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                  table_number=6,
                                  key_size=12,
                                  multi_probe_level=1)
            
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Eşleştirme yap
        matches = matcher.match(desc1, desc2)
        
        # Distance'a göre sırala
        matches = sorted(matches, key=lambda x: x.distance)
        
        return matches
    
    def find_homography_matches(self, kp1, kp2, matches, min_matches=10):
        """Homography kullanarak iyi eşleşmeleri bul"""
        if len(matches) < min_matches:
            return [], None
        
        # İyi eşleşmeleri al
        good_matches = matches[:min(50, len(matches))]
        
        # Noktaları çıkar
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Homography hesapla
        try:
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                cv2.RANSAC, 5.0)
            
            # İnlier matches
            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
            
            return inlier_matches, homography
        except:
            return [], None

class ContourDetector:
    """Contour tabanlı tespit sınıfı"""
    
    def __init__(self):
        self.shape_detector = ShapeDetector()
    
    def detect_contours(self, image, min_area=100):
        """Contour'ları tespit et"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold (Otsu)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Contour detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Area filtresi
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return filtered_contours, thresh
    
    def analyze_contours(self, contours):
        """Contour'ları analiz et"""
        results = []
        
        for contour in contours:
            # Temel özellikler
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Extent (contour area / bounding rect area)
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # Solidity (contour area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Shape detection
            shape = self.shape_detector.detect_shape(contour)
            
            results.append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'bbox': (x, y, w, h),
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                'shape': shape
            })
        
        return results

class ShapeDetector:
    """Şekil tespit sınıfı"""
    
    def detect_shape(self, contour):
        """Contour'dan şekil tespit et"""
        # Contour'u approxmiate et
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        vertices = len(approx)
        
        # Şekil sınıflandırması
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # Kare/dikdörtgen ayrımı
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 0.95 <= aspect_ratio <= 1.05:
                return "square"
            else:
                return "rectangle"
        elif vertices == 5:
            return "pentagon"
        elif vertices > 5:
            # Daire/ellips kontrolü
            area = cv2.contourArea(contour)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = math.pi * (radius ** 2)
            
            if area / circle_area > 0.7:
                return "circle"
            else:
                return "polygon"
        else:
            return "unknown"

def ornek_1_feature_detection_comparison():
    """
    Örnek 1: Farklı özellik detector'larının karşılaştırması
    """
    print("\n🎯 Örnek 1: Feature Detection Karşılaştırması")
    print("=" * 50)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = FeatureDetector()
    available_detectors = list(detector.detectors.keys())
    
    if not available_detectors:
        print("❌ Hiç detector oluşturulamadı!")
        return
    
    current_detector = available_detectors[0]
    detector_index = 0
    
    print(f"📋 Mevcut detector'lar: {available_detectors}")
    print("🎮 Kontroller: 1-4 detector değiştir, ESC çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Feature detection
        keypoints, descriptors = detector.detect_features(frame, current_detector)
        
        detection_time = (time.time() - start_time) * 1000
        
        # Keypoints çiz
        if keypoints:
            frame_with_kp = cv2.drawKeypoints(frame, keypoints, None, 
                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            frame_with_kp = frame.copy()
        
        # Bilgi paneli
        cv2.putText(frame_with_kp, f"Detector: {current_detector}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_kp, f"Keypoints: {len(keypoints) if keypoints else 0}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_kp, f"Time: {detection_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detector listesi
        for i, det_name in enumerate(available_detectors):
            color = (0, 255, 0) if det_name == current_detector else (255, 255, 255)
            cv2.putText(frame_with_kp, f"{i+1}: {det_name}", 
                       (frame.shape[1] - 120, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow('Feature Detection Comparison', frame_with_kp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key >= ord('1') and key <= ord('4'):
            idx = key - ord('1')
            if idx < len(available_detectors):
                current_detector = available_detectors[idx]
                print(f"🔄 Detector: {current_detector}")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_feature_matching():
    """
    Örnek 2: Feature matching ve object recognition
    """
    print("\n🎯 Örnek 2: Feature Matching")
    print("=" * 30)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = FeatureDetector()
    
    # Template selection
    template_mode = True
    template_image = None
    template_kp = None
    template_desc = None
    
    current_detector = 'ORB'
    current_matcher = 'BF'
    
    # Template selection variables
    selecting = False
    selection_start = None
    selection_roi = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selection_start, selection_roi
        
        if template_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                selection_start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                selection_roi = (*selection_start, x - selection_start[0], y - selection_start[1])
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                if selection_roi and abs(selection_roi[2]) > 50 and abs(selection_roi[3]) > 50:
                    return True
        return False
    
    cv2.namedWindow('Feature Matching')
    cv2.setMouseCallback('Feature Matching', mouse_callback)
    
    print("🎯 Feature Matching")
    print("1. Template seçin (sürükleyerek)")
    print("2. SPACE ile matching moduna geçin")
    print("Kontroller: d:detector, m:matcher")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        if template_mode:
            # Template seçim modu
            cv2.putText(display_frame, "TEMPLATE SECIM MODU", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Obje sürükleyerek seçin", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Selection box
            if selection_roi:
                x1, y1, w, h = selection_roi
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Template seçildi mi kontrol et
                if mouse_callback(cv2.EVENT_LBUTTONUP, 0, 0, 0, None):
                    # Template'i kaydet
                    x1, y1, w, h = selection_roi
                    if w < 0:
                        x1, w = x1 + w, -w
                    if h < 0:
                        y1, h = y1 + h, -h
                    
                    template_image = frame[y1:y1+h, x1:x1+w].copy()
                    template_kp, template_desc = detector.detect_features(template_image, current_detector)
                    
                    if template_kp and len(template_kp) > 10:
                        print(f"✅ Template seçildi: {len(template_kp)} keypoint")
                        template_mode = False
                    else:
                        print("⚠️ Yeterli keypoint bulunamadı, başka bir alan seçin")
                    
                    selection_roi = None
        
        else:
            # Matching modu
            if template_image is not None:
                start_time = time.time()
                
                # Current frame features
                frame_kp, frame_desc = detector.detect_features(frame, current_detector)
                
                if frame_kp and frame_desc is not None:
                    # Feature matching
                    matches = detector.match_features(template_desc, frame_desc, 
                                                    current_matcher, current_detector)
                    
                    # Homography ile iyi eşleşmeleri bul
                    good_matches, homography = detector.find_homography_matches(
                        template_kp, frame_kp, matches)
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Template'in köşelerini hesapla
                    if homography is not None and len(good_matches) > 10:
                        h, w = template_image.shape[:2]
                        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        
                        # Transform corners
                        transformed_corners = cv2.perspectiveTransform(corners, homography)
                        
                        # Bounding box çiz
                        cv2.polylines(display_frame, [np.int32(transformed_corners)], 
                                    True, (0, 255, 0), 3)
                        
                        # Center point
                        center = np.mean(transformed_corners.reshape(-1, 2), axis=0)
                        cv2.circle(display_frame, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)
                    
                    # Match lines çiz (sadece good matches)
                    if len(good_matches) > 0:
                        # Mini template preview
                        template_small = cv2.resize(template_image, (100, 100))
                        h_small, w_small = template_small.shape[:2]
                        display_frame[10:10+h_small, frame.shape[1]-w_small-10:frame.shape[1]-10] = template_small
                        cv2.rectangle(display_frame, 
                                    (frame.shape[1]-w_small-10, 10),
                                    (frame.shape[1]-10, 10+h_small), (255, 255, 255), 2)
                
                else:
                    processing_time = (time.time() - start_time) * 1000
                
                # Bilgi paneli
                cv2.putText(display_frame, f"MATCHING MODU", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Detector: {current_detector}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Matcher: {current_matcher}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if 'matches' in locals():
                    cv2.putText(display_frame, f"Total Matches: {len(matches)}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Good Matches: {len(good_matches)}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f"Time: {processing_time:.1f}ms", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Feature Matching', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):
            if template_image is not None:
                template_mode = not template_mode
                mode = "TEMPLATE" if template_mode else "MATCHING"
                print(f"🔄 Mod: {mode}")
        elif key == ord('d'):
            # Detector değiştir
            available = list(detector.detectors.keys())
            current_idx = available.index(current_detector)
            current_detector = available[(current_idx + 1) % len(available)]
            print(f"🔄 Detector: {current_detector}")
        elif key == ord('m'):
            # Matcher değiştir
            current_matcher = 'FLANN' if current_matcher == 'BF' else 'BF'
            print(f"🔄 Matcher: {current_matcher}")
        elif key == ord('r'):
            # Reset
            template_mode = True
            template_image = None
            template_kp = None
            template_desc = None
            print("🔄 Template sıfırlandı")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_contour_analysis():
    """
    Örnek 3: Contour analizi ve şekil tespiti
    """
    print("\n🎯 Örnek 3: Contour Analizi")
    print("=" * 30)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    contour_detector = ContourDetector()
    
    min_area = 500
    
    print("📐 Contour analizi - şekilli objeler gösterin!")
    print("Kontroller: +/-: min area, ESC: çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Contour detection
        contours, thresh = contour_detector.detect_contours(frame, min_area)
        
        # Contour analizi
        contour_analysis = contour_detector.analyze_contours(contours)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Sonuçları çiz
        for i, result in enumerate(contour_analysis):
            contour = result['contour']
            bbox = result['bbox']
            shape = result['shape']
            area = result['area']
            
            # Contour çiz
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            # Center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Label
            label = f"{shape} ({int(area)})"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Detaylı bilgi (ilk 3 contour için)
            if i < 3:
                info_x = 10
                info_y = 150 + i * 120
                
                cv2.putText(frame, f"Contour {i+1}:", (info_x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Shape: {shape}", (info_x, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Area: {int(area)}", (info_x, info_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"AR: {result['aspect_ratio']:.2f}", (info_x, info_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Extent: {result['extent']:.2f}", (info_x, info_y + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Solidity: {result['solidity']:.2f}", (info_x, info_y + 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Ana bilgi paneli
        cv2.putText(frame, f"Contours: {len(contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Min Area: {min_area}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {processing_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Threshold görüntüsü (küçük pencere)
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh_small = cv2.resize(thresh_colored, (160, 120))
        
        frame[frame.shape[0]-130:frame.shape[0]-10, frame.shape[1]-170:frame.shape[1]-10] = thresh_small
        cv2.rectangle(frame, (frame.shape[1]-170, frame.shape[0]-130), 
                     (frame.shape[1]-10, frame.shape[0]-10), (255, 255, 255), 2)
        cv2.putText(frame, "Threshold", (frame.shape[1]-165, frame.shape[0]-135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Contour Analysis', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('+') or key == ord('='):
            min_area = min(2000, min_area + 100)
        elif key == ord('-'):
            min_area = max(100, min_area - 100)
    
    cap.release()
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🔍 OpenCV Özellik Tabanlı Tespit Demo")
        print("="*50)
        print("1. 🎯 Feature Detection Karşılaştırması")
        print("2. 🔗 Feature Matching & Object Recognition")
        print("3. 📐 Contour Analizi & Şekil Tespiti")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-3): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_feature_detection_comparison()
            elif secim == "2":
                ornek_2_feature_matching()
            elif secim == "3":
                ornek_3_contour_analysis()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-3 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🔍 OpenCV Özellik Tabanlı Tespit")
    print("Bu modül feature-based object detection tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (önerilen)")
    print("\n📝 Not: SIFT patent korumalı olabilir, ORB açık kaynak alternatifidir.")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. ORB SIFT'e göre daha hızlı ama daha az robust
# 2. Feature matching homography ile iyileştirilebilir
# 3. Contour analysis şekil tespiti için etkili
# 4. FLANN matcher büyük descriptor setleri için hızlı
# 5. Template selection interaktif daha kullanışlı