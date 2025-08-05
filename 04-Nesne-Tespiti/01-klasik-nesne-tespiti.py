#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ Klasik Nesne Tespiti - OpenCV Classical Object Detection
==========================================================

Bu modül klasik nesne tespit yöntemlerini kapsar:
- Haar Cascade Classifiers
- HOG (Histogram of Oriented Gradients) 
- Template Matching
- Contour-based Detection

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

class HaarCascadeDetector:
    """Haar Cascade tabanlı nesne tespit sınıfı"""
    
    def __init__(self):
        self.detectors = {}
        self.load_cascade_files()
        
    def load_cascade_files(self):
        """Haar cascade dosyalarını yükle"""
        # OpenCV ile gelen hazır cascade'ler
        cascade_files = {
            'face': 'haarcascade_frontalface_alt.xml',
            'eye': 'haarcascade_eye.xml',
            'smile': 'haarcascade_smile.xml',
            'profile_face': 'haarcascade_profileface.xml',
            'fullbody': 'haarcascade_fullbody.xml',
            'upperbody': 'haarcascade_upperbody.xml'
        }
        
        for name, filename in cascade_files.items():
            try:
                cascade_path = cv2.data.haarcascades + filename
                self.detectors[name] = cv2.CascadeClassifier(cascade_path)
                print(f"✅ {name} cascade yüklendi")
            except Exception as e:
                print(f"⚠️ {name} cascade yüklenemedi: {e}")
    
    def detect(self, frame, detector_type='face', scale_factor=1.1, min_neighbors=5):
        """
        Nesne tespiti yap
        
        Args:
            frame: İşlenecek frame
            detector_type: Detector türü (face, eye, smile, etc.)
            scale_factor: Scale factor parametresi
            min_neighbors: Minimum komşu parametresi
            
        Returns:
            list: Tespit edilen nesnelerin koordinatları [(x, y, w, h), ...]
        """
        if detector_type not in self.detectors:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Histogram esitleme (opsiyonel iyilestirme)
        gray = cv2.equalizeHist(gray)
        
        # Tespit yap
        detections = self.detectors[detector_type].detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return detections

class HOGDetector:
    """HOG tabanlı nesne tespit sınıfı"""
    
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        # Onceden egitilmis insan detector'u yukle
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
    def detect_people(self, frame, win_stride=(8, 8), padding=(16, 16), scale=1.05):
        """
        İnsan tespiti yap
        
        Args:
            frame: İşlenecek frame
            win_stride: Window stride parametresi
            padding: Padding parametresi
            scale: Scale parametresi
            
        Returns:
            tuple: (locations, weights) - tespit edilen lokasyonlar ve ağırlıklar
        """
        # HOG detection
        locations, weights = self.hog.detectMultiScale(
            frame,
            winStride=win_stride,
            padding=padding,
            scale=scale
        )
        
        return locations, weights
    
    def non_max_suppression(self, boxes, weights, threshold=0.3):
        """Non-maximum suppression uygula"""
        if len(boxes) == 0:
            return []
        
        # OpenCV'nin NMS fonksiyonunu kullan
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            weights.tolist(),
            score_threshold=0.5,
            nms_threshold=threshold
        )
        
        if len(indices) > 0:
            return boxes[indices.flatten()]
        return []

class TemplateMatchingDetector:
    """Template matching tabanlı nesne tespit sınıfı"""
    
    def __init__(self):
        self.templates = {}
        
    def add_template(self, name, template_image):
        """Template ekle"""
        if len(template_image.shape) == 3:
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_image.copy()
        
        self.templates[name] = template_gray
        print(f"✅ Template '{name}' eklendi: {template_gray.shape}")
    
    def load_template_from_file(self, name, file_path):
        """Dosyadan template yükle"""
        if os.path.exists(file_path):
            template = cv2.imread(file_path)
            if template is not None:
                self.add_template(name, template)
                return True
        print(f"⚠️ Template dosyası bulunamadı: {file_path}")
        return False
    
    def match_template(self, frame, template_name, method=cv2.TM_CCOEFF_NORMED, threshold=0.8):
        """
        Template matching yap
        
        Args:
            frame: İşlenecek frame
            template_name: Template adı
            method: Matching metodu
            threshold: Tespit threshold'u
            
        Returns:
            list: Tespit edilen lokasyonlar [(x, y, w, h, confidence), ...]
        """
        if template_name not in self.templates:
            return []
        
        template = self.templates[template_name]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(gray, template, method)
        
        # Threshold uzerindeki esleslmeleri bul
        locations = np.where(result >= threshold)
        
        h, w = template.shape
        matches = []
        
        for pt in zip(*locations[::-1]):  # x, y koordinatları
            confidence = result[pt[1], pt[0]]
            matches.append((pt[0], pt[1], w, h, confidence))
        
        return matches
    
    def multi_scale_template_matching(self, frame, template_name, scales=None, threshold=0.8):
        """Multi-scale template matching"""
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        if template_name not in self.templates:
            return []
        
        template = self.templates[template_name]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        all_matches = []
        
        for scale in scales:
            # Template'i scale et
            scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
            
            if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                continue
            
            # Matching yap
            result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            h, w = scaled_template.shape
            
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                all_matches.append((pt[0], pt[1], w, h, confidence, scale))
        
        return all_matches

def ornek_1_haar_cascade_detection():
    """
    Örnek 1: Haar Cascade ile nesne tespiti
    """
    print("\n🎯 Örnek 1: Haar Cascade Detection")
    print("=" * 40)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam acilamadi!")
        return
    
    detector = HaarCascadeDetector()
    
    # Mevcut detector'ları listele
    print("📋 Mevcut detector'lar:")
    for name in detector.detectors.keys():
        print(f"   - {name}")
    
    current_detector = 'face'
    detector_names = list(detector.detectors.keys())
    detector_index = 0
    
    # Parametreler
    scale_factor = 1.1
    min_neighbors = 5
    
    print(f"\n🎮 Kontroller:")
    print("1-6: Detector degistir")
    print("+/-: Scale factor")
    print("n/m: Min neighbors")
    print("ESC: Cikis")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Tespit yap
        detections = detector.detect(frame, current_detector, scale_factor, min_neighbors)
        
        detection_time = (time.time() - start_time) * 1000
        
        # Sonuçları çiz
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, current_detector, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Bilgi paneli
        cv2.putText(frame, f"Detector: {current_detector}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Scale Factor: {scale_factor:.2f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Min Neighbors: {min_neighbors}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {detection_time:.1f}ms", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detector listesi (sağ alt)
        y_offset = frame.shape[0] - 100
        for i, name in enumerate(detector_names):
            color = (0, 255, 0) if name == current_detector else (255, 255, 255)
            cv2.putText(frame, f"{i+1}: {name}", (frame.shape[1] - 150, y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow('Haar Cascade Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key >= ord('1') and key <= ord('6'):
            idx = key - ord('1')
            if idx < len(detector_names):
                current_detector = detector_names[idx]
                print(f"🔄 Detector: {current_detector}")
        elif key == ord('+') or key == ord('='):
            scale_factor = min(2.0, scale_factor + 0.05)
        elif key == ord('-'):
            scale_factor = max(1.05, scale_factor - 0.05)
        elif key == ord('n'):
            min_neighbors = max(1, min_neighbors - 1)
        elif key == ord('m'):
            min_neighbors = min(10, min_neighbors + 1)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_hog_people_detection():
    """
    Örnek 2: HOG ile insan tespiti
    """
    print("\n🎯 Örnek 2: HOG People Detection")
    print("=" * 35)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam acilamadi!")
        return
    
    detector = HOGDetector()
    
    # Parametreler
    use_nms = True
    nms_threshold = 0.3
    
    print("🚶 HOG insan tespiti - kameranin onunde durun!")
    print("n: NMS on/off, +/-: threshold, ESC: cikis")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # İnsan tespiti
        locations, weights = detector.detect_people(frame)
        
        # Non-maximum suppression
        if use_nms and len(locations) > 0:
            final_locations = detector.non_max_suppression(locations, weights, nms_threshold)
        else:
            final_locations = locations
        
        detection_time = (time.time() - start_time) * 1000
        
        # Tum tespitleri ciz (acik mavi - NMS oncesi)
        for (x, y, w, h) in locations:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        
        # Final tespitleri ciz (yesil - NMS sonrasi)
        for (x, y, w, h) in final_locations:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Bilgi paneli
        cv2.putText(frame, f"Raw Detections: {len(locations)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Final Detections: {len(final_locations)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"NMS: {'ON' if use_nms else 'OFF'}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"NMS Threshold: {nms_threshold:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {detection_time:.1f}ms", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('HOG People Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('n'):
            use_nms = not use_nms
            print(f"🔄 NMS: {'ON' if use_nms else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            nms_threshold = min(0.9, nms_threshold + 0.05)
        elif key == ord('-'):
            nms_threshold = max(0.1, nms_threshold - 0.05)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_template_matching():
    """
    Örnek 3: Template matching
    """
    print("\n🎯 Örnek 3: Template Matching")
    print("=" * 30)
    
    # Video kaynağı
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam acilamadi!")
        return
    
    detector = TemplateMatchingDetector()
    
    # Template selection mode
    template_selection_mode = True
    template_roi = None
    selecting = False
    selection_start = None
    current_template = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, selection_start, template_roi
        
        if template_selection_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                selecting = True
                selection_start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and selecting:
                template_roi = (*selection_start, x - selection_start[0], y - selection_start[1])
            elif event == cv2.EVENT_LBUTTONUP:
                selecting = False
                if template_roi and abs(template_roi[2]) > 20 and abs(template_roi[3]) > 20:
                    # Template'i kaydet
                    frame = param
                    x1, y1, w, h = template_roi
                    if w < 0:
                        x1, w = x1 + w, -w
                    if h < 0:
                        y1, h = y1 + h, -h
                    
                    template_image = frame[y1:y1+h, x1:x1+w]
                    detector.add_template("user_template", template_image)
                    current_template = "user_template"
                    print("✅ Template secildi!")
                    return True
        return False
    
    cv2.namedWindow('Template Matching')
    cv2.setMouseCallback('Template Matching', mouse_callback)
    
    # Parametreler
    threshold = 0.8
    use_multi_scale = False
    
    print("🎯 Template Matching")
    print("1. Once template secin (surukleyerek)")
    print("2. SPACE ile detection moduna gecin")
    print("Kontroller: t:threshold, s:multi-scale, r:reset")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.setMouseCallback('Template Matching', lambda *args: mouse_callback(*args, frame))
        
        display_frame = frame.copy()
        
        if template_selection_mode:
            # Template secim modu
            cv2.putText(display_frame, "TEMPLATE SECIM MODU", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Surukleyerek template secin", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "SPACE: Detection moduna gec", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Secim kutusunu ciz
            if template_roi:
                x1, y1, w, h = template_roi
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        else:
            # Detection modu
            if current_template:
                start_time = time.time()
                
                if use_multi_scale:
                    matches = detector.multi_scale_template_matching(frame, current_template, threshold=threshold)
                else:
                    matches = detector.match_template(frame, current_template, threshold=threshold)
                
                detection_time = (time.time() - start_time) * 1000
                
                # Esleslmeleri ciz
                for match in matches:
                    if use_multi_scale:
                        x, y, w, h, confidence, scale = match
                        color = (0, 255, 0)
                        label = f"{confidence:.2f} (x{scale:.1f})"
                    else:
                        x, y, w, h, confidence = match
                        color = (0, 255, 0)
                        label = f"{confidence:.2f}"
                    
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Bilgi paneli
                cv2.putText(display_frame, f"DETECTION MODU", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Matches: {len(matches)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Threshold: {threshold:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Multi-scale: {'ON' if use_multi_scale else 'OFF'}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Time: {detection_time:.1f}ms", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Template secilmedi!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Template Matching', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):
            if current_template:
                template_selection_mode = not template_selection_mode
                mode = "SELECTION" if template_selection_mode else "DETECTION"
                print(f"🔄 Mod: {mode}")
        elif key == ord('t'):
            try:
                new_threshold = float(input(f"Yeni threshold (mevcut: {threshold:.2f}): "))
                threshold = max(0.1, min(1.0, new_threshold))
                print(f"📊 Threshold: {threshold:.2f}")
            except:
                print("Gecersiz deger!")
        elif key == ord('s'):
            use_multi_scale = not use_multi_scale
            print(f"🔄 Multi-scale: {'ON' if use_multi_scale else 'OFF'}")
        elif key == ord('r'):
            template_selection_mode = True
            current_template = None
            template_roi = None
            print("🔄 Template sifirlandi")
    
    cap.release()
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🏛️ OpenCV Klasik Nesne Tespiti Demo")
        print("="*50)
        print("1. 👤 Haar Cascade Detection (Yüz, Göz, vs.)")
        print("2. 🚶 HOG People Detection")
        print("3. 🎯 Template Matching")
        print("0. ❌ Cikis")
        
        try:
            secim = input("\nSeciminizi yapin (0-3): ").strip()
            
            if secim == "0":
                print("👋 Gorusmek uzere!")
                break
            elif secim == "1":
                ornek_1_haar_cascade_detection()
            elif secim == "2":
                ornek_2_hog_people_detection()
            elif secim == "3":
                ornek_3_template_matching()
            else:
                print("❌ Gecersiz secim! Lutfen 0-3 arasinda bir sayi girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandirildi.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🏛️ OpenCV Klasik Nesne Tespiti")
    print("Bu modul geleneksel nesne tespit yontemlerini ogretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (onerilen)")
    print("\n📝 Not: Haar cascade dosyalari OpenCV ile birlikte gelir.")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Haar cascades hızlı ama sınırlı doğruluk
# 2. HOG insanlar için çok etkili
# 3. Template matching basit nesneler için ideal
# 4. NMS overlapping detection'ları temizler
# 5. Multi-scale farklı boyutları yakalar