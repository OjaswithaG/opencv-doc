#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Alıştırma 1 Çözümü: Hibrit Tespit Sistemi
===========================================

Bu çözüm farklı nesne tespit yöntemlerini birleştiren
kapsamlı bir hibrit sistem örneğidir.

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import time
import json
from collections import defaultdict, deque
from datetime import datetime

class FaceDetector:
    """Yüz tespit sınıfı"""
    
    def __init__(self):
        try:
            # Haar cascade yükle
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.available = True
        except:
            self.available = False
            print("⚠️ Face detector yüklenemedi")
    
    def detect(self, frame):
        """Yüz tespiti yap"""
        if not self.available:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            results.append({
                'type': 'face',
                'id': f'face_{i}',
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'confidence': 0.8,
                'area': w * h
            })
        
        return results

class ShapeDetector:
    """Şekil tespit sınıfı"""
    
    def __init__(self):
        self.min_area = 500
        self.available = True
    
    def detect(self, frame):
        """Şekil tespiti yap"""
        if not self.available:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Contour detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Approximate contour
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Shape classification
            vertices = len(approx)
            if vertices == 3:
                shape_name = "triangle"
            elif vertices == 4:
                aspect_ratio = w / h if h > 0 else 0
                if 0.85 <= aspect_ratio <= 1.15:
                    shape_name = "square"
                else:
                    shape_name = "rectangle"
            elif vertices > 6:
                # Check if circular
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                if circularity > 0.7:
                    shape_name = "circle"
                else:
                    shape_name = f"polygon_{vertices}"
            else:
                shape_name = f"polygon_{vertices}"
            
            results.append({
                'type': 'shape',
                'id': f'shape_{i}',
                'shape': shape_name,
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'confidence': 0.7,
                'area': area,
                'vertices': vertices
            })
        
        return results

class ColorDetector:
    """Renk tabanlı tespit sınıfı"""
    
    def __init__(self):
        # HSV renk aralıkları
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'red2': ([170, 50, 50], [180, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'yellow': ([20, 50, 50], [40, 255, 255])
        }
        self.min_area = 500
        self.available = True
    
    def detect(self, frame):
        """Renk tabanlı tespit yap"""
        if not self.available:
            return []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = []
        
        for color_name, (lower, upper) in self.color_ranges.items():
            if color_name == 'red2':  # Skip duplicate red
                continue
            
            # Renk maskesi
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Kırmızı için özel durum
            if color_name == 'red':
                lower2, upper2 = self.color_ranges['red2']
                mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                mask = cv2.bitwise_or(mask, mask2)
            
            # Morfolojik işlemler
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Contour bulma
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                results.append({
                    'type': 'color',
                    'id': f'color_{color_name}_{i}',
                    'color': color_name,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'confidence': 0.6,
                    'area': area
                })
        
        return results

class QRDetector:
    """QR kod tespit sınıfı"""
    
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()
        self.available = True
    
    def detect(self, frame):
        """QR kod tespiti yap"""
        if not self.available:
            return []
        
        try:
            data, bbox, _ = self.qr_detector.detectAndDecode(frame)
            
            if data and bbox is not None:
                bbox = bbox.astype(int)
                
                # Bounding rectangle hesapla
                x_min, y_min = np.min(bbox, axis=0)
                x_max, y_max = np.max(bbox, axis=0)
                
                return [{
                    'type': 'qr',
                    'id': 'qr_0',
                    'data': data,
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'center': ((x_min + x_max)//2, (y_min + y_max)//2),
                    'confidence': 1.0,
                    'area': (x_max - x_min) * (y_max - y_min)
                }]
        except:
            pass
        
        return []

class HybridDetectionSystem:
    """Hibrit tespit sistemi ana sınıfı"""
    
    def __init__(self):
        # Detection modülleri
        self.detectors = {
            'face': FaceDetector(),
            'shape': ShapeDetector(), 
            'color': ColorDetector(),
            'qr': QRDetector()
        }
        
        # Detection modları
        self.modes = {
            'ALL': ['face', 'shape', 'color', 'qr'],
            'SELECTIVE': [],
            'PERFORMANCE': ['face', 'color'],  # Hızlı modlar
            'ACCURACY': ['face', 'qr']         # Doğru modlar
        }
        
        self.current_mode = 'ALL'
        self.active_detectors = self.modes[self.current_mode].copy()
        
        # İstatistikler
        self.stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'detection_counts': defaultdict(int),
            'processing_times': defaultdict(list),
            'start_time': time.time()
        }
        
        # Detection geçmişi
        self.detection_history = deque(maxlen=1000)
        
        # Renk kodlaması
        self.detection_colors = {
            'face': (0, 255, 0),      # Green
            'shape': (255, 0, 0),     # Blue
            'color': (0, 0, 255),     # Red
            'qr': (255, 255, 0)       # Cyan
        }
        
        # UI state
        self.show_statistics = True
        self.paused = False
    
    def set_mode(self, mode):
        """Detection modunu değiştir"""
        if mode in self.modes:
            self.current_mode = mode
            if mode == 'SELECTIVE':
                # SELECTIVE modda mevcut active_detectors'ı koru
                pass
            else:
                self.active_detectors = self.modes[mode].copy()
            return True
        return False
    
    def toggle_detector(self, detector_name):
        """Detector'ı aktif/pasif yap"""
        if detector_name in self.detectors:
            if detector_name in self.active_detectors:
                self.active_detectors.remove(detector_name)
            else:
                self.active_detectors.append(detector_name)
                # SELECTIVE mod'a geç
                self.current_mode = 'SELECTIVE'
            return True
        return False
    
    def detect_all(self, frame):
        """Tüm aktif detector'ları çalıştır"""
        if self.paused:
            return {}
        
        all_results = {}
        self.stats['total_frames'] += 1
        
        for detector_name in self.active_detectors:
            if detector_name not in self.detectors:
                continue
                
            detector = self.detectors[detector_name]
            if not detector.available:
                continue
            
            start_time = time.time()
            
            try:
                results = detector.detect(frame)
                detection_time = (time.time() - start_time) * 1000
                
                all_results[detector_name] = results
                self.stats['detection_counts'][detector_name] += len(results)
                self.stats['processing_times'][detector_name].append(detection_time)
                
                # Processing time listesini sınırla
                if len(self.stats['processing_times'][detector_name]) > 50:
                    self.stats['processing_times'][detector_name] = self.stats['processing_times'][detector_name][-50:]
                
            except Exception as e:
                print(f"⚠️ {detector_name} detector hatası: {e}")
                all_results[detector_name] = []
        
        # Detection history'ye ekle
        if any(len(results) > 0 for results in all_results.values()):
            self.stats['successful_frames'] += 1
            
            self.detection_history.append({
                'timestamp': time.time(),
                'results': all_results,
                'frame_count': self.stats['total_frames']
            })
        
        return all_results
    
    def draw_results(self, frame, results):
        """Tespit sonuçlarını çiz"""
        output_frame = frame.copy()
        
        for detector_name, detections in results.items():
            color = self.detection_colors.get(detector_name, (255, 255, 255))
            
            for detection in detections:
                bbox = detection['bbox']
                center = detection['center']
                detection_type = detection['type']
                confidence = detection.get('confidence', 0.0)
                
                x, y, w, h = bbox
                
                # Bounding box
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
                
                # Center point
                cv2.circle(output_frame, center, 5, color, -1)
                
                # Label
                label_parts = [detection_type]
                
                if 'shape' in detection:
                    label_parts.append(detection['shape'])
                elif 'color' in detection:
                    label_parts.append(detection['color'])
                elif 'data' in detection:
                    label_parts.append(detection['data'][:10] + "...")
                
                if confidence > 0:
                    label_parts.append(f"{confidence:.2f}")
                
                label = " ".join(label_parts)
                
                # Label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(output_frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
                
                # Label text
                cv2.putText(output_frame, label, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return output_frame
    
    def draw_ui(self, frame):
        """UI bilgilerini çiz"""
        if not self.show_statistics:
            return frame
        
        h, w = frame.shape[:2]
        
        # Ana bilgi paneli
        cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Aktif detector'lar
        active_text = f"Active: {', '.join(self.active_detectors)}"
        cv2.putText(frame, active_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # İstatistikler
        total_detections = sum(self.stats['detection_counts'].values())
        cv2.putText(frame, f"Total Detections: {total_detections}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        success_rate = 0
        if self.stats['total_frames'] > 0:
            success_rate = self.stats['successful_frames'] / self.stats['total_frames'] * 100
        cv2.putText(frame, f"Success Rate: {success_rate:.1f}%", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Performance bilgileri
        y_offset = 150
        for detector_name, times in self.stats['processing_times'].items():
            if times and detector_name in self.active_detectors:
                avg_time = sum(times) / len(times)
                count = self.stats['detection_counts'][detector_name]
                color = self.detection_colors.get(detector_name, (255, 255, 255))
                
                cv2.putText(frame, f"{detector_name}: {avg_time:.1f}ms ({count})", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y_offset += 25
        
        # Kontrol bilgileri
        control_y = h - 120
        cv2.putText(frame, "Controls:", (10, control_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, "1-4: Toggle detectors", (10, control_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "m: Mode, s: Stats, e: Export", (10, control_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "SPACE: Pause, ESC: Exit", (10, control_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Pause indicator
        if self.paused:
            cv2.putText(frame, "PAUSED", (w//2 - 50, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        
        return frame
    
    def export_results(self, filename=None):
        """Sonuçları JSON olarak dışa aktar"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hybrid_detection_results_{timestamp}.json"
        
        # İstatistik özeti
        runtime = time.time() - self.stats['start_time']
        avg_processing_times = {}
        
        for detector_name, times in self.stats['processing_times'].items():
            if times:
                avg_processing_times[detector_name] = {
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times)
                }
        
        export_data = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': runtime,
                'mode': self.current_mode,
                'active_detectors': self.active_detectors
            },
            'statistics': {
                'total_frames': self.stats['total_frames'],
                'successful_frames': self.stats['successful_frames'],
                'success_rate': self.stats['successful_frames'] / max(1, self.stats['total_frames']),
                'detection_counts': dict(self.stats['detection_counts']),
                'avg_processing_times': avg_processing_times
            },
            'detection_history': list(self.detection_history)[-100:]  # Son 100 detection
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Sonuçlar dışa aktarıldı: {filename}")
            return True
        except Exception as e:
            print(f"❌ Dışa aktarma hatası: {e}")
            return False
    
    def reset_statistics(self):
        """İstatistikleri sıfırla"""
        self.stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'detection_counts': defaultdict(int),
            'processing_times': defaultdict(list),
            'start_time': time.time()
        }
        self.detection_history.clear()
        print("🔄 İstatistikler sıfırlandı")

def main():
    """Ana program"""
    print("🔍 Hibrit Tespit Sistemi")
    print("=" * 30)
    
    # Sistem başlat
    system = HybridDetectionSystem()
    
    # Webcam başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    print("📱 Sistem başlatıldı!")
    print("Kontroller:")
    print("  1-4: Detection modüllerini aç/kapat")
    print("  m: Mode değiştir (ALL/SELECTIVE/PERFORMANCE/ACCURACY)")
    print("  s: İstatistikleri göster/gizle")
    print("  e: Sonuçları dışa aktar")
    print("  r: İstatistikleri sıfırla")
    print("  SPACE: Duraklat/Devam et")
    print("  ESC: Çıkış")
    
    # Ana döngü
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame okunamadı!")
            break
        
        # Tespitleri yap
        results = system.detect_all(frame)
        
        # Sonuçları çiz
        display_frame = system.draw_results(frame, results)
        display_frame = system.draw_ui(display_frame)
        
        # Göster
        cv2.imshow('Hybrid Detection System', display_frame)
        
        # Klavye kontrolü
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            system.toggle_detector('face')
            print(f"🔄 Face detector: {'ON' if 'face' in system.active_detectors else 'OFF'}")
        elif key == ord('2'):
            system.toggle_detector('shape')
            print(f"🔄 Shape detector: {'ON' if 'shape' in system.active_detectors else 'OFF'}")
        elif key == ord('3'):
            system.toggle_detector('color')
            print(f"🔄 Color detector: {'ON' if 'color' in system.active_detectors else 'OFF'}")
        elif key == ord('4'):
            system.toggle_detector('qr')
            print(f"🔄 QR detector: {'ON' if 'qr' in system.active_detectors else 'OFF'}")
        elif key == ord('m'):
            # Mode cycling
            modes = ['ALL', 'SELECTIVE', 'PERFORMANCE', 'ACCURACY']
            current_idx = modes.index(system.current_mode)
            next_mode = modes[(current_idx + 1) % len(modes)]
            system.set_mode(next_mode)
            print(f"🔄 Mode: {next_mode}")
        elif key == ord('s'):
            system.show_statistics = not system.show_statistics
            print(f"🔄 Statistics: {'ON' if system.show_statistics else 'OFF'}")
        elif key == ord('e'):
            system.export_results()
        elif key == ord('r'):
            system.reset_statistics()
        elif key == ord(' '):  # SPACE
            system.paused = not system.paused
            print(f"🔄 {'PAUSED' if system.paused else 'RESUMED'}")
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()
    
    # Final export
    print("\n📊 Final İstatistikler:")
    runtime = time.time() - system.stats['start_time']
    print(f"Çalışma süresi: {runtime:.1f} saniye")
    print(f"Toplam frame: {system.stats['total_frames']}")
    print(f"Başarılı frame: {system.stats['successful_frames']}")
    
    if system.stats['total_frames'] > 0:
        success_rate = system.stats['successful_frames'] / system.stats['total_frames'] * 100
        print(f"Başarı oranı: {success_rate:.1f}%")
    
    print(f"Toplam tespit: {sum(system.stats['detection_counts'].values())}")
    
    # Otomatik export
    if system.detection_history:
        system.export_results()
    
    print("👋 Sistem kapatıldı.")

if __name__ == "__main__":
    main()

# 📝 ÇÖZÜM NOTLARI:
# 1. Modular tasarım - her detector ayrı sınıf
# 2. Mode-based operation - farklı kullanım senaryoları
# 3. Performance monitoring - detaylı istatistikler
# 4. Error handling - robust system design
# 5. Export functionality - veri analizi için JSON export