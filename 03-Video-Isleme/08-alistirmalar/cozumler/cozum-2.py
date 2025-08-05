#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 Alıştırma 2 Çözümü: Real-time Video Filtreleme ve Analiz
==========================================================

Bu dosya alıştırma-2.py için örnek çözümdür.
Kendi çözümünüzü yapmaya çalıştıktan sonra referans olarak kullanın.

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import time
from collections import deque
import threading

class VideoFilterSystem:
    """Real-time video filtreleme ve analiz sistemi - ÇÖZÜM"""
    
    def __init__(self):
        # Video capture
        self.cap = None
        self.is_running = False
        
        # Filter kernels
        self.kernels = {
            'Original': None,
            'Blur': np.ones((5,5), np.float32) / 25,
            'Sharpen': np.array([[-1,-1,-1],
                               [-1, 9,-1], 
                               [-1,-1,-1]]),
            'Edge': np.array([[-1,-1,-1],
                            [-1, 8,-1],
                            [-1,-1,-1]]),
            'Emboss': np.array([[-2,-1,0],
                              [-1, 1,1],
                              [ 0, 1,2]]),
        }
        
        # Filter parameters
        self.current_filter = 'Original'
        self.filter_intensity = 5.0  # 1.0 - 10.0 arası
        self.filter_list = list(self.kernels.keys())
        self.filter_index = 0
        
        # Performance tracking
        self.fps_counter = FPS_Counter()
        self.processing_times = deque(maxlen=30)
        self.start_time = time.time()
        
        # Histogram
        self.show_histogram = True
        self.histogram_update_rate = 5  # Her 5 frame'de bir güncelle
        self.frame_counter = 0
        self.current_histogram = None
        
        # Frame analysis
        self.frame_stats = {
            'brightness': 0.0,
            'contrast': 0.0,
            'motion_intensity': 0.0
        }
        
        # Previous frame for motion detection
        self.prev_gray = None
        
        # Display windows
        self.window_names = {
            'original': 'Original Video',
            'filtered': 'Filtered Video', 
            'histogram': 'Real-time Histogram',
            'stats': 'Frame Statistics'
        }
    
    def calculate_histogram(self, frame):
        """Real-time histogram hesaplama"""
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        
        return hist_b, hist_g, hist_r
    
    def draw_histogram(self, hist_b, hist_g, hist_r):
        """Histogram görselleştirme"""
        height, width = 400, 512
        hist_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Normalize histograms
        hist_b = cv2.normalize(hist_b, None, 0, height, cv2.NORM_MINMAX)
        hist_g = cv2.normalize(hist_g, None, 0, height, cv2.NORM_MINMAX)
        hist_r = cv2.normalize(hist_r, None, 0, height, cv2.NORM_MINMAX)
        
        # Draw histogram lines
        bin_width = width // 256
        
        for i in range(256):
            x = i * bin_width
            
            # Blue channel
            cv2.line(hist_image, 
                    (x, height), 
                    (x, height - int(hist_b[i])), 
                    (255, 0, 0), 1)
            
            # Green channel  
            cv2.line(hist_image, 
                    (x, height), 
                    (x, height - int(hist_g[i])), 
                    (0, 255, 0), 1)
            
            # Red channel
            cv2.line(hist_image, 
                    (x, height), 
                    (x, height - int(hist_r[i])), 
                    (0, 0, 255), 1)
        
        return hist_image
    
    def calculate_frame_stats(self, frame):
        """Frame istatistiklerini hesapla"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness (ortalama parlaklık)
        brightness = np.mean(gray)
        
        # Contrast (standart sapma)
        contrast = np.std(gray)
        
        # Motion intensity (önceki frame ile fark)
        motion_intensity = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            motion_intensity = np.mean(diff)
        
        # Previous frame'i güncelle
        self.prev_gray = gray.copy()
        
        return {
            'brightness': brightness,
            'contrast': contrast, 
            'motion_intensity': motion_intensity
        }
    
    def apply_filter(self, frame, filter_name, intensity=1.0):
        """Filter uygulama"""
        if filter_name == 'Original':
            return frame.copy()
        
        kernel = self.kernels.get(filter_name)
        if kernel is None:
            return frame.copy()
        
        # Intensity ile kernel'i scale et
        scaled_kernel = kernel * (intensity / 5.0)  # 5.0 default intensity
        
        # Filter uygula (veri tipi problemlerini önle)
        frame_float = frame.astype(np.float32)
        filtered = cv2.filter2D(frame_float, -1, scaled_kernel)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)
        
        return filtered
    
    def draw_ui_overlay(self, frame):
        """UI overlay bilgilerini çiz"""
        overlay_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Sistem bilgileri (sol üst)
        filter_text = f"Filter: {self.current_filter} (Intensity: {self.filter_intensity:.1f})"
        cv2.putText(overlay_frame, filter_text, (10, 25), font, 0.6, (0, 255, 0), 2)
        
        # FPS bilgisi
        current_fps = self.fps_counter.get_fps()
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(overlay_frame, fps_text, (10, 50), font, 0.6, (255, 255, 0), 2)
        
        # Processing time
        avg_process_time = np.mean(self.processing_times) if self.processing_times else 0
        process_text = f"Process: {avg_process_time*1000:.1f}ms"
        cv2.putText(overlay_frame, process_text, (10, 75), font, 0.6, (255, 255, 0), 2)
        
        # Frame statistics (sağ üst)
        stats_y = 25
        stats_x = overlay_frame.shape[1] - 250
        
        brightness_text = f"Brightness: {self.frame_stats['brightness']:.1f}"
        contrast_text = f"Contrast: {self.frame_stats['contrast']:.1f}"
        motion_text = f"Motion: {self.frame_stats['motion_intensity']:.1f}"
        
        cv2.putText(overlay_frame, brightness_text, (stats_x, stats_y), font, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_frame, contrast_text, (stats_x, stats_y+20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay_frame, motion_text, (stats_x, stats_y+40), font, 0.5, (255, 255, 255), 1)
        
        # Kontrol bilgileri (alt)
        controls_text = "Space:Filter I/D:Intensity H:Histogram R:Reset Q:Quit"
        text_size = cv2.getTextSize(controls_text, font, 0.5, 1)[0]
        text_x = (overlay_frame.shape[1] - text_size[0]) // 2
        cv2.putText(overlay_frame, controls_text, (text_x, overlay_frame.shape[0] - 10), 
                   font, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def process_frame(self, frame):
        """Ana frame işleme fonksiyonu"""
        process_start = time.time()
        
        # Frame statistics hesapla
        self.frame_stats = self.calculate_frame_stats(frame)
        
        # Filter uygula
        filtered_frame = self.apply_filter(frame, self.current_filter, self.filter_intensity)
        
        # Histogram hesapla (her N frame'de bir)
        if self.frame_counter % self.histogram_update_rate == 0:
            hist_b, hist_g, hist_r = self.calculate_histogram(frame)
            self.current_histogram = self.draw_histogram(hist_b, hist_g, hist_r)
        
        # UI overlay ekle
        original_with_overlay = self.draw_ui_overlay(frame)
        filtered_with_overlay = self.draw_ui_overlay(filtered_frame)
        
        # Processing time kaydet
        process_time = time.time() - process_start
        self.processing_times.append(process_time)
        
        self.frame_counter += 1
        
        return original_with_overlay, filtered_with_overlay
    
    def handle_keyboard_input(self, key):
        """Keyboard input handling"""
        # Filter değiştirme (Space ve N tuşları)
        if key == ord(' ') or key == ord('n') or key == ord('N'):
            self.filter_index = (self.filter_index + 1) % len(self.filter_list)
            self.current_filter = self.filter_list[self.filter_index]
            print(f"🔄 Filter değiştirildi: {self.current_filter}")
        
        # Intensity kontrolü (I: artır, D: azalt)
        elif key == ord('i') or key == ord('I'):
            self.filter_intensity = min(10.0, self.filter_intensity + 0.5)
            print(f"📈 Intensity: {self.filter_intensity:.1f}")
        elif key == ord('d') or key == ord('D'):
            self.filter_intensity = max(1.0, self.filter_intensity - 0.5)
            print(f"📉 Intensity: {self.filter_intensity:.1f}")
        
        # Histogram toggle (H tuşu)
        elif key == ord('h') or key == ord('H'):
            self.show_histogram = not self.show_histogram
            status = "Açık" if self.show_histogram else "Kapalı"
            print(f"🔄 Histogram: {status}")
        
        # Reset (R tuşu)
        elif key == ord('r') or key == ord('R'):
            self.current_filter = 'Original'
            self.filter_index = 0
            self.filter_intensity = 5.0
            print("🔄 Sistem reset edildi")
    
    def create_windows(self):
        """Display windowlarını oluştur"""
        cv2.namedWindow(self.window_names['original'], cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_names['filtered'], cv2.WINDOW_NORMAL)
        
        if self.show_histogram:
            cv2.namedWindow(self.window_names['histogram'], cv2.WINDOW_NORMAL)
        
        # Position windows
        cv2.moveWindow(self.window_names['original'], 100, 100)
        cv2.moveWindow(self.window_names['filtered'], 750, 100)
        if self.show_histogram:
            cv2.moveWindow(self.window_names['histogram'], 100, 600)
    
    def run(self):
        """Ana çalıştırma fonksiyonu"""
        print("🎬 Real-time Video Filtreleme Sistemi")
        print("=" * 40)
        
        # Webcam'i başlat
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
        
        # Webcam ayarları
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Windows oluştur
        self.create_windows()
        
        print("📷 Kamera başlatıldı")
        print("🎯 Filter sistemi hazır")
        print("\n🎮 Kontroller:")
        print("   Space/N: Sonraki filter")
        print("   I: Intensity artır")
        print("   D: Intensity azalt") 
        print("   H: Histogram aç/kapat")
        print("   R: Reset")
        print("   Q: Çıkış")
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Frame okunamadı!")
                    break
                
                # Frame'i işle
                original_overlay, filtered_overlay = self.process_frame(frame)
                
                # Sonuçları göster
                cv2.imshow(self.window_names['original'], original_overlay)
                cv2.imshow(self.window_names['filtered'], filtered_overlay)
                
                # Histogram göster (eğer aktifse)
                if self.show_histogram and self.current_histogram is not None:
                    cv2.imshow(self.window_names['histogram'], self.current_histogram)
                
                # Keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                else:
                    self.handle_keyboard_input(key)
                
                # FPS update
                self.fps_counter.update()
        
        except KeyboardInterrupt:
            print("\n⚠️ Kullanıcı tarafından durduruldu")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Temizlik işlemleri"""
        print("🔄 Sistem kapatılıyor...")
        
        # Webcam'i serbest bırak
        if self.cap:
            self.cap.release()
        
        # Windows'ları kapat
        cv2.destroyAllWindows()
        
        # Final statistics yazdır
        session_duration = time.time() - self.start_time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_fps = self.fps_counter.get_fps()
        
        print(f"📊 Session Özeti:")
        print(f"   Süre: {session_duration:.1f} saniye")
        print(f"   Ortalama FPS: {avg_fps:.1f}")
        print(f"   Ortalama işlem süresi: {avg_processing_time*1000:.1f}ms")
        print(f"   Toplam frame: {self.frame_counter}")
        print(f"   Son filter: {self.current_filter}")
        
        self.is_running = False

class FPS_Counter:
    """FPS Counter class - ÇÖZÜM"""
    
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)
        self.window_size = window_size
    
    def update(self):
        """FPS güncelleme"""
        self.times.append(time.time())
    
    def get_fps(self):
        """FPS hesaplama"""
        if len(self.times) < 2:
            return 0.0
        
        time_diff = self.times[-1] - self.times[0]
        if time_diff == 0:
            return 0.0
        
        return (len(self.times) - 1) / time_diff

def main():
    """Ana fonksiyon - ÇÖZÜM"""
    filter_system = VideoFilterSystem()
    filter_system.run()

if __name__ == "__main__":
    main()

# 📝 ÇÖZÜM NOTLARI:
# 1. Filter kernels düzgün tanımlandı
# 2. Real-time histogram hesaplama optimize edildi
# 3. Frame statistics doğru hesaplandı  
# 4. UI overlay sistematik olarak düzenlendi
# 5. Performance monitoring implement edildi
# 6. Keyboard controls responsive şekilde çalışıyor

# 🎯 PERFORMANS NOTLARI:
# - Histogram her 5 frame'de bir güncelleniyor (optimize edilmiş)
# - Filter kernels intensity ile scale ediliyor
# - FPS counter sliding window kullanıyor
# - Memory management deque ile optimize edilmiş

# 🚀 İYİLEŞTİRME ÖNERİLERİ:
# - Custom kernel editor eklenebilir
# - Video recording functionality
# - Advanced filter combinations
# - Threading ile performance artırımı
# - GPU acceleration desteği