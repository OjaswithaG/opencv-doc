#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 Alıştırma 2: Real-time Video Filtreleme ve Analiz
==================================================

Bu alıştırmada real-time video akışı üzerinde çeşitli filtreler
ve analiz teknikleri uygulayacaksınız.

GÖREV: Aşağıdaki TODO kısımlarını tamamlayın!

Yazan: [ADINIZI BURAYA YAZIN]
Tarih: 2024
"""

import cv2
import numpy as np
import time
from collections import deque
import threading

class VideoFilterSystem:
    """Real-time video filtreleme ve analiz sistemi"""
    
    def __init__(self):
        # Video capture
        self.cap = None
        self.is_running = False
        
        # TODO 1: Filter kernels tanımlayın
        # Blur, Sharpen, Edge, ve Emboss kernelleri oluşturun
        self.kernels = {
            'Original': None,
            'Blur': None,      # BURAYA KOD YAZIN - np.ones((5,5), np.float32) / 25
            'Sharpen': None,   # BURAYA KOD YAZIN - sharpen kernel
            'Edge': None,      # BURAYA KOD YAZIN - edge detection kernel
            'Emboss': None     # BURAYA KOD YAZIN - emboss kernel
        }
        
        # TODO 2: Filter parametrelerini ayarlayın
        self.current_filter = 'Original'
        self.filter_intensity = 5.0  # 1.0 - 10.0 arası
        self.filter_list = list(self.kernels.keys())
        self.filter_index = 0
        
        # TODO 3: Performance tracking için değişkenler
        self.fps_counter = None  # BURAYA FPS counter class instance yazın
        self.processing_times = deque(maxlen=30)
        self.start_time = time.time()
        
        # TODO 4: Histogram için değişkenler
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
    
    def initialize_kernels(self):
        """
        TODO 5: Filter kernels'larını initialize edin
        
        Blur: Gaussian blur kernel
        Sharpen: Sharpening kernel  
        Edge: Edge detection kernel
        Emboss: Emboss effect kernel
        """
        # TODO 5a: Blur kernel (5x5 averaging)
        # self.kernels['Blur'] = ?
        
        # TODO 5b: Sharpen kernel
        # self.kernels['Sharpen'] = np.array([...])
        
        # TODO 5c: Edge detection kernel
        # self.kernels['Edge'] = ?
        
        # TODO 5d: Emboss kernel  
        # self.kernels['Emboss'] = ?
        
        print("✅ Filter kernels initialized")
    
    def calculate_histogram(self, frame):
        """
        TODO 6: Real-time histogram hesaplama
        
        Args:
            frame: BGR frame
            
        Returns:
            tuple: (hist_b, hist_g, hist_r) - her kanal için histogram
        """
        # TODO 6a: Her kanal için histogram hesaplayın
        # cv2.calcHist kullanın: [frame], [channel], None, [256], [0, 256]
        hist_b = None  # BURAYA KOD YAZIN
        hist_g = None  # BURAYA KOD YAZIN  
        hist_r = None  # BURAYA KOD YAZIN
        
        return hist_b, hist_g, hist_r
    
    def draw_histogram(self, hist_b, hist_g, hist_r):
        """
        TODO 7: Histogram görselleştirme
        
        Args:
            hist_b, hist_g, hist_r: Her kanal için histogram
            
        Returns:
            np.array: Histogram görselleştirme resmi
        """
        height, width = 400, 512
        hist_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # TODO 7a: Histogram değerlerini normalize edin
        # cv2.normalize kullanın: (histogram, None, 0, height, cv2.NORM_MINMAX)
        # BURAYA KOD YAZIN
        
        # TODO 7b: Her kanal için histogram çizgilerini çizin
        # cv2.line kullanarak her bin için çizgi çizin
        # Mavi: hist_b, Yeşil: hist_g, Kırmızı: hist_r
        # BURAYA KOD YAZIN
        
        return hist_image
    
    def calculate_frame_stats(self, frame):
        """
        TODO 8: Frame istatistiklerini hesapla
        
        Args:
            frame: BGR frame
            
        Returns:
            dict: Frame statistics
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # TODO 8a: Brightness (ortalama parlaklık)
        brightness = 0.0  # BURAYA KOD YAZIN - np.mean(gray)
        
        # TODO 8b: Contrast (standart sapma)
        contrast = 0.0    # BURAYA KOD YAZIN - np.std(gray)
        
        # TODO 8c: Motion intensity (önceki frame ile fark)
        motion_intensity = 0.0
        if self.prev_gray is not None:
            # Frame difference hesaplayın
            # BURAYA KOD YAZIN
            pass
        
        # Previous frame'i güncelle
        self.prev_gray = gray.copy()
        
        return {
            'brightness': brightness,
            'contrast': contrast, 
            'motion_intensity': motion_intensity
        }
    
    def apply_filter(self, frame, filter_name, intensity=1.0):
        """
        TODO 9: Filter uygulama
        
        Args:
            frame: Input frame
            filter_name: Filter adı
            intensity: Filter yoğunluğu (1.0-10.0)
            
        Returns:
            np.array: Filtered frame
        """
        if filter_name == 'Original':
            return frame.copy()
        
        kernel = self.kernels.get(filter_name)
        if kernel is None:
            return frame.copy()
        
        # TODO 9a: Intensity ile kernel'i scale edin
        # scaled_kernel = kernel * (intensity / 5.0)  # 5.0 default intensity
        scaled_kernel = None  # BURAYA KOD YAZIN
        
        # TODO 9b: Filter uygulayın
        # cv2.filter2D kullanın: (frame, -1, scaled_kernel)
        # Veri tipi problemlerini önlemek için frame'i float32'ye çevirin
        filtered = frame.copy()  # BURAYA KOD YAZIN
        
        return filtered
    
    def draw_ui_overlay(self, frame):
        """
        TODO 10: UI overlay bilgilerini çiz
        
        Args:
            frame: Frame üzerine bilgi yazılacak
            
        Returns:
            np.array: Overlay'li frame
        """
        overlay_frame = frame.copy()
        
        # TODO 10a: Sistem bilgileri (sol üst)
        # Filter adı, intensity, FPS
        filter_text = f"Filter: {self.current_filter} (Intensity: {self.filter_intensity:.1f})"
        # cv2.putText ile yazın
        # BURAYA KOD YAZIN
        
        # TODO 10b: FPS bilgisi
        current_fps = 0.0  # BURAYA FPS hesaplaması yazın
        fps_text = f"FPS: {current_fps:.1f}"
        # BURAYA KOD YAZIN
        
        # TODO 10c: Frame statistics (sağ üst)
        stats_y = 30
        stats_x = overlay_frame.shape[1] - 250
        
        brightness_text = f"Brightness: {self.frame_stats['brightness']:.1f}"
        contrast_text = f"Contrast: {self.frame_stats['contrast']:.1f}"
        motion_text = f"Motion: {self.frame_stats['motion_intensity']:.1f}"
        
        # Bu bilgileri sağ üst köşeye yazın
        # BURAYA KOD YAZIN
        
        # TODO 10d: Kontrol bilgileri (alt)
        controls_text = "Kontroller: Q:Çıkış Space:Filter N:Next I/D:Intensity H:Histogram"
        # Alt ortaya yazın
        # BURAYA KOD YAZIN
        
        return overlay_frame
    
    def process_frame(self, frame):
        """
        TODO 11: Ana frame işleme fonksiyonu
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (original_with_overlay, filtered_frame)
        """
        process_start = time.time()
        
        # TODO 11a: Frame statistics hesapla
        self.frame_stats = self.calculate_frame_stats(frame)
        
        # TODO 11b: Filter uygula
        filtered_frame = self.apply_filter(frame, self.current_filter, self.filter_intensity)
        
        # TODO 11c: Histogram hesapla (her N frame'de bir)
        if self.frame_counter % self.histogram_update_rate == 0:
            # Histogram hesaplayın ve self.current_histogram'a atayın
            # BURAYA KOD YAZIN
            pass
        
        # TODO 11d: UI overlay ekle
        original_with_overlay = self.draw_ui_overlay(frame)
        filtered_with_overlay = self.draw_ui_overlay(filtered_frame)
        
        # TODO 11e: Processing time kaydet
        process_time = time.time() - process_start
        self.processing_times.append(process_time)
        
        self.frame_counter += 1
        
        return original_with_overlay, filtered_with_overlay
    
    def handle_keyboard_input(self, key):
        """
        TODO 12: Keyboard input handling
        
        Args:
            key: Pressed key code
        """
        # TODO 12a: Filter değiştirme (Space ve N tuşları)
        if key == ord(' ') or key == ord('n') or key == ord('N'):
            # Bir sonraki filter'a geç
            # BURAYA KOD YAZIN
            pass
        
        # TODO 12b: Intensity kontrolü (I: artır, D: azalt)
        elif key == ord('i') or key == ord('I'):
            # Intensity'yi artır (max 10.0)
            # BURAYA KOD YAZIN
            pass
        elif key == ord('d') or key == ord('D'):
            # Intensity'yi azalt (min 1.0)
            # BURAYA KOD YAZIN  
            pass
        
        # TODO 12c: Histogram toggle (H tuşu)
        elif key == ord('h') or key == ord('H'):
            # Histogram gösterimini aç/kapat
            # BURAYA KOD YAZIN
            pass
        
        # TODO 12d: Reset (R tuşu)
        elif key == ord('r') or key == ord('R'):
            # Sistemi reset et (Original filter, default intensity)
            # BURAYA KOD YAZIN
            pass
    
    def create_windows(self):
        """Display windowlarını oluştur"""
        for window_name in self.window_names.values():
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
    
    def run(self):
        """
        TODO 13: Ana çalıştırma fonksiyonu
        """
        print("🎬 Real-time Video Filtreleme Sistemi")
        print("=" * 40)
        
        # TODO 13a: Webcam'i başlat
        self.cap = None  # BURAYA cv2.VideoCapture(0) yazın
        
        if not self.cap or not self.cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
        
        # TODO 13b: Webcam ayarları
        # Resolution ve FPS ayarlayın
        # BURAYA KOD YAZIN
        
        # TODO 13c: Kernels'ları initialize et
        # self.initialize_kernels() çağırın
        # BURAYA KOD YAZIN
        
        # TODO 13d: FPS counter oluştur
        # self.fps_counter = FPS_Counter() # (FPS_Counter class'ını da yazmanız gerekecek)
        # BURAYA KOD YAZIN
        
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
                # TODO 13e: Frame oku
                ret, frame = False, None  # BURAYA self.cap.read() yazın
                
                if not ret:
                    print("❌ Frame okunamadı!")
                    break
                
                # TODO 13f: Frame'i işle
                original_overlay, filtered_overlay = self.process_frame(frame)
                
                # TODO 13g: Sonuçları göster
                # Original ve filtered frame'leri gösterin
                # BURAYA KOD YAZIN
                
                # TODO 13h: Histogram göster (eğer aktifse)
                if self.show_histogram and self.current_histogram is not None:
                    # Histogram window'da gösterin
                    # BURAYA KOD YAZIN
                    pass
                
                # TODO 13i: Keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                else:
                    self.handle_keyboard_input(key)
                
                # TODO 13j: FPS update
                # self.fps_counter.update() çağırın
                # BURAYA KOD YAZIN
        
        except KeyboardInterrupt:
            print("\n⚠️ Kullanıcı tarafından durduruldu")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        TODO 14: Temizlik işlemleri
        """
        print("🔄 Sistem kapatılıyor...")
        
        # TODO 14a: Webcam'i serbest bırak
        # BURAYA KOD YAZIN
        
        # TODO 14b: Windows'ları kapat
        # BURAYA KOD YAZIN
        
        # TODO 14c: Final statistics yazdır
        session_duration = time.time() - self.start_time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        print(f"📊 Session Özeti:")
        print(f"   Süre: {session_duration:.1f} saniye")
        print(f"   Ortalama işlem süresi: {avg_processing_time*1000:.1f}ms")
        print(f"   Toplam frame: {self.frame_counter}")
        
        self.is_running = False

class FPS_Counter:
    """
    TODO 15: FPS Counter class'ını implement edin
    
    Bu class real-time FPS hesaplaması yapacak
    """
    def __init__(self, window_size=30):
        # TODO 15a: Initialize variables
        # BURAYA KOD YAZIN
        pass
    
    def update(self):
        """
        TODO 15b: FPS güncelleme
        """
        # TODO: Current time'ı listeye ekle
        # TODO: Window size'ı aş varsa eski değerleri sil
        # BURAYA KOD YAZIN
        pass
    
    def get_fps(self):
        """
        TODO 15c: FPS hesaplama
        
        Returns:
            float: Current FPS
        """
        # TODO: Son N frame'in süresinden FPS hesapla
        # BURAYA KOD YAZIN
        return 0.0

def main():
    """Ana fonksiyon"""
    # TODO 16: VideoFilterSystem'i oluştur ve çalıştır
    filter_system = VideoFilterSystem()
    filter_system.run()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. TODO kısımlarını sırayla tamamlayın
# 2. Filter kernels'larını doğru tanımlayın
# 3. Histogram hesaplamasını optimize edin
# 4. Performance metrics'leri takip edin
# 5. UI overlay'lerini temiz tutun

# 🎯 TEST ÖNERİLERİ:
# - Farklı filtreleri deneyin
# - Intensity değerlerini test edin
# - Uzun süre çalıştırın (performance test)
# - Histogram doğruluğunu kontrol edin

# 🚀 BONUS ÖZELLIKLER:
# - Custom kernel editor
# - Video recording functionality
# - Advanced statistics
# - GPU acceleration