#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 Alıştırma 1 Çözümü: Video Güvenlik Sistemi
============================================

Bu dosya alıştırma-1.py için örnek çözümdür.
Kendi çözümünüzü yapmaya çalıştıktan sonra referans olarak kullanın.

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path

class SecuritySystem:
    """Video güvenlik sistemi sınıfı - ÇÖZÜM"""
    
    def __init__(self):
        # Temel parametreler
        self.is_active = True
        self.is_recording = False
        self.motion_detected = False
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )
        
        # Hareket algılama parametreleri
        self.min_motion_area = 1000
        self.learning_rate = 0.01
        self.idle_timeout = 3.0
        
        # Kayıt yönetimi
        self.video_writer = None
        self.current_filename = None
        self.last_motion_time = 0
        self.recording_start_time = 0
        
        # İstatistikler
        self.stats = {
            'total_alarms': 0,
            'total_recording_time': 0,
            'session_start': time.time()
        }
        
        # Kayıt klasörü oluştur
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
    
    def detect_motion(self, frame):
        """Frame'de hareket algıla"""
        # Background subtraction
        motion_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Gürültü temizleme
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Contour detection
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Toplam hareket alanı
        total_motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 100)
        
        # Hareket tespit
        motion_detected = total_motion_area > self.min_motion_area
        
        return motion_detected, total_motion_area, motion_mask
    
    def start_recording(self, frame):
        """Video kaydını başlat"""
        if self.is_recording:
            return
        
        # Dosya adı oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = f"motion_{timestamp}.avi"
        
        # Video writer oluştur
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        full_path = self.recordings_dir / self.current_filename
        self.video_writer = cv2.VideoWriter(str(full_path), fourcc, 20.0, (width, height))
        
        if self.video_writer and self.video_writer.isOpened():
            self.is_recording = True
            self.recording_start_time = time.time()
            print(f"🎬 Kayıt başladı: {self.current_filename}")
        else:
            print("❌ Video writer oluşturulamadı!")
    
    def stop_recording(self):
        """Video kaydını durdur"""
        if not self.is_recording or not self.video_writer:
            return
        
        # Kayıt süresi
        recording_duration = time.time() - self.recording_start_time
        
        # Video writer kapat
        self.video_writer.release()
        self.video_writer = None
        self.is_recording = False
        
        # İstatistikleri güncelle
        self.stats['total_recording_time'] += recording_duration
        
        print(f"⏹️ Kayıt durduruldu: {recording_duration:.1f} saniye")
    
    def update_motion_state(self, motion_detected):
        """Hareket durumunu güncelle"""
        current_time = time.time()
        
        if motion_detected:
            if not self.motion_detected:
                # Yeni hareket başlangıcı
                print("🚨 HAREKET TESPİT EDİLDİ!")
                self.stats['total_alarms'] += 1
            
            self.last_motion_time = current_time
            self.motion_detected = True
            
        else:
            if self.motion_detected:
                # Timeout kontrolü
                time_since_motion = current_time - self.last_motion_time
                
                if time_since_motion > self.idle_timeout:
                    self.motion_detected = False
                    print("✅ Hareket sona erdi")
    
    def process_frame(self, frame):
        """Frame'i işle"""
        if not self.is_active:
            return frame
        
        # Hareket algılama
        motion_detected, motion_area, motion_mask = self.detect_motion(frame)
        
        # Hareket durumu güncelle
        self.update_motion_state(motion_detected)
        
        # Kayıt kararı
        if self.motion_detected and not self.is_recording:
            self.start_recording(frame)
        elif not self.motion_detected and self.is_recording:
            self.stop_recording()
        
        # Frame kaydet
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
        
        # UI çiz
        display_frame = self.draw_ui(frame, motion_detected, motion_area, motion_mask)
        
        return display_frame
    
    def draw_ui(self, frame, motion_detected, motion_area, motion_mask):
        """Kullanıcı arayüzünü çiz"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Sistem durumu
        status_text = "AKTIF" if self.is_active else "PASIF"
        status_color = (0, 255, 0) if self.is_active else (0, 0, 255)
        cv2.putText(display_frame, f"Sistem: {status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Hareket durumu
        if motion_detected:
            # Kırmızı çerçeve
            cv2.rectangle(display_frame, (0, 0), (w-1, h-1), (0, 0, 255), 3)
            cv2.putText(display_frame, "HAREKET!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Kayıt durumu
        if self.is_recording:
            cv2.putText(display_frame, "KAYIT YAPILIYOR", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            recording_duration = time.time() - self.recording_start_time
            cv2.putText(display_frame, f"Süre: {recording_duration:.1f}s", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # İstatistikler (sağ üst)
        stats_x = w - 250
        session_duration = time.time() - self.stats['session_start']
        
        cv2.putText(display_frame, f"Alarmlar: {self.stats['total_alarms']}", (stats_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Kayıt: {self.stats['total_recording_time']:.1f}s", (stats_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Session: {session_duration:.0f}s", (stats_x, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hareket alanı
        if motion_area > 0:
            cv2.putText(display_frame, f"Hareket: {motion_area}px", (10, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Motion mask küçük pencere
        if motion_mask is not None:
            mask_small = cv2.resize(motion_mask, (160, 120))
            mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            
            # Sağ alt köşe
            y_start = h - 130
            x_start = w - 170
            display_frame[y_start:y_start+120, x_start:x_start+160] = mask_colored
            cv2.rectangle(display_frame, (x_start, y_start), (x_start+160, y_start+120), (255, 255, 255), 2)
            cv2.putText(display_frame, "Motion", (x_start, y_start-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Kontroller
        cv2.putText(display_frame, "Q:Çıkış R:Reset S:On/Off", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame
    
    def reset_system(self):
        """Sistemi sıfırla"""
        print("🔄 Sistem sıfırlanıyor...")
        
        # Kayıt durdur
        if self.is_recording:
            self.stop_recording()
        
        # Background subtractor yenile
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )
        
        # Durumları sıfırla
        self.motion_detected = False
        self.last_motion_time = 0
        
        print("✅ Sistem sıfırlandı")
    
    def cleanup(self):
        """Sistem kapatma"""
        print("🔒 Güvenlik sistemi kapatılıyor...")
        
        # Kayıt durdur
        if self.is_recording:
            self.stop_recording()
        
        # İstatistikler
        session_duration = time.time() - self.stats['session_start']
        print(f"📊 Session Özeti:")
        print(f"   Süre: {session_duration:.1f} saniye")
        print(f"   Toplam Alarm: {self.stats['total_alarms']}")
        print(f"   Toplam Kayıt: {self.stats['total_recording_time']:.1f} saniye")
        
        # Log dosyası (bonus)
        log_file = Path("security_log.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()}: Session {session_duration:.1f}s, "
                   f"Alarms {self.stats['total_alarms']}, "
                   f"Recording {self.stats['total_recording_time']:.1f}s\n")

def main():
    """Ana fonksiyon - ÇÖZÜM"""
    print("🔒 Video Güvenlik Sistemi")
    print("=" * 40)
    
    # Webcam başlat
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    # Webcam ayarları
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Güvenlik sistemi
    security_system = SecuritySystem()
    
    print("📷 Webcam bağlandı")
    print("🎯 Background model öğreniliyor...")
    print("⚡ Sistem aktif - hareket izleniyor")
    print("\n🎮 Kontroller:")
    print("   Q: Çıkış")
    print("   R: Reset")
    print("   S: Sistem On/Off")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Frame okunamadı!")
                break
            
            # Frame işle
            processed_frame = security_system.process_frame(frame)
            
            # Göster
            cv2.imshow('Video Güvenlik Sistemi', processed_frame)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                security_system.reset_system()
            elif key == ord('s') or key == ord('S'):
                security_system.is_active = not security_system.is_active
                status = "AKTIF" if security_system.is_active else "PASIF"
                print(f"🔄 Sistem durumu: {status}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Kullanıcı tarafından durduruldu")
    
    finally:
        security_system.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# 📝 ÇÖZÜM NOTLARI:
# 1. Background subtraction için MOG2 kullanıldı
# 2. Morphological operations ile gürültü temizlendi
# 3. Contour area hesaplaması ile hareket alanı bulundu
# 4. State machine ile kayıt kontrolü yapıldı
# 5. UI elemanları sistematik olarak yerleştirildi
# 6. Error handling ve cleanup düzgün implement edildi

# 🎯 PERFORMANS NOTLARI:
# - Frame boyutu 640x480 optimal
# - Learning rate 0.01 çoğu durumda iyi
# - Min motion area frame boyutuna göre ayarlanabilir
# - Idle timeout çok kısa olmamalı (false positive)

# 🚀 İYİLEŞTİRME ÖNERİLERİ:
# - ROI (Region of Interest) desteği
# - Multiple zone detection
# - Configuration file
# - Email/SMS notifications
# - Web dashboard