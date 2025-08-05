#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔒 Alıştırma 1: Video Güvenlik Sistemi
=====================================

Bu alıştırmada webcam tabanlı bir güvenlik sistemi geliştireceksiniz.
Sistem hareket algıladığında otomatik kayıt yapacak ve alarm verecek.

GÖREV: Aşağıdaki TODO kısımlarını tamamlayın!

Yazan: [ADINIZI BURAYA YAZIN]
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path

class SecuritySystem:
    """Video güvenlik sistemi sınıfı"""
    
    def __init__(self):
        # Temel parametreler
        self.is_active = True
        self.is_recording = False
        self.motion_detected = False
        
        # TODO 1: Background subtractor'ı initialize edin
        # İpucu: cv2.createBackgroundSubtractorMOG2() kullanın
        # detectShadows=True, varThreshold=50, history=500 parametreleri ile
        self.bg_subtractor = None  # BURAYA KOD YAZIN
        
        # TODO 2: Hareket algılama parametrelerini ayarlayın
        # Minimum hareket alanı (piksel cinsinden)
        self.min_motion_area = 0  # BURAYA DEĞER YAZIN (örn: 1000)
        
        # Learning rate (0.0-1.0 arası)
        self.learning_rate = 0.0  # BURAYA DEĞER YAZIN (örn: 0.01)
        
        # Idle timeout (saniye) - hareket bittikten sonra kayıt ne kadar süre devam etsin
        self.idle_timeout = 0.0  # BURAYA DEĞER YAZIN (örn: 3.0)
        
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
        """
        Frame'de hareket algıla
        
        TODO 3: Bu fonksiyonu tamamlayın
        
        Args:
            frame: İşlenecek frame
            
        Returns:
            tuple: (motion_detected: bool, motion_area: int, motion_mask: np.array)
        """
        # TODO 3a: Background subtraction uygulayın
        # self.bg_subtractor.apply() fonksiyonunu kullanın
        # learning_rate parametresini de geçin
        motion_mask = None  # BURAYA KOD YAZIN
        
        if motion_mask is None:
            # Frame boyutlarını güvenli şekilde al
            if len(frame.shape) == 3:
                return False, 0, np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            else:
                return False, 0, np.zeros_like(frame)
        
        # TODO 3b: Gürültüyü temizleyin (morphological operations)
        # cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) ile kernel oluşturun
        # cv2.morphologyEx ile MORPH_OPEN ve MORPH_CLOSE uygulayın
        kernel = None  # BURAYA KOD YAZIN
        # motion_mask = cv2.morphologyEx(...)  # BURAYA KOD YAZIN
        # motion_mask = cv2.morphologyEx(...)  # BURAYA KOD YAZIN
        
        # TODO 3c: Contour'ları bulun ve toplam alanı hesaplayın
        # cv2.findContours kullanın (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        contours = []  # BURAYA KOD YAZIN
        
        # TODO 3d: Toplam hareket alanını hesaplayın
        # Her contour için cv2.contourArea() hesaplayın ve toplayın
        total_motion_area = 0  # BURAYA KOD YAZIN
        
        # TODO 3e: Hareket tespit edildi mi kontrol edin
        motion_detected = False  # BURAYA KOD YAZIN (total_motion_area > self.min_motion_area)
        
        return motion_detected, total_motion_area, motion_mask
    
    def start_recording(self, frame):
        """
        Video kaydını başlat
        
        TODO 4: Bu fonksiyonu tamamlayın
        """
        if self.is_recording:
            return
        
        # TODO 4a: Dosya adını oluşturun
        # datetime.now().strftime() kullanarak "motion_YYYYMMDD_HHMMSS.avi" formatında
        timestamp = ""  # BURAYA KOD YAZIN
        self.current_filename = ""  # BURAYA KOD YAZIN
        
        # TODO 4b: Video writer'ı oluşturun
        # cv2.VideoWriter_fourcc(*'XVID') ile codec
        # 20.0 FPS, frame boyutları (width, height)
        height, width = frame.shape[:2]
        fourcc = None  # BURAYA KOD YAZIN
        full_path = self.recordings_dir / self.current_filename
        self.video_writer = None  # BURAYA KOD YAZIN
        
        if self.video_writer and self.video_writer.isOpened():
            self.is_recording = True
            self.recording_start_time = time.time()
            print(f"🎬 Kayıt başladı: {self.current_filename}")
        else:
            print("❌ Video writer oluşturulamadı!")
    
    def stop_recording(self):
        """
        Video kaydını durdur
        
        TODO 5: Bu fonksiyonu tamamlayın
        """
        if not self.is_recording or not self.video_writer:
            return
        
        # TODO 5a: Kayıt süresini hesaplayın
        recording_duration = 0.0  # BURAYA KOD YAZIN
        
        # TODO 5b: Video writer'ı kapatın
        # self.video_writer.release() çağırın
        # BURAYA KOD YAZIN
        
        # TODO 5c: Değişkenleri sıfırlayın
        self.video_writer = None
        self.is_recording = False
        
        # TODO 5d: İstatistikleri güncelleyin
        # self.stats['total_recording_time'] += recording_duration
        # BURAYA KOD YAZIN
        
        print(f"⏹️ Kayıt durduruldu: {recording_duration:.1f} saniye")
    
    def update_motion_state(self, motion_detected):
        """
        Hareket durumunu güncelle ve kayıt kararı ver
        
        TODO 6: Bu fonksiyonu tamamlayın
        """
        current_time = time.time()
        
        if motion_detected:
            # TODO 6a: Hareket tespit edildi
            if not self.motion_detected:
                # Yeni hareket başlangıcı
                print("🚨 HAREKET TESPİT EDİLDİ!")
                # İstatistik güncelleyin
                # BURAYA KOD YAZIN
            
            # TODO 6b: Son hareket zamanını güncelleyin
            # BURAYA KOD YAZIN
            
            # Motion state'i güncelleyin
            self.motion_detected = True
            
        else:
            # TODO 6c: Hareket yok - timeout kontrolü yapın
            if self.motion_detected:
                # Önceden hareket vardı, şimdi yok
                # Son hareketten bu yana geçen süreyi kontrol edin
                time_since_motion = 0.0  # BURAYA KOD YAZIN
                
                if time_since_motion > self.idle_timeout:
                    # Timeout aşıldı, hareket durumunu false yap
                    # BURAYA KOD YAZIN
                    print("✅ Hareket sona erdi")
    
    def process_frame(self, frame):
        """
        Frame'i işle - ana processing fonksiyonu
        
        TODO 7: Bu fonksiyonu tamamlayın
        """
        if not self.is_active:
            return frame
        
        # TODO 7a: Hareket algılama
        motion_detected, motion_area, motion_mask = self.detect_motion(frame)
        
        # TODO 7b: Hareket durumunu güncelle
        # self.update_motion_state() çağırın
        # BURAYA KOD YAZIN
        pass
        
        # TODO 7c: Kayıt kararı
        if self.motion_detected and not self.is_recording:
            # Hareket var ama kayıt yok - kayıt başlat
            # BURAYA KOD YAZIN
            pass
        elif not self.motion_detected and self.is_recording:
            # Hareket yok ama kayıt var - kayıt durdur
            # BURAYA KOD YAZIN
            pass
        
        # TODO 7d: Frame'i kaydet (eğer kayıt aktifse)
        if self.is_recording and self.video_writer:
            # BURAYA KOD YAZIN
            pass
        
        # TODO 7e: Görselleştirme için frame'i hazırla
        display_frame = self.draw_ui(frame, motion_detected, motion_area, motion_mask)
        
        return display_frame
    
    def draw_ui(self, frame, motion_detected, motion_area, motion_mask):
        """
        Kullanıcı arayüzünü çiz
        
        TODO 8: Bu fonksiyonu tamamlayın
        """
        display_frame = frame.copy()
        
        # TODO 8a: Sistem durumu
        status_text = "AKTIF" if self.is_active else "PASIF"
        status_color = (0, 255, 0) if self.is_active else (0, 0, 255)
        # cv2.putText ile sol üst köşeye yazın
        # BURAYA KOD YAZIN
        
        # TODO 8b: Hareket durumu
        if motion_detected:
            # Kırmızı çerçeve çizin (frame'in kenarları)
            # cv2.rectangle ile
            # BURAYA KOD YAZIN
            
            # "HAREKET!" yazısı
            # BURAYA KOD YAZIN
            pass
        
        # TODO 8c: Kayıt durumu
        if self.is_recording:
            # "KAYIT YAPILIYOR" yazısı (kırmızı)
            # Sol üst, status_text altına
            # BURAYA KOD YAZIN
            
            # Kayıt süresi
            recording_duration = time.time() - self.recording_start_time
            duration_text = f"Süre: {recording_duration:.1f}s"
            # BURAYA KOD YAZIN
            pass
        
        # TODO 8d: İstatistikler (sağ üst)
        stats_x = display_frame.shape[1] - 200
        # Toplam alarm sayısı
        # Toplam kayıt süresi
        # Session süresi
        # BURAYA KOD YAZIN
        
        # TODO 8e: Hareket alanı bilgisi
        if motion_area > 0:
            area_text = f"Hareket Alanı: {motion_area}px"
            # Sol alt köşede gösterin
            # BURAYA KOD YAZIN
        
        # TODO 8f: Motion mask'i küçük pencerede göster
        if motion_mask is not None:
            # Motion mask'i resize edin (örn: 160x120)
            # Sağ alt köşeye yerleştirin
            # Çerçeve çizin
            # BURAYA KOD YAZIN
            pass
        
        # TODO 8g: Kontrol bilgileri (alt ortada)
        controls_text = "Kontroller: Q:Çıkış R:Reset S:Sistem On/Off"
        # Alt ortada gösterin
        # BURAYA KOD YAZIN
        
        return display_frame
    
    def reset_system(self):
        """
        Sistemi sıfırla
        
        TODO 9: Bu fonksiyonu tamamlayın
        """
        print("🔄 Sistem sıfırlanıyor...")
        
        # TODO 9a: Aktif kayıt varsa durdur
        # BURAYA KOD YAZIN
        
        # TODO 9b: Background subtractor'ı yeniden oluştur
        # BURAYA KOD YAZIN
        
        # TODO 9c: Durumları sıfırla
        self.motion_detected = False
        self.last_motion_time = 0
        # BURAYA KOD YAZIN
        
        # TODO 9d: İstatistikleri sıfırla (opsiyonel)
        # BURAYA KOD YAZIN
        
        print("✅ Sistem sıfırlandı")
    
    def cleanup(self):
        """
        Sistem kapatma işlemleri
        
        TODO 10: Bu fonksiyonu tamamlayın
        """
        print("🔒 Güvenlik sistemi kapatılıyor...")
        
        # TODO 10a: Aktif kayıt varsa durdur
        # BURAYA KOD YAZIN
        
        # TODO 10b: Final istatistikleri yazdır
        session_duration = time.time() - self.stats['session_start']
        print(f"📊 Session Özeti:")
        print(f"   Süre: {session_duration:.1f} saniye")
        print(f"   Toplam Alarm: {self.stats['total_alarms']}")
        print(f"   Toplam Kayıt: {self.stats['total_recording_time']:.1f} saniye")
        
        # TODO 10c: Log dosyası yazın (bonus)
        # Tarih, süre, alarm sayısı, kayıt süresi
        # BURAYA KOD YAZIN (opsiyonel)

def main():
    """Ana fonksiyon"""
    print("🔒 Video Güvenlik Sistemi")
    print("=" * 40)
    
    # TODO 11: Webcam'i başlatın
    cap = None  # BURAYA KOD YAZIN (cv2.VideoCapture(0))
    
    if cap is None:
        print("❌ Webcam initialize edilmedi!")
        return
    
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    # TODO 12: Webcam ayarları (opsiyonel)
    # Çözünürlük ayarlayın (örn: 640x480)
    # FPS ayarlayın
    # BURAYA KOD YAZIN
    
    # Güvenlik sistemini başlat
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
            # TODO 13: Frame okuyun
            ret, frame = False, None  # BURAYA KOD YAZIN - cap.read() kullanın
            
            if not ret or frame is None:
                print("❌ Frame okunamadı!")
                break
            
            # TODO 14: Frame'i işleyin
            processed_frame = None  # BURAYA KOD YAZIN (security_system.process_frame(frame))
            
            # TODO 15: Sonucu gösterin
            # cv2.imshow ile
            # BURAYA KOD YAZIN
            
            # Geçici çözüm - processed_frame None ise original frame göster
            if processed_frame is not None:
                display_frame = processed_frame
            else:
                display_frame = frame
                
            # TODO: cv2.imshow('Video Güvenlik Sistemi', display_frame) yazın
            
            # TODO 16: Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                # Çıkış
                break
            elif key == ord('r') or key == ord('R'):
                # Reset
                # BURAYA KOD YAZIN
                pass
            elif key == ord('s') or key == ord('S'):
                # Sistem on/off toggle
                security_system.is_active = not security_system.is_active
                status = "AKTIF" if security_system.is_active else "PASIF"
                print(f"🔄 Sistem durumu: {status}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Kullanıcı tarafından durduruldu")
    
    finally:
        # TODO 17: Temizlik işlemleri
        # security_system.cleanup() çağırın
        # cap.release() ve cv2.destroyAllWindows() çağırın
        # BURAYA KOD YAZIN
        pass

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. TODO kısımlarını sırayla tamamlayın
# 2. Her TODO'dan sonra test edin
# 3. Hata mesajlarını debug için kullanın
# 4. Frame boyutlarını kontrol edin
# 5. Webcam'i release etmeyi unutmayın

# 🎯 TEST ÖNERİLERİ:
# - Kameranın önünde el sallayın
# - Farklı boyutlarda objeler deneyin
# - Uzun süre çalıştırın (memory leak kontrolü)
# - Webcam'i kapatıp açın

# 🚀 BONUS ÖZELLIKLER:
# - Configuration dosyası
# - Multiple detection zones
# - Email/SMS alarm
# - Web interface