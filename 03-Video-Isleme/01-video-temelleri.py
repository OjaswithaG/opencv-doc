#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 Video Temelleri - OpenCV Video İşleme
========================================

Bu modül OpenCV ile video işlemenin temellerini kapsar:
- Video dosyası okuma ve yazma
- Video özellikleri (FPS, çözünürlük, codec)
- Frame ekstraktı ve manipülasyonu
- Webcam erişimi ve kontrolü
- Video metadata analizi

Yazan: Eren Terzi
Tarih: 2025
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

# Video işleme için yardımcı fonksiyonlar
def video_bilgilerini_goster(video_path):
    """
    Video dosyasının temel bilgilerini gösterir
    
    Args:
        video_path (str): Video dosyasının yolu
    """
    print(f"\n📹 Video Bilgileri: {video_path}")
    print("-" * 50)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Video dosyası açılamadı!")
        return
    
    # Video özelliklerini al
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Süre hesapla
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"📏 Çözünürlük: {width} x {height}")
    print(f"🎞️ FPS: {fps:.2f}")
    print(f"📊 Toplam Frame: {frame_count}")
    print(f"⏱️ Süre: {duration:.2f} saniye")
    print(f"💾 Codec: {int(cap.get(cv2.CAP_PROP_FOURCC))}")
    
    cap.release()

def ornek_1_video_okuma():
    """
    Örnek 1: Temel video dosyası okuma ve gösterme
    """
    print("\n🎯 Örnek 1: Video Dosyası Okuma")
    print("=" * 40)
    
    # Test video dosyası oluştur (eğer yoksa)
    test_video_path = create_test_video()
    
    # Video dosyasını aç
    cap = cv2.VideoCapture(test_video_path)
    
    if not cap.isOpened():
        print("❌ Video dosyası açılamadı!")
        return
    
    # Video bilgilerini göster
    video_bilgilerini_goster(test_video_path)
    
    print("\n▶️ Video oynatılıyor... (ESC ile çıkış)")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✅ Video sona erdi veya frame okunamadı")
            break
        
        # Frame'i göster
        cv2.imshow('Video Oynatıcı', frame)
        
        # ESC tuşu ile çıkış
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Space ile duraklat
            cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_webcam_erisimi():
    """
    Örnek 2: Webcam erişimi ve kontrol
    """
    print("\n🎯 Örnek 2: Webcam Erişimi")
    print("=" * 30)
    
    # Webcam'i aç (0 = varsayılan kamera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        print("💡 İpucu: Webcam bağlı mı? Diğer uygulamalar kullanıyor mu?")
        return
    
    # Webcam ayarları
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("📷 Webcam başlatıldı")
    print("Kontroller:")
    print("  - ESC: Çıkış")
    print("  - s: Screenshot al")
    print("  - r: Kayıt başlat/durdur")
    
    recording = False
    out = None
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Frame okunamadı")
            break
        
        # Frame'i çevir (ayna etkisi)
        frame = cv2.flip(frame, 1)
        
        # Kayıt durumu göster
        if recording:
            cv2.putText(frame, "🔴 RECORDING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Frame'i göster
        cv2.imshow('Webcam', frame)
        
        # Tuş kontrolü
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Screenshot
            screenshot_path = f"screenshot_{screenshot_count:03d}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot kaydedildi: {screenshot_path}")
            screenshot_count += 1
        elif key == ord('r'):  # Record
            if not recording:
                # Kayıt başlat
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('webcam_recording.avi', fourcc, 20.0, (640, 480))
                recording = True
                print("🎬 Kayıt başladı")
            else:
                # Kayıt durdur
                recording = False
                if out:
                    out.release()
                print("⏹️ Kayıt durduruldu")
        
        # Kayıt yap
        if recording and out:
            out.write(frame)
    
    # Kaynakları temizle
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def ornek_3_video_yazma():
    """
    Örnek 3: Video dosyası oluşturma ve yazma
    """
    print("\n🎯 Örnek 3: Video Dosyası Yazma")
    print("=" * 35)
    
    # Video parametreleri
    width, height = 640, 480
    fps = 30
    duration = 5  # saniye
    total_frames = fps * duration
    
    # Video codec ve writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('generated_video.avi', fourcc, fps, (width, height))
    
    print(f"🎬 Video oluşturuluyor: {width}x{height}, {fps} FPS, {duration}s")
    print("📊 İlerleme:")
    
    for frame_num in range(total_frames):
        # Animasyonlu frame oluştur
        frame = create_animated_frame(frame_num, width, height, total_frames)
        
        # Frame'i videoya yaz
        out.write(frame)
        
        # İlerleme göster
        progress = (frame_num + 1) / total_frames * 100
        if frame_num % 10 == 0:
            print(f"  Frame {frame_num + 1}/{total_frames} (%{progress:.1f})")
    
    out.release()
    print("✅ Video oluşturuldu: generated_video.avi")

def ornek_4_frame_ekstrakti():
    """
    Örnek 4: Video'dan frame çıkarma ve kaydetme
    """
    print("\n🎯 Örnek 4: Frame Ekstraktı")
    print("=" * 30)
    
    # Test video dosyasını kullan
    test_video_path = create_test_video()
    cap = cv2.VideoCapture(test_video_path)
    
    if not cap.isOpened():
        print("❌ Video dosyası açılamadı!")
        return
    
    # Output klasörü oluştur
    output_dir = Path("extracted_frames")
    output_dir.mkdir(exist_ok=True)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video: {frame_count} frame, {fps:.2f} FPS")
    print(f"📁 Çıktı klasörü: {output_dir}")
    
    # Her 30 frame'de bir kaydet (yaklaşık her saniye)
    extract_interval = max(1, int(fps))
    frame_num = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Belirli aralıklarla frame kaydet
        if frame_num % extract_interval == 0:
            timestamp = frame_num / fps
            filename = output_dir / f"frame_{extracted_count:04d}_t{timestamp:.2f}s.jpg"
            cv2.imwrite(str(filename), frame)
            extracted_count += 1
            print(f"💾 Frame kaydedildi: {filename.name}")
        
        frame_num += 1
    
    cap.release()
    print(f"✅ Toplam {extracted_count} frame çıkarıldı")

def ornek_5_video_manipulasyonu():
    """
    Örnek 5: Video hızı değiştirme ve manipülasyon
    """
    print("\n🎯 Örnek 5: Video Manipülasyonu")
    print("=" * 35)
    
    # Test video dosyası
    test_video_path = create_test_video()
    cap = cv2.VideoCapture(test_video_path)
    
    if not cap.isOpened():
        print("❌ Video dosyası açılamadı!")
        return
    
    # Orijinal video özellikleri
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Orijinal FPS: {original_fps:.2f}")
    
    # Farklı hızlarda videolar oluştur
    speed_factors = [0.5, 2.0, 4.0]  # 0.5x, 2x, 4x hız
    
    for speed in speed_factors:
        print(f"\n🎬 {speed}x hızında video oluşturuluyor...")
        
        # Video writer
        output_fps = original_fps * speed
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = f'video_{speed}x_speed.avi'
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Başa dön
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Hızlı video için frame skip
            if speed > 1.0:
                # Her N frame'den birini al
                if frame_count % int(speed) != 0:
                    frame_count += 1
                    continue
            
            # Yavaş video için frame tekrar
            repeat_count = max(1, int(1 / speed))
            for _ in range(repeat_count):
                out.write(frame)
            
            frame_count += 1
        
        out.release()
        print(f"✅ Oluşturuldu: {output_path}")
    
    cap.release()

def ornek_6_video_ozellikleri():
    """
    Örnek 6: Video özellikleri analizi ve değiştirme
    """
    print("\n🎯 Örnek 6: Video Özellikleri Analizi")
    print("=" * 40)
    
    # Test video dosyası
    test_video_path = create_test_video()
    cap = cv2.VideoCapture(test_video_path)
    
    if not cap.isOpened():
        print("❌ Video dosyası açılamadı!")
        return
    
    print("📊 Tüm Video Özellikleri:")
    print("-" * 30)
    
    # Önemli özellikler
    properties = {
        'Frame Width': cv2.CAP_PROP_FRAME_WIDTH,
        'Frame Height': cv2.CAP_PROP_FRAME_HEIGHT,
        'FPS': cv2.CAP_PROP_FPS,
        'Frame Count': cv2.CAP_PROP_FRAME_COUNT,
        'Codec (FourCC)': cv2.CAP_PROP_FOURCC,
        'Position (ms)': cv2.CAP_PROP_POS_MSEC,
        'Position (frames)': cv2.CAP_PROP_POS_FRAMES,
        'Brightness': cv2.CAP_PROP_BRIGHTNESS,
        'Contrast': cv2.CAP_PROP_CONTRAST,
        'Saturation': cv2.CAP_PROP_SATURATION,
        'Buffer Size': cv2.CAP_PROP_BUFFERSIZE,
    }
    
    for prop_name, prop_id in properties.items():
        value = cap.get(prop_id)
        if prop_name == 'Codec (FourCC)':
            # FourCC'yi string'e çevir
            fourcc_int = int(value)
            fourcc_str = ''.join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            print(f"{prop_name:20}: {fourcc_str} ({fourcc_int})")
        else:
            print(f"{prop_name:20}: {value}")
    
    # Video süresini hesapla
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\n⏱️ Hesaplanan Süre: {duration:.2f} saniye")
    print(f"📏 Çözünürlük: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    cap.release()

def create_test_video():
    """
    Test için basit bir video dosyası oluşturur
    
    Returns:
        str: Oluşturulan video dosyasının yolu
    """
    video_path = "test_video.avi"
    
    # Eğer video zaten varsa, yeniden oluşturma
    if os.path.exists(video_path):
        return video_path
    
    print("🎬 Test videosu oluşturuluyor...")
    
    # Video parametreleri
    width, height = 320, 240
    fps = 15
    duration = 3  # saniye
    total_frames = fps * duration
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Basit animasyonlu frame oluştur
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Arka plan rengi (animasyonlu)
        bg_color = int(50 + 50 * np.sin(frame_num * 0.1))
        frame[:] = (bg_color, bg_color // 2, bg_color // 3)
        
        # Hareket eden daire
        center_x = int(width // 2 + width // 4 * np.cos(frame_num * 0.2))
        center_y = int(height // 2 + height // 4 * np.sin(frame_num * 0.15))
        radius = int(20 + 10 * np.sin(frame_num * 0.3))
        
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
        
        # Frame numarasını yaz
        cv2.putText(frame, f"Frame: {frame_num + 1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Test videosu oluşturuldu: {video_path}")
    return video_path

def create_animated_frame(frame_num, width, height, total_frames):
    """
    Animasyonlu frame oluşturur
    
    Args:
        frame_num (int): Frame numarası
        width (int): Frame genişliği
        height (int): Frame yüksekliği
        total_frames (int): Toplam frame sayısı
    
    Returns:
        numpy.ndarray: Oluşturulan frame
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # İlerleme yüzdesi
    progress = frame_num / total_frames
    
    # Arka plan gradient
    for y in range(height):
        for x in range(width):
            r = int(100 + 100 * np.sin(x * 0.01 + progress * 10))
            g = int(100 + 100 * np.cos(y * 0.01 + progress * 8))
            b = int(100 + 100 * np.sin((x + y) * 0.005 + progress * 6))
            frame[y, x] = [max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r))]
    
    # Hareket eden objeler
    # Merkez çember
    center_x = int(width // 2 + width // 4 * np.cos(progress * 4 * np.pi))
    center_y = int(height // 2 + height // 4 * np.sin(progress * 4 * np.pi))
    radius = int(30 + 20 * np.sin(progress * 8 * np.pi))
    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
    
    # Dönen dikdörtgen
    rect_center = (width // 4, height // 4)
    rect_size = (60, 40)
    angle = progress * 360
    
    # Dikdörtgen köşelerini hesapla
    box = cv2.boxPoints(((rect_center[0], rect_center[1]), rect_size, angle))
    box = np.int0(box)
    cv2.drawContours(frame, [box], 0, (0, 255, 255), -1)
    
    # İlerleme çubuğu
    bar_width = int(width * 0.8)
    bar_height = 20
    bar_x = (width - bar_width) // 2
    bar_y = height - 40
    
    # Arka plan
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    # İlerleme
    progress_width = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
    
    # Frame numarası ve süre
    time_text = f"Frame: {frame_num + 1}/{total_frames} ({progress * 100:.1f}%)"
    cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def demo_menu():
    """
    Demo menüsü - kullanıcıdan seçim alır
    """
    while True:
        print("\n" + "="*50)
        print("🎬 OpenCV Video Temelleri Demo")
        print("="*50)
        print("1. 📹 Video Dosyası Okuma")
        print("2. 📷 Webcam Erişimi")
        print("3. 🎬 Video Dosyası Yazma")
        print("4. 🖼️ Frame Ekstraktı")
        print("5. ⚡ Video Manipülasyonu (Hız)")
        print("6. 📊 Video Özellikleri Analizi")
        print("7. 🔧 Test Videosu Oluştur")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-7): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_video_okuma()
            elif secim == "2":
                ornek_2_webcam_erisimi()
            elif secim == "3":
                ornek_3_video_yazma()
            elif secim == "4":
                ornek_4_frame_ekstrakti()
            elif secim == "5":
                ornek_5_video_manipulasyonu()
            elif secim == "6":
                ornek_6_video_ozellikleri()
            elif secim == "7":
                create_test_video()
                print("✅ Test videosu hazır!")
            else:
                print("❌ Geçersiz seçim! Lütfen 0-7 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🎬 OpenCV Video Temelleri")
    print("Bu modül video işleme temellerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (opsiyonel)")
    
    # Demo menüsünü başlat
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Video codec desteği sisteme bağlıdır
# 2. Webcam erişimi için kamera gereklidir
# 3. Büyük videolarla dikkatli olun (RAM kullanımı)
# 4. Video yazarken disk alanını kontrol edin
# 5. FPS değerleri performansı etkiler