#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 Test Video Oluşturucu
=======================

Video işleme alıştırmaları için test videoları oluşturur.
Webcam olmadığında veya kontrollü test senaryoları için kullanılır.

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import math
import random
from pathlib import Path

class TestVideoGenerator:
    """Test video oluşturucu sınıfı"""
    
    def __init__(self, width=640, height=480, fps=20):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = Path("test_videos")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_motion_test_video(self, duration=30):
        """Hareket testi için video oluştur"""
        filename = self.output_dir / "motion_test.avi"
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(filename), fourcc, self.fps, (self.width, self.height))
        
        total_frames = duration * self.fps
        
        print(f"🎬 Hareket test videosu oluşturuluyor: {duration}s")
        
        for frame_num in range(total_frames):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Arka plan (hafif gürültü)
            noise = np.random.normal(20, 5, (self.height, self.width, 3))
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
            
            # Hareket eden daire
            t = frame_num / self.fps
            
            # Sinüs hareketi
            x = int(self.width/2 + (self.width/3) * math.sin(t * 0.5))
            y = int(self.height/2 + (self.height/4) * math.cos(t * 0.3))
            
            # Radius değişimi
            radius = int(20 + 15 * math.sin(t * 2))
            
            cv2.circle(frame, (x, y), radius, (255, 255, 255), -1)
            
            # İkinci nesne (farklı hareket)
            if frame_num > total_frames // 3:
                x2 = int(self.width/4 + (self.width/2) * math.cos(t * 0.8))
                y2 = int(self.height/3 + (self.height/3) * math.sin(t * 0.6))
                cv2.rectangle(frame, (x2-15, y2-15), (x2+15, y2+15), (0, 255, 255), -1)
            
            # Rastgele gürültü (test için)
            if random.random() < 0.05:  # %5 şans
                noise_x = random.randint(50, self.width-50)
                noise_y = random.randint(50, self.height-50)
                cv2.circle(frame, (noise_x, noise_y), 3, (128, 128, 128), -1)
            
            out.write(frame)
            
            if frame_num % (self.fps * 5) == 0:
                print(f"  Frame {frame_num}/{total_frames}")
        
        out.release()
        print(f"✅ Video oluşturuldu: {filename}")
        return str(filename)
    
    def create_tracking_test_video(self, duration=60):
        """Nesne takibi için test videosu"""
        filename = self.output_dir / "tracking_test.avi"
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(filename), fourcc, self.fps, (self.width, self.height))
        
        total_frames = duration * self.fps
        
        print(f"🎯 Tracking test videosu oluşturuluyor: {duration}s")
        
        # Çoklu nesne parametreleri
        objects = [
            {'color': (255, 0, 0), 'size': 25, 'speed': 1.0, 'path': 'circle'},
            {'color': (0, 255, 0), 'size': 20, 'speed': 0.8, 'path': 'linear'},
            {'color': (0, 0, 255), 'size': 30, 'speed': 1.2, 'path': 'zigzag'},
            {'color': (255, 255, 0), 'size': 15, 'speed': 0.6, 'path': 'random'}
        ]
        
        for frame_num in range(total_frames):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Arka plan texture
            for i in range(0, self.width, 40):
                cv2.line(frame, (i, 0), (i, self.height), (30, 30, 30), 1)
            for i in range(0, self.height, 40):
                cv2.line(frame, (0, i), (self.width, i), (30, 30, 30), 1)
            
            t = frame_num / self.fps
            
            for i, obj in enumerate(objects):
                # Farklı başlangıç zamanları
                start_time = i * 5
                if t < start_time:
                    continue
                
                obj_t = t - start_time
                
                # Hareket türüne göre pozisyon
                if obj['path'] == 'circle':
                    x = int(self.width/2 + 150 * math.cos(obj_t * obj['speed']))
                    y = int(self.height/2 + 100 * math.sin(obj_t * obj['speed']))
                elif obj['path'] == 'linear':
                    x = int((obj_t * obj['speed'] * 50) % self.width)
                    y = int(self.height/2 + 50 * math.sin(obj_t * 0.5))
                elif obj['path'] == 'zigzag':
                    x = int((obj_t * obj['speed'] * 30) % self.width)
                    y = int(self.height/2 + 80 * math.sin(obj_t * obj['speed'] * 3))
                else:  # random
                    x = int(self.width/2 + 100 * math.sin(obj_t * obj['speed'] * 0.7))
                    y = int(self.height/2 + 100 * math.cos(obj_t * obj['speed'] * 0.3))
                
                # Ekran sınırları
                x = max(obj['size'], min(self.width - obj['size'], x))
                y = max(obj['size'], min(self.height - obj['size'], y))
                
                # Nesneyi çiz
                cv2.circle(frame, (x, y), obj['size'], obj['color'], -1)
                cv2.circle(frame, (x, y), obj['size'], (255, 255, 255), 2)
                
                # ID numarası
                cv2.putText(frame, str(i+1), (x-5, y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Crossing line (test için)
            cv2.line(frame, (self.width//2, 0), (self.width//2, self.height), (255, 255, 255), 2)
            cv2.putText(frame, "CROSSING LINE", (self.width//2 + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
            
            if frame_num % (self.fps * 10) == 0:
                print(f"  Frame {frame_num}/{total_frames}")
        
        out.release()
        print(f"✅ Video oluşturuldu: {filename}")
        return str(filename)
    
    def create_quality_test_video(self, duration=45):
        """Video kalitesi testi için video"""
        filename = self.output_dir / "quality_test.avi"
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(filename), fourcc, self.fps, (self.width, self.height))
        
        total_frames = duration * self.fps
        
        print(f"📊 Kalite test videosu oluşturuluyor: {duration}s")
        
        for frame_num in range(total_frames):
            t = frame_num / self.fps
            phase = int(t / 10)  # Her 10 saniyede faz değişimi
            
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            if phase == 0:
                # Yüksek kalite, düşük hareket
                for i in range(20):
                    x = random.randint(50, self.width-50)
                    y = random.randint(50, self.height-50)
                    cv2.circle(frame, (x, y), 5, (100, 150, 200), -1)
                
            elif phase == 1:
                # Orta kalite, orta hareket
                noise = np.random.normal(100, 30, (self.height, self.width, 3))
                frame = np.clip(noise, 0, 255).astype(np.uint8)
                
                # Hareket eden pattern
                offset = int(t * 50) % self.width
                cv2.rectangle(frame, (offset-20, self.height//2-20), 
                             (offset+20, self.height//2+20), (255, 255, 255), -1)
                
            elif phase == 2:
                # Düşük kalite, yüksek hareket
                # Çok gürültülü
                noise = np.random.normal(128, 60, (self.height, self.width, 3))
                frame = np.clip(noise, 0, 255).astype(np.uint8)
                
                # Hızlı hareket
                for i in range(10):
                    x = int((t * 200 + i * 50) % self.width)
                    y = int(self.height/2 + 100 * math.sin(t * 3 + i))
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
                
            elif phase == 3:
                # Scene change test
                color_base = int(128 + 127 * math.sin(t * 2))
                frame[:] = (color_base, color_base//2, color_base//3)
                
                # Gradient
                for y in range(self.height):
                    intensity = int(255 * y / self.height)
                    frame[y, :] = (intensity, intensity//2, intensity//3)
            
            else:
                # Kompleks sahne
                # Checkerboard pattern
                tile_size = 20
                for y in range(0, self.height, tile_size):
                    for x in range(0, self.width, tile_size):
                        if ((x//tile_size) + (y//tile_size)) % 2:
                            cv2.rectangle(frame, (x, y), (x+tile_size, y+tile_size), 
                                        (255, 255, 255), -1)
                
                # Moving overlay
                overlay_x = int((t * 30) % self.width)
                cv2.rectangle(frame, (overlay_x, 0), (overlay_x+50, self.height), 
                             (255, 0, 0), -1)
            
            # Phase bilgisi
            cv2.putText(frame, f"Phase {phase+1}/5 - Time: {t:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            
            if frame_num % (self.fps * 5) == 0:
                print(f"  Frame {frame_num}/{total_frames}")
        
        out.release()
        print(f"✅ Video oluşturuldu: {filename}")
        return str(filename)

def main():
    """Ana fonksiyon"""
    print("🎬 Test Video Oluşturucu")
    print("=" * 40)
    
    generator = TestVideoGenerator(width=640, height=480, fps=20)
    
    print("Hangi test videosunu oluşturmak istiyorsunuz?")
    print("1. Hareket Algılama Testi (30s)")
    print("2. Nesne Takibi Testi (60s)")
    print("3. Video Kalitesi Testi (45s)")
    print("4. Tümü")
    
    choice = input("\nSeçiminizi yapın (1-4): ").strip()
    
    if choice == "1":
        generator.create_motion_test_video()
    elif choice == "2":
        generator.create_tracking_test_video()
    elif choice == "3":
        generator.create_quality_test_video()
    elif choice == "4":
        print("\n🎬 Tüm test videoları oluşturuluyor...")
        generator.create_motion_test_video()
        generator.create_tracking_test_video()
        generator.create_quality_test_video()
        print("\n✅ Tüm videolar oluşturuldu!")
    else:
        print("❌ Geçersiz seçim!")
        return
    
    print(f"\n📁 Videolar şu klasörde: {generator.output_dir}")
    print("🎯 Alıştırmalarınızda bu videoları kullanabilirsiniz!")

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Bu araç webcam olmadığında test için kullanılır
# 2. Farklı senaryolar için parametreler ayarlanabilir
# 3. Video codec sistemde desteklenmiyorsa MJPG deneyin
# 4. Büyük videolar için disk alanını kontrol edin
# 5. FPS değeri performance'ı etkiler