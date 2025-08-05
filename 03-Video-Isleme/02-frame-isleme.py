#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖼️ Frame İşleme - OpenCV Video Frame Manipülasyonu
================================================

Bu modül video frame'lerinin işlenmesi konularını kapsar:
- Frame-by-frame video işleme
- Frame manipülasyonu ve transformasyonlar  
- Frame buffer yönetimi
- Synchronized frame işleme
- Frame interpolation ve blending

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from collections import deque
from pathlib import Path
import threading
import queue

class FrameProcessor:
    """Frame işleme sınıfı"""
    
    def __init__(self, buffer_size=10):
        self.frame_buffer = deque(maxlen=buffer_size)
        self.processed_buffer = deque(maxlen=buffer_size)
        self.frame_count = 0
        
    def add_frame(self, frame):
        """Buffer'a frame ekle"""
        self.frame_buffer.append(frame.copy())
        self.frame_count += 1
    
    def get_current_frame(self):
        """Mevcut frame'i al"""
        return self.frame_buffer[-1] if self.frame_buffer else None
    
    def get_previous_frame(self, steps_back=1):
        """Önceki frame'i al"""
        if len(self.frame_buffer) > steps_back:
            return self.frame_buffer[-(steps_back + 1)]
        return None

def ornek_1_frame_by_frame_islreme():
    """
    Örnek 1: Frame-by-frame video işleme
    """
    print("\n🎯 Örnek 1: Frame-by-Frame İşleme")
    print("=" * 40)
    
    # Test video yolu (önceki modülden)
    video_path = "test_video.avi"
    if not os.path.exists(video_path):
        print("❌ Test videosu bulunamadı. Önce 01-video-temelleri.py çalıştırın.")
        return
    
    cap = cv2.VideoCapture(video_path)
    processor = FrameProcessor()
    
    print("📝 İşleme seçenekleri:")
    print("1: Orijinal")
    print("2: Gri tonlama")
    print("3: Blur")
    print("4: Kenar algılama")
    print("5: Histogram eşitleme")
    print("ESC: Çıkış, Space: Duraklat")
    
    processing_mode = 1
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Video sona erdi, başa dön
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            processor.add_frame(frame)
            
            # Frame işleme modlarına göre işle
            if processing_mode == 1:
                processed_frame = frame.copy()
                mode_text = "Orijinal"
            elif processing_mode == 2:
                processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                mode_text = "Gri Tonlama"
            elif processing_mode == 3:
                processed_frame = cv2.GaussianBlur(frame, (15, 15), 0)
                mode_text = "Gaussian Blur"
            elif processing_mode == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                mode_text = "Kenar Algılama"
            elif processing_mode == 5:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
                processed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                mode_text = "Histogram Eşitleme"
            
            # Bilgi metni ekle
            cv2.putText(processed_frame, f"Mod: {mode_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Frame: {processor.frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if paused:
                cv2.putText(processed_frame, "PAUSED", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Frame İşleme', processed_frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Space
            paused = not paused
        elif key >= ord('1') and key <= ord('5'):
            processing_mode = key - ord('0')
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_frame_differencing():
    """
    Örnek 2: Frame farkı alma (Frame Differencing)
    """
    print("\n🎯 Örnek 2: Frame Differencing")
    print("=" * 35)
    
    # Webcam kullan (daha iyi sonuç için)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    # İlk frame'i al
    ret, previous_frame = cap.read()
    if not ret:
        print("❌ İlk frame alınamadı!")
        return
    
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    
    print("👋 Kameranın önünde hareket edin!")
    print("ESC: Çıkış, 't': Threshold ayarla")
    
    threshold_value = 30
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Frame farkını hesapla
        frame_diff = cv2.absdiff(previous_gray, current_gray)
        
        # Threshold uygula
        _, threshold_diff = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Morfolojik işlemler (gürültü azaltma)
        kernel = np.ones((3, 3), np.uint8)
        threshold_diff = cv2.morphologyEx(threshold_diff, cv2.MORPH_OPENING, kernel)
        threshold_diff = cv2.morphologyEx(threshold_diff, cv2.MORPH_CLOSING, kernel)
        
        # Hareket alanını hesapla
        motion_area = cv2.countNonZero(threshold_diff)
        motion_percentage = (motion_area / (current_frame.shape[0] * current_frame.shape[1])) * 100
        
        # Görselleştirme
        # Orijinal frame
        display_frame = current_frame.copy()
        cv2.putText(display_frame, f"Hareket: %{motion_percentage:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Threshold: {threshold_value}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame diff'i renkli yap
        threshold_diff_colored = cv2.cvtColor(threshold_diff, cv2.COLOR_GRAY2BGR)
        
        # Yan yana göster
        combined = np.hstack((display_frame, threshold_diff_colored))
        cv2.imshow('Frame Differencing - Orijinal | Hareket', combined)
        
        # Önceki frame'i güncelle
        previous_gray = current_gray.copy()
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('t'):
            # Threshold ayarlama
            new_threshold = input(f"Yeni threshold değeri (mevcut: {threshold_value}): ")
            try:
                threshold_value = int(new_threshold)
                threshold_value = max(1, min(255, threshold_value))
            except:
                print("Geçersiz değer!")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_frame_blending():
    """
    Örnek 3: Frame blending ve interpolation
    """
    print("\n🎯 Örnek 3: Frame Blending")
    print("=" * 30)
    
    # Test video
    video_path = "test_video.avi"
    if not os.path.exists(video_path):
        print("❌ Test videosu bulunamadı.")
        return
    
    cap = cv2.VideoCapture(video_path)
    processor = FrameProcessor(buffer_size=5)
    
    print("Blending modları:")
    print("1: Normal")
    print("2: Ortalama blend (3 frame)")
    print("3: Hareket blur")
    print("4: Ghost effect")
    
    blend_mode = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        processor.add_frame(frame)
        
        if blend_mode == 1:
            # Normal frame
            result = frame.copy()
            mode_text = "Normal"
            
        elif blend_mode == 2 and len(processor.frame_buffer) >= 3:
            # Son 3 frame'in ortalaması
            frames = list(processor.frame_buffer)[-3:]
            result = np.zeros_like(frame, dtype=np.float32)
            for f in frames:
                result += f.astype(np.float32)
            result = (result / len(frames)).astype(np.uint8)
            mode_text = "Ortalama Blend"
            
        elif blend_mode == 3 and len(processor.frame_buffer) >= 2:
            # Hareket blur efekti
            current = processor.get_current_frame().astype(np.float32)
            previous = processor.get_previous_frame(1).astype(np.float32)
            # Ağırlıklı ortalama
            result = (0.7 * current + 0.3 * previous).astype(np.uint8)
            mode_text = "Hareket Blur"
            
        elif blend_mode == 4 and len(processor.frame_buffer) >= 3:
            # Ghost efekti - önceki frame'leri şeffaf olarak bindirme
            result = processor.get_current_frame().copy().astype(np.float32)
            if len(processor.frame_buffer) >= 2:
                prev1 = processor.get_previous_frame(1).astype(np.float32)
                result = 0.8 * result + 0.2 * prev1
            if len(processor.frame_buffer) >= 3:
                prev2 = processor.get_previous_frame(2).astype(np.float32)
                result = 0.9 * result + 0.1 * prev2
            result = result.astype(np.uint8)
            mode_text = "Ghost Effect"
        else:
            result = frame.copy()
            mode_text = "Loading..."
        
        # Bilgi ekle
        cv2.putText(result, f"Mod: {mode_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Buffer: {len(processor.frame_buffer)}/5", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Frame Blending', result)
        
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break
        elif key >= ord('1') and key <= ord('4'):
            blend_mode = key - ord('0')
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_4_frame_buffer_yonetimi():
    """
    Örnek 4: Frame buffer yönetimi ve optimize işleme
    """
    print("\n🎯 Örnek 4: Frame Buffer Yönetimi")
    print("=" * 40)
    
    class OptimizedFrameProcessor:
        def __init__(self, buffer_size=20):
            self.buffer = deque(maxlen=buffer_size)
            self.processed_cache = {}
            self.frame_index = 0
        
        def add_frame(self, frame):
            self.buffer.append({
                'frame': frame,
                'index': self.frame_index,
                'timestamp': time.time()
            })
            self.frame_index += 1
        
        def get_frame_stats(self):
            if not self.buffer:
                return {}
            
            # Buffer istatistikleri
            current_time = time.time()
            frame_times = [f['timestamp'] for f in self.buffer]
            
            if len(frame_times) > 1:
                fps = len(frame_times) / (max(frame_times) - min(frame_times))
                avg_interval = (max(frame_times) - min(frame_times)) / (len(frame_times) - 1)
            else:
                fps = 0
                avg_interval = 0
            
            return {
                'buffer_size': len(self.buffer),
                'max_buffer': self.buffer.maxlen,
                'fps': fps,
                'avg_interval': avg_interval,
                'cache_size': len(self.processed_cache)
            }
        
        def process_with_history(self, process_func, history_length=5):
            if len(self.buffer) < history_length:
                return None
            
            # Son N frame'i al
            recent_frames = [item['frame'] for item in list(self.buffer)[-history_length:]]
            return process_func(recent_frames)
    
    # Webcam ile test
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Webcam yoksa test video kullan
        cap = cv2.VideoCapture("test_video.avi")
    
    processor = OptimizedFrameProcessor(buffer_size=30)
    
    def temporal_average(frames):
        """Temporal averaging işlemi"""
        if not frames:
            return frames[0]
        
        result = np.zeros_like(frames[0], dtype=np.float32)
        for frame in frames:
            result += frame.astype(np.float32)
        return (result / len(frames)).astype(np.uint8)
    
    def motion_magnitude(frames):
        """Hareket büyüklüğü hesaplama"""
        if len(frames) < 2:
            return frames[0]
        
        # Son iki frame arasındaki fark
        current = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
        previous = cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(current, previous)
        # Farkı renkli göster
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        return diff_colored
    
    process_mode = 1
    start_time = time.time()
    
    print("İşleme modları:")
    print("1: Normal")
    print("2: Temporal Average (5 frame)")
    print("3: Motion Magnitude")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Test video için döngü
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        processor.add_frame(frame)
        
        # İşleme moduna göre frame işle
        if process_mode == 1:
            result = frame.copy()
        elif process_mode == 2:
            result = processor.process_with_history(temporal_average, 5)
            if result is None:
                result = frame.copy()
        elif process_mode == 3:
            result = processor.process_with_history(motion_magnitude, 2)
            if result is None:
                result = frame.copy()
        
        # İstatistikleri al
        stats = processor.get_frame_stats()
        
        # Bilgileri göster
        y_offset = 30
        cv2.putText(result, f"Mod: {process_mode}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(result, f"Buffer: {stats.get('buffer_size', 0)}/{stats.get('max_buffer', 0)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(result, f"FPS: {stats.get('fps', 0):.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        # Çalışma süresi
        elapsed = time.time() - start_time
        cv2.putText(result, f"Süre: {elapsed:.1f}s", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Frame Buffer Yönetimi', result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key >= ord('1') and key <= ord('3'):
            process_mode = key - ord('0')
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_5_threaded_processing():
    """
    Örnek 5: Multi-threaded frame işleme
    """
    print("\n🎯 Örnek 5: Multi-threaded İşleme")
    print("=" * 40)
    
    class ThreadedFrameProcessor:
        def __init__(self):
            self.input_queue = queue.Queue(maxsize=10)
            self.output_queue = queue.Queue(maxsize=10)
            self.running = False
            self.process_thread = None
            self.stats = {
                'processed_frames': 0,
                'processing_time': 0,
                'queue_full_count': 0
            }
        
        def start_processing(self):
            self.running = True
            self.process_thread = threading.Thread(target=self._process_worker)
            self.process_thread.start()
        
        def stop_processing(self):
            self.running = False
            if self.process_thread:
                self.process_thread.join()
        
        def add_frame(self, frame):
            try:
                self.input_queue.put(frame, block=False)
                return True
            except queue.Full:
                self.stats['queue_full_count'] += 1
                return False
        
        def get_processed_frame(self):
            try:
                return self.output_queue.get(block=False)
            except queue.Empty:
                return None
        
        def _process_worker(self):
            while self.running:
                try:
                    frame = self.input_queue.get(timeout=0.1)
                    
                    # İşleme başlangıcı
                    start_time = time.time()
                    
                    # Yoğun işleme simülasyonu - birden fazla filtre
                    processed = frame.copy()
                    
                    # 1. Gaussian blur
                    processed = cv2.GaussianBlur(processed, (5, 5), 0)
                    
                    # 2. Edge detection
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    
                    # 3. Edge'leri orijinalle birleştir
                    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    processed = cv2.addWeighted(processed, 0.7, edges_colored, 0.3, 0)
                    
                    # İşleme süresi
                    process_time = time.time() - start_time
                    self.stats['processing_time'] += process_time
                    self.stats['processed_frames'] += 1
                    
                    # İstatistikleri frame'e ekle
                    avg_time = self.stats['processing_time'] / self.stats['processed_frames']
                    cv2.putText(processed, f"Avg Process: {avg_time*1000:.1f}ms", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(processed, f"Frames: {self.stats['processed_frames']}", (10, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(processed, f"Queue Full: {self.stats['queue_full_count']}", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Sonucu output queue'ya ekle
                    try:
                        self.output_queue.put(processed, block=False)
                    except queue.Full:
                        # Output queue dolu, eski frame'i at
                        try:
                            self.output_queue.get(block=False)
                            self.output_queue.put(processed, block=False)
                        except queue.Empty:
                            pass
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"İşleme hatası: {e}")
    
    # Webcam ile test
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.avi")
    
    processor = ThreadedFrameProcessor()
    processor.start_processing()
    
    print("📊 Threaded processing başlatıldı")
    print("ESC: Çıkış")
    
    last_fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Video döngüsü
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Frame'i işleme kuyruğuna ekle
            frame_added = processor.add_frame(frame)
            
            # İşlenmiş frame'i al
            processed_frame = processor.get_processed_frame()
            
            # Gösterilecek frame'i belirle
            if processed_frame is not None:
                display_frame = processed_frame
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, "İşleniyor...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # FPS hesaplama
            fps_counter += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                current_fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time
            
            # FPS bilgisini ekle
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, display_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Queue durumlarını göster
            input_size = processor.input_queue.qsize()
            output_size = processor.output_queue.qsize()
            cv2.putText(display_frame, f"Queue: {input_size}|{output_size}", (10, display_frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Threaded Processing', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    
    finally:
        processor.stop_processing()
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 İşleme İstatistikleri:")
        print(f"   Toplam işlenen frame: {processor.stats['processed_frames']}")
        print(f"   Ortalama işleme süresi: {processor.stats['processing_time']/max(1, processor.stats['processed_frames'])*1000:.1f}ms")
        print(f"   Queue dolu sayısı: {processor.stats['queue_full_count']}")

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🖼️ OpenCV Frame İşleme Demo")
        print("="*50)
        print("1. 🎞️ Frame-by-Frame İşleme")
        print("2. 📊 Frame Differencing")
        print("3. 🌈 Frame Blending & Interpolation")
        print("4. 💾 Frame Buffer Yönetimi")
        print("5. ⚡ Multi-threaded İşleme")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-5): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_frame_by_frame_islreme()
            elif secim == "2":
                ornek_2_frame_differencing()
            elif secim == "3":
                ornek_3_frame_blending()
            elif secim == "4":  
                ornek_4_frame_buffer_yonetimi()
            elif secim == "5":
                ornek_5_threaded_processing()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-5 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🖼️ OpenCV Frame İşleme")
    print("Bu modül frame-by-frame video işleme tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Threading (built-in)")
    print("   - Webcam (bazı örnekler için)")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Frame buffer boyutu RAM kullanımını etkiler
# 2. Multi-threading performansı artırabilir
# 3. Queue boyutları sistemin hızına göre ayarlayın
# 4. Frame differencing hareket algılama için temeldir
# 5. Blending efektleri yaratıcı video düzenleme için kullanışlıdır