"""
Hareket Algılama ve Güvenlik Sistemi
OpenCV ile Hareket Algılama ve Güvenlik Projesi

Bu proje, hareket algılama tekniklerini kullanarak
güvenlik sistemi geliştirir.

Yazar: OpenCV Türkiye Topluluğu
Tarih: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

class HareketAlgilamaSistemi:
    """Hareket algılama ve güvenlik sistemi"""
    
    def __init__(self):
        """Sistem başlatma"""
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        
        # Hareket algılama parametreleri
        self.min_area = 500  # Minimum hareket alanı
        self.threshold = 25   # Hareket eşiği
        
        # Güvenlik ayarları
        self.recording = False
        self.alerts = []
        self.motion_history = []
        
        # Zaman takibi
        self.last_motion_time = None
        self.motion_duration = 0
        
    def hareket_tespit(self, frame):
        """Hareket tespit et"""
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Gürültü azaltma
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Kontur tespit
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > self.min_area:
                # Kontur çerçevesi
                x, y, w, h = cv2.boundingRect(contour)
                
                # Hareket bölgesi
                motion_regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })
                
        return motion_regions, fg_mask
        
    def hareket_analizi(self, motion_regions, frame):
        """Hareket analizi yap"""
        current_time = time.time()
        
        # Hareket tespit edildi mi?
        motion_detected = len(motion_regions) > 0
        
        if motion_detected:
            if self.last_motion_time is None:
                # Yeni hareket başladı
                self.last_motion_time = current_time
                self.motion_duration = 0
                
                # Uyarı oluştur
                alert = {
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'type': 'motion_start',
                    'regions': len(motion_regions),
                    'total_area': sum(r['area'] for r in motion_regions)
                }
                self.alerts.append(alert)
                
            else:
                # Hareket devam ediyor
                self.motion_duration = current_time - self.last_motion_time
                
        else:
            if self.last_motion_time is not None:
                # Hareket bitti
                motion_end_time = current_time
                total_duration = motion_end_time - self.last_motion_time
                
                # Hareket geçmişine ekle
                self.motion_history.append({
                    'start_time': self.last_motion_time,
                    'end_time': motion_end_time,
                    'duration': total_duration
                })
                
                self.last_motion_time = None
                self.motion_duration = 0
                
        return motion_detected
        
    def hareket_gorselleştir(self, frame, motion_regions, motion_detected):
        """Hareket görselleştirme"""
        # Hareket bölgelerini çiz
        for region in motion_regions:
            x, y, w, h = region['bbox']
            center_x, center_y = region['center']
            
            # Çerçeve çiz
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Merkez nokta
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Alan bilgisi
            cv2.putText(frame, f"Area: {region['area']}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # Hareket durumu
        if motion_detected:
            status_color = (0, 0, 255)  # Kırmızı
            status_text = "HAREKET TESPIT"
            
            # Süre bilgisi
            if self.motion_duration > 0:
                duration_text = f"Sure: {self.motion_duration:.1f}s"
                cv2.putText(frame, duration_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            status_color = (0, 255, 0)  # Yeşil
            status_text = "SESSIZ"
            
        # Durum metni
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Bölge sayısı
        cv2.putText(frame, f"Bolge: {len(motion_regions)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    def kayit_baslat(self):
        """Video kayıt başlat"""
        if not self.recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"motion_recording_{timestamp}.avi"
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            
            self.recording = True
            print(f"📹 Kayıt başlatıldı: {filename}")
            
    def kayit_durdur(self):
        """Video kayıt durdur"""
        if self.recording:
            self.video_writer.release()
            self.recording = False
            print("📹 Kayıt durduruldu")
            
    def gercek_zamanli_izleme(self):
        """Gerçek zamanlı hareket izleme"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
            
        print("📹 Gerçek zamanlı hareket izleme başlatıldı!")
        print("🔑 Kontroller:")
        print("  'q' - Çıkış")
        print("  'r' - Kayıt başlat/durdur")
        print("  's' - Ekran görüntüsü al")
        print("  'a' - Uyarı ayarları")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Frame boyutunu ayarla
            frame = cv2.resize(frame, (640, 480))
            
            # Hareket tespit
            motion_regions, fg_mask = self.hareket_tespit(frame)
            
            # Hareket analizi
            motion_detected = self.hareket_analizi(motion_regions, frame)
            
            # Görselleştirme
            self.hareket_gorselleştir(frame, motion_regions, motion_detected)
            
            # Kayıt
            if self.recording:
                self.video_writer.write(frame)
                cv2.putText(frame, "REC", (frame.shape[1]-80, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Zaman bilgisi
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, current_time, (frame.shape[1]-120, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Görüntüleri göster
            cv2.imshow('Hareket Algılama Sistemi', frame)
            cv2.imshow('Motion Mask', fg_mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if self.recording:
                    self.kayit_durdur()
                else:
                    self.kayit_baslat()
            elif key == ord('s'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"motion_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Ekran görüntüsü kaydedildi: {filename}")
            elif key == ord('a'):
                self.uyari_ayarlari()
                
        # Kayıt durdur
        if self.recording:
            self.kayit_durdur()
            
        cap.release()
        cv2.destroyAllWindows()
        
    def uyari_ayarlari(self):
        """Uyarı ayarları"""
        print("\n⚙️ UYARI AYARLARI")
        print("=" * 50)
        
        try:
            min_area = input(f"Minimum hareket alanı (şu an: {self.min_area}): ").strip()
            if min_area:
                self.min_area = int(min_area)
                
            threshold = input(f"Hareket eşiği (şu an: {self.threshold}): ").strip()
            if threshold:
                self.threshold = int(threshold)
                
            print("✅ Ayarlar güncellendi!")
            
        except ValueError:
            print("❌ Geçersiz değer!")
            
    def istatistikler(self):
        """Hareket istatistikleri"""
        print("\n📊 HAREKET İSTATİSTİKLERİ")
        print("=" * 50)
        
        total_alerts = len(self.alerts)
        total_motions = len(self.motion_history)
        
        print(f"🚨 Toplam uyarı: {total_alerts}")
        print(f"📈 Toplam hareket: {total_motions}")
        
        if total_motions > 0:
            avg_duration = sum(m['duration'] for m in self.motion_history) / total_motions
            print(f"⏱️ Ortalama hareket süresi: {avg_duration:.2f} saniye")
            
            max_duration = max(m['duration'] for m in self.motion_history)
            print(f"⏱️ En uzun hareket: {max_duration:.2f} saniye")
            
        if total_alerts > 0:
            print(f"\n📋 Son 5 uyarı:")
            for alert in self.alerts[-5:]:
                print(f"  🚨 {alert['timestamp']} - {alert['type']} ({alert['regions']} bölge)")
                
    def hareket_gecmisi(self):
        """Hareket geçmişi"""
        print("\n📋 HAREKET GEÇMİŞİ")
        print("=" * 50)
        
        if not self.motion_history:
            print("❌ Henüz hareket kaydedilmedi!")
            return
            
        print("📅 Hareket kayıtları:")
        for i, motion in enumerate(self.motion_history[-10:], 1):
            start_time = datetime.datetime.fromtimestamp(motion['start_time']).strftime("%H:%M:%S")
            end_time = datetime.datetime.fromtimestamp(motion['end_time']).strftime("%H:%M:%S")
            duration = motion['duration']
            
            print(f"  {i}. {start_time} - {end_time} ({duration:.1f}s)")
            
    def veri_kaydet(self, filename="motion_data.json"):
        """Veriyi kaydet"""
        data = {
            'alerts': self.alerts,
            'motion_history': self.motion_history,
            'settings': {
                'min_area': self.min_area,
                'threshold': self.threshold
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Veri kaydedildi: {filename}")
        
    def veri_yukle(self, filename="motion_data.json"):
        """Veriyi yükle"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.alerts = data.get('alerts', [])
            self.motion_history = data.get('motion_history', [])
            
            settings = data.get('settings', {})
            self.min_area = settings.get('min_area', 500)
            self.threshold = settings.get('threshold', 25)
            
            print(f"✅ Veri yüklendi: {filename}")
        else:
            print(f"❌ Dosya bulunamadı: {filename}")
            
    def performans_testi(self, test_duration=30):
        """Performans testi"""
        print(f"⚡ {test_duration} saniyelik performans testi başlatılıyor...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
            
        start_time = time.time()
        frame_count = 0
        motion_count = 0
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.resize(frame, (640, 480))
            
            # Hareket tespit
            motion_regions, _ = self.hareket_tespit(frame)
            motion_detected = self.hareket_analizi(motion_regions, frame)
            
            frame_count += 1
            if motion_detected:
                motion_count += 1
                
        cap.release()
        
        # Sonuçları hesapla
        total_time = time.time() - start_time
        fps = frame_count / total_time
        motion_rate = (motion_count / frame_count) * 100 if frame_count > 0 else 0
        
        print("\n📊 PERFORMANS SONUÇLARI")
        print("=" * 50)
        print(f"⏱️ Test süresi: {total_time:.1f} saniye")
        print(f"📹 Toplam frame: {frame_count}")
        print(f"🚀 FPS: {fps:.1f}")
        print(f"🚨 Hareket tespit oranı: {motion_rate:.1f}%")
        
    def demo_modu(self):
        """Demo modu"""
        print("🎯 Demo modu başlatılıyor...")
        print("💡 Bu mod, örnek hareket verileriyle sistemi test eder")
        
        # Demo verileri oluştur
        demo_alerts = [
            {
                'timestamp': '2025-01-15 10:30:15',
                'type': 'motion_start',
                'regions': 2,
                'total_area': 1500
            },
            {
                'timestamp': '2025-01-15 10:32:45',
                'type': 'motion_start',
                'regions': 1,
                'total_area': 800
            }
        ]
        
        demo_motions = [
            {
                'start_time': time.time() - 300,
                'end_time': time.time() - 280,
                'duration': 20.0
            },
            {
                'start_time': time.time() - 150,
                'end_time': time.time() - 140,
                'duration': 10.0
            }
        ]
        
        self.alerts = demo_alerts
        self.motion_history = demo_motions
        
        print("✅ Demo verileri yüklendi!")
        self.istatistikler()

def demo_menu():
    """Demo menüsü"""
    sistem = HareketAlgilamaSistemi()
    
    # Veri yüklemeyi dene
    sistem.veri_yukle()
    
    while True:
        print("\n" + "="*60)
        print("🚨 HAREKET ALGILAMA VE GÜVENLİK SİSTEMİ")
        print("="*60)
        print("1. 📹 Gerçek Zamanlı İzleme")
        print("2. 📊 İstatistikler")
        print("3. 📋 Hareket Geçmişi")
        print("4. 💾 Veri Kaydet")
        print("5. 📂 Veri Yükle")
        print("6. ⚡ Performans Testi")
        print("7. 🎯 Demo Modu")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-7): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                sistem.gercek_zamanli_izleme()
            elif secim == "2":
                sistem.istatistikler()
            elif secim == "3":
                sistem.hareket_gecmisi()
            elif secim == "4":
                sistem.veri_kaydet()
            elif secim == "5":
                filename = input("Veri dosya adını girin: ").strip()
                if filename:
                    sistem.veri_yukle(filename)
                else:
                    sistem.veri_yukle()
            elif secim == "6":
                duration = input("Test süresi (saniye): ").strip()
                try:
                    test_duration = int(duration) if duration else 30
                    sistem.performans_testi(test_duration)
                except ValueError:
                    print("❌ Geçersiz süre!")
            elif secim == "7":
                sistem.demo_modu()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-7 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n👋 Program sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    print("🚨 Hareket Algılama ve Güvenlik Sistemi")
    print("=" * 60)
    print("Bu proje, hareket algılama ve güvenlik tekniklerini uygular.")
    print("📚 Öğrenilecek Konular:")
    print("  • Background subtraction")
    print("  • Motion detection algoritmaları")
    print("  • Video kayıt ve işleme")
    print("  • Gerçek zamanlı izleme")
    print("  • Güvenlik sistemi tasarımı")
    
    demo_menu() 