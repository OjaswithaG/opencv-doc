"""
Plaka Tanıma Sistemi
OpenCV ile Plaka Tanıma ve OCR Projesi

Bu proje, araç plakalarını tespit edip tanıma tekniklerini
kullanarak gerçek zamanlı bir sistem geliştirir.

Yazar: Eren Terzi
Tarih: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import re
import time
import os
import warnings
warnings.filterwarnings('ignore')

class PlakaTanimaSistemi:
    """Plaka tanıma sistemi"""
    
    def __init__(self):
        """Sistem başlatma"""
        # Tesseract OCR yolu (Windows için)
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Plaka tespit için cascade classifier
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
        
        # Türk plaka formatı regex
        self.plate_pattern = re.compile(r'^[0-9]{2}\s*[A-Z]{1,3}\s*[0-9]{2,4}$')
        
        # Tanınan plakalar veritabanı
        self.recognized_plates = {}
        
    def goruntu_on_isleme(self, frame):
        """Görüntü ön işleme"""
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Kenar tespit
        edges = cv2.Canny(gray, 30, 200)
        
        return gray, edges
        
    def kontur_tespit(self, edges):
        """Kontur tespit et"""
        # Konturları bul
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Konturları boyuta göre sırala
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_contours = []
        
        for contour in contours:
            # Kontur çevresini hesapla
            perimeter = cv2.arcLength(contour, True)
            
            # Yaklaşık kontur
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Dikdörtgen şeklinde olanları seç
            if len(approx) == 4:
                plate_contours.append(approx)
                
        return plate_contours
        
    def plaka_roi_cikar(self, frame, contour):
        """Plaka ROI çıkar"""
        x, y, w, h = cv2.boundingRect(contour)
        
        # ROI çıkar
        plate_roi = frame[y:y+h, x:x+w]
        
        return plate_roi, (x, y, w, h)
        
    def plaka_metin_tanima(self, plate_roi):
        """Plaka metnini tanı"""
        try:
            # Görüntüyü PIL formatına çevir
            pil_image = Image.fromarray(plate_roi)
            
            # OCR ayarları
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            
            # OCR uygula
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            # Metni temizle
            text = re.sub(r'[^A-Z0-9\s]', '', text.upper())
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            print(f"OCR hatası: {e}")
            return ""
            
    def plaka_dogrula(self, text):
        """Plaka formatını doğrula"""
        if not text:
            return False, ""
            
        # Türk plaka formatını kontrol et
        if self.plate_pattern.match(text):
            return True, text
            
        # Alternatif formatlar
        # 34ABC123 formatı
        alt_pattern = re.compile(r'^[0-9]{2}[A-Z]{1,3}[0-9]{2,4}$')
        if alt_pattern.match(text):
            return True, text
            
        return False, text
        
    def plaka_gorselleştir(self, frame, plate_roi, plate_coords, text, is_valid):
        """Plaka görselleştirme"""
        x, y, w, h = plate_coords
        
        # Plaka çerçevesi çiz
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Metin ekle
        if text:
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Geçerlilik durumu
        status = "GECERLI" if is_valid else "GECERSIZ"
        cv2.putText(frame, status, (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                   
    def plaka_tespit_klasik(self, frame):
        """Klasik yöntemle plaka tespit"""
        gray, edges = self.goruntu_on_isleme(frame)
        
        # Kontur tespit
        plate_contours = self.kontur_tespit(edges)
        
        detected_plates = []
        
        for contour in plate_contours:
            # ROI çıkar
            plate_roi, coords = self.plaka_roi_cikar(frame, contour)
            
            # Boyut kontrolü
            x, y, w, h = coords
            if w < 100 or h < 20:  # Çok küçük alanları atla
                continue
                
            # Metin tanıma
            text = self.plaka_metin_tanima(plate_roi)
            
            # Plaka doğrulama
            is_valid, clean_text = self.plaka_dogrula(text)
            
            if text:
                detected_plates.append({
                    'roi': plate_roi,
                    'coords': coords,
                    'text': clean_text,
                    'is_valid': is_valid
                })
                
        return detected_plates
        
    def plaka_tespit_cascade(self, frame):
        """Cascade classifier ile plaka tespit"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Plaka tespit
        plates = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 30)
        )
        
        detected_plates = []
        
        for (x, y, w, h) in plates:
            # ROI çıkar
            plate_roi = frame[y:y+h, x:x+w]
            
            # Metin tanıma
            text = self.plaka_metin_tanima(plate_roi)
            
            # Plaka doğrulama
            is_valid, clean_text = self.plaka_dogrula(text)
            
            if text:
                detected_plates.append({
                    'roi': plate_roi,
                    'coords': (x, y, w, h),
                    'text': clean_text,
                    'is_valid': is_valid
                })
                
        return detected_plates
        
    def gercek_zamanli_tanima(self):
        """Gerçek zamanlı plaka tanıma"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
            
        print("📹 Gerçek zamanlı plaka tanıma başlatıldı!")
        print("🔑 Kontroller:")
        print("  'q' - Çıkış")
        print("  's' - Ekran görüntüsü al")
        print("  'c' - Cascade modunu değiştir")
        print("  'k' - Klasik modunu değiştir")
        
        use_cascade = True
        detection_mode = "Cascade"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Plaka tespit
            if use_cascade:
                detected_plates = self.plaka_tespit_cascade(frame)
            else:
                detected_plates = self.plaka_tespit_klasik(frame)
                
            # Sonuçları görselleştir
            for plate in detected_plates:
                self.plaka_gorselleştir(
                    frame, 
                    plate['roi'], 
                    plate['coords'], 
                    plate['text'], 
                    plate['is_valid']
                )
                
                # Geçerli plakaları kaydet
                if plate['is_valid'] and plate['text']:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.recognized_plates[plate['text']] = timestamp
                    
            # Mod bilgisi
            cv2.putText(frame, f"Mod: {detection_mode}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Tespit edilen plaka sayısı
            cv2.putText(frame, f"Tespit: {len(detected_plates)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Plaka Tanıma Sistemi', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                use_cascade = True
                detection_mode = "Cascade"
                print("🔄 Cascade modu aktif")
            elif key == ord('k'):
                use_cascade = False
                detection_mode = "Klasik"
                print("🔄 Klasik modu aktif")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"plaka_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Ekran görüntüsü kaydedildi: {filename}")
                
        cap.release()
        cv2.destroyAllWindows()
        
    def resim_analizi(self, image_path):
        """Resim dosyasından plaka analizi"""
        if not os.path.exists(image_path):
            print(f"❌ Dosya bulunamadı: {image_path}")
            return
            
        # Resmi yükle
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Resim yüklenemedi: {image_path}")
            return
            
        print(f"📸 Resim analiz ediliyor: {image_path}")
        
        # Her iki yöntemi dene
        methods = [
            ("Cascade", self.plaka_tespit_cascade),
            ("Klasik", self.plaka_tespit_klasik)
        ]
        
        for method_name, method_func in methods:
            print(f"\n🔍 {method_name} yöntemi:")
            detected_plates = method_func(frame)
            
            if detected_plates:
                for i, plate in enumerate(detected_plates):
                    print(f"  Plaka {i+1}: {plate['text']} ({'Geçerli' if plate['is_valid'] else 'Geçersiz'})")
                    
                    # Plaka ROI'sini göster
                    cv2.imshow(f'Plaka {i+1} - {method_name}', plate['roi'])
            else:
                print("  Plaka tespit edilemedi")
                
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def plaka_veritabani(self):
        """Tanınan plakaları listele"""
        print("\n📋 TANINAN PLAKALAR")
        print("=" * 50)
        
        if not self.recognized_plates:
            print("❌ Henüz plaka tanınmadı!")
            return
            
        for plate, timestamp in self.recognized_plates.items():
            print(f"  🚗 {plate} - {timestamp}")
            
    def plaka_istatistikleri(self):
        """Plaka tanıma istatistikleri"""
        print("\n📊 PLAKA TANIMA İSTATİSTİKLERİ")
        print("=" * 50)
        
        total_plates = len(self.recognized_plates)
        valid_plates = len([p for p in self.recognized_plates.keys() if self.plaka_dogrula(p)[0]])
        
        print(f"📈 Toplam tanınan plaka: {total_plates}")
        print(f"✅ Geçerli plaka: {valid_plates}")
        print(f"❌ Geçersiz plaka: {total_plates - valid_plates}")
        
        if total_plates > 0:
            success_rate = (valid_plates / total_plates) * 100
            print(f"🎯 Başarı oranı: {success_rate:.1f}%")
            
    def plaka_test_verisi_olustur(self):
        """Test verisi oluştur"""
        print("🧪 Test verisi oluşturuluyor...")
        
        # Örnek plaka görüntüleri oluştur
        test_plates = [
            "34ABC123",
            "06XYZ789", 
            "35DEF456",
            "01GHI012"
        ]
        
        for i, plate_text in enumerate(test_plates):
            # Plaka görüntüsü oluştur
            img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            
            # Metin ekle
            cv2.putText(img, plate_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            
            # Kaydet
            filename = f"test_plaka_{i+1}.jpg"
            cv2.imwrite(filename, img)
            print(f"✅ {filename} oluşturuldu")
            
    def performans_testi(self, test_images):
        """Performans testi yap"""
        print("⚡ Performans testi başlatılıyor...")
        
        results = {
            'cascade': {'total': 0, 'success': 0, 'time': 0},
            'klasik': {'total': 0, 'success': 0, 'time': 0}
        }
        
        methods = [
            ("cascade", self.plaka_tespit_cascade),
            ("klasik", self.plaka_tespit_klasik)
        ]
        
        for method_name, method_func in methods:
            print(f"\n🔍 {method_name.upper()} yöntemi test ediliyor...")
            
            for image_path in test_images:
                if not os.path.exists(image_path):
                    continue
                    
                frame = cv2.imread(image_path)
                if frame is None:
                    continue
                    
                # Zaman ölçümü
                start_time = time.time()
                detected_plates = method_func(frame)
                end_time = time.time()
                
                results[method_name]['total'] += 1
                results[method_name]['time'] += (end_time - start_time)
                
                if detected_plates:
                    results[method_name]['success'] += 1
                    
        # Sonuçları göster
        print("\n📊 PERFORMANS SONUÇLARI")
        print("=" * 50)
        
        for method, data in results.items():
            if data['total'] > 0:
                success_rate = (data['success'] / data['total']) * 100
                avg_time = data['time'] / data['total']
                
                print(f"\n{method.upper()}:")
                print(f"  Başarı oranı: {success_rate:.1f}%")
                print(f"  Ortalama süre: {avg_time:.3f} saniye")
                print(f"  Toplam test: {data['total']}")

def demo_menu():
    """Demo menüsü"""
    sistem = PlakaTanimaSistemi()
    
    while True:
        print("\n" + "="*60)
        print("🚗 PLAKA TANIMA SİSTEMİ")
        print("="*60)
        print("1. 📹 Gerçek Zamanlı Tanıma")
        print("2. 📸 Resim Analizi")
        print("3. 📋 Plaka Veritabanı")
        print("4. 📊 İstatistikler")
        print("5. 🧪 Test Verisi Oluştur")
        print("6. ⚡ Performans Testi")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-6): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                sistem.gercek_zamanli_tanima()
            elif secim == "2":
                image_path = input("Resim dosya yolunu girin: ").strip()
                if image_path:
                    sistem.resim_analizi(image_path)
                else:
                    print("❌ Dosya yolu belirtilmedi!")
            elif secim == "3":
                sistem.plaka_veritabani()
            elif secim == "4":
                sistem.plaka_istatistikleri()
            elif secim == "5":
                sistem.plaka_test_verisi_olustur()
            elif secim == "6":
                test_images = ["test_plaka_1.jpg", "test_plaka_2.jpg", 
                              "test_plaka_3.jpg", "test_plaka_4.jpg"]
                sistem.performans_testi(test_images)
            else:
                print("❌ Geçersiz seçim! Lütfen 0-6 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n👋 Program sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    print("🚗 Plaka Tanıma Sistemi")
    print("=" * 60)
    print("Bu proje, plaka tanıma ve OCR tekniklerini uygular.")
    print("📚 Öğrenilecek Konular:")
    print("  • Görüntü ön işleme teknikleri")
    print("  • Kontur tespit ve analiz")
    print("  • OCR (Optical Character Recognition)")
    print("  • Gerçek zamanlı video işleme")
    print("  • Performans optimizasyonu")
    

    demo_menu() 
