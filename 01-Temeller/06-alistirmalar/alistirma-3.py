#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Alıştırma 3: Mini Proje - Akıllı Fotoğraf Düzenleyici
=======================================================

Bu alıştırma, öğrendiklerinizi birleştirerek kapsamlı bir mini proje
geliştirmenizi sağlar. Gerçek dünya uygulamasına yakın bir deneyim.

Zorluk Seviyesi: ⚡ İleri
Tahmini Süre: 45-60 dakika
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class FotografDuzenleyici:
    """Akıllı Fotoğraf Düzenleyici Sınıfı"""
    
    def __init__(self):
        self.orijinal_resim = None
        self.guncel_resim = None
        self.gecmis = []  # Geri alma için
        self.output_dir = Path("ciktiler")
        self.output_dir.mkdir(exist_ok=True)
    
    def resim_yukle(self, dosya_yolu):
        """Resim yükleme fonksiyonu"""
        # TODO: Bu fonksiyonu tamamlayın
        pass
    
    def geri_al(self):
        """Son işlemi geri al"""
        # TODO: Bu fonksiyonu tamamlayın
        pass
    
    def otomatik_duzelt(self):
        """Otomatik düzeltme fonksiyonu"""
        # TODO: Bu fonksiyonu tamamlayın
        pass

def gorev_1_sinif_tasarimi():
    """
    🎯 GÖREV 1: Fotoğraf Düzenleyici Sınıfını Tamamlama
    
    Yapılacaklar:
    1. FotografDuzenleyici sınıfındaki boş fonksiyonları tamamlayın:
       - resim_yukle(): Güvenli resim yükleme
       - geri_al(): İşlem geçmişi ile geri alma
       - otomatik_duzelt(): Parlaklık/kontrast otomatik ayarlama
    
    2. Ek olarak şu fonksiyonları ekleyin:
       - parlaklik_ayarla(deger): Parlaklık ayarlama (-100, +100)
       - kontrast_ayarla(deger): Kontrast ayarlama (0.5, 3.0)
       - bulaniklik_ekle(kuvvet): Gaussian blur uygulama
       - keskinlestir(): Unsharp masking ile keskinleştirme
    
    İpucu: Her işlemde geçmişe kaydedin
    """
    print("🎯 GÖREV 1: Sınıf Tasarımı")
    print("-" * 30)
    
    # TODO: FotografDuzenleyici sınıfını tamamlayın
    # İpucu: self.gecmis.append(self.guncel_resim.copy())
    
    print("✅ Sınıf fonksiyonlarınızı tamamlayın")

def gorev_2_batch_islem():
    """
    🎯 GÖREV 2: Toplu Resim İşleme (Batch Processing)
    
    Yapılacaklar:
    1. test-resimleri/ klasöründeki tüm resimleri otomatik işleyin
    2. Her resim için şu işlemleri uygulayın:
       - Otomatik kontrast düzeltme
       - Boyut standardizasyonu (800x600)
       - Dosya formatını JPEG'e çevirme
       - Çıktıları 'islenmis/' klasörüne kaydetme
    
    3. İşlem istatistikleri gösterin:
       - İşlenen dosya sayısı
       - Toplam işlem süresi
       - Ortalama dosya boyutu değişimi
    
    İpucu: os.listdir() ve for döngüsü kullanın
    """
    print("\\n🎯 GÖREV 2: Batch İşlem")
    print("-" * 25)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: time.time() ile süre ölçümü
    # Path.glob("*.jpg") ile dosya filtreleme
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_3_kalite_analizi():
    """
    🎯 GÖREV 3: Resim Kalite Analizi
    
    Yapılacaklar:
    1. Bir resmin kalitesini analiz eden fonksiyon yazın:
       - Parlaklık seviyesi (çok karanlık/aydınlık uyarısı)
       - Kontrast seviyesi (düşük kontrast tespiti)
       - Bulanıklık tespiti (Laplacian variance)
       - Renk dağılımı analizi
    
    2. Kalite raporunu hem konsola hem dosyaya yazdırın
    3. Önerilen düzeltmeleri listeleyin
    4. Düzeltme öncesi/sonrası karşılaştırma gösterin
    
    İpucu: cv2.Laplacian() ile bulanıklık tespiti
    """
    print("\\n🎯 GÖREV 3: Kalite Analizi")
    print("-" * 30)
    
    def kalite_analiz_et(resim):
        """Resim kalitesini analiz eder"""
        # TODO: Kalite analiz fonksiyonu
        pass
    
    def oneri_ver(analiz_sonucu):
        """Analiz sonucuna göre öneri verir"""
        # TODO: Öneri fonksiyonu
        pass
    
    # TODO: Test resimleri üzerinde kalite analizi yapın
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_4_filtreleme_sistemi():
    """
    🎯 GÖREV 4: Gelişmiş Filterleme Sistemi
    
    Yapılacaklar:
    1. En az 5 farklı fotoğraf filtresi oluşturun:
       - Vintage (eski fotoğraf efekti)
       - Soğuk ton (mavi vurgu)
       - Sıcak ton (sarı/turuncu vurgu)
       - Yüksek kontrast (dramatik)
       - Yumuşak (soft focus)
    
    2. Her filtre için:
       - Özel renk transformasyonu
       - Kontrast/parlaklık ayarı
       - Özel efektler (vignette, grain)
    
    3. Filtreleri karşılaştırmalı gösterin
    4. En iyi filtreyi otomatik seçme algoritması
    
    İpucu: LUT (Look-Up Table) kullanabilirsiniz
    """
    print("\\n🎯 GÖREV 4: Filtreleme Sistemi")
    print("-" * 35)
    
    def vintage_filtre(resim):
        """Vintage fotoğraf efekti"""
        # TODO: Vintage filtre implementasyonu
        pass
    
    def soguk_ton_filtre(resim):
        """Soğuk ton filtresi"""
        # TODO: Soğuk ton implementasyonu
        pass
    
    # TODO: Diğer filtreleri de implementeyin
    # TODO: Filtre karşılaştırma sistemi
    
    pass

def gorev_5_interaktif_arayuz():
    """
    🎯 GÖREV 5: İnteraktif Kullanıcı Arayüzü
    
    Yapılacaklar:
    1. OpenCV trackbar'ları ile kontrol paneli:
       - Parlaklık kontrolü (-100, +100)
       - Kontrast kontrolü (0.5, 3.0)
       - Doygunluk kontrolü (0.0, 2.0)
       - Sıcaklık kontrolü (3000K, 8000K)
    
    2. Klavye kısayolları:
       - 'r': Reset (sıfırla)
       - 's': Save (kaydet)
       - 'u': Undo (geri al)
       - 'f': Filter menüsü
       - ESC: Çıkış
    
    3. Real-time önizleme
    4. Mouse ile bölgesel düzeltme (sadece seçili alan)
    
    İpucu: cv2.createTrackbar() ve callback fonksiyonları
    """
    print("\\n🎯 GÖREV 5: İnteraktif Arayüz")
    print("-" * 35)
    
    def trackbar_callback(val):
        """Trackbar değişiklik callback'i"""
        # TODO: Real-time güncelleme
        pass
    
    def mouse_callback(event, x, y, flags, param):
        """Mouse olayları callback'i"""
        # TODO: Bölgesel düzeltme
        pass
    
    # TODO: İnteraktif arayüz implementasyonu
    pass

def gorev_6_performans_optimizasyonu():
    """
    🎯 GÖREV 6: Performans Optimizasyonu ve Benchmarking
    
    Yapılacaklar:
    1. Farklı algoritmaları karşılaştırın:
       - Resize yöntemleri (INTER_LINEAR vs INTER_CUBIC vs INTER_AREA)
       - Blur yöntemleri (Gaussian vs Box vs Bilateral)
    
    2. Performans metrikleri:
       - İşlem süresi ölçümü
       - Bellek kullanımı
       - Kalite skoru (PSNR/SSIM)
    
    3. Optimizasyon teknikleri:
       - NumPy vectorization
       - OpenCV optimize fonksiyonlar
       - Multi-threading (opsiyonel)
    
    4. Sonuçları grafik halinde gösterin
    
    İpucu: time.perf_counter() ve memory_profiler
    """
    print("\\n🎯 GÖREV 6: Performans Optimizasyonu")
    print("-" * 40)
    
    def benchmark_resize(resim, boyutlar):
        """Resize yöntemlerini karşılaştır"""
        # TODO: Benchmark implementasyonu
        pass
    
    def kalite_metrikler(orijinal, islenmis):
        """Kalite metriklerini hesapla"""
        # TODO: PSNR, SSIM hesaplama
        pass
    
    # TODO: Performans testleri ve grafikler
    pass

def bonus_gorev_ai_destekli_duzenleme():
    """
    🤖 BONUS GÖREV: AI Destekli Düzenleme
    
    İleri seviye AI özellikler (opsiyonel):
    1. Otomatik yüz güzelleştirme
    2. Nesne tabanlı düzenleme
    3. Style transfer (sanat stili aktarımı)
    4. Akıllı kırpma (kompozisyon kuralları)
    5. Otomatik renk düzeltme (deep learning)
    
    Not: Bu görev ileri OpenCV bilgisi gerektirir
    """
    print("\\n🤖 BONUS GÖREV: AI Destekli Düzenleme")
    print("-" * 45)
    print("Bu görev ileri seviye AI teknikleri içerir!")
    
    # TODO: AI özellikler (opsiyonel)
    pass

def test_verileri_olustur():
    """Test için çeşitli resimler oluşturur"""
    test_dir = Path("test-resimleri")
    test_dir.mkdir(exist_ok=True)
    
    # 1. Düşük kontrastlı resim
    dusuk = np.random.randint(80, 120, (300, 400, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "dusuk-kontrast.jpg"), dusuk)
    
    # 2. Aşırı parlak resim
    parlak = np.ones((300, 400, 3), dtype=np.uint8) * 200
    # Biraz varyasyon ekle
    for i in range(400):
        parlak[:, i] = parlak[:, i] + np.random.randint(-20, 20)
    cv2.imwrite(str(test_dir / "asiri-parlak.jpg"), parlak)
    
    # 3. Karanlık resim
    karanlik = np.ones((300, 400, 3), dtype=np.uint8) * 50
    cv2.rectangle(karanlik, (100, 100), (300, 200), (100, 100, 100), -1)
    cv2.imwrite(str(test_dir / "karanlik.jpg"), karanlik)
    
    # 4. Renkli test resmi
    renkli = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(renkli, (0, 0), (133, 300), (255, 0, 0), -1)      # Mavi
    cv2.rectangle(renkli, (133, 0), (266, 300), (0, 255, 0), -1)    # Yeşil
    cv2.rectangle(renkli, (266, 0), (400, 300), (0, 0, 255), -1)    # Kırmızı
    cv2.imwrite(str(test_dir / "renkli-test.jpg"), renkli)
    
    # 5. Gürültülü resim
    temiz = np.ones((300, 400, 3), dtype=np.uint8) * 128
    gurultu = np.random.normal(0, 30, temiz.shape)
    gurultulu = np.clip(temiz + gurultu, 0, 255).astype(np.uint8)
    cv2.imwrite(str(test_dir / "gurultulu.jpg"), gurultulu)
    
    print("✅ Test verileri oluşturuldu: test-resimleri/ klasörü")

def main():
    """Ana program - proje yöneticisi"""
    print("🎯 OpenCV Alıştırma 3: Mini Proje - Akıllı Fotoğraf Düzenleyici")
    print("=" * 75)
    print("Bu kapsamlı projede öğrendiklerinizi birleştirerek gerçek bir uygulama yapacaksınız.\\n")
    
    # Test verilerini oluştur
    test_verileri_olustur()
    
    # Proje menüsü
    while True:
        print("\\n" + "="*50)
        print("📸 Akıllı Fotoğraf Düzenleyici - Proje Menüsü")
        print("="*50)
        print("1. Sınıf Tasarımını Tamamla")
        print("2. Batch İşlem Sistemi")
        print("3. Kalite Analizi Modülü")
        print("4. Filtreleme Sistemi")
        print("5. İnteraktif Arayüz")
        print("6. Performans Optimizasyonu")
        print("7. BONUS: AI Destekli Düzenleme")
        print("8. Tüm Modülleri Test Et")
        print("0. Çıkış")
        
        secim = input("\\nLütfen bir modül seçin (0-8): ").strip()
        
        if secim == "0":
            print("🎉 Proje tamamlandı! Harika iş çıkardınız!")
            break
        elif secim == "1":
            gorev_1_sinif_tasarimi()
        elif secim == "2":
            gorev_2_batch_islem()
        elif secim == "3":
            gorev_3_kalite_analizi()
        elif secim == "4":
            gorev_4_filtreleme_sistemi()
        elif secim == "5":
            gorev_5_interaktif_arayuz()
        elif secim == "6":
            gorev_6_performans_optimizasyonu()
        elif secim == "7":
            bonus_gorev_ai_destekli_duzenleme()
        elif secim == "8":
            print("🧪 Tüm modüller test ediliyor...")
            # Burada tüm modüllerin entegrasyonu test edilebilir
        else:
            print("❌ Geçersiz seçim! Lütfen 0-8 arasında bir sayı girin.")

if __name__ == "__main__":
    main()

# 📝 PROJE NOTLARI:
# 1. Bu bir mini proje olduğu için modüler yaklaşım benimseyin
# 2. Her modülü bağımsız test edin
# 3. Kod kalitesine dikkat edin (fonksiyon, sınıf tasarımı)
# 4. Kullanıcı deneyimini ön planda tutun
# 5. Hata kontrolü yapmayı unutmayın
# 6. Çözüm için cozumler/cozum-3.py dosyasına bakabilirsiniz