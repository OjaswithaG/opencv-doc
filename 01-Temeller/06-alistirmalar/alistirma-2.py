#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Alıştırma 2: Resim İşlemleri ve Renk Uzayları
===============================================

Bu alıştırma resim okuma/yazma, renk uzayı dönüşümleri ve temel
resim manipülasyon işlemlerini içerir.

Zorluk Seviyesi: 🚀 Orta
Tahmini Süre: 25-30 dakika
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def gorev_1_resim_okuma_kaydetme():
    """
    🎯 GÖREV 1: Resim Okuma ve Farklı Formatlarda Kaydetme
    
    Yapılacaklar:
    1. examples/gradyan.jpg dosyasını okuyun (yoksa oluşturun)
    2. Resmin bilgilerini yazdırın (boyut, tip, kanal)
    3. Aynı resmi şu formatlarda kaydedin:
       - PNG formatında (yüksek kalite)
       - BMP formatında (sıkıştırmasız)
       - JPEG formatında %75 kalite ile
    4. Dosya boyutlarını karşılaştırın ve yazdırın
    
    İpucu: cv2.imwrite() fonksiyonunda kalite parametresi kullanın
    """
    print("🎯 GÖREV 1: Resim Okuma ve Kaydetme")
    print("-" * 40)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: Önce resim var mı kontrol edin, yoksa oluşturun
    # cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_2_renk_uzayi_donusumleri():
    """
    🎯 GÖREV 2: Renk Uzayı Dönüşümleri
    
    Yapılacaklar:
    1. Renkli bir test resmi oluşturun veya yükleyin
    2. BGR'dan şu renk uzaylarına dönüştürün:
       - RGB
       - HSV  
       - LAB
       - Gri tonlama
    3. Tüm dönüşümleri 2x3 subplot ile gösterin
    4. Her renk uzayının avantajlarını yorumlayın
    
    İpucu: cv2.cvtColor() ve matplotlib.pyplot.subplot() kullanın
    """
    print("\\n🎯 GÖREV 2: Renk Uzayı Dönüşümleri")
    print("-" * 40)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: plt.subplot(2, 3, 1) ile subplot oluşturun
    # cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ile dönüştürün
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_3_hsv_renk_filtreleme():
    """
    🎯 GÖREV 3: HSV ile Renk Filtreleme
    
    Yapılacaklar:
    1. Farklı renklerde geometrik şekiller içeren resim oluşturun
    2. HSV renk uzayına dönüştürün
    3. Sadece kırmızı renkteki nesneleri filtreleyin
    4. Sadece mavi renkteki nesneleri filtreleyin  
    5. Her filtreleme sonucunu gösterin
    6. Bonus: İnteraktif renk seçici yapın (trackbar ile)
    
    İpucu: cv2.inRange() ve cv2.bitwise_and() kullanın
    """
    print("\\n🎯 GÖREV 3: HSV ile Renk Filtreleme")
    print("-" * 40)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: Kırmızı için iki aralık gerekli (0-10 ve 170-180)
    # HSV aralıkları: H(0-179), S(0-255), V(0-255)
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_4_piksel_manipulasyonu():
    """
    🎯 GÖREV 4: Piksel Seviyesi Manipülasyon
    
    Yapılacaklar:
    1. 300x400 boyutunda beyaz bir resim oluşturun
    2. Sol yarısının parlaklığını %50 azaltın
    3. Sağ yarısının parlaklığını %50 artırın
    4. Merkeze 50x50 boyutunda kırmızı bir kare çizin
    5. Resmin 4 köşesine farklı renkler ekleyin
    6. ROI (Region of Interest) kullanarak merkez bölgeyi kopyalayın
    
    İpucu: Numpy array indexing ve OpenCV geometrik şekiller
    """
    print("\\n🎯 GÖREV 4: Piksel Manipülasyonu")
    print("-" * 35)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: resim[y1:y2, x1:x2] ile bölge seçimi
    # cv2.rectangle(), cv2.circle() ile şekil çizimi
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_5_histogram_analizi():
    """
    🎯 GÖREV 5: Histogram Analizi ve Düzeltme
    
    Yapılacaklar:
    1. Düşük kontrastlı bir resim oluşturun veya yükleyin
    2. Orijinal resmin histogramını çizin
    3. Histogram eşitleme uygulayın
    4. Eşitlenmiş resmin histogramını çizin
    5. Orijinal ve düzeltilmiş resimleri karşılaştırın
    6. BGR kanallarının ayrı histogramlarını gösterin
    
    İpucu: cv2.calcHist() ve cv2.equalizeHist() kullanın
    """
    print("\\n🎯 GÖREV 5: Histogram Analizi")
    print("-" * 30)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.calcHist([image], [0], None, [256], [0,256])
    # plt.plot() ile histogram çizimi
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_6_resim_matematigi():
    """
    🎯 GÖREV 6: Resim Matematiği ve Birleştirme
    
    Yapılacaklar:
    1. İki farklı resim oluşturun veya yükleyin
    2. Resimleri aynı boyuta getirin
    3. Şu işlemleri uygulayın ve sonuçları gösterin:
       - Toplama (cv2.add)
       - Çıkarma (cv2.subtract)
       - Harmanlanma (cv2.addWeighted)
       - Bitwise AND, OR, XOR
    4. Her işlemin sonucunu açıklayın
    
    İpucu: cv2 matemtik fonksiyonları taşmayı önler
    """
    print("\\n🎯 GÖREV 6: Resim Matematiği")
    print("-" * 30)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.add() vs numpy + farkını test edin
    # cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
    
    pass  # Bu satırı silin ve kodunuzu yazın

def bonus_gorev_mini_fotograf_editoru():
    """
    🎨 BONUS GÖREV: Mini Fotoğraf Editörü
    
    Interaktif bir fotoğraf editörü oluşturun:
    1. Resim yükleme fonksiyonu
    2. Trackbar'larla parlaklık/kontrast ayarlama
    3. Farklı renk uzaylarına geçiş
    4. Basit filtreler (blur, sharpen)
    5. Sonucu kaydetme
    
    Bu tamamen yaratıcılığınıza kalmış!
    """
    print("\\n🎨 BONUS GÖREV: Mini Fotoğraf Editörü")
    print("-" * 45)
    print("Yaratıcılığınızı konuşturun!")
    
    # TODO: Buraya yaratıcı kodunuzu yazın
    # Örnek: Trackbar'larla interaktif editör
    
    pass  # Bu satırı silin ve kodunuzu yazın

def test_resimleri_olustur():
    """Test için gerekli resimleri oluşturur"""
    test_dir = Path("test-resimleri")
    test_dir.mkdir(exist_ok=True)
    
    # 1. Gradyan resmi
    gradyan = np.zeros((200, 300, 3), dtype=np.uint8)
    for i in range(300):
        r = int(255 * i / 300)
        g = int(255 * (1 - i / 300))
        b = 128
        gradyan[:, i] = [b, g, r]
    cv2.imwrite(str(test_dir / "gradyan.jpg"), gradyan)
    
    # 2. Düşük kontrastlı resim
    dusuk_kontrast = np.random.randint(100, 156, (200, 300, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "dusuk-kontrast.jpg"), dusuk_kontrast)
    
    # 3. Geometrik şekiller
    sekiller = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(sekiller, (50, 50), (150, 150), (0, 0, 255), -1)    # Kırmızı
    cv2.rectangle(sekiller, (200, 50), (300, 150), (255, 0, 0), -1)   # Mavi
    cv2.circle(sekiller, (100, 225), 40, (0, 255, 0), -1)             # Yeşil
    cv2.circle(sekiller, (250, 225), 40, (0, 255, 255), -1)           # Sarı
    cv2.imwrite(str(test_dir / "renkli-sekiller.png"), sekiller)
    
    print("✅ Test resimleri oluşturuldu: test-resimleri/ klasörü")

def main():
    """Ana program - tüm görevleri çalıştırır"""
    print("🎯 OpenCV Alıştırma 2: Resim İşlemleri ve Renk Uzayları")
    print("=" * 65)
    print("Bu alıştırmada resim işleme ve renk uzaylarını öğreneceksiniz.\\n")
    
    # Test resimlerini oluştur
    test_resimleri_olustur()
    
    try:
        # Görevleri sırayla çalıştır
        gorev_1_resim_okuma_kaydetme()
        gorev_2_renk_uzayi_donusumleri()
        gorev_3_hsv_renk_filtreleme()
        gorev_4_piksel_manipulasyonu()
        gorev_5_histogram_analizi()
        gorev_6_resim_matematigi()
        
        # Bonus görev (opsiyonel)
        bonus_cevap = input("\\nBonus görevi yapmak ister misiniz? (e/h): ")
        if bonus_cevap.lower() == 'e':
            bonus_gorev_mini_fotograf_editoru()
        
        print("\\n🎉 Tebrikler! Alıştırma 2'yi tamamladınız!")
        print("✅ Öğrendikleriniz:")
        print("   - Resim okuma/yazma ve format dönüşümleri")
        print("   - Renk uzayları ve dönüşümler")
        print("   - HSV ile renk filtreleme")
        print("   - Piksel seviyesi manipülasyon")
        print("   - Histogram analizi ve düzeltme")
        print("   - Resim matematiksel işlemleri")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        print("💡 İpucu: Hata mesajını dikkatlice okuyun ve kodunuzu kontrol edin.")

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Her görevde 'pass' satırını silip kendi kodunuzu yazın
# 2. Test resimleri otomatik oluşturulur
# 3. Matplotlib kullanarak görselleştirme yapın
# 4. HSV renk aralıklarına dikkat edin
# 5. Çözüm için cozumler/cozum-2.py dosyasına bakabilirsiniz