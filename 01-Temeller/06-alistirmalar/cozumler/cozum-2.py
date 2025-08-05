#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ Çözüm 2: Resim İşlemleri ve Renk Uzayları
===========================================

Bu dosya, Alıştırma 2'nin örnek çözümlerini içerir.
Resim işleme ve renk uzayları konularında detaylı örnekler.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def gorev_1_resim_okuma_kaydetme():
    """
    ✅ ÇÖZÜM 1: Resim Okuma ve Farklı Formatlarda Kaydetme
    """
    print("🎯 GÖREV 1: Resim Okuma ve Kaydetme")
    print("-" * 40)
    
    # Test resmi oluştur veya yükle
    test_dir = Path("test-resimleri")
    test_dir.mkdir(exist_ok=True)
    
    gradyan_yolu = test_dir / "gradyan.jpg"
    
    # Gradyan resmi oluştur (yoksa)
    if not gradyan_yolu.exists():
        gradyan = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(300):
            r = int(255 * i / 300)
            g = int(255 * (1 - i / 300))
            b = 128
            gradyan[:, i] = [b, g, r]
        cv2.imwrite(str(gradyan_yolu), gradyan)
        print("✅ Gradyan resmi oluşturuldu")
    
    # Resmi oku
    resim = cv2.imread(str(gradyan_yolu))
    if resim is None:
        print("❌ Resim okunamadı!")
        return
    
    # Resim bilgilerini yazdır
    print("📊 Resim Bilgileri:")
    print(f"   Boyut: {resim.shape}")
    print(f"   Veri tipi: {resim.dtype}")
    print(f"   Kanal sayısı: {resim.shape[2] if len(resim.shape) == 3 else 1}")
    
    # Farklı formatlarda kaydet
    output_dir = Path("ciktiler")
    output_dir.mkdir(exist_ok=True)
    
    # PNG (yüksek kalite, kayıpsız)
    png_yolu = output_dir / "output.png"
    cv2.imwrite(str(png_yolu), resim, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    # BMP (sıkıştırmasız)
    bmp_yolu = output_dir / "output.bmp"
    cv2.imwrite(str(bmp_yolu), resim)
    
    # JPEG (%75 kalite)
    jpg_yolu = output_dir / "output_75.jpg"
    cv2.imwrite(str(jpg_yolu), resim, [cv2.IMWRITE_JPEG_QUALITY, 75])
    
    # Dosya boyutlarını karşılaştır
    print("\\n💾 Dosya Boyutları:")
    for dosya_yolu in [png_yolu, bmp_yolu, jpg_yolu]:
        if dosya_yolu.exists():
            boyut = os.path.getsize(dosya_yolu) / 1024  # KB
            print(f"   {dosya_yolu.name}: {boyut:.1f} KB")
    
    print("✅ Resimler farklı formatlarda kaydedildi!")

def gorev_2_renk_uzayi_donusumleri():
    """
    ✅ ÇÖZÜM 2: Renk Uzayı Dönüşümleri
    """
    print("\\n🎯 GÖREV 2: Renk Uzayı Dönüşümleri")
    print("-" * 40)
    
    # Renkli test resmi oluştur
    test_resim = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Renkli bölgeler ekle
    test_resim[50:150, 50:100] = [0, 0, 255]    # Kırmızı
    test_resim[50:150, 100:150] = [0, 255, 0]   # Yeşil  
    test_resim[50:150, 150:200] = [255, 0, 0]   # Mavi
    test_resim[50:150, 200:250] = [0, 255, 255] # Sarı
    
    # Renk uzayı dönüşümleri
    rgb_resim = cv2.cvtColor(test_resim, cv2.COLOR_BGR2RGB)
    hsv_resim = cv2.cvtColor(test_resim, cv2.COLOR_BGR2HSV)
    lab_resim = cv2.cvtColor(test_resim, cv2.COLOR_BGR2LAB)
    gri_resim = cv2.cvtColor(test_resim, cv2.COLOR_BGR2GRAY)
    
    # 2x3 subplot ile göster
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Renk Uzayı Dönüşümleri', fontsize=16)
    
    # Orijinal (RGB formatında göster)
    axes[0, 0].imshow(rgb_resim)
    axes[0, 0].set_title('RGB (Orijinal)')
    axes[0, 0].axis('off')
    
    # HSV
    axes[0, 1].imshow(hsv_resim)
    axes[0, 1].set_title('HSV\\n(Renk filtreleme için ideal)')
    axes[0, 1].axis('off')
    
    # LAB
    axes[0, 2].imshow(lab_resim)
    axes[0, 2].set_title('LAB\\n(Işık bağımsız)')
    axes[0, 2].axis('off')
    
    # Gri tonlama
    axes[1, 0].imshow(gri_resim, cmap='gray')
    axes[1, 0].set_title('Gri Tonlama\\n(Tek kanal)')
    axes[1, 0].axis('off')
    
    # HSV kanalları ayrı ayrı
    h, s, v = cv2.split(hsv_resim)
    axes[1, 1].imshow(s, cmap='gray')
    axes[1, 1].set_title('HSV - Saturation\\n(Doygunluk)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(v, cmap='gray')
    axes[1, 2].set_title('HSV - Value\\n(Parlaklık)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Renk uzayı yorumları
    print("\\n🎨 Renk Uzayı Avantajları:")
    print("   RGB: Web ve ekran gösterimi için ideal")
    print("   HSV: Renk filtreleme ve maskeleme için mükemmel")
    print("   LAB: Renk farkı hesaplama ve profesyonel baskı")
    print("   Gri: Hız ve basitlik, many algorithms için")

def gorev_3_hsv_renk_filtreleme():
    """
    ✅ ÇÖZÜM 3: HSV ile Renk Filtreleme
    """
    print("\\n🎯 GÖREV 3: HSV ile Renk Filtreleme")
    print("-" * 40)
    
    # Geometrik şekiller içeren resim oluştur
    resim = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Farklı renklerde şekiller
    cv2.rectangle(resim, (50, 50), (150, 150), (0, 0, 255), -1)    # Kırmızı
    cv2.rectangle(resim, (200, 50), (300, 150), (255, 0, 0), -1)   # Mavi
    cv2.circle(resim, (100, 225), 40, (0, 255, 0), -1)             # Yeşil
    cv2.circle(resim, (250, 225), 40, (0, 255, 255), -1)           # Sarı
    
    # HSV'ye dönüştür
    hsv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
    
    # Kırmızı renk filtreleme (iki aralık gerekli)
    # Kırmızı alt aralık (0-10)
    alt_kirmizi1 = np.array([0, 120, 70])
    ust_kirmizi1 = np.array([10, 255, 255])
    maske_kirmizi1 = cv2.inRange(hsv_resim, alt_kirmizi1, ust_kirmizi1)
    
    # Kırmızı üst aralık (170-180)  
    alt_kirmizi2 = np.array([170, 120, 70])
    ust_kirmizi2 = np.array([180, 255, 255])
    maske_kirmizi2 = cv2.inRange(hsv_resim, alt_kirmizi2, ust_kirmizi2)
    
    # Kırmızı maskelerini birleştir
    maske_kirmizi = cv2.bitwise_or(maske_kirmizi1, maske_kirmizi2)
    kirmizi_filtrelenmis = cv2.bitwise_and(resim, resim, mask=maske_kirmizi)
    
    # Mavi renk filtreleme
    alt_mavi = np.array([100, 150, 0])
    ust_mavi = np.array([140, 255, 255])
    maske_mavi = cv2.inRange(hsv_resim, alt_mavi, ust_mavi)
    mavi_filtrelenmis = cv2.bitwise_and(resim, resim, mask=maske_mavi)
    
    # Sonuçları göster
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Orijinal Resim')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(kirmizi_filtrelenmis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Kırmızı Filtresi')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(mavi_filtrelenmis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Mavi Filtresi')
    axes[1, 0].axis('off')
    
    # Maske gösterimi
    axes[1, 1].imshow(maske_kirmizi, cmap='gray')
    axes[1, 1].set_title('Kırmızı Maske')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ HSV renk filtreleme tamamlandı!")
    print("💡 İpucu: Kırmızı renk için iki aralık gerekli (HSV'de 0° ve 180° civarı)")

def gorev_4_piksel_manipulasyonu():
    """
    ✅ ÇÖZÜM 4: Piksel Seviyesi Manipülasyon
    """
    print("\\n🎯 GÖREV 4: Piksel Manipülasyonu")
    print("-" * 35)
    
    # 300x400 beyaz resim
    resim = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Sol yarının parlaklığını %50 azalt
    sol_yarim = resim[:, :200].copy()
    resim[:, :200] = np.clip(sol_yarim * 0.5, 0, 255).astype(np.uint8)
    
    # Sağ yarının parlaklığını %50 artır
    sag_yarim = resim[:, 200:].copy()
    resim[:, 200:] = np.clip(sag_yarim * 1.5, 0, 255).astype(np.uint8)
    
    # Merkeze 50x50 kırmızı kare
    merkez_x, merkez_y = 200, 150
    resim[merkez_y-25:merkez_y+25, merkez_x-25:merkez_x+25] = [0, 0, 255]
    
    # 4 köşeye farklı renkler
    resim[0:30, 0:30] = [255, 0, 0]        # Sol üst - Mavi
    resim[0:30, 370:400] = [0, 255, 0]     # Sağ üst - Yeşil
    resim[270:300, 0:30] = [0, 255, 255]   # Sol alt - Sarı
    resim[270:300, 370:400] = [255, 0, 255] # Sağ alt - Magenta
    
    # ROI (Region of Interest) - merkez bölgesini kopyala
    roi = resim[100:200, 150:250].copy()
    
    # ROI'yi sol üst köşeye yapıştır
    resim[50:150, 50:150] = roi
    
    # Sonucu göster
    cv2.imshow('Piksel Manipulasyonu', resim)
    print("Piksel manipülasyonu gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ Piksel manipülasyonu tamamlandı!")
    print("💡 ROI (Region of Interest) - ilgilenilen bölge kopyalandı")

def gorev_5_histogram_analizi():
    """
    ✅ ÇÖZÜM 5: Histogram Analizi ve Düzeltme
    """
    print("\\n🎯 GÖREV 5: Histogram Analizi")
    print("-" * 30)
    
    # Düşük kontrastlı resim oluştur
    dusuk_kontrast = np.random.randint(80, 120, (200, 300, 3), dtype=np.uint8)
    
    # Gri tonlama dönüşümü
    gri_orijinal = cv2.cvtColor(dusuk_kontrast, cv2.COLOR_BGR2GRAY)
    
    # Histogram eşitleme uygula
    gri_esitlenmis = cv2.equalizeHist(gri_orijinal)
    
    # Renkli resim için histogram eşitleme (YUV üzerinden)
    yuv = cv2.cvtColor(dusuk_kontrast, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    renkli_esitlenmis = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # Histogramları hesapla
    hist_orijinal = cv2.calcHist([gri_orijinal], [0], None, [256], [0, 256])
    hist_esitlenmis = cv2.calcHist([gri_esitlenmis], [0], None, [256], [0, 256])
    
    # BGR kanalları histogramı
    b, g, r = cv2.split(dusuk_kontrast)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    
    # Sonuçları görselleştir
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    # Orijinal ve düzeltilmiş resimler
    axes[0, 0].imshow(gri_orijinal, cmap='gray')
    axes[0, 0].set_title('Orijinal (Düşük Kontrast)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gri_esitlenmis, cmap='gray')
    axes[0, 1].set_title('Histogram Eşitlenmiş')
    axes[0, 1].axis('off')
    
    # Histogramlar
    axes[1, 0].plot(hist_orijinal, color='black')
    axes[1, 0].set_title('Orijinal Histogram')
    axes[1, 0].set_xlabel('Piksel Değeri')
    axes[1, 0].set_ylabel('Frekans')
    
    axes[1, 1].plot(hist_esitlenmis, color='black')
    axes[1, 1].set_title('Eşitlenmiş Histogram')
    axes[1, 1].set_xlabel('Piksel Değeri')
    axes[1, 1].set_ylabel('Frekans')
    
    # BGR kanalları histogramı
    axes[2, 0].plot(hist_b, color='blue', label='Mavi', alpha=0.7)
    axes[2, 0].plot(hist_g, color='green', label='Yeşil', alpha=0.7)
    axes[2, 0].plot(hist_r, color='red', label='Kırmızı', alpha=0.7)
    axes[2, 0].set_title('BGR Kanalları Histogramı')
    axes[2, 0].set_xlabel('Piksel Değeri')
    axes[2, 0].set_ylabel('Frekans')
    axes[2, 0].legend()
    
    # Renkli eşitlenmiş
    axes[2, 1].imshow(cv2.cvtColor(renkli_esitlenmis, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('Renkli Histogram Eşitleme')
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Histogram analizi tamamlandı!")
    print("💡 Histogram eşitleme kontrast artırır ama renkleri değiştirebilir")

def gorev_6_resim_matematigi():
    """
    ✅ ÇÖZÜM 6: Resim Matematiği ve Birleştirme
    """
    print("\\n🎯 GÖREV 6: Resim Matematiği")
    print("-" * 30)
    
    # İki farklı resim oluştur
    resim1 = np.ones((200, 300, 3), dtype=np.uint8) * 100
    cv2.rectangle(resim1, (50, 50), (150, 150), (255, 255, 255), -1)
    
    resim2 = np.ones((200, 300, 3), dtype=np.uint8) * 50
    cv2.circle(resim2, (150, 100), 60, (200, 200, 200), -1)
    
    # Matematiksel işlemler
    # OpenCV güvenli işlemler (taşma kontrolü)
    toplam_cv = cv2.add(resim1, resim2)
    cikarma_cv = cv2.subtract(resim1, resim2)
    harmanlama = cv2.addWeighted(resim1, 0.7, resim2, 0.3, 0)
    
    # NumPy işlemler (taşma riski var)
    toplam_np = resim1 + resim2
    
    # Bitwise işlemler
    resim1_binary = cv2.threshold(resim1, 127, 255, cv2.THRESH_BINARY)[1]
    resim2_binary = cv2.threshold(resim2, 127, 255, cv2.THRESH_BINARY)[1]
    
    bitwise_and = cv2.bitwise_and(resim1_binary, resim2_binary)
    bitwise_or = cv2.bitwise_or(resim1_binary, resim2_binary)
    bitwise_xor = cv2.bitwise_xor(resim1_binary, resim2_binary)
    
    # Sonuçları göster
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Orijinal resimler
    axes[0, 0].imshow(cv2.cvtColor(resim1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Resim 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(resim2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Resim 2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(harmanlama, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Harmanlama (70% + 30%)')
    axes[0, 2].axis('off')
    
    # Aritmetik işlemler
    axes[1, 0].imshow(cv2.cvtColor(toplam_cv, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('CV2 Toplama (Güvenli)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(toplam_np, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('NumPy Toplama (Taşma Risk)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(cikarma_cv, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('CV2 Çıkarma')
    axes[1, 2].axis('off')
    
    # Bitwise işlemler
    axes[2, 0].imshow(cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('Bitwise AND')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(cv2.cvtColor(bitwise_or, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('Bitwise OR')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(cv2.cvtColor(bitwise_xor, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title('Bitwise XOR')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Taşma kontrolü demonstrasyonu
    print("\\n🔍 Taşma Kontrolü Karşılaştırması:")
    print(f"CV2 toplama max değer: {np.max(toplam_cv)}")
    print(f"NumPy toplama max değer: {np.max(toplam_np)}")
    print("💡 OpenCV fonksiyonları taşmayı önler, NumPy'da taşabilir (255'i geçerse 0'dan başlar)")
    
    print("✅ Resim matematiği tamamlandı!")

def bonus_gorev_mini_fotograf_editoru():
    """
    🎨 BONUS ÇÖZÜM: Mini Fotoğraf Editörü
    """
    print("\\n🎨 BONUS GÖREV: Mini Fotoğraf Editörü")
    print("-" * 45)
    
    # Basit bir fotoğraf editörü
    resim = np.ones((300, 400, 3), dtype=np.uint8) * 128
    cv2.rectangle(resim, (100, 100), (300, 200), (200, 150, 100), -1)
    
    pencere_adi = 'Mini Fotograf Editoru'
    cv2.namedWindow(pencere_adi)
    
    # Trackbar'ları oluştur
    cv2.createTrackbar('Parlaklik', pencere_adi, 50, 100, lambda x: None)
    cv2.createTrackbar('Kontrast', pencere_adi, 50, 100, lambda x: None)
    cv2.createTrackbar('Blur', pencere_adi, 0, 20, lambda x: None)
    
    orijinal_resim = resim.copy()
    
    print("🎛️ Kontroller:")
    print("   Trackbar'ları kullanarak ayar yapın")
    print("   'r': Reset")
    print("   's': Save")
    print("   ESC: Çıkış")
    
    while True:
        # Trackbar değerlerini al
        parlaklik = cv2.getTrackbarPos('Parlaklik', pencere_adi) - 50
        kontrast = cv2.getTrackbarPos('Kontrast', pencere_adi) / 50.0
        blur_val = cv2.getTrackbarPos('Blur', pencere_adi)
        
        # İşlemleri uygula
        islenmis = orijinal_resim.copy()
        
        # Parlaklık ve kontrast
        islenmis = cv2.convertScaleAbs(islenmis, alpha=kontrast, beta=parlaklik)
        
        # Bulanıklaştırma
        if blur_val > 0:
            ksize = blur_val * 2 + 1  # Tek sayı olması gerekli
            islenmis = cv2.GaussianBlur(islenmis, (ksize, ksize), 0)
        
        # Göster
        cv2.imshow(pencere_adi, islenmis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset
            cv2.setTrackbarPos('Parlaklik', pencere_adi, 50)
            cv2.setTrackbarPos('Kontrast', pencere_adi, 50)
            cv2.setTrackbarPos('Blur', pencere_adi, 0)
        elif key == ord('s'):  # Save
            cv2.imwrite('editlenmis_resim.jpg', islenmis)
            print("✅ Resim 'editlenmis_resim.jpg' olarak kaydedildi!")
    
    cv2.destroyAllWindows()
    print("🎉 Mini fotoğraf editörü kapatıldı!")

def main():
    """Ana çözüm programı"""
    print("✅ OpenCV Alıştırma 2 - ÇÖZÜMLER")
    print("=" * 45)
    print("Resim işlemleri ve renk uzayları çözümleri\\n")
    
    try:
        # Çözümleri sırayla göster
        gorev_1_resim_okuma_kaydetme()
        gorev_2_renk_uzayi_donusumleri()
        gorev_3_hsv_renk_filtreleme()
        gorev_4_piksel_manipulasyonu()
        gorev_5_histogram_analizi()
        gorev_6_resim_matematigi()
        
        # Bonus görev
        bonus_cevap = input("\\nBonus editör çözümünü görmek ister misiniz? (e/h): ")
        if bonus_cevap.lower() == 'e':
            bonus_gorev_mini_fotograf_editoru()
        
        print("\\n🎉 Tüm çözümler gösterildi!")
        print("\\n📚 Önemli Öğrenme Notları:")
        print("   - Dosya boyutu: PNG > BMP > JPEG (kalite sırasıyla)")
        print("   - HSV renk filtreleme için ideal (özellikle H kanalı)")
        print("   - Histogram eşitleme kontrast artırır")
        print("   - OpenCV matematik fonksiyonları güvenli (taşma kontrolü)")
        print("   - ROI kullanarak sadece ilgilenilen bölgeyi işleyin")
        print("   - Matplotlib'te BGR→RGB dönüşümü gerekli")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# 📝 ÇÖZÜM NOTLARI:
# 1. Bu çözümler detaylı açıklamalar içerir
# 2. Her adım adım açıklanmıştır
# 3. Görselleştirme önemlidir - matplotlib kullanın
# 4. Hata kontrolü yapmayı unutmayın
# 5. Test resimlerini otomatik oluşturun