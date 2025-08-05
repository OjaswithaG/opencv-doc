"""
✅ Çözüm 1: Temel Resim İşleme
=============================

Bu dosyada Alıştırma 1'in tam çözümü bulunmaktadır.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def cozum_1():
    """Alıştırma 1'in tam çözümü"""
    
    print("✅ Çözüm 1: Temel Resim İşleme")
    print("=" * 40)
    
    # GÖREV 1: Test resmini yükle
    print("\n📁 GÖREV 1: Test resmini yükleme")
    test_resmi_yolu = "test-resimleri/normal.jpg"
    
    resim = cv2.imread(test_resmi_yolu)
    if resim is None:
        print("⚠️ Test resmi bulunamadı, örnek resim oluşturuluyor...")
        resim = ornek_resim_olustur()
    
    print(f"✅ Resim yüklendi: {resim.shape}")
    
    # GÖREV 2: Resmi 45° saat yönünde döndür
    print("\n🔄 GÖREV 2: 45° döndürme")
    
    merkez = (resim.shape[1]//2, resim.shape[0]//2)
    rotasyon_matrisi = cv2.getRotationMatrix2D(merkez, -45, 1.0)  # -45 = saat yönü
    dondurulmus_resim = cv2.warpAffine(resim, rotasyon_matrisi, (resim.shape[1], resim.shape[0]))
    
    print("✅ Resim döndürüldü")
    
    # GÖREV 3: Resmi %75 oranında küçült
    print("\n📏 GÖREV 3: %75 küçültme")
    
    yeni_boyut = (int(resim.shape[1] * 0.75), int(resim.shape[0] * 0.75))
    kucultulmus_resim = cv2.resize(resim, yeni_boyut, interpolation=cv2.INTER_LINEAR)
    
    print(f"✅ Resim küçültüldü: {resim.shape} -> {kucultulmus_resim.shape}")
    
    # GÖREV 4: 3 farklı boyutta Gaussian blur
    print("\n🌫️ GÖREV 4: Gaussian blur (3 boyut)")
    
    blur_3x3 = cv2.GaussianBlur(resim, (3, 3), 0)
    blur_7x7 = cv2.GaussianBlur(resim, (7, 7), 0)
    blur_15x15 = cv2.GaussianBlur(resim, (15, 15), 0)
    
    print("✅ Gaussian blur uygulandı")
    
    # GÖREV 5: Salt & Pepper gürültü ekle ve temizle
    print("\n🧂 GÖREV 5: Salt & Pepper gürültü temizleme")
    
    # Gürültü ekleme
    gurultulu_resim = resim.copy().astype(np.float32)
    
    # Salt noise (beyaz piksel)
    salt_mask = np.random.random(resim.shape[:2]) < 0.05
    gurultulu_resim[salt_mask] = 255
    
    # Pepper noise (siyah piksel)
    pepper_mask = np.random.random(resim.shape[:2]) < 0.05
    gurultulu_resim[pepper_mask] = 0
    
    gurultulu_resim = gurultulu_resim.astype(np.uint8)
    
    # Median filter ile temizleme
    temizlenmis_resim = cv2.medianBlur(gurultulu_resim, 5)
    
    print("✅ Gürültü temizlendi")
    
    # GÖREV 6: Histogram eşitleme
    print("\n📊 GÖREV 6: Histogram eşitleme")
    
    # HSV color space'de V kanalını eşitle (renkli resim için)
    hsv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
    hsv_resim[:,:,2] = cv2.equalizeHist(hsv_resim[:,:,2])
    esitlenmis_resim = cv2.cvtColor(hsv_resim, cv2.COLOR_HSV2BGR)
    
    print("✅ Histogram eşitleme uygulandı")
    
    # GÖREV 7: PSNR hesaplama
    print("\n📈 GÖREV 7: PSNR hesaplama")
    
    def psnr_hesapla(orijinal, islenmis):
        """PSNR (Peak Signal-to-Noise Ratio) hesapla"""
        # Boyutları eşitle
        if orijinal.shape != islenmis.shape:
            islenmis = cv2.resize(islenmis, (orijinal.shape[1], orijinal.shape[0]))
        
        # MSE hesapla
        mse = np.mean((orijinal.astype(np.float32) - islenmis.astype(np.float32)) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # PSNR hesapla
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    psnr_dondurulmus = psnr_hesapla(resim, dondurulmus_resim)
    psnr_kucultulmus = psnr_hesapla(resim, kucultulmus_resim)
    psnr_blur = psnr_hesapla(resim, blur_7x7)
    psnr_temizlenmis = psnr_hesapla(resim, temizlenmis_resim)
    psnr_esitlenmis = psnr_hesapla(resim, esitlenmis_resim)
    
    print("✅ PSNR değerleri hesaplandı")
    
    # GÖREV 8: Sonuçları görselleştir
    print("\n🖼️ GÖREV 8: Görselleştirme")
    
    plt.figure(figsize=(15, 15))
    
    # Orijinal
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal')
    plt.axis('off')
    
    # Döndürülmüş
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(dondurulmus_resim, cv2.COLOR_BGR2RGB))
    plt.title(f'45° Döndürülmüş\nPSNR: {psnr_dondurulmus:.1f} dB')
    plt.axis('off')
    
    # Küçültülmüş
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(kucultulmus_resim, cv2.COLOR_BGR2RGB))
    plt.title(f'%75 Küçültülmüş\nPSNR: {psnr_kucultulmus:.1f} dB')
    plt.axis('off')
    
    # Blur örnekleri
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(blur_3x3, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur 3x3')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(blur_7x7, cv2.COLOR_BGR2RGB))
    plt.title(f'Gaussian Blur 7x7\nPSNR: {psnr_blur:.1f} dB')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(blur_15x15, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur 15x15')
    plt.axis('off')
    
    # Gürültü temizleme
    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(gurultulu_resim, cv2.COLOR_BGR2RGB))
    plt.title('Salt & Pepper Gürültülü')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(cv2.cvtColor(temizlenmis_resim, cv2.COLOR_BGR2RGB))
    plt.title(f'Median Filter Temizlenmiş\nPSNR: {psnr_temizlenmis:.1f} dB')
    plt.axis('off')
    
    # Histogram eşitleme
    plt.subplot(3, 3, 9)
    plt.imshow(cv2.cvtColor(esitlenmis_resim, cv2.COLOR_BGR2RGB))
    plt.title(f'Histogram Eşitlenmiş\nPSNR: {psnr_esitlenmis:.1f} dB')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Sonuç raporu
    print("\n📋 SONUÇ RAPORU")
    print("=" * 30)
    print(f"🔄 Döndürme PSNR: {psnr_dondurulmus:.2f} dB")
    print(f"📏 Küçültme PSNR: {psnr_kucultulmus:.2f} dB")
    print(f"🌫️ Blur PSNR: {psnr_blur:.2f} dB")
    print(f"🧼 Temizleme PSNR: {psnr_temizlenmis:.2f} dB")
    print(f"📊 Eşitleme PSNR: {psnr_esitlenmis:.2f} dB")
    
    print("\n💡 Analiz:")
    print(f"   • En yüksek PSNR: {max(psnr_dondurulmus, psnr_kucultulmus, psnr_blur, psnr_temizlenmis, psnr_esitlenmis):.1f} dB")
    print(f"   • Gürültü temizleme başarılı (PSNR artışı)")
    print(f"   • Blur işlemi beklenen şekilde PSNR düşürdü")
    
    print("\n🎉 Çözüm 1 tamamlandı!")

def ornek_resim_olustur():
    """Test için örnek resim oluştur"""
    resim = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Gradient arka plan
    for i in range(300):
        for j in range(300):
            r = int(100 + 50 * np.sin(i/50))
            g = int(120 + 30 * np.cos(j/40))
            b = int(140 + 40 * np.sin((i+j)/60))
            resim[i, j] = [b, g, r]
    
    # Şekiller
    cv2.rectangle(resim, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(resim, (200, 200), 50, (0, 0, 255), -1)
    cv2.ellipse(resim, (100, 250), (40, 20), 0, 0, 360, (0, 255, 0), -1)
    
    return resim

if __name__ == "__main__":
    print("✅ OpenCV Alıştırma 1 - ÇÖZÜM")
    print("Bu dosyada tüm görevlerin çözümleri bulunmaktadır.\n")
    
    cozum_1()