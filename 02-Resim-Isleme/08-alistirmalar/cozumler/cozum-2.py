"""
✅ Çözüm 2: İleri Resim İyileştirme
=================================

Bu dosyada Alıştırma 2'nin tam çözümü bulunmaktadır.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

def cozum_2():
    """Alıştırma 2'nin tam çözümü"""
    
    print("✅ Çözüm 2: İleri Resim İyileştirme")
    print("=" * 40)
    
    # GÖREV 1: Karma gürültülü resmi yükle
    print("\n📁 GÖREV 1: Karma gürültülü resim yükleme")
    
    # Karma gürültülü resim oluştur (test için)
    orijinal_resim = ornek_resim_olustur()
    karma_gurultulu = karma_gurultu_ekle(orijinal_resim)
    
    print(f"✅ Karma gürültülü resim hazır: {karma_gurultulu.shape}")
    
    # GÖREV 2: Multi-tip gürültü temizleme
    print("\n🧼 GÖREV 2: Multi-tip gürültü temizleme")
    
    def multi_gurultu_temizleme(resim):
        """Hem Gaussian hem Salt&Pepper gürültüsünü temizle"""
        
        # 1. Önce median filter (salt&pepper için)
        median_filtered = cv2.medianBlur(resim, 5)
        
        # 2. Sonra bilateral filter (Gaussian için ve kenarları koruyarak)
        bilateral_filtered = cv2.bilateralFilter(median_filtered, 9, 75, 75)
        
        # 3. Son olarak hafif Gaussian blur (kalite iyileştirme)
        final_result = cv2.GaussianBlur(bilateral_filtered, (3, 3), 0)
        
        return final_result
    
    temizlenmis_resim = multi_gurultu_temizleme(karma_gurultulu)
    print("✅ Multi-tip gürültü temizlendi")
    
    # GÖREV 3: Otomatik kontrast ayarlama
    print("\n⚡ GÖREV 3: Otomatik kontrast ayarlama")
    
    def otomatik_kontrast(resim, percentile_low=2, percentile_high=98):
        """Histogram stretching ile otomatik kontrast ayarlama"""
        
        ayarlanmis = resim.copy().astype(np.float32)
        
        # Her kanal için işlem yap
        for i in range(resim.shape[2]):
            kanal = ayarlanmis[:,:,i]
            
            # Düşük ve yüksek percentile bulun
            low_val = np.percentile(kanal, percentile_low)
            high_val = np.percentile(kanal, percentile_high)
            
            # Histogram stretching
            if high_val != low_val:
                kanal = 255 * (kanal - low_val) / (high_val - low_val)
                kanal = np.clip(kanal, 0, 255)
            
            ayarlanmis[:,:,i] = kanal
        
        return ayarlanmis.astype(np.uint8)
    
    kontrast_ayarli = otomatik_kontrast(temizlenmis_resim)
    print("✅ Otomatik kontrast ayarlandı")
    
    # GÖREV 4: CLAHE uygulaması
    print("\n🎯 GÖREV 4: CLAHE uygulaması")
    
    def clahe_uygula(resim, clip_limit=2.0, tile_grid_size=(8,8)):
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        
        # 1. Resmi LAB color space'e çevir
        lab_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2LAB)
        
        # 2. CLAHE objesi oluştur
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # 3. L kanalına CLAHE uygula
        lab_resim[:,:,0] = clahe.apply(lab_resim[:,:,0])
        
        # 4. Tekrar BGR'ye çevir
        clahe_sonuc = cv2.cvtColor(lab_resim, cv2.COLOR_LAB2BGR)
        
        return clahe_sonuc
    
    clahe_resim = clahe_uygula(kontrast_ayarli)
    print("✅ CLAHE uygulandı")
    
    # GÖREV 5: Gamma düzeltme
    print("\n🌟 GÖREV 5: Gamma düzeltme")
    
    def otomatik_gamma_duzeltme(resim):
        """Otomatik gamma düzeltme"""
        
        # 1. Resmin ortalama parlaklığını hesapla
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        ortalama_parlaklik = np.mean(gray) / 255.0
        
        # 2. Gamma değerini hesapla
        if ortalama_parlaklik < 0.3:
            gamma = 0.7  # Karanlık resim - aydınlat
        elif ortalama_parlaklik > 0.7:
            gamma = 1.3  # Parlak resim - koyulaştır
        else:
            gamma = 1.0  # Normal parlıklık
        
        # 3. LUT (Look-Up Table) oluştur
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)])
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        
        # 4. LUT'u uygula
        gamma_duzeltilmis = cv2.LUT(resim, lut)
        
        return gamma_duzeltilmis, gamma
    
    gamma_resim, kullanilan_gamma = otomatik_gamma_duzeltme(clahe_resim)
    print(f"✅ Gamma düzeltme uygulandı (γ={kullanilan_gamma:.2f})")
    
    # GÖREV 6: Morfolojik şekil analizi
    print("\n🔧 GÖREV 6: Morfolojik şekil analizi")
    
    def morfolojik_analiz(resim):
        """Morfolojik işlemlerle şekil temizleme"""
        
        # 1. Resmi ikili (binary) hale getir
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Kernel oluştur
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 3. Opening ile küçük gürültüleri temizle
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 4. Closing ile şekillerdeki boşlukları doldur
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 5. Sonucu orijinal resimle birleştir
        # Maske oluştur
        mask = closing > 0
        morfolojik_sonuc = resim.copy()
        
        # Maskenin tersini al ve arka planı biraz koyulaştır
        ters_mask = ~mask
        morfolojik_sonuc[ters_mask] = (morfolojik_sonuc[ters_mask] * 0.8).astype(np.uint8)
        
        return morfolojik_sonuc
    
    morfoloji_resim = morfolojik_analiz(gamma_resim)
    print("✅ Morfolojik analiz tamamlandı")
    
    # GÖREV 7: Filtreleme pipeline oluştur
    print("\n🚀 GÖREV 7: Filtreleme pipeline")
    
    def gelismis_pipeline(resim):
        """Tüm işlemleri birleştiren pipeline"""
        
        print("  🔄 Pipeline başlatılıyor...")
        start_time = time.time()
        
        # 1. Gürültü ön temizleme
        adim1 = multi_gurultu_temizleme(resim)
        
        # 2. Kontrast optimizasyonu
        adim2 = otomatik_kontrast(adim1)
        
        # 3. Adaptif histogram eşitleme
        adim3 = clahe_uygula(adim2)
        
        # 4. Gamma düzeltme
        adim4, _ = otomatik_gamma_duzeltme(adim3)
        
        # 5. Son rötuş filtreleme
        final_sonuc = cv2.bilateralFilter(adim4, 5, 50, 50)
        
        end_time = time.time()
        islem_suresi = end_time - start_time
        
        print(f"  ✅ Pipeline tamamlandı ({islem_suresi:.2f}s)")
        return final_sonuc, islem_suresi
    
    pipeline_sonuc, sure = gelismis_pipeline(karma_gurultulu)
    
    # GÖREV 8: Performans analizi
    print("\n📊 GÖREV 8: Performans analizi")
    
    def kalite_analizi(orijinal, islenmis, isim):
        """Kalite metrikleri hesapla"""
        
        # Boyutları eşitle
        if orijinal.shape != islenmis.shape:
            islenmis = cv2.resize(islenmis, (orijinal.shape[1], orijinal.shape[0]))
        
        # MSE hesapla
        mse = np.mean((orijinal.astype(np.float32) - islenmis.astype(np.float32)) ** 2)
        
        # PSNR hesapla
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        print(f"  {isim}:")
        print(f"    PSNR: {psnr:.2f} dB")
        print(f"    MSE: {mse:.2f}")
    
    print("📈 Kalite Karşılaştırması:")
    kalite_analizi(orijinal_resim, temizlenmis_resim, "Gürültü Temizleme")
    kalite_analizi(orijinal_resim, kontrast_ayarli, "Kontrast Ayarlama")
    kalite_analizi(orijinal_resim, clahe_resim, "CLAHE")
    kalite_analizi(orijinal_resim, gamma_resim, "Gamma Düzeltme")
    kalite_analizi(orijinal_resim, pipeline_sonuc, "Full Pipeline")
    
    # Görselleştirme
    print("\n🖼️ Sonuçları görselleştirme...")
    
    plt.figure(figsize=(18, 12))
    
    sonuclar = [
        (orijinal_resim, "Orijinal"),
        (karma_gurultulu, "Karma Gürültülü"),
        (temizlenmis_resim, "Gürültü Temizlendi"),
        (kontrast_ayarli, "Kontrast Ayarlı"),
        (clahe_resim, "CLAHE"),
        (gamma_resim, "Gamma Düzeltilmiş"),
        (morfoloji_resim, "Morfoloji"),
        (pipeline_sonuc, "Final Pipeline"),
    ]
    
    # Her sonucu subplot'ta göster
    for i, (resim, baslik) in enumerate(sonuclar):
        plt.subplot(3, 3, i+1)
        plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
        plt.title(baslik)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎉 Çözüm 2 tamamlandı!")
    print(f"⏱️ Toplam işlem süresi: {sure:.2f} saniye")

def ornek_resim_olustur():
    """Test için örnek resim oluştur"""
    resim = np.zeros((250, 250, 3), dtype=np.uint8)
    
    # Arka plan gradient
    for i in range(250):
        for j in range(250):
            r = int(80 + 60 * np.sin(i/40) * np.cos(j/40))
            g = int(90 + 50 * np.cos(i/30))
            b = int(100 + 40 * np.sin(j/35))
            resim[i, j] = [b, g, r]
    
    # Geometrik şekiller
    cv2.rectangle(resim, (30, 30), (120, 120), (200, 200, 200), -1)
    cv2.circle(resim, (180, 180), 40, (150, 150, 255), -1)
    cv2.ellipse(resim, (70, 200), (30, 15), 45, 0, 360, (255, 150, 150), -1)
    
    return resim

def karma_gurultu_ekle(resim):
    """Karma gürültü ekle (Gaussian + Salt&Pepper)"""
    gurultulu = resim.astype(np.float32)
    
    # Gaussian gürültü
    gaussian = np.random.normal(0, 15, resim.shape)
    gurultulu += gaussian
    
    # Salt & Pepper gürültü
    salt_mask = np.random.random(resim.shape[:2]) < 0.03
    pepper_mask = np.random.random(resim.shape[:2]) < 0.03
    
    gurultulu[salt_mask] = 255
    gurultulu[pepper_mask] = 0
    
    return np.clip(gurultulu, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    print("✅ OpenCV Alıştırma 2 - ÇÖZÜM")
    print("Bu dosyada tüm görevlerin çözümleri bulunmaktadır.\n")
    
    cozum_2()