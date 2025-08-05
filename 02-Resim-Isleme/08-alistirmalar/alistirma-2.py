"""
🎯 Alıştırma 2: İleri Resim İyileştirme
======================================

Zorluk: ⭐⭐⭐ (İleri)
Süre: 60-90 dakika
Konular: Gürültü azaltma, kontrast düzeltme, morfoloji

Bu alıştırmada ileri seviye resim iyileştirme tekniklerini uygulayacaksınız.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

def alistirma_2():
    """
    🎯 GÖREV 2: İleri Resim İyileştirme Alıştırması
    
    Bu alıştırmada şu görevleri tamamlamanız gerekiyor:
    
    1. ✅ Karma gürültülü resmi yükleyin
    2. 🧼 Multi-tip gürültü temizleme (Gaussian + Salt&Pepper)
    3. ⚡ Otomatik kontrast ayarlama implementasyonu
    4. 🎯 CLAHE (Adaptive Histogram Equalization) uygulaması
    5. 🌟 Gamma düzeltme ile parlaklık optimizasyonu
    6. 🔧 Morfolojik işlemlerle şekil analizi
    7. 🚀 Filtreleme pipeline oluşturma
    8. 📊 Performans analizi ve karşılaştırma
    """
    
    print("🎯 Alıştırma 2: İleri Resim İyileştirme")
    print("=" * 45)
    
    # GÖREV 1: Karma gürültülü resmi yükle
    print("\n📁 GÖREV 1: Karma gürültülü resim yükleme")
    
    # Karma gürültülü resim oluştur (test için)
    orijinal_resim = ornek_resim_olustur()
    karma_gurultulu = karma_gurultu_ekle(orijinal_resim)
    
    print(f"✅ Karma gürültülü resim hazır: {karma_gurultulu.shape}")
    
    # GÖREV 2: Multi-tip gürültü temizleme
    print("\n🧼 GÖREV 2: Multi-tip gürültü temizleme")
    
    def multi_gurultu_temizleme(resim):
        """
        TODO: Hem Gaussian hem Salt&Pepper gürültüsünü temizleyin
        
        Önerilen yaklaşım:
        1. Önce median filter (salt&pepper için)
        2. Sonra bilateral filter (Gaussian için)
        3. Son olarak hafif Gaussian blur (kalite iyileştirme)
        
        İpucu: Farklı sıralamaları deneyin!
        """
        
        # TODO: Burayı doldurun
        temizlenmis = resim.copy()  # BUNU DEĞİŞTİRİN!
        
        return temizlenmis
    
    temizlenmis_resim = multi_gurultu_temizleme(karma_gurultulu)
    print("✅ Multi-tip gürültü temizlendi")
    
    # GÖREV 3: Otomatik kontrast ayarlama
    print("\n⚡ GÖREV 3: Otomatik kontrast ayarlama")
    
    def otomatik_kontrast(resim, percentile_low=2, percentile_high=98):
        """
        TODO: Histogram stretching ile otomatik kontrast ayarlama
        
        Algoritma:
        1. Her kanal için düşük ve yüksek percentile bulun
        2. Bu değerler arasındaki aralığı 0-255'e çekin
        3. Formül: yeni_değer = 255 * (eski - min) / (max - min)
        """
        
        # TODO: Burayı doldurun
        ayarlanmis = resim.copy()  # BUNU DEĞİŞTİRİN!
        
        return ayarlanmis
    
    kontrast_ayarli = otomatik_kontrast(temizlenmis_resim)
    print("✅ Otomatik kontrast ayarlandı")
    
    # GÖREV 4: CLAHE uygulaması
    print("\n🎯 GÖREV 4: CLAHE uygulaması")
    
    def clahe_uygula(resim, clip_limit=2.0, tile_grid_size=(8,8)):
        """
        TODO: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        İpucu:
        1. Resmi LAB color space'e çevirin
        2. L kanalına CLAHE uygulayın
        3. Tekrar BGR'ye çevirin
        """
        
        # TODO: Burayı doldurun
        clahe_sonuc = resim.copy()  # BUNU DEĞİŞTİRİN!
        
        return clahe_sonuc
    
    clahe_resim = clahe_uygula(kontrast_ayarli)
    print("✅ CLAHE uygulandı")
    
    # GÖREV 5: Gamma düzeltme
    print("\n🌟 GÖREV 5: Gamma düzeltme")
    
    def otomatik_gamma_duzeltme(resim):
        """
        TODO: Otomatik gamma düzeltme
        
        Algoritma:
        1. Resmin ortalama parlaklığını hesaplayın
        2. Eğer çok karanlıksa gamma < 1 (aydınlatma)
        3. Eğer çok parlaksa gamma > 1 (koyulaştırma)
        4. LUT (Look-Up Table) oluşturup uygulayın
        """
        
        # TODO: Burayı doldurun
        gamma_duzeltilmis = resim.copy()  # BUNU DEĞİŞTİRİN!
        gamma_degeri = 1.0  # BUNU HESAPLAYIN!
        
        return gamma_duzeltilmis, gamma_degeri
    
    gamma_resim, kullanilan_gamma = otomatik_gamma_duzeltme(clahe_resim)
    print(f"✅ Gamma düzeltme uygulandı (γ={kullanilan_gamma:.2f})")
    
    # GÖREV 6: Morfolojik şekil analizi
    print("\n🔧 GÖREV 6: Morfolojik şekil analizi")
    
    def morfolojik_analiz(resim):
        """
        TODO: Morfolojik işlemlerle şekil temizleme
        
        Yapılacaklar:
        1. Resmi ikili (binary) hale getirin
        2. Opening ile küçük gürültüleri temizleyin
        3. Closing ile şekillerdeki boşlukları doldurun
        4. Sonucu orijinal resimle birleştirin
        """
        
        # TODO: Burayı doldurun
        morfolojik_sonuc = resim.copy()  # BUNU DEĞİŞTİRİN!
        
        return morfolojik_sonuc
    
    morfoloji_resim = morfolojik_analiz(gamma_resim)
    print("✅ Morfolojik analiz tamamlandı")
    
    # GÖREV 7: Filtreleme pipeline oluştur
    print("\n🚀 GÖREV 7: Filtreleme pipeline")
    
    def gelismis_pipeline(resim):
        """
        TODO: Tüm işlemleri birleştiren pipeline
        
        Pipeline adımları:
        1. Gürültü ön temizleme
        2. Kontrast optimizasyonu
        3. Adaptif histogram eşitleme
        4. Gamma düzeltme
        5. Son rötuş filtreleme
        """
        
        print("  🔄 Pipeline başlatılıyor...")
        start_time = time.time()
        
        # TODO: Tüm işlemleri sırayla uygulayın
        # adim1 = ???
        # adim2 = ???
        # ...
        
        # Geçici çözüm
        final_sonuc = resim.copy()  # BUNU DEĞİŞTİRİN!
        
        end_time = time.time()
        islem_suresi = end_time - start_time
        
        print(f"  ✅ Pipeline tamamlandı ({islem_suresi:.2f}s)")
        return final_sonuc, islem_suresi
    
    pipeline_sonuc, sure = gelismis_pipeline(karma_gurultulu)
    
    # GÖREV 8: Performans analizi
    print("\n📊 GÖREV 8: Performans analizi")
    
    def kalite_analizi(orijinal, islenmis, isim):
        """Kalite metrikleri hesapla"""
        
        # TODO: PSNR, MSE ve SSIM hesaplayın
        # PSNR hesabı için önceki alıştırmayı kullanabilirsiniz
        
        psnr = 0.0  # BUNU HESAPLAYIN!
        mse = 0.0   # BUNU HESAPLAYIN!
        
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
    
    # TODO: 3x3 subplot ile tüm sonuçları gösterin
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
    
    # TODO: Her sonucu subplot'ta gösterin
    for i, (resim, baslik) in enumerate(sonuclar):
        plt.subplot(3, 3, i+1)
        # TODO: Resmi gösterin ve başlık ekleyin
        
        # Geçici çözüm
        plt.text(0.5, 0.5, f'{baslik}\n(TODO: Resim ekleyin)', 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(baslik)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎉 Alıştırma 2 tamamlandı!")
    print(f"⏱️ Toplam işlem süresi: {sure:.2f} saniye")
    print("\nℹ️ Çözümü görmek için: python cozumler/cozum-2.py")

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

def kontrol_listesi():
    """Alıştırma kontrol listesi"""
    print("\n✅ KONTROL LİSTESİ")
    print("=" * 30)
    print("[ ] Karma gürültü başarıyla temizlendi")
    print("[ ] Otomatik kontrast ayarlama çalışıyor")
    print("[ ] CLAHE doğru uygulandı")
    print("[ ] Gamma düzeltme otomatik hesaplanıyor")
    print("[ ] Morfolojik işlemler uygulandı")
    print("[ ] Pipeline tüm adımları içeriyor")
    print("[ ] Kalite metrikleri hesaplandı")
    print("[ ] Sonuçlar görselleştirildi")
    print("\n🎯 Hepsini tamamladıysanız çözümle karşılaştırın!")

if __name__ == "__main__":
    print("🎓 OpenCV İleri Resim İşleme Alıştırmaları")
    print("Bu alıştırmada ileri seviye resim iyileştirme tekniklerini öğreneceksiniz.\n")
    
    try:
        alistirma_2()
        kontrol_listesi()
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        print("\n💡 İpuçları:")
        print("   • Numpy array işlemlerinde veri tiplerini kontrol edin")
        print("   • CLAHE için LAB color space kullanmayı unutmayın")
        print("   • Gamma LUT hesaplamasında overflow kontrolü yapın")
        print("   • Morfolojik işlemler için binary threshold gerekebilir")