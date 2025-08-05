"""
🎯 Alıştırma 1: Temel Resim İşleme
================================

Zorluk: ⭐⭐ (Orta)
Süre: 45-60 dakika
Konular: Geometrik transformasyon, filtreleme, histogram

Bu alıştırmada temel resim işleme tekniklerini uygulayacaksınız.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def allistirma_1():
    """
    🎯 GÖREV 1: Temel Resim İşleme Alıştırması
    
    Bu alıştırmada şu görevleri tamamlamanız gerekiyor:
    
    1. ✅ Test resmini yükleyin
    2. 🔄 Resmi 45° saat yönünde döndürün
    3. 📏 Resmi %75 oranında küçültün
    4. 🌫️ 3 farklı boyutta Gaussian blur uygulayın (3x3, 7x7, 15x15)
    5. 🧂 Salt & Pepper gürültü ekleyip median filter ile temizleyin
    6. 📊 Histogram eşitleme uygulayın
    7. 📈 PSNR hesaplayarak kalite karşılaştırması yapın
    8. 🖼️ Tüm sonuçları görselleştirin
    """
    
    print("🎯 Alıştırma 1: Temel Resim İşleme")
    print("=" * 40)
    
    # GÖREV 1: Test resmini yükle
    print("\n📁 GÖREV 1: Test resmini yükleme")
    test_resmi_yolu = "test-resimleri/normal.jpg"
    
    # TODO: Burada cv2.imread() kullanarak resmi yükleyin
    # resim = ???
    
    # Eğer resim bulunamazsa test resmi oluştur
    if 'resim' not in locals() or resim is None:
        print("⚠️ Test resmi bulunamadı, örnek resim oluşturuluyor...")
        resim = ornek_resim_olustur()
    
    print(f"✅ Resim yüklendi: {resim.shape}")
    
    # GÖREV 2: Resmi 45° saat yönünde döndür
    print("\n🔄 GÖREV 2: 45° döndürme")
    
    # TODO: cv2.getRotationMatrix2D() ve cv2.warpAffine() kullanın
    # İpucu: Merkez nokta resmin ortası olmalı
    # merkez = (resim.shape[1]//2, resim.shape[0]//2)
    # rotasyon_matrisi = ???
    # dondurulmus_resim = ???
    
    # Geçici çözüm (siz kendi kodunuzu yazın)
    dondurulmus_resim = resim.copy()  # BUNU DEĞİŞTİRİN!
    
    print("✅ Resim döndürüldü")
    
    # GÖREV 3: Resmi %75 oranında küçült
    print("\n📏 GÖREV 3: %75 küçültme")
    
    # TODO: cv2.resize() kullanın
    # İpucu: Yeni boyut = (genişlik * 0.75, yükseklik * 0.75)
    # kucultulmus_resim = ???
    
    # Geçici çözüm (siz kendi kodunuzu yazın)
    kucultulmus_resim = resim.copy()  # BUNU DEĞİŞTİRİN!
    
    print("✅ Resim küçültüldü")
    
    # GÖREV 4: 3 farklı boyutta Gaussian blur
    print("\n🌫️ GÖREV 4: Gaussian blur (3 boyut)")
    
    # TODO: cv2.GaussianBlur() ile 3 farklı kernel boyutu
    # blur_3x3 = ???
    # blur_7x7 = ???
    # blur_15x15 = ???
    
    # Geçici çözümler (siz kendi kodlarınızı yazın)
    blur_3x3 = resim.copy()    # BUNU DEĞİŞTİRİN!
    blur_7x7 = resim.copy()    # BUNU DEĞİŞTİRİN!
    blur_15x15 = resim.copy()  # BUNU DEĞİŞTİRİN!
    
    print("✅ Gaussian blur uygulandı")
    
    # GÖREV 5: Salt & Pepper gürültü ekle ve temizle
    print("\n🧂 GÖREV 5: Salt & Pepper gürültü temizleme")
    
    # Gürültü ekleme (bu kısım hazır)
    gurultulu_resim = resim.copy().astype(np.float32)
    
    # Salt noise (beyaz piksel)
    salt_mask = np.random.random(resim.shape[:2]) < 0.05
    gurultulu_resim[salt_mask] = 255
    
    # Pepper noise (siyah piksel)
    pepper_mask = np.random.random(resim.shape[:2]) < 0.05
    gurultulu_resim[pepper_mask] = 0
    
    gurultulu_resim = gurultulu_resim.astype(np.uint8)
    
    # TODO: cv2.medianBlur() ile gürültüyü temizleyin
    # temizlenmis_resim = ???
    
    # Geçici çözüm (siz kendi kodunuzu yazın)
    temizlenmis_resim = gurultulu_resim.copy()  # BUNU DEĞİŞTİRİN!
    
    print("✅ Gürültü temizlendi")
    
    # GÖREV 6: Histogram eşitleme
    print("\n📊 GÖREV 6: Histogram eşitleme")
    
    # TODO: cv2.equalizeHist() kullanın (önce gri seviyeye çevirin)
    # gri_resim = ???
    # esitlenmis_gri = ???
    # esitlenmis_resim = ??? (gri'yi tekrar renkli yapın)
    
    # Geçici çözüm (siz kendi kodunuzu yazın)
    esitlenmis_resim = resim.copy()  # BUNU DEĞİŞTİRİN!
    
    print("✅ Histogram eşitleme uygulandı")
    
    # GÖREV 7: PSNR hesaplama
    print("\n📈 GÖREV 7: PSNR hesaplama")
    
    def psnr_hesapla(orijinal, islenmis):
        """PSNR (Peak Signal-to-Noise Ratio) hesapla"""
        # TODO: PSNR formülünü uygulayın
        # MSE = Mean Squared Error = ortalama((orijinal - islenmis)²)
        # PSNR = 20 * log10(255 / sqrt(MSE))
        
        # Geçici çözüm (siz kendi kodunuzu yazın)
        return 0.0  # BUNU DEĞİŞTİRİN!
    
    # TODO: Her işlem için PSNR hesaplayın
    psnr_dondurulmus = psnr_hesapla(resim, dondurulmus_resim)
    psnr_kucultulmus = psnr_hesapla(resim, kucultulmus_resim)
    psnr_blur = psnr_hesapla(resim, blur_7x7)
    psnr_temizlenmis = psnr_hesapla(resim, temizlenmis_resim)
    psnr_esitlenmis = psnr_hesapla(resim, esitlenmis_resim)
    
    print("✅ PSNR değerleri hesaplandı")
    
    # GÖREV 8: Sonuçları görselleştir
    print("\n🖼️ GÖREV 8: Görselleştirme")
    
    # TODO: Matplotlib ile 3x3 subplot oluşturun ve tüm sonuçları gösterin
    # İpucu: plt.subplot(3, 3, i) kullanın
    
    plt.figure(figsize=(15, 15))
    
    # TODO: Her subplot için resim ve başlık ekleyin
    # Örnek:
    # plt.subplot(3, 3, 1)
    # plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    # plt.title('Orijinal')
    # plt.axis('off')
    
    # Geçici çözüm - siz doldurunu!
    plt.subplot(3, 3, 1)
    plt.text(0.5, 0.5, 'Orijinal\n(TODO: Resim ekleyin)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Orijinal')
    plt.axis('off')
    
    # Diğer subplotları da ekleyin...
    
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
    
    print("\n🎉 Alıştırma 1 tamamlandı!")
    print("\nℹ️ Çözümü görmek için: python cozumler/cozum-1.py")

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

def kontrol_listesi():
    """Alıştırma kontrol listesi"""
    print("\n✅ KONTROL LİSTESİ")
    print("=" * 30)
    print("[ ] Resim başarıyla yüklendi")
    print("[ ] 45° döndürme çalışıyor")
    print("[ ] %75 küçültme doğru boyutta")
    print("[ ] 3 farklı Gaussian blur uygulandı")
    print("[ ] Salt & Pepper gürültü temizlendi")
    print("[ ] Histogram eşitleme uygulandı")
    print("[ ] PSNR değerleri hesaplandı")
    print("[ ] Tüm sonuçlar görselleştirildi")
    print("\n🎯 Hepsini tamamladıysanız çözümle karşılaştırın!")

if __name__ == "__main__":
    print("🎓 OpenCV Resim İşleme Alıştırmaları")
    print("Bu alıştırmada temel resim işleme tekniklerini öğreneceksiniz.\n")
    
    try:
        allistirma_1()
        kontrol_listesi()
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        print("\n💡 İpuçları:")
        print("   • Test resimlerini oluşturdunuz mu? (python test-resimleri/resim_olusturucu.py)")
        print("   • Tüm import'lar doğru mu?")
        print("   • Değişken adları doğru yazıldı mı?")