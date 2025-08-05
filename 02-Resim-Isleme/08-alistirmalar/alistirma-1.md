# 🎯 Alıştırma 1: Temel Resim İşleme

**Zorluk:** ⭐⭐ (Orta)  
**Süre:** 45-60 dakika  
**Konular:** Geometrik transformasyon, filtreleme, histogram  

Bu alıştırmada temel resim işleme tekniklerini uygulayacaksınız.

## 📚 Gerekli Kütüphaneler

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

## 🎯 Görevler

Bu alıştırmada şu görevleri tamamlamanız gerekiyor:

1. ✅ Test resmini yükleyin
2. 🔄 Resmi 45° saat yönünde döndürün
3. 📏 Resmi %75 oranında küçültün
4. 🌫️ 3 farklı boyutta Gaussian blur uygulayın (3x3, 7x7, 15x15)
5. 🧂 Salt & Pepper gürültü ekleyip median filter ile temizleyin
6. 📊 Histogram eşitleme uygulayın
7. 📈 PSNR hesaplayarak kalite karşılaştırması yapın
8. 🖼️ Tüm sonuçları görselleştirin

---

## 📁 GÖREV 1: Test Resmini Yükleme

Test resmi yolunu belirleyin ve resmi yükleyin:

```python
test_resmi_yolu = "test-resimleri/normal.jpg"

# TODO: Burada cv2.imread() kullanarak resmi yükleyin
# resim = ???

# Eğer resim bulunamazsa test resmi oluştur
if 'resim' not in locals() or resim is None:
    print("⚠️ Test resmi bulunamadı, örnek resim oluşturuluyor...")
    resim = ornek_resim_olustur()  # Bu fonksiyon altta tanımlı
```

**İpuçları:**
- `cv2.imread()` fonksiyonunu kullanın
- Resim yüklenmezse None döner, kontrol edin
- `resim.shape` ile boyutları yazdırın

---

## 🔄 GÖREV 2: 45° Döndürme

Resmi 45° saat yönünde döndürün:

```python
# TODO: cv2.getRotationMatrix2D() ve cv2.warpAffine() kullanın
# İpucu: Merkez nokta resmin ortası olmalı
# merkez = (resim.shape[1]//2, resim.shape[0]//2)
# rotasyon_matrisi = ???
# dondurulmus_resim = ???
```

**İpuçları:**
- Merkez nokta: `(genişlik//2, yükseklik//2)`
- Saat yönünde döndürme için negatif açı kullanın
- `cv2.getRotationMatrix2D(merkez, açı, ölçek)`
- `cv2.warpAffine(resim, matris, (genişlik, yükseklik))`

---

## 📏 GÖREV 3: %75 Küçültme

Resmi %75 oranında küçültün:

```python
# TODO: cv2.resize() kullanın
# İpucu: Yeni boyut = (genişlik * 0.75, yükseklik * 0.75)
# kucultulmus_resim = ???
```

**İpuçları:**
- `cv2.resize(resim, (yeni_genişlik, yeni_yükseklik))`
- Boyutları tam sayı yapın: `int(genişlik * 0.75)`
- Interpolasyon yöntemi: `cv2.INTER_LINEAR`

---

## 🌫️ GÖREV 4: Gaussian Blur (3 Boyut)

3 farklı boyutta Gaussian blur uygulayın:

```python
# TODO: cv2.GaussianBlur() ile 3 farklı kernel boyutu
# blur_3x3 = ???
# blur_7x7 = ???
# blur_15x15 = ???
```

**İpuçları:**
- `cv2.GaussianBlur(resim, (kernel_size, kernel_size), sigmaX)`
- Kernel boyutu tek sayı olmalı
- Sigma değeri 0 olabilir (otomatik hesaplanır)

---

## 🧂 GÖREV 5: Salt & Pepper Gürültü Temizleme

Gürültü ekleme kısmı hazır, sadece temizleme yapın:

```python
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
```

**İpuçları:**
- `cv2.medianBlur(resim, kernel_size)`
- Salt & Pepper gürültü için median filter en etkilidir
- Kernel boyutu 3, 5 veya 7 olabilir

---

## 📊 GÖREV 6: Histogram Eşitleme

Histogram eşitleme uygulayın:

```python
# TODO: cv2.equalizeHist() kullanın (önce gri seviyeye çevirin)
# gri_resim = ???
# esitlenmis_gri = ???
# esitlenmis_resim = ??? (gri'yi tekrar renkli yapın)
```

**İpuçları:**
- Önce `cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)` ile gri yap
- `cv2.equalizeHist()` sadece gri resimlerle çalışır
- Sonucu renkli yapmak için `cv2.cvtColor(gri, cv2.COLOR_GRAY2BGR)`

---

## 📈 GÖREV 7: PSNR Hesaplama

PSNR (Peak Signal-to-Noise Ratio) hesaplama fonksiyonu yazın:

```python
def psnr_hesapla(orijinal, islenmis):
    """PSNR (Peak Signal-to-Noise Ratio) hesapla"""
    # TODO: PSNR formülünü uygulayın
    # MSE = Mean Squared Error = ortalama((orijinal - islenmis)²)
    # PSNR = 20 * log10(255 / sqrt(MSE))
    
    # Boyutları eşitle
    if orijinal.shape != islenmis.shape:
        islenmis = cv2.resize(islenmis, (orijinal.shape[1], orijinal.shape[0]))
    
    # TODO: MSE hesaplayın
    # TODO: PSNR hesaplayın
    return 0.0  # BUNU DEĞİŞTİRİN!

# Her işlem için PSNR hesaplayın
psnr_dondurulmus = psnr_hesapla(resim, dondurulmus_resim)
psnr_kucultulmus = psnr_hesapla(resim, kucultulmus_resim)
psnr_blur = psnr_hesapla(resim, blur_7x7)
psnr_temizlenmis = psnr_hesapla(resim, temizlenmis_resim)
psnr_esitlenmis = psnr_hesapla(resim, esitlenmis_resim)
```

**İpuçları:**
- MSE: `np.mean((img1.astype(float) - img2.astype(float)) ** 2)`
- PSNR: `20 * np.log10(255.0 / np.sqrt(mse))`
- MSE = 0 ise PSNR = ∞

---

## 🖼️ GÖREV 8: Görselleştirme

Matplotlib ile 3x3 subplot oluşturun ve tüm sonuçları gösterin:

```python
plt.figure(figsize=(15, 15))

# TODO: Her subplot için resim ve başlık ekleyin
# Örnek:
# plt.subplot(3, 3, 1)
# plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
# plt.title('Orijinal')
# plt.axis('off')

# TODO: Tüm sonuçları gösterin:
# 1. Orijinal
# 2. Döndürülmüş + PSNR
# 3. Küçültülmüş + PSNR
# 4. Blur 3x3
# 5. Blur 7x7 + PSNR
# 6. Blur 15x15
# 7. Gürültülü
# 8. Temizlenmiş + PSNR
# 9. Histogram Eşitlenmiş + PSNR

plt.tight_layout()
plt.show()
```

**İpuçları:**
- OpenCV BGR, matplotlib RGB kullanır: `cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)`
- `plt.axis('off')` ile eksenleri gizleyin
- PSNR değerlerini başlıkta gösterin

---

## 🔨 Yardımcı Fonksiyonlar

### Örnek Resim Oluşturucu

```python
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
```

---

## ✅ Kontrol Listesi

Alıştırmayı tamamladıktan sonra kontrol edin:

- [ ] Resim başarıyla yüklendi
- [ ] 45° döndürme çalışıyor
- [ ] %75 küçültme doğru boyutta
- [ ] 3 farklı Gaussian blur uygulandı
- [ ] Salt & Pepper gürültü temizlendi
- [ ] Histogram eşitleme uygulandı
- [ ] PSNR değerleri hesaplandı
- [ ] Tüm sonuçlar görselleştirildi

## 📋 Sonuç Raporu

Programınız şu formatta rapor üretmelidir:

```
📋 SONUÇ RAPORU
==============================
🔄 Döndürme PSNR: XX.XX dB
📏 Küçültme PSNR: XX.XX dB
🌫️ Blur PSNR: XX.XX dB
🧼 Temizleme PSNR: XX.XX dB
📊 Eşitleme PSNR: XX.XX dB
```

## 💡 İpuçları

### Genel İpuçları
- Her adımda sonuçları kontrol edin
- Hata mesajlarını dikkatlice okuyun
- Test resimlerini oluşturdunuz mu?
- Değişken adlarını doğru yazdınız mı?

### Teknik İpuçları
- Veri tiplerini kontrol edin (uint8, float32)
- OpenCV BGR, matplotlib RGB kullanır
- Kernel boyutları tek sayı olmalı
- Boyut uyumsuzluklarına dikkat edin

### Hata Durumları
- Test resmi bulunamıyorsa örnek resim oluşturulur
- Import hataları varsa kütüphaneleri kontrol edin
- PSNR hesaplamasında sıfıra bölme durumunu kontrol edin

## 🎯 Çözüm

Tamamladıktan sonra çözümle karşılaştırın:
```bash
python cozumler/cozum-1.py
```

## 🚀 Bonus Görevler

Temel alıştırmayı tamamladıysanız bunları deneyin:

- [ ] İnteraktif parametre ayarlama
- [ ] Batch processing (birden fazla resim)
- [ ] Farklı dosya formatları desteği
- [ ] Histogram görselleştirme
- [ ] Otomatik threshold hesaplama

---

*Bu alıştırma Eren Terzi tarafından hazırlanmıştır.*