# 🎯 Alıştırma 2: İleri Resim İyileştirme

**Zorluk:** ⭐⭐⭐ (İleri)  
**Süre:** 60-90 dakika  
**Konular:** Gürültü azaltma, kontrast düzeltme, morfoloji  

Bu alıştırmada ileri seviye resim iyileştirme tekniklerini uygulayacaksınız.

## 📚 Gerekli Kütüphaneler

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
```

## 🎯 Görevler

Bu alıştırmada şu görevleri tamamlamanız gerekiyor:

1. ✅ Karma gürültülü resmi yükleyin
2. 🧼 Multi-tip gürültü temizleme (Gaussian + Salt&Pepper)
3. ⚡ Otomatik kontrast ayarlama implementasyonu
4. 🎯 CLAHE (Adaptive Histogram Equalization) uygulaması
5. 🌟 Gamma düzeltme ile parlaklık optimizasyonu
6. 🔧 Morfolojik işlemlerle şekil analizi
7. 🚀 Filtreleme pipeline oluşturma
8. 📊 Performans analizi ve karşılaştırma

---

## 📁 GÖREV 1: Karma Gürültülü Resim Yükleme

Test için karma gürültülü resim oluşturun:

```python
# Test için karma gürültülü resim oluştur
orijinal_resim = ornek_resim_olustur()  # Yardımcı fonksiyon
karma_gurultulu = karma_gurultu_ekle(orijinal_resim)  # Yardımcı fonksiyon

print(f"✅ Karma gürültülü resim hazır: {karma_gurultulu.shape}")
```

**Not:** Yardımcı fonksiyonlar dökümanın sonunda tanımlıdır.

---

## 🧼 GÖREV 2: Multi-tip Gürültü Temizleme

Hem Gaussian hem Salt&Pepper gürültüsünü temizleyin:

```python
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
```

**İpuçları:**
- `cv2.medianBlur()` salt&pepper için etkilidir
- `cv2.bilateralFilter()` kenarları koruyarak Gaussian gürültüyü azaltır
- `cv2.GaussianBlur()` son rötuş için
- Sıralama önemlidir - farklı kombinasyonları deneyin

---

## ⚡ GÖREV 3: Otomatik Kontrast Ayarlama

Histogram stretching ile otomatik kontrast ayarlama:

```python
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
```

**İpuçları:**
- `np.percentile(array, percentile)` kullanın
- Her kanal (B, G, R) için ayrı ayrı işlem yapın
- Sıfıra bölme durumuna dikkat edin
- Değerleri 0-255 aralığında tutun

---

## 🎯 GÖREV 4: CLAHE Uygulaması

CLAHE (Contrast Limited Adaptive Histogram Equalization) uygulayın:

```python
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
```

**İpuçları:**
- `cv2.cvtColor(resim, cv2.COLOR_BGR2LAB)` için LAB'a çevirin
- `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`
- Sadece L kanalına uygulayın: `lab[:,:,0] = clahe.apply(lab[:,:,0])`
- `cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)` ile geri çevirin

---

## 🌟 GÖREV 5: Gamma Düzeltme

Otomatik gamma düzeltme ile parlaklık optimizasyonu:

```python
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
```

**İpuçları:**
- Ortalama parlaklık: `np.mean(cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)) / 255.0`
- Gamma < 1: aydınlatma, Gamma > 1: koyulaştırma
- LUT: `np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)])`
- `cv2.LUT(resim, lut)` ile uygulayın

---

## 🔧 GÖREV 6: Morfolojik Şekil Analizi

Morfolojik işlemlerle şekil temizleme:

```python
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
```

**İpuçları:**
- Binary: `cv2.threshold()` veya `cv2.adaptiveThreshold()`
- Kernel: `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))`
- Opening: `cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)`
- Closing: `cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)`

---

## 🚀 GÖREV 7: Filtreleme Pipeline

Tüm işlemleri birleştiren pipeline oluşturun:

```python
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
```

**İpuçları:**
- Yukarıda yazdığınız fonksiyonları sırayla çağırın
- İşlem süresini ölçün
- Her adımda sonucu bir sonrakine geçirin

---

## 📊 GÖREV 8: Performans Analizi

Kalite metrikleri hesaplayın ve karşılaştırın:

```python
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
```

---

## 🖼️ Görselleştirme

Tüm sonuçları 3x3 subplot ile görselleştirin:

```python
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
    plt.title(baslik)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

---

## 🔨 Yardımcı Fonksiyonlar

### Örnek Resim Oluşturucu

```python
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
```

### Karma Gürültü Oluşturucu

```python
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
```

---

## ✅ Kontrol Listesi

Alıştırmayı tamamladıktan sonra kontrol edin:

- [ ] Karma gürültü başarıyla temizlendi
- [ ] Otomatik kontrast ayarlama çalışıyor
- [ ] CLAHE doğru uygulandı
- [ ] Gamma düzeltme otomatik hesaplanıyor
- [ ] Morfolojik işlemler uygulandı
- [ ] Pipeline tüm adımları içeriyor
- [ ] Kalite metrikleri hesaplandı
- [ ] Sonuçlar görselleştirildi

## 💡 İpuçları

### Genel İpuçları
- Her adımda ara sonuçları kontrol edin
- Parametreleri optimizasyon yapın
- Pipeline'da adım sırasını deneyin
- Görsel sonuçları karşılaştırın

### Teknik İpuçları
- Numpy array işlemlerinde veri tiplerini kontrol edin
- CLAHE için LAB color space kullanmayı unutmayın
- Gamma LUT hesaplamasında overflow kontrolü yapın
- Morfolojik işlemler için binary threshold gerekebilir

### Performans İpuçları
- `time.time()` ile süre ölçümü yapın
- Büyük resimlerle test ederken dikkatli olun
- Pipeline'ı optimize etmek için profiling yapın

## 🚀 Bonus Görevler

Temel alıştırmayı tamamladıysanız bunları deneyin:

- [ ] Adaptive filtering
- [ ] Multi-scale processing
- [ ] Custom gürültü modelleri
- [ ] İnteraktif parametre ayarlama
- [ ] Batch processing
- [ ] Real-time filtering

## 🎯 Çözüm

Tamamladıktan sonra çözümle karşılaştırın:
```bash
python cozumler/cozum-2.py
```

---

*Bu alıştırma Eren Terzi tarafından hazırlanmıştır.*