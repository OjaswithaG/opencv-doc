# 🎯 Alıştırma 2: Resim İşlemleri ve Renk Uzayları

**Zorluk Seviyesi:** 🚀 Orta  
**Tahmini Süre:** 25-30 dakika  

Bu alıştırma resim okuma/yazma, renk uzayı dönüşümleri ve temel resim manipülasyon işlemlerini içerir.

## 📚 Gerekli Kütüphaneler

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

## 🎯 Görevler

Bu alıştırmada resim işleme ve renk uzaylarını öğreneceksiniz.

---

## 🎯 GÖREV 1: Resim Okuma ve Farklı Formatlarda Kaydetme

Resim okuma/yazma işlemleri ve farklı format karşılaştırması yapın.

### Yapılacaklar:
1. `examples/gradyan.jpg` dosyasını okuyun (yoksa oluşturun)
2. Resmin bilgilerini yazdırın (boyut, tip, kanal)
3. Aynı resmi şu formatlarda kaydedin:
   - PNG formatında (yüksek kalite)
   - BMP formatında (sıkıştırmasız)
   - JPEG formatında %75 kalite ile
4. Dosya boyutlarını karşılaştırın ve yazdırın

### Kodunuzu Yazın:
```python
def gorev_1_resim_okuma_kaydetme():
    print("🎯 GÖREV 1: Resim Okuma ve Kaydetme")
    print("-" * 40)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: Önce resim var mı kontrol edin, yoksa oluşturun
    # cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- `cv2.imread('dosya_yolu')` ile resim okuma
- `cv2.imwrite()` fonksiyonunda kalite parametresi kullanın
- JPEG kalite: `[cv2.IMWRITE_JPEG_QUALITY, 75]`
- PNG kalite: `[cv2.IMWRITE_PNG_COMPRESSION, 1]`
- `os.path.getsize()` ile dosya boyutu
- Resim bilgileri: `resim.shape`, `resim.dtype`

---

## 🎯 GÖREV 2: Renk Uzayı Dönüşümleri

Farklı renk uzayları arasında dönüşüm yapın ve görselleştirin.

### Yapılacaklar:
1. Renkli bir test resmi oluşturun veya yükleyin
2. BGR'dan şu renk uzaylarına dönüştürün:
   - RGB
   - HSV  
   - LAB
   - Gri tonlama
3. Tüm dönüşümleri 2x3 subplot ile gösterin
4. Her renk uzayının avantajlarını yorumlayın

### Kodunuzu Yazın:
```python
def gorev_2_renk_uzayi_donusumleri():
    print("\n🎯 GÖREV 2: Renk Uzayı Dönüşümleri")
    print("-" * 40)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: plt.subplot(2, 3, 1) ile subplot oluşturun
    # cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ile dönüştürün
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- `cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)` ile dönüşüm
- Renk uzayları:
  - `cv2.COLOR_BGR2RGB`
  - `cv2.COLOR_BGR2HSV`
  - `cv2.COLOR_BGR2LAB`
  - `cv2.COLOR_BGR2GRAY`
- `plt.subplot(2, 3, i)` ile görselleştirme
- `plt.imshow()` RGB, gri için `cmap='gray'`

### Renk Uzayları Avantajları:
- **RGB**: İnsan gözü, ekran görüntüleme
- **HSV**: Renk tabanlı segmentasyon
- **LAB**: Renk farkı hesaplama
- **Gri**: Hesaplama hızı, bellek tasarrufu

---

## 🎯 GÖREV 3: HSV ile Renk Filtreleme

HSV renk uzayında belirli renkleri filtreleme yapın.

### Yapılacaklar:
1. Farklı renklerde geometrik şekiller içeren resim oluşturun
2. HSV renk uzayına dönüştürün
3. Sadece kırmızı renkteki nesneleri filtreleyin
4. Sadece mavi renkteki nesneleri filtreleyin  
5. Her filtreleme sonucunu gösterin
6. Bonus: İnteraktif renk seçici yapın (trackbar ile)

### Kodunuzu Yazın:
```python
def gorev_3_hsv_renk_filtreleme():
    print("\n🎯 GÖREV 3: HSV ile Renk Filtreleme")
    print("-" * 40)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: Kırmızı için iki aralık gerekli (0-10 ve 170-180)
    # HSV aralıkları: H(0-179), S(0-255), V(0-255)
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- HSV aralıkları: H(0-179), S(0-255), V(0-255)
- Kırmızı renk iki aralıkta: (0-10) ve (170-179)
- `cv2.inRange(hsv, alt_sınır, üst_sınır)` ile maske
- `cv2.bitwise_and(resim, resim, mask=maske)` ile filtreleme
- Renk aralıkları:
  ```python
  # Kırmızı
  alt_kırmızı1 = np.array([0, 50, 50])
  ust_kırmızı1 = np.array([10, 255, 255])
  alt_kırmızı2 = np.array([170, 50, 50])
  ust_kırmızı2 = np.array([180, 255, 255])
  
  # Mavi
  alt_mavi = np.array([100, 50, 50])
  ust_mavi = np.array([130, 255, 255])
  ```

---

## 🎯 GÖREV 4: Piksel Seviyesi Manipülasyon

Piksel düzeyinde işlemler ve ROI (Region of Interest) kullanımı.

### Yapılacaklar:
1. 300x400 boyutunda beyaz bir resim oluşturun
2. Sol yarısının parlaklığını %50 azaltın
3. Sağ yarısının parlaklığını %50 artırın
4. Merkeze 50x50 boyutunda kırmızı bir kare çizin
5. Resmin 4 köşesine farklı renkler ekleyin
6. ROI (Region of Interest) kullanarak merkez bölgeyi kopyalayın

### Kodunuzu Yazın:
```python
def gorev_4_piksel_manipulasyonu():
    print("\n🎯 GÖREV 4: Piksel Manipülasyonu")
    print("-" * 35)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: resim[y1:y2, x1:x2] ile bölge seçimi
    # cv2.rectangle(), cv2.circle() ile şekil çizimi
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- ROI seçimi: `resim[y1:y2, x1:x2]`
- Parlaklık azaltma: `resim * 0.5`
- Parlaklık artırma: `np.clip(resim * 1.5, 0, 255)`
- Kare çizimi: `cv2.rectangle(resim, (x1,y1), (x2,y2), renk, -1)`
- Köşelere renk ekleme: indeksleme ile
- ROI kopyalama: `roi = resim[y1:y2, x1:x2].copy()`

---

## 🎯 GÖREV 5: Histogram Analizi ve Düzeltme

Histogram hesaplama, görselleştirme ve eşitleme işlemleri.

### Yapılacaklar:
1. Düşük kontrastlı bir resim oluşturun veya yükleyin
2. Orijinal resmin histogramını çizin
3. Histogram eşitleme uygulayın
4. Eşitlenmiş resmin histogramını çizin
5. Orijinal ve düzeltilmiş resimleri karşılaştırın
6. BGR kanallarının ayrı histogramlarını gösterin

### Kodunuzu Yazın:
```python
def gorev_5_histogram_analizi():
    print("\n🎯 GÖREV 5: Histogram Analizi")
    print("-" * 30)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.calcHist([image], [0], None, [256], [0,256])
    # plt.plot() ile histogram çizimi
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- Histogram hesaplama: `cv2.calcHist([resim], [kanal], None, [256], [0,256])`
- Gri resim için: `cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)`
- Histogram eşitleme: `cv2.equalizeHist(gri_resim)`
- Renkli histogram eşitleme: her kanalı ayrı ayrı
- Histogram çizimi: `plt.plot(histogram)`
- BGR kanalları için:
  ```python
  renkler = ['b', 'g', 'r']
  for i, renk in enumerate(renkler):
      hist = cv2.calcHist([resim], [i], None, [256], [0,256])
      plt.plot(hist, color=renk)
  ```

---

## 🎯 GÖREV 6: Resim Matematiği ve Birleştirme

Matematiksel işlemler ve resim birleştirme operasyonları.

### Yapılacaklar:
1. İki farklı resim oluşturun veya yükleyin
2. Resimleri aynı boyuta getirin
3. Şu işlemleri uygulayın ve sonuçları gösterin:
   - Toplama (cv2.add)
   - Çıkarma (cv2.subtract)
   - Harmanlanma (cv2.addWeighted)
   - Bitwise AND, OR, XOR
4. Her işlemin sonucunu açıklayın

### Kodunuzu Yazın:
```python
def gorev_6_resim_matematigi():
    print("\n🎯 GÖREV 6: Resim Matematiği")
    print("-" * 30)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.add() vs numpy + farkını test edin
    # cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- OpenCV fonksiyonları taşmayı önler (overflow protection)
- `cv2.add(resim1, resim2)` vs `resim1 + resim2`
- `cv2.subtract(resim1, resim2)` 
- `cv2.addWeighted(resim1, alpha, resim2, beta, gamma)`
- Bitwise işlemler:
  - `cv2.bitwise_and(resim1, resim2)`
  - `cv2.bitwise_or(resim1, resim2)`
  - `cv2.bitwise_xor(resim1, resim2)`

### İşlem Açıklamaları:
- **Add**: Pixel değerleri toplanır, taşma önlenir
- **Subtract**: Pixel değerleri çıkarılır
- **AddWeighted**: Ağırlıklı toplama (alpha blending)
- **Bitwise AND**: Maske işlemleri için kullanılır
- **Bitwise OR**: Resim birleştirme
- **Bitwise XOR**: Fark belirleme

---

## 🎨 BONUS GÖREV: Mini Fotoğraf Editörü

İnteraktif bir fotoğraf editörü oluşturun - yaratıcılığınıza kalmış!

### Özellikler:
1. Resim yükleme fonksiyonu
2. Trackbar'larla parlaklık/kontrast ayarlama
3. Farklı renk uzaylarına geçiş
4. Basit filtreler (blur, sharpen)
5. Sonucu kaydetme

### Kodunuzu Yazın:
```python
def bonus_gorev_mini_fotograf_editoru():
    print("\n🎨 BONUS GÖREV: Mini Fotoğraf Editörü")
    print("-" * 45)
    print("Yaratıcılığınızı konuşturun!")
    
    # TODO: Buraya yaratıcı kodunuzu yazın
    # Örnek: Trackbar'larla interaktif editör
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**Öneriler:**
- `cv2.createTrackbar()` ile interaktif kontroller
- `cv2.getTrackbarPos()` ile değer okuma
- Callback fonksiyonları ile real-time güncelleme
- Farklı filtreler: blur, sharpen, edge detection
- Kaydetme ve yükleme fonksiyonları

---

## 🔨 Yardımcı Fonksiyonlar

### Test Resimleri Oluşturucu

```python
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
```

---

## 🖥️ Ana Program

```python
def main():
    """Ana program - tüm görevleri çalıştırır"""
    print("🎯 OpenCV Alıştırma 2: Resim İşlemleri ve Renk Uzayları")
    print("=" * 65)
    print("Bu alıştırmada resim işleme ve renk uzaylarını öğreneceksiniz.\n")
    
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
        bonus_cevap = input("\nBonus görevi yapmak ister misiniz? (e/h): ")
        if bonus_cevap.lower() == 'e':
            bonus_gorev_mini_fotograf_editoru()
        
        print("\n🎉 Tebrikler! Alıştırma 2'yi tamamladınız!")
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
```

---

## ✅ Kontrol Listesi

Alıştırmayı tamamladıktan sonra kontrol edin:

- [ ] Resim okuma/yazma ve format karşılaştırması yapabiliyorum
- [ ] Farklı renk uzayları arasında dönüşüm yapabiliyorum
- [ ] HSV ile renk filtreleme uygulayabiliyorum
- [ ] Piksel seviyesi manipülasyon yapabiliyorum
- [ ] Histogram analizi ve eşitleme uygulayabiliyorum
- [ ] Resim matematiksel işlemleri kullanabiliyorum
- [ ] (Bonus) İnteraktif fotoğraf editörü oluşturdum

## 💡 İpuçları

### Genel İpuçları
- Her görevde `pass` satırını silip kendi kodunuzu yazın
- Test resimleri otomatik oluşturulur
- Matplotlib kullanarak görselleştirme yapın
- HSV renk aralıklarına dikkat edin

### Teknik İpuçları
- OpenCV BGR, matplotlib RGB kullanır
- HSV aralıkları: H(0-179), S(0-255), V(0-255)
- Kırmızı renk iki aralıkta bulunur
- cv2 matematik fonksiyonları taşmayı önler
- ROI işlemleri için indeksleme kullanın

### Hata Durumları
- Dosya bulunamama: test resimlerini oluşturun
- Boyut uyumsuzlukları: `cv2.resize()` kullanın
- Renk uzayı hataları: doğru conversion flag'i kullanın
- Histogram hataları: gri resim gerekebilir

## 🎯 Çözüm

Takıldığınızda çözüm dosyasına bakabilirsiniz:
```bash
python cozumler/cozum-2.py
```

## 🚀 Sonraki Adım

Bu alıştırmayı tamamladıktan sonra **Alıştırma 3: Mini Proje - Akıllı Fotoğraf Düzenleyici** adımına geçebilirsiniz.

---

*Bu alıştırma Eren Terzi tarafından hazırlanmıştır.*