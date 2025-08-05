# 🖼️ OpenCV Resim İşleme

Bu bölümde görüntü işleme temellerini öğreneceksiniz. Daha karmaşık işlemler, filtreler ve transformasyonlar ile resimlerinizi manipüle etmeyi keşfedeceksiniz.

## 📚 Bu Bölümde Öğrenecekleriniz

- ✅ Geometrik transformasyonlar (döndürme, ölçekleme, kayma)
- ✅ Resim filtreleme (bulanıklaştırma, keskinleştirme)
- ✅ Morfolojik işlemler (erozyon, dilatasyon)
- ✅ Histogram işlemleri ve eşitleme
- ✅ Kontrast ve parlaklık ayarlama
- ✅ Gürültü azaltma teknikleri
- ✅ Kenar algılama algoritmaları

## 📖 İçindekiler

### 1. [Geometrik Transformasyonlar](01-geometrik-transformasyonlar.py)
- Döndürme (rotation)
- Ölçekleme (scaling)
- Öteleme (translation)
- Perspektif düzeltme
- Affine transformasyonlar

### 2. [Resim Filtreleme](02-resim-filtreleme.py)
- Gaussian blur (Gaussian bulanıklaştırma)
- Motion blur (hareket bulanıklaştırması)
- Median filter (medyan filtresi)
- Bilateral filter (bilateral filtre)
- Custom kernel'lar

### 3. [Morfolojik İşlemler](03-morfolojik-islemler.py)
- Erozyon (erosion)
- Dilatasyon (dilation) 
- Açma (opening)
- Kapama (closing)
- Gradient ve Top-hat

### 4. [Histogram İşlemleri](04-histogram-islemleri.py)
- Histogram hesaplama
- Histogram eşitleme
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Histogram karşılaştırma

### 5. [Kontrast ve Parlaklık](05-kontrast-parlaklik.py)
- Lineer transformasyonlar
- Gamma düzeltme
- Logaritmik transformasyonlar
- Otomatik kontrast ayarlama

### 6. [Gürültü Azaltma](06-gurultu-azaltma.py)
- Gaussian gürültü
- Salt & pepper gürültü
- Non-local means denoising
- Wiener filtresi

### 7. [Kenar Algılama](07-kenar-algilama.py)
- Sobel operatörü
- Canny kenar algılama
- Laplacian of Gaussian
- Scharr operatörü

### 8. [Pratik Alıştırmalar](08-alistirmalar/)
- Fotoğraf düzenleme araçları
- Resim iyileştirme
- Sanat efektleri
- Mini projeler

## 🎯 Öğrenme Hedefleri

Bu bölümü tamamladıktan sonra:

- [x] Resimleri geometrik olarak transform edebileceksiniz
- [x] Çeşitli filtreleme teknikleri uygulayabileceksiniz
- [x] Histogram analizi ve düzeltme yapabileceksiniz
- [x] Gürültülü resimleri temizleyebileceksiniz
- [x] Kenar algılama algoritmaları kullanabileceksiniz
- [x] Profesyonel görüntü işleme teknikleri uygulayabileceksiniz

## 💻 Gereksinimler

```bash
# Temel kütüphaneler
pip install opencv-python
pip install numpy
pip install matplotlib

# Ek kütüphaneler (opsiyonel)
pip install scikit-image  # Karşılaştırma için
pip install pillow        # PIL desteği
```

## ⚡ Hızlı Başlangıç

İlk transform örneği:

```python
import cv2
import numpy as np

# Resim yükle
resim = cv2.imread('ornek.jpg')

# 45 derece döndür
merkez = (resim.shape[1]//2, resim.shape[0]//2)
rotasyon_matrisi = cv2.getRotationMatrix2D(merkez, 45, 1.0)
dondurulmus = cv2.warpAffine(resim, rotasyon_matrisi, 
                             (resim.shape[1], resim.shape[0]))

# Sonucu göster
cv2.imshow('Döndürülmüş', dondurulmus)
cv2.waitKey(0)
```

## 📁 Dosya Yapısı

```
02-Resim-Isleme/
├── README.md                           # Bu dosya
├── 01-geometrik-transformasyonlar.py   # Geometrik dönüşümler
├── 02-resim-filtreleme.py             # Filtreler ve bulanıklaştırma
├── 03-morfolojik-islemler.py          # Morfolojik operasyonlar
├── 04-histogram-islemleri.py          # Histogram analizi
├── 05-kontrast-parlaklik.py           # Kontrast ve parlaklık
├── 06-gurultu-azaltma.py              # Gürültü temizleme
├── 07-kenar-algilama.py               # Kenar tespit algoritmaları
├── 08-alistirmalar/                   # Pratik alıştırmalar
│   ├── README.md
│   ├── alistirma-1.py
│   ├── alistirma-2.py
│   ├── cozumler/
│   └── test-resimleri/
└── examples/                          # Örnek resimler
    ├── sample-noisy.jpg
    ├── sample-blur.jpg
    └── sample-low-contrast.jpg
```

## 🔍 Temel Kavramlar

### 🔄 **Geometrik Transformasyonlar**
Resimlerin geometrik özelliklerini değiştiren işlemler:
- **Affine**: Döndürme, ölçekleme, kayma
- **Perspektif**: 3D görünüm düzeltme
- **Elastic**: Esnek deformasyonlar

### 🎛️ **Filtreler**
Piksel komşuluklarına dayalı işlemler:
- **Low-pass**: Gürültü azaltma, bulanıklaştırma
- **High-pass**: Keskinleştirme, kenar vurgulama
- **Band-pass**: Belirli frekans aralıkları

### 📊 **Histogram**
Piksel değerlerinin dağılımı:
- **Eşitleme**: Kontrast iyileştirme
- **Germe**: Dinamik aralık artırma
- **Karşılaştırma**: Benzerlik ölçümü

## 💡 İpuçları

### 🎯 **Performans İpuçları**
- Büyük resimler için önce boyut küçültün
- Batch işlemlerde vectorized operasyonlar kullanın
- GPU desteği için OpenCV'nin GPU modüllerini kullanın

### 🔧 **Kalite İpuçları**
- Her zaman orijinal resmin kopyasını tutun
- İşlem sırasını optimize edin (irreversible işlemler sonda)
- Parametre değerlerini test resimlerle ayarlayın

### 🐛 **Debug İpuçları**
- Ara sonuçları görselleştirin
- Histogram kontrolü yapın
- Veri tipi taşmalarına dikkat edin

## 🎨 Pratik Projeler

Bu bölümde yapabileceğiniz projeler:

1. **📸 Fotoğraf Editörü**
   - Parlaklık/kontrast ayarlama
   - Filtre efektleri
   - Histogram düzeltme

2. **🔧 Belge Düzeltici**
   - Perspektif düzeltme
   - Kontrast artırma
   - Gürültü temizleme

3. **🎭 Sanat Efektleri**
   - Sketch efekti
   - Oil painting efekti
   - Vintage filtreleri

## 🚀 Sonraki Adım

Bu bölümü tamamladıktan sonra [`03-Video-Isleme/`](../03-Video-Isleme/) bölümüne geçebilirsiniz.

---

**📚 Öğrenme Tavsiyesi**: Her tekniği önce teorik olarak anlayın, sonra pratik yapın. Farklı parametrelerle deneyim yaparak etkileri gözlemleyin!

**⚠️ Dikkat**: Resim işleme CPU yoğun işlemler olabilir. Büyük resimlerle çalışırken sabırlı olun!