# 🧱 OpenCV Temel Veri Yapıları

OpenCV'de görüntüler ve veriler belirli yapılarda saklanır. Bu bölümde bu temel yapıları öğreneceksiniz.

## 📊 NumPy ndarray

OpenCV Python'da **NumPy ndarray** yapısını kullanır. Bu, N-boyutlu dizi anlamına gelir.

### 🔍 Temel Özellikler

```python
import cv2
import numpy as np

# Basit bir resim oluştur
resim = np.zeros((300, 400, 3), dtype=np.uint8)

print("Resim şekli:", resim.shape)        # (yükseklik, genişlik, kanal)
print("Veri tipi:", resim.dtype)          # uint8
print("Boyut sayısı:", resim.ndim)        # 3
print("Toplam piksel:", resim.size)       # 300 * 400 * 3 = 360000
```

### 📐 Resim Boyutları

```python
yukseklik, genislik, kanal_sayisi = resim.shape

print(f"Yükseklik: {yukseklik} piksel")
print(f"Genişlik: {genislik} piksel") 
print(f"Kanal sayısı: {kanal_sayisi}")
```

## 🎨 Renk Kanalları

### 3 Kanallı (BGR) Resimler
```python
# BGR (Blue, Green, Red) formatı
bgr_resim = np.zeros((100, 100, 3), dtype=np.uint8)

# Mavi kanalı
bgr_resim[:, :, 0] = 255  # Mavi
# Yeşil kanalı  
bgr_resim[:, :, 1] = 128  # Yeşil
# Kırmızı kanalı
bgr_resim[:, :, 2] = 64   # Kırmızı
```

### Tek Kanallı (Gri Tonlama) Resimler
```python
# Gri tonlama resim
gri_resim = np.zeros((100, 100), dtype=np.uint8)
print("Gri resim şekli:", gri_resim.shape)  # (100, 100)
```

## 🔢 Veri Tipleri

OpenCV'de yaygın kullanılan veri tipleri:

| Veri Tipi | Açıklama | Değer Aralığı |
|-----------|----------|---------------|
| `uint8` | 8-bit işaretsiz tam sayı | 0-255 |
| `int8` | 8-bit işaretli tam sayı | -128 ile 127 |
| `uint16` | 16-bit işaretsiz tam sayı | 0-65535 |
| `int16` | 16-bit işaretli tam sayı | -32768 ile 32767 |
| `float32` | 32-bit kayan nokta | Herhangi bir değer |
| `float64` | 64-bit kayan nokta | Herhangi bir değer |

### 💡 Veri Tipi Seçimi

```python
# Standart resimler için
resim_8bit = np.zeros((100, 100, 3), dtype=np.uint8)

# Yüksek hassasiyetli işlemler için
resim_float = np.zeros((100, 100, 3), dtype=np.float32)

# Veri tipi dönüşümü
float_resim = resim_8bit.astype(np.float32) / 255.0
```

## 🎯 Piksel Erişimi

### Tek Piksel İşlemleri

```python
# Piksel değeri okuma
piksel_degeri = resim[50, 100]  # (y, x) koordinatında
print("Piksel değeri:", piksel_degeri)

# BGR piksel değeri okuma
b, g, r = resim[50, 100]
print(f"Mavi: {b}, Yeşil: {g}, Kırmızı: {r}")

# Piksel değeri yazma
resim[50, 100] = [255, 0, 0]  # Mavi piksel
```

### Bölge İşlemleri

```python
# Dikdörtgen bölge seçimi
bolge = resim[50:150, 100:200]  # [y1:y2, x1:x2]

# Bölgeye değer atama
resim[50:150, 100:200] = [0, 255, 0]  # Yeşil dikdörtgen

# Bölgeyi kopyalama
kopya_bolge = resim[0:100, 0:100].copy()
```

## 🔄 Dizi İşlemleri

### Resim Oluşturma Yöntemleri

```python
# Siyah resim (sıfırlarla dolu)
siyah = np.zeros((300, 400, 3), dtype=np.uint8)

# Beyaz resim (255'lerle dolu)
beyaz = np.ones((300, 400, 3), dtype=np.uint8) * 255

# Rastgele resim
rastgele = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

# Belirli bir değerle dolu resim
gri = np.full((300, 400, 3), 128, dtype=np.uint8)
```

### Matematiksel İşlemler

```python
# Aritmetik işlemler
resim1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
resim2 = np.ones((100, 100, 3), dtype=np.uint8) * 50

toplam = cv2.add(resim1, resim2)      # Güvenli toplama
fark = cv2.subtract(resim1, resim2)   # Güvenli çıkarma
carpim = cv2.multiply(resim1, resim2) # Çarpma

# NumPy işlemleri (taşma kontrolü yok)
toplam_numpy = resim1 + resim2
```

## 🖼️ Kanal İşlemleri

### Kanalları Ayırma

```python
# BGR kanallarını ayır
b, g, r = cv2.split(resim)

print("Mavi kanal şekli:", b.shape)   # (yükseklik, genişlik)
print("Yeşil kanal şekli:", g.shape) # (yükseklik, genişlik)  
print("Kırmızı kanal şekli:", r.shape) # (yükseklik, genişlik)
```

### Kanalları Birleştirme

```python
# Kanalları tekrar birleştir
birlesik_resim = cv2.merge([b, g, r])

# Tek kanallı resmi 3 kanala dönüştür
tek_kanal = np.zeros((100, 100), dtype=np.uint8)
uc_kanal = cv2.merge([tek_kanal, tek_kanal, tek_kanal])
```

## 📏 Boyut İşlemleri

### Resim Boyutunu Değiştirme

```python
# Boyut değiştirme
yeni_boyut = cv2.resize(resim, (800, 600))  # (genişlik, yükseklik)

# Orantılı boyutlandırma
scale_percent = 50  # %50 küçültme
genislik = int(resim.shape[1] * scale_percent / 100)
yukseklik = int(resim.shape[0] * scale_percent / 100)
kucuk_resim = cv2.resize(resim, (genislik, yukseklik))
```

### Resim Kırpma

```python
# Resmi kırp (y1:y2, x1:x2)
kirpik_resim = resim[50:250, 100:300]
```

## 🔍 Pratik Örnekler

### Örnek 1: Renk Kanallarını Görselleştirme

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Renkli resim oluştur
resim = cv2.imread('örnek_resim.jpg')

# Kanalları ayır
b, g, r = cv2.split(resim)

# Matplotlib ile göster (RGB formatında)
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
plt.title('Orijinal')

plt.subplot(1, 4, 2)
plt.imshow(r, cmap='Reds')
plt.title('Kırmızı Kanal')

plt.subplot(1, 4, 3)
plt.imshow(g, cmap='Greens')
plt.title('Yeşil Kanal')

plt.subplot(1, 4, 4)
plt.imshow(b, cmap='Blues')
plt.title('Mavi Kanal')

plt.show()
```

### Örnek 2: Piksel Değerlerini Analiz Etme

```python
# Resim istatistikleri
print("Minimum değer:", np.min(resim))
print("Maksimum değer:", np.max(resim))
print("Ortalama değer:", np.mean(resim))
print("Standart sapma:", np.std(resim))

# Kanal bazında istatistikler
for i, renk in enumerate(['Mavi', 'Yeşil', 'Kırmızı']):
    kanal = resim[:, :, i]
    print(f"{renk} kanal ortalaması: {np.mean(kanal):.2f}")
```

## 📝 Önemli Notlar

### ⚠️ Dikkat Edilmesi Gerekenler

1. **Koordinat Sistemi**: OpenCV'de (y, x) sıralaması kullanılır
2. **Renk Sırası**: OpenCV BGR, Matplotlib RGB kullanır
3. **Veri Tipi**: uint8 (0-255) en yaygın kullanılan tip
4. **Bellek Yönetimi**: Büyük resimlerle çalışırken bellek kullanımına dikkat

### 💡 İpuçları

- `resim.copy()` kullanarak güvenli kopyalar oluşturun
- Matematiksel işlemlerde OpenCV fonksiyonlarını tercih edin
- Veri tipi dönüşümlerinde hassasiyet kaybına dikkat edin
- Büyük resimlerle çalışırken bellek sınırlarını göz önünde bulundurun

## 🚀 Sonraki Adım

Temel veri yapılarını öğrendiğiniz için artık [Resim İşlemleri](04-resim-islemleri.py) konusuna geçebilirsiniz!

---

**🎯 Bu bölümde öğrendikleriniz:**
- NumPy ndarray yapısı
- Renk kanalları ve veri tipleri  
- Piksel erişimi ve manipülasyonu
- Dizi işlemleri ve boyut kontrolü