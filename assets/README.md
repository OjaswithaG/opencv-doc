# 📁 Assets - Örnek Dosyalar

Bu klasör, OpenCV dokümantasyonunda kullanılan örnek resim ve video dosyalarını içerir.

## 📂 İçerik

### 🖼️ Resim Dosyaları

#### **Test Resimleri**
- `sample-1.jpg` - Genel amaçlı test resmi (landscape)
- `sample-2.jpg` - Portre test resmi
- `sample-3.png` - Geometrik şekiller (test için)
- `sample-4.bmp` - Yüksek kaliteli test resmi

#### **Özel Amaçlı Resimler**
- `noisy-image.jpg` - Gurultulu resim (gürültü azaltma testleri için)
- `low-contrast.jpg` - Düşük kontrastlı resim (histogram testleri için)
- `blurry-image.jpg` - Bulanık resim (keskinleştirme testleri için)
- `geometric-shapes.png` - Geometrik şekiller (contour testleri için)

#### **Renk Test Resimleri**
- `color-wheel.png` - Renk tekerleği (renk uzayı testleri için)
- `gradient.jpg` - Renk gradyanı
- `rgb-test.png` - RGB test deseni

#### **Nesne Tespiti için**
- `faces.jpg` - Yüz tespiti testleri için
- `cars.jpg` - Araç tespiti testleri için
- `coins.jpg` - Çember/nesne tespiti için
- `documents.jpg` - Belge işleme testleri için

### 🎥 Video Dosyaları

- `sample-video.mp4` - Genel test videosu (30 fps, 10 saniye)
- `traffic.mp4` - Trafik videosu (hareket tespiti için)
- `walking-person.mp4` - Yürüyen insan (nesne takibi için)

### 📊 Kalibrasyon Dosyaları

- `chessboard.png` - Satranç tahtası (kamera kalibrasyonu için)
- `calibration-*.jpg` - Çeşitli kalibrasyon görüntüleri

## 🚀 Kullanım

### Python'da Dosya Yolu

```python
import cv2
import os

# Assets klasörü yolu
assets_path = os.path.join(os.path.dirname(__file__), '..', 'assets')

# Resim yükleme
resim_yolu = os.path.join(assets_path, 'sample-1.jpg')
resim = cv2.imread(resim_yolu)

# Video yükleme  
video_yolu = os.path.join(assets_path, 'sample-video.mp4')
video = cv2.VideoCapture(video_yolu)
```

### Pathlib ile (Önerilen)

```python
from pathlib import Path
import cv2

# Assets klasörü
assets_dir = Path(__file__).parent.parent / 'assets'

# Resim yükleme
resim = cv2.imread(str(assets_dir / 'sample-1.jpg'))

# Video yükleme
video = cv2.VideoCapture(str(assets_dir / 'sample-video.mp4'))
```

## 📊 Dosya Boyutları ve Formatlar

| Dosya | Format | Boyut | Çözünürlük | Açıklama |
|-------|--------|-------|------------|----------|
| sample-1.jpg | JPEG | ~200KB | 640x480 | Genel test |
| sample-2.jpg | JPEG | ~150KB | 480x640 | Portre |
| sample-3.png | PNG | ~50KB | 400x400 | Geometrik |
| noisy-image.jpg | JPEG | ~180KB | 512x512 | Gurultulu |
| sample-video.mp4 | MP4 | ~2MB | 640x480 | Test videosu |

## 🎨 Resim Oluşturma Kodları

Kendi test resimlerinizi oluşturmak için:

```python
import cv2
import numpy as np

def create_test_images():
    """Test resimleri oluşturur"""
    
    # 1. Geometrik şekiller
    shapes = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(shapes, (50, 50), (150, 150), (0, 255, 0), -1)
    cv2.circle(shapes, (300, 100), 50, (255, 0, 0), -1)
    cv2.line(shapes, (50, 300), (350, 350), (0, 0, 255), 5)
    cv2.imwrite('assets/geometric-shapes.png', shapes)
    
    # 2. Renk gradyanı
    gradient = np.zeros((200, 400, 3), dtype=np.uint8)
    for i in range(400):
        gradient[:, i] = [i*255//400, (400-i)*255//400, 128]
    cv2.imwrite('assets/gradient.jpg', gradient)
    
    # 3. Gürültülü resim
    clean = np.ones((300, 300, 3), dtype=np.uint8) * 128
    noise = np.random.normal(0, 25, clean.shape)
    noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)
    cv2.imwrite('assets/noisy-image.jpg', noisy)

if __name__ == "__main__":
    create_test_images()
```

## 🔒 Telif Hakları

- Tüm örnek dosyalar eğitim amaçlı olarak oluşturulmuştur
- Ticari kullanım için kendi resimlerinizi kullanın
- Bazı resimler açık kaynak veritabanlarından alınmıştır

## 📝 Dosya Ekleme Rehberi

Yeni dosya eklerken:

1. **Uygun boyut**: Çok büyük dosyalar eklemeyin (max 5MB)
2. **Yaygın formatlar**: JPG, PNG, MP4 tercih edin
3. **Açıklayıcı isimler**: Dosya adı kullanım amacını belirtsin
4. **README güncelleme**: Bu dosyayı güncelleyin

## 🗂️ Klasör Yapısı

```
assets/
├── README.md              # Bu dosya
├── images/               # Resim dosyaları
│   ├── samples/         # Genel örnek resimler
│   ├── test/           # Test resimleri
│   ├── noisy/          # Gürültülü resimler
│   └── calibration/    # Kalibrasyon resimleri
├── videos/              # Video dosyaları
│   ├── samples/        # Örnek videolar
│   └── test/          # Test videoları
└── data/               # Diğer veri dosyaları
    ├── models/         # ML modelleri
    └── configs/        # Konfigürasyon dosyaları
```

---

**💡 İpucu**: Kendi projelerinizde bu asset'leri kullanabilirsiniz, ancak gerçek uygulamalarda telif hakları konusunda dikkatli olun!

**📌 Not**: Büyük dosyalar Git LFS kullanılarak yönetilmektedir.