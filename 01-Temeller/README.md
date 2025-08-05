# 🔰 OpenCV Temelleri

Bu bölüm OpenCV'nin temellerini öğrenmek isteyenler için hazırlanmıştır. Sıfırdan başlayarak OpenCV'nin ne olduğunu, nasıl kurulacağını ve temel kavramları öğreneceksiniz.

## 📚 Bu Bölümde Öğrenecekleriniz

- ✅ OpenCV nedir ve neden kullanılır?
- ✅ OpenCV kurulumu (Python, C++, Java)
- ✅ İlk OpenCV programınızı yazma
- ✅ Temel veri yapıları (Mat, ndarray)
- ✅ Resim okuma, gösterme ve kaydetme
- ✅ Renk uzayları ve dönüşümler
- ✅ Temel görüntü özellikleri

## 📖 İçindekiler

### 1. [Giriş ve Kurulum](01-giris-ve-kurulum.md)
- OpenCV nedir?
- Tarihçesi ve kullanım alanları
- Python için kurulum
- Geliştirme ortamı kurulumu

### 2. [İlk OpenCV Programı](02-ilk-program.py)
- Basit resim gösterme
- Kütüphane import etme
- Hata kontrolü

### 3. [Temel Veri Yapıları](03-temel-veri-yapilari.md)
- NumPy ndarray
- Mat sınıfı (C++)
- Piksel erişimi
- Veri tipleri

### 4. [Resim İşlemleri](04-resim-islemleri.py)
- Resim okuma ve yazma
- Farklı formatlar (JPG, PNG, BMP)
- Resim boyutları ve özellikler

### 5. [Renk Uzayları](05-renk-uzaylari.py)
- RGB, BGR, HSV, LAB
- Renk uzayı dönüşümleri
- Gri tonlama dönüşümü

### 6. [Pratik Alıştırmalar](06-alistirmalar/)
- Temel işlemler pratiği
- Mini projeler
- Çözüm örnekleri

## 🎯 Öğrenme Hedefleri

Bu bölümü tamamladıktan sonra:

- [x] OpenCV'yi başarıyla kurabileceksiniz
- [x] Basit resim okuma/yazma işlemleri yapabileceksiniz  
- [x] Farklı renk uzayları arasında dönüşüm yapabileceksiniz
- [x] Temel veri yapılarını anlayacaksınız
- [x] Bir sonraki bölüme geçmeye hazır olacaksınız

## 💻 Gereksinimler

```bash
Python 3.7+
pip install opencv-python
pip install numpy
pip install matplotlib
```

## ⚡ Hızlı Başlangıç

1. **Kurulum kontrolü:**
```python
import cv2
print(cv2.__version__)
```

2. **İlk programınız:**
```python
import cv2
import numpy as np

# Siyah bir resim oluştur
img = np.zeros((300, 300, 3), dtype=np.uint8)

# Resmi göster
cv2.imshow('İlk OpenCV Programım', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 📁 Dosya Yapısı

```
01-Temeller/
├── README.md                    # Bu dosya
├── 01-giris-ve-kurulum.md      # OpenCV tanıtım ve kurulum
├── 02-ilk-program.py           # İlk OpenCV programı
├── 03-temel-veri-yapilari.md   # Veri yapıları açıklaması
├── 04-resim-islemleri.py       # Resim okuma/yazma
├── 05-renk-uzaylari.py         # Renk uzayı dönüşümleri
├── 06-alistirmalar/            # Pratik alıştırmalar
│   ├── alistirma-1.py
│   ├── alistirma-2.py
│   ├── cozumler/
│   └── README.md
└── examples/                    # Örnek resimler
    ├── sample1.jpg
    ├── sample2.png
    └── README.md
```

## 🚀 Sonraki Adım

Bu bölümü tamamladıktan sonra [`02-Resim-Isleme/`](../02-Resim-Isleme/) bölümüne geçebilirsiniz.

---

**💡 İpucu:** Her kodu çalıştırırken nelerin değiştiğini gözlemleyin ve kendi deneyimlerinizi yapın!

**⚠️ Dikkat:** Kodları çalıştırmadan önce gerekli resim dosyalarının doğru konumda olduğundan emin olun.