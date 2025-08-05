# 🖼️ Test Resimleri

Bu klasör, OpenCV alıştırmaları için özel olarak hazırlanmış test resimlerini içerir.

## 📂 İçerik

### 🎨 **Temel Test Resimleri**
- `gradyan.jpg` - Renk gradyanı (renk uzayı testleri için)
- `normal.jpg` - Normal kaliteli resim (genel testler için)
- `renkli-sekiller.png` - Geometrik şekiller (renk filtreleme için)

### 📊 **Kalite Test Resimleri**  
- `dusuk-kontrast.jpg` - Düşük kontrastlı resim (histogram testleri için)
- `yuksek-kontrast.jpg` - Yüksek kontrastlı resim (kontrast testleri için)
- `asiri-parlak.jpg` - Aşırı parlak resim (parlaklık testleri için)
- `karanlik.jpg` - Çok karanlık resim (parlaklık testleri için)

### 🔧 **İşleme Test Resimleri**
- `gurultulu.jpg` - Gürültülü resim (gürültü azaltma için)
- `bulanik.jpg` - Bulanık resim (keskinleştirme için)

### 🧪 **Özel Test Desenleri**
- `satranc-tahtasi.png` - Satranç tahtası (kamera kalibrasyonu için)
- `renk-testi.png` - RGB renk şeritleri (renk testleri için)
- `cizgi-testi.png` - Çizgi desenleri (kenar algılama için)

### 🎯 **Uygulama Test Resimleri**
- `yuz-ornegi.jpg` - Basit yüz çizimi (yüz tespiti testleri için)
- `belge-ornegi.jpg` - Belge simülasyonu (belge işleme için)
- `para-tespiti.jpg` - Para örnekleri (nesne tespiti için)

## 🚀 Kullanım

### Python'da Test Resmi Yükleme
```python
import cv2

# Test resmi yükle
resim = cv2.imread('test-resimleri/gradyan.jpg')

# Kontrol et
if resim is not None:
    print("✅ Resim başarıyla yüklendi!")
    cv2.imshow('Test Resmi', resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### Alıştırmalarda Kullanım
Bu resimler alıştırmalarda otomatik olarak kullanılır:
- Alıştırma 1: `normal.jpg`, `renkli-sekiller.png`
- Alıştırma 2: `gradyan.jpg`, `dusuk-kontrast.jpg`, `gurultulu.jpg`
- Alıştırma 3: Tüm resimler (batch işlem için)

## 🛠️ Resim Özellikleri

| Resim | Boyut | Format | Özellik |
|-------|-------|---------|---------|
| gradyan.jpg | 300x400 | JPEG | Renk geçişleri |
| dusuk-kontrast.jpg | 250x350 | JPEG | Dar değer aralığı (80-120) |
| renkli-sekiller.png | 400x500 | PNG | 6 farklı renk |
| gurultulu.jpg | 300x400 | JPEG | Gaussian gürültü (σ=25) |
| bulanik.jpg | 300x400 | JPEG | Gaussian blur (15x15) |

## 📝 Notlar

- Tüm resimler programatik olarak oluşturulmuştur
- Telif hakkı sorunu yoktur
- Eğitim amaçlı kullanım için optimize edilmiştir
- İhtiyaç halinde `resim_olusturucu.py` ile yeniden oluşturulabilir

---

**💡 İpucu:** Kendi test resimlerinizi de ekleyebilirsiniz!
