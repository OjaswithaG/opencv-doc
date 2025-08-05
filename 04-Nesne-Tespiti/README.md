# 🎯 Nesne Tespiti - OpenCV Object Detection

Bu bölüm OpenCV ile nesne tespiti (object detection) tekniklerini kapsar. Klasik algoritmalardan modern derin öğrenme yaklaşımlarına kadar geniş bir yelpazede nesne tespit etme yöntemlerini öğreneceksiniz.

## 📚 Modül İçeriği

### 🏛️ Klasik Yaklaşımlar
- **Haar Cascade Classifiers** - Yüz, göz, gülümseme tespiti
- **HOG + SVM** - İnsan (pedestrian) tespiti
- **Template Matching** - Şablon eşleştirme
- **Contour-Based Detection** - Şekil tabanlı tespit

### 🔍 Özellik Tabanlı Yaklaşımlar
- **SIFT (Scale-Invariant Feature Transform)** - Ölçek değişmez özellikler
- **ORB (Oriented FAST and Rotated BRIEF)** - Hızlı özellik tespiti
- **Feature Matching** - Özellik eşleştirme
- **Homography** - Perspektif dönüşümleri

### 🎨 Renk ve Şekil Tabanlı
- **Color-based Detection** - Renk aralığı ile tespit
- **HSV Color Space** - Renk uzayı dönüşümleri
- **Shape Detection** - Geometrik şekil tespiti
- **Morphological Operations** - Morfolojik işlemler

### 🧠 Modern Yaklaşımlar
- **DNN (Deep Neural Networks)** - Derin sinir ağları
- **Pre-trained Models** - Önceden eğitilmiş modeller
- **YOLO Integration** - Real-time object detection
- **MobileNet** - Hafif mobil modeller

### 📱 Pratik Uygulamalar
- **QR Code & Barcode Reading** - QR kod ve barkod okuma
- **Face Recognition** - Yüz tanıma
- **Document Detection** - Belge tespiti
- **Logo Detection** - Logo tanıma

## 🎯 Öğrenme Hedefleri

Bu bölümü tamamladığınızda şunları yapabileceksiniz:

### Temel Seviye
- ✅ Haar cascade ile yüz tespiti
- ✅ Template matching ile nesne bulma
- ✅ Renk tabanlı nesne ayırma
- ✅ Temel şekil (daire, kare) tespiti
- ✅ QR kod okuma

### Orta Seviye
- 🎯 HOG ile insan tespiti
- 🎯 SIFT/ORB ile özellik eşleştirme
- 🎯 Contour analizi ile karmaşık şekiller
- 🎯 Multi-scale detection
- 🎯 Performance optimization

### İleri Seviye
- 🚀 DNN modelleri kullanma
- 🚀 Custom classifier eğitme
- 🚀 Real-time detection systems
- 🚀 Multi-object tracking integration
- 🚀 Production deployment

## 📋 Modül Yapısı

```
04-Nesne-Tespiti/
├── README.md                          # Bu dosya
├── 01-klasik-nesne-tespiti.py        # Haar, HOG, Template matching
├── 02-ozellik-tabanli-tespit.py      # SIFT, ORB, feature matching
├── 03-yuz-tespit.py                  # Face, eye, smile detection
├── 04-sekil-tespit.py                # Geometric shape detection
├── 05-renk-tabanli-tespit.py         # Color-based detection
├── 06-dnn-nesne-tespiti.py           # Modern DNN approaches
├── 07-qr-barkod-okuma.py             # QR codes and barcodes
├── 08-alistirmalar/                   # Pratik alıştırmalar
│   ├── README.md                     # Alıştırma rehberi
│   ├── alistirma-1.py               # Yüz tanıma sistemi
│   ├── alistirma-2.py               # Logo tespit sistemi
│   ├── alistirma-3.py               # Plaka okuma sistemi
│   └── cozumler/                    # Örnek çözümler
└── models/                           # Pre-trained modeller
    ├── README.md                     # Model açıklamaları
    ├── haarcascade_frontalface.xml  # Yüz tespiti
    ├── haarcascade_eye.xml          # Göz tespiti
    └── download_models.py           # Model indirme scripti
```

## 🔄 Klasik vs Modern Yaklaşımlar

### 🏛️ Klasik Yöntemler

| Yöntem | Hız | Doğruluk | Kullanım | Avantajlar |
|--------|-----|----------|----------|-----------|
| **Haar Cascades** | ⚡⚡⚡ | ⭐⭐ | Yüz/Göz | Hızlı, hafif |
| **HOG + SVM** | ⚡⚡ | ⭐⭐⭐ | İnsan | Güvenilir |
| **Template Match** | ⚡⚡⚡ | ⭐⭐ | Şablon | Basit |
| **SIFT/ORB** | ⚡ | ⭐⭐⭐⭐ | Özellik | Robust |

### 🧠 Modern Yöntemler

| Model | Hız | Doğruluk | GPU | Kullanım |
|-------|-----|----------|-----|----------|
| **YOLO** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Önerilen | Real-time |
| **SSD** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Opsiyonel | Mobil |
| **R-CNN** | ⚡ | ⭐⭐⭐⭐⭐ | Gerekli | Doğruluk |
| **MobileNet** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Hayır | Edge |

## 🛠️ Kurulum ve Gereksinimler

### Temel Gereksinimler
```bash
pip install opencv-python
pip install numpy
pip install matplotlib
```

### İleri Seviye (DNN için)
```bash
pip install opencv-contrib-python  # DNN desteği
pip install tensorflow             # TensorFlow modelleri için
pip install torch torchvision     # PyTorch modelleri için
```

### Model Dosyaları
```bash
# Haar cascade modelleri (OpenCV ile gelir)
# QR kod decoder için
pip install pyzbar

# DNN modelleri (otomatik indirilecek)
python models/download_models.py
```

## 🎮 Demo ve Test

Her modül interaktif demo içerir:

```python
# Temel kullanım
python 01-klasik-nesne-tespiti.py

# Webcam ile test
python 03-yuz-tespit.py

# Batch processing
python 06-dnn-nesne-tespiti.py --input videos/ --output results/
```

## 📊 Performance Karşılaştırması

### Hız Testi (640x480, CPU)
- **Haar Cascade**: ~30 FPS
- **HOG + SVM**: ~15 FPS  
- **SIFT Matching**: ~5 FPS
- **DNN (CPU)**: ~2 FPS
- **DNN (GPU)**: ~25 FPS

### Doğruluk Testi (mAP@0.5)
- **Haar Cascade**: 85% (yüzler için)
- **HOG + SVM**: 78% (insanlar için)
- **YOLO v5**: 95% (genel objeler)
- **Custom Training**: %90+ (spesifik domain)

## 🎯 Kullanım Alanları

### 🏢 Endüstriyel Uygulamalar
- **Kalite Kontrol**: Ürün defekt tespiti
- **Güvenlik**: Kişi/araç takibi
- **Otomasyon**: Robot görüş sistemleri
- **Envanter**: Stok sayımı

### 📱 Tüketici Uygulamaları
- **Mobil**: AR filtreleri
- **Fotoğraf**: Otomatik etiketleme
- **Güvenlik**: Akıllı kameralar
- **Otomotiv**: ADAS sistemleri

### 🏥 Özel Alanlar
- **Tıp**: Medikal görüntü analizi
- **Tarım**: Mahsul analizi
- **Çevre**: Vahşi yaşam takibi
- **Eğitim**: İnteraktif öğrenme

## 🚀 Performans Optimizasyonu

### CPU Optimizasyonu
```python
# Multi-threading
cv2.setNumThreads(4)

# Image pyramid kullanma
detector.setScaleFactor(1.1)

# ROI (Region of Interest)
roi = frame[y:y+h, x:x+w]
```

### Memory Optimizasyonu
```python
# Frame resize
frame = cv2.resize(frame, (320, 240))

# Grayscale conversion
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Buffer reuse
buffer = np.empty((240, 320), dtype=np.uint8)
```

### GPU Kullanımı
```python
# OpenCV DNN GPU backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

## 🔧 Debugging ve Troubleshooting

### Yaygın Problemler
1. **Model bulunamadı**: `models/` klasörünü kontrol edin
2. **Yavaş performans**: Frame boyutunu küçültün
3. **False positives**: Threshold değerlerini ayarlayın
4. **GPU hatası**: CUDA kurulumunu kontrol edin

### Debug Teknikleri
```python
# Bounding box visualize
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Confidence score
cv2.putText(frame, f"{confidence:.2f}", (x, y-10), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Processing time
start_time = time.time()
# ... detection code ...
print(f"Detection time: {(time.time() - start_time)*1000:.1f}ms")
```

## 📚 Ek Kaynaklar

### Dokümantasyon
- [OpenCV Object Detection Tutorial](https://docs.opencv.org/4.x/d2/d64/tutorial_table_of_content_objdetect.html)
- [Deep Learning with OpenCV](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)

### Model Repositories
- [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo)
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [TensorFlow Model Garden](https://github.com/tensorflow/models)

### Research Papers
- Viola-Jones (2001): Robust Real-time Object Detection
- Dalal-Triggs (2005): HOG for Human Detection
- YOLO (2016): You Only Look Once
- SSD (2016): Single Shot MultiBox Detector

## 🎓 Başarı Kriterleri

### Temel Seviye ✅
- [ ] Haar cascade ile yüz tespit edebilme
- [ ] Template matching ile nesne bulma
- [ ] Renk filtreleme ile nesne ayırma
- [ ] QR kod okuyabilme

### Orta Seviye 🎯
- [ ] Custom Haar cascade eğitme
- [ ] SIFT ile özellik eşleştirme
- [ ] Multi-scale detection
- [ ] Performance optimization

### İleri Seviye 🚀
- [ ] DNN modeli entegrasyonu
- [ ] Real-time detection sistemi
- [ ] Custom dataset ile training
- [ ] Production deployment

---

**Nesne Tespiti ile güçlü görüş sistemleri geliştirin! 🎯**

*Hazırlayan: Eren Terzi - 2024*