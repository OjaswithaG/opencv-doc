# 07-Projeler

Bu bölüm, OpenCV ve makine öğrenmesi tekniklerini kullanarak gerçek dünya projelerini içerir. Her proje, öğrenilen teknikleri pratik uygulamalarda kullanmayı amaçlar.

## 🎯 Proje Kategorileri

### 1. Görüntü İşleme Projeleri
- **01-yuz-tanima-sistemi.py**: Yüz tanıma ve duygu analizi
- **02-plaka-tanima.py**: Plaka tanıma sistemi
- **03-dokuman-tarama.py**: Belge tarama ve OCR
- **04-renk-tespit.py**: Renk tabanlı nesne tespiti

### 2. Video İşleme Projeleri
- **05-hareket-algilama.py**: Hareket algılama ve güvenlik sistemi
- **06-nesne-takibi.py**: Nesne takip sistemi
- **07-video-stabilizasyon.py**: Video stabilizasyon
- **08-arka-plan-cikarma.py**: Arka plan çıkarma ve değiştirme

### 3. Makine Öğrenmesi Projeleri
- **09-gesture-kontrol.py**: El hareketi ile kontrol sistemi
- **10-nesne-siniflandirma.py**: Gerçek zamanlı nesne sınıflandırma
- **11-anomali-tespit.py**: Anomali tespit sistemi
- **12-optik-karakter-tanima.py**: OCR ve metin tanıma

### 4. İleri Seviye Projeler
- **13-3d-pose-estimation.py**: 3D poz tahmini
- **14-segmentation.py**: Görüntü segmentasyonu
- **15-style-transfer.py**: Stil transferi
- **16-face-swap.py**: Yüz değiştirme

## 🛠️ Proje Gereksinimleri

### Temel Kütüphaneler
```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### İleri Seviye Kütüphaneler
```bash
pip install tensorflow keras mediapipe dlib
pip install pytesseract pillow imutils
```

### Opsiyonel Kütüphaneler
```bash
pip install opencv-contrib-python
pip install face-recognition
pip install easyocr
```

## 📚 Proje Detayları

### Görüntü İşleme Projeleri

#### 01. Yüz Tanıma Sistemi
- **Amaç**: Yüz tanıma ve duygu analizi
- **Teknikler**: Haar Cascade, LBPH, CNN
- **Uygulama**: Güvenlik sistemleri, kişiselleştirme

#### 02. Plaka Tanıma
- **Amaç**: Araç plakası tanıma
- **Teknikler**: Kenar tespit, OCR, karakter tanıma
- **Uygulama**: Trafik yönetimi, otopark sistemleri

#### 03. Belge Tarama
- **Amaç**: Belge tarama ve metin çıkarma
- **Teknikler**: Perspektif dönüşüm, OCR
- **Uygulama**: Dijital arşivleme, form işleme

#### 04. Renk Tespit
- **Amaç**: Renk tabanlı nesne tespiti
- **Teknikler**: HSV renk uzayı, kontur tespit
- **Uygulama**: Endüstriyel kalite kontrol

### Video İşleme Projeleri

#### 05. Hareket Algılama
- **Amaç**: Hareket algılama ve güvenlik
- **Teknikler**: Background subtraction, frame differencing
- **Uygulama**: Güvenlik kameraları, akıllı ev sistemleri

#### 06. Nesne Takibi
- **Amaç**: Nesne takip sistemi
- **Teknikler**: Kalman filter, mean shift, camshift
- **Uygulama**: Robotik, otonom araçlar

#### 07. Video Stabilizasyon
- **Amaç**: Video stabilizasyonu
- **Teknikler**: Motion estimation, frame alignment
- **Uygulama**: Video düzenleme, drone kameraları

#### 08. Arka Plan Çıkarma
- **Amaç**: Arka plan çıkarma ve değiştirme
- **Teknikler**: GrabCut, deep learning
- **Uygulama**: Video konferans, sinema

### Makine Öğrenmesi Projeleri

#### 09. Gesture Kontrol
- **Amaç**: El hareketi ile kontrol
- **Teknikler**: Hand tracking, gesture recognition
- **Uygulama**: Oyun kontrolü, sunum kontrolü

#### 10. Nesne Sınıflandırma
- **Amaç**: Gerçek zamanlı nesne sınıflandırma
- **Teknikler**: CNN, transfer learning
- **Uygulama**: Akıllı kameralar, robotik

#### 11. Anomali Tespit
- **Amaç**: Anomali tespit sistemi
- **Teknikler**: Autoencoder, one-class SVM
- **Uygulama**: Endüstriyel kalite kontrol, güvenlik

#### 12. Optik Karakter Tanıma
- **Amaç**: OCR ve metin tanıma
- **Teknikler**: Tesseract, deep learning OCR
- **Uygulama**: Belge işleme, dijitalleştirme

### İleri Seviye Projeler

#### 13. 3D Poz Tahmini
- **Amaç**: 3D poz tahmini
- **Teknikler**: MediaPipe, pose estimation
- **Uygulama**: Spor analizi, rehabilitasyon

#### 14. Görüntü Segmentasyonu
- **Amaç**: Görüntü segmentasyonu
- **Teknikler**: U-Net, DeepLab, Mask R-CNN
- **Uygulama**: Tıbbi görüntüleme, otonom araçlar

#### 15. Stil Transferi
- **Amaç**: Görüntü stil transferi
- **Teknikler**: Neural style transfer, GAN
- **Uygulama**: Sanat, eğlence

#### 16. Yüz Değiştirme
- **Amaç**: Yüz değiştirme uygulaması
- **Teknikler**: Face detection, face swap
- **Uygulama**: Eğlence, sinema

## 🎓 Öğrenme Yolu

### Başlangıç Seviyesi
1. **01-yuz-tanima-sistemi.py**: Temel görüntü işleme
2. **05-hareket-algilama.py**: Video işleme temelleri
3. **09-gesture-kontrol.py**: Makine öğrenmesi uygulaması

### Orta Seviye
1. **02-plaka-tanima.py**: Gelişmiş görüntü işleme
2. **06-nesne-takibi.py**: Nesne takip algoritmaları
3. **10-nesne-siniflandirma.py**: CNN uygulamaları

### İleri Seviye
1. **13-3d-pose-estimation.py**: 3D görüntü işleme
2. **14-segmentation.py**: Derin öğrenme segmentasyonu
3. **15-style-transfer.py**: GAN ve stil transferi

## 🔧 Proje Geliştirme Süreci

### 1. Planlama
- Proje amacını belirleme
- Gereksinimleri analiz etme
- Teknoloji seçimi
- Zaman planlaması

### 2. Geliştirme
- Veri toplama ve hazırlama
- Algoritma geliştirme
- Model eğitimi
- Test ve optimizasyon

### 3. Deployment
- Model optimizasyonu
- Performans testleri
- Kullanıcı arayüzü
- Dokümantasyon

## 📊 Performans Metrikleri

### Görüntü İşleme
- Doğruluk oranı
- İşlem hızı (FPS)
- Bellek kullanımı
- CPU/GPU optimizasyonu

### Video İşleme
- Gerçek zamanlı performans
- Latency
- Frame rate
- Kalite kaybı

### Makine Öğrenmesi
- Model doğruluğu
- Eğitim süresi
- Inference süresi
- Model boyutu

## 🎨 Kullanıcı Arayüzü

### GUI Tasarımı
- Tkinter ile basit arayüzler
- PyQt ile gelişmiş arayüzler
- Web tabanlı arayüzler
- Mobil uygulama entegrasyonu

### Görselleştirme
- Gerçek zamanlı görselleştirme
- Grafik ve çizelgeler
- 3D görselleştirme
- İnteraktif kontroller

## 🚀 Deployment ve Dağıtım

### Model Optimizasyonu
- Model quantization
- TensorRT optimizasyonu
- ONNX dönüşümü
- Edge deployment

### Cloud Deployment
- AWS SageMaker
- Google Cloud AI
- Azure Machine Learning
- Docker containerization

### Edge Computing
- Raspberry Pi deployment
- Mobile deployment
- IoT cihazları
- Embedded systems

## 📈 Proje Geliştirme İpuçları

### Kod Organizasyonu
- Modüler yapı
- Sınıf tabanlı tasarım
- Configuration management
- Error handling

### Performans Optimizasyonu
- GPU acceleration
- Multi-threading
- Memory management
- Algorithm optimization

### Test ve Debugging
- Unit testing
- Integration testing
- Performance profiling
- Error logging

## 🤝 Katkıda Bulunma

### Proje Geliştirme
1. Yeni proje önerisi
2. Mevcut projeleri geliştirme
3. Bug fix ve optimizasyon
4. Dokümantasyon iyileştirme

### Topluluk Katılımı
- GitHub pull requests
- Issue reporting
- Code review
- Mentoring

## 📚 Ek Kaynaklar

### Dokümantasyon
- [OpenCV Python Tutorials](https://docs.opencv.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Araştırma Makaleleri
- Computer Vision Papers
- Deep Learning Applications
- Real-time Systems
- Edge Computing

### Online Kurslar
- Coursera: Computer Vision
- Udacity: Computer Vision Nanodegree
- edX: Computer Vision
- Fast.ai: Practical Deep Learning

---

**Not**: Bu projeler, OpenCV ve makine öğrenmesi konularında pratik deneyim kazanmanızı sağlar. Her proje, gerçek dünya uygulamalarında kullanılabilecek teknikler içerir. 