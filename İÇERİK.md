# 🔥 OpenCV ile Görüntü İşleme - Sıfırdan İleri Seviyeye

## 📖 Bu Dokümantasyon Hakkında

Bu kapsamlı kaynak, **OpenCV** (Open Source Computer Vision Library) ile görüntü işleme öğrenmek isteyen herkes için hazırlanmıştır. **Bilgisayarlı görü** (Computer Vision) ve **görüntü işleme** (Image Processing) alanında teorik temelleri ve pratik uygulamaları birleştiren sistematik bir yaklaşımla ilerler.

### 🧠 OpenCV Nedir?

**OpenCV** (Open Source Computer Vision Library), Intel tarafından geliştirilen ve şu anda tüm dünyaca kullanılan açık kaynaklı bir bilgisayarlı görü kütüphanesidir. **3000'den fazla algoritma** içerir ve şu alanlarda kullanılır:

- **Görüntü İşleme**: Filtreleme, transformasyon, iyileştirme
- **Bilgisayarlı Görü**: Nesne tanıma, yüz tanıma, hareket algılama  
- **Makine Öğrenmesi**: Özellik çıkarma, sınıflandırma, kümeleme
- **Derin Öğrenme**: CNN modelleri, transfer learning
- **Video Analizi**: Hareket takibi, arka plan çıkarma
- **3D Görüş**: Stereo vision, kamera kalibrasyonu

### 🌟 Neden OpenCV?

- ✅ **Performans**: C/C++ ile yazılmış, optimize edilmiş algoritmalar
- ✅ **Çok Dilli**: Python, C++, Java, C#, JavaScript desteği
- ✅ **Çapraz Platform**: Windows, Linux, macOS, Android, iOS
- ✅ **Geniş Topluluk**: Milyonlarca geliştirici, sürekli güncellemeler
- ✅ **Endüstri Standardı**: Google, Microsoft, Intel gibi şirketlerde kullanılır
- ✅ **Ücretsiz**: Apache 2.0 lisansı ile tamamen açık kaynak

## 🎯 Kimler İçin?

### 👨‍💻 **Yazılım Geliştiriciler**
- **Backend Developers**: API'lerde görüntü işleme servisleri
- **Mobile Developers**: AR/VR uygulamaları, kamera filtreleri
- **Web Developers**: Browser'da gerçek zamanlı görüntü işleme
- **Game Developers**: Oyunlarda görüntü tanıma, hareket kontrolü

### 🎓 **Öğrenciler ve Akademisyenler**
- **Bilgisayar Mühendisliği**: Görüntü işleme, yapay zeka dersleri
- **Elektrik-Elektronik Mühendisliği**: Sinyal işleme, kontrol sistemleri
- **Endüstri Mühendisliği**: Kalite kontrol, otomasyon sistemleri
- **Tıp Öğrencileri**: Medikal görüntü analizi, teşhis sistemleri

### 🔬 **Araştırmacılar**
- **Akademik Araştırma**: Bilgisayarlı görü, robotik, AI
- **R&D Mühendisleri**: Yeni algoritma geliştirme, prototyping
- **Patent Geliştirme**: Görüntü işleme tabanlı buluşlar
- **Bilimsel Analiz**: Mikroskopi, uydu görüntüleri, teleskop verileri

### 🤖 **AI/ML Uzmanları**
- **Data Scientists**: Görüntü verisi ön işleme, feature extraction
- **ML Engineers**: Model deployment, real-time inference
- **Deep Learning Researchers**: CNN, GAN, Vision Transformers
- **Computer Vision Engineers**: Production-ready vision systems

## 📚 İçindekiler ve Öğrenme Yolu

Bu dokümantasyon **6 ana seviye** halinde organize edilmiştir. Her seviye bir öncekinin üzerine kurulur ve **160+ örnek**, **50+ algoritma** ve **20+ proje** içerir.

---

### 🔰 **TEMEL SEVİYE (Level 1-2)**

#### [`01-Temeller/`](01-Temeller/) - Temel Kavramlar ve Kurulum 🛠️

**📋 Teorik Temeller:**
- **Bilgisayarlı Görü Nedir?** - Dijital görüntülerin matematiksel temsili
- **Piksel ve Koordinat Sistemleri** - (x,y) koordinatları, RGB/BGR renk modelleri
- **Görüntü Dosya Formatları** - JPEG, PNG, TIFF, BMP karşılaştırması
- **Bellek Yönetimi** - Mat objesi, pointer'lar, memory leak'lerden kaçınma

**🔧 Pratik Uygulamalar:**
- **Çoklu Platform Kurulum** - Windows, Linux, macOS, Anaconda, Docker
- **IDE Konfigürasyonu** - VS Code, PyCharm, Jupyter Lab optimizasyonu
- **Temel I/O İşlemleri** - `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()`
- **Veri Yapısı Çevirileri** - OpenCV Mat ↔ NumPy ndarray ↔ PIL Image

**🎨 Renk Uzayları ve Dönüşümler:**
- **RGB (Red-Green-Blue)** - Ekran tabanlı, additive color model
- **BGR (Blue-Green-Red)** - OpenCV'nin varsayılan formatı
- **HSV (Hue-Saturation-Value)** - Renk seçimi ve segmentation için ideal
- **LAB (Lightness-A-B)** - Perceptual uniform, renk mesafesi hesapları
- **YUV/YCrCb** - Video compression, broadcast standardları
- **Grayscale** - Tek kanal, hız optimizasyonu için

---

#### [`02-Resim-Isleme/`](02-Resim-Isleme/) - Görüntü İşleme Temelleri 🖼️

**📊 Matematiksel Temeller:**
- **Konvolüsyon Teorisi** - Kernel'lar, sliding window, padding
- **Fourier Transformu** - Frekans domain, filtreleme teorisi
- **Morfometri** - Set theory, struktural element'ler
- **Histogram Analizi** - Probability distribution, statistical moments

**🔄 Geometrik Transformasyonlar:**
- **Affine Transform** - Döndürme, ölçekleme, shearing, translation
  - *Denklem*: `[x', y'] = M * [x, y, 1]ᵀ` (3x3 matris)
- **Perspektif Transform** - 3D projeksiyon, homografi matrisi
  - *Uygulama*: Belge tarama, drone görüntüleri düzeltme
- **Elastic Deformation** - Non-linear, medical imaging
- **Polar Koordinat** - Dairesel simetri, radar görüntüleri

**🌫️ Filtreleme Teknikleri:**
- **Spatial Domain Filtering:**
  - *Gaussian Filter* - `G(x,y) = (1/2πσ²)e^(-(x²+y²)/2σ²)`
  - *Box Filter* - Uniform averaging, hızlı blur
  - *Median Filter* - Non-linear, salt&pepper gürültü
  - *Bilateral Filter* - Edge-preserving, `w(i,j) = exp(-(d²/2σd²))exp(-(r²/2σr²))`

- **Frequency Domain Filtering:**
  - *Low-pass* - Gürültü azaltma, yumuşatma
  - *High-pass* - Keskinleştirme, edge enhancement
  - *Band-pass* - Belirli frekans aralıkları
  - *Notch Filters* - Periyodik gürültü temizleme

**⚗️ Morfolojik İşlemler:**
- **Binary Morphology:**
  - *Erosion* - `A ⊖ B = {z ∈ E | (B)z ⊆ A}`  
  - *Dilation* - `A ⊕ B = {z ∈ E | (B̂)z ∩ A ≠ ∅}`
  - *Opening* - `A ○ B = (A ⊖ B) ⊕ B` (noise removal)
  - *Closing* - `A • B = (A ⊕ B) ⊖ B` (gap filling)

- **Advanced Operations:**
  - *Morphological Gradient* - Edge detection
  - *Top-hat/Black-hat* - Small object detection
  - *Skeletonization* - Shape representation
  - *Distance Transform* - Medial axis transform

**📈 Histogram İşlemleri:**
- **Histogram Equalization** - Kontrast iyileştirme
  - *Formül*: `T(rk) = (L-1) * Σ(k,j=0) P(rj)`
- **CLAHE** - Contrast Limited Adaptive HE, local enhancement
- **Histogram Matching** - İki görüntü arasında tonal transfer
- **Multi-dimensional Histograms** - RGB, HSV, LAB joint distribution

**✨ Kontrast ve Parlaklık:**
- **Point Operations:**
  - *Linear* - `s = αr + β` (alpha=contrast, beta=brightness)
  - *Gamma Correction* - `s = cr^γ` (monitor calibration)
  - *Logarithmic* - `s = c*log(1+r)` (dynamic range compression)

- **Adaptive Methods:**
  - *Auto-contrast* - Min-max stretching
  - *Histogram Stretching* - Percentile-based
  - *Tone Mapping* - HDR → LDR conversion

**🧹 Gürültü Azaltma:**
- **Gürültü Modelleri:**
  - *Gaussian* - Thermal noise, N(μ,σ²)
  - *Salt & Pepper* - Impulse noise, dead pixels
  - *Speckle* - Multiplicative, radar/ultrasound
  - *Poisson* - Shot noise, low light conditions

- **Denoising Algorithms:**
  - *Wiener Filter* - Optimal MMSE, frequency domain
  - *Non-local Means* - Patch-based similarity
  - *BM3D* - Block-matching 3D collaborative filtering
  - *Wavelet Denoising* - Multi-resolution analysis

**🔍 Kenar Algılama:**
- **Gradient-based Methods:**
  - *Sobel* - `Gx = [-1,0,1; -2,0,2; -1,0,1]`
  - *Prewitt* - Uniform weighting
  - *Scharr* - Optimized coefficients
  - *Roberts* - Minimal 2x2 cross-gradient

- **Second Derivative:**
  - *Laplacian* - `∇²f = ∂²f/∂x² + ∂²f/∂y²`
  - *Laplacian of Gaussian* - LoG, blob detection
  - *Zero-crossing* - Edge localization

- **Advanced Edge Detection:**
  - *Canny Algorithm* - Multi-stage, optimal edge detector
    1. Gaussian smoothing
    2. Gradient computation
    3. Non-maximum suppression
    4. Double thresholding
    5. Edge tracking by hysteresis

---

### 🚀 **ORTA SEVİYE (Level 3-4)**

#### [`03-Video-Isleme/`](03-Video-Isleme/) - Video İşleme ve Kamera 🎥

**⏯️ Video Teorisi:**
- **Temporal Processing** - Frame sequence analysis, motion vectors
- **Codec'ler** - H.264, H.265, VP9, compression algorithms
- **Frame Rate** - FPS, temporal aliasing, motion blur
- **Color Spaces in Video** - YUV420, broadcast standards

**📹 Gerçek Zamanlı İşleme:**
- **Camera Interface** - DirectShow, V4L2, GStreamer
- **Buffer Management** - Triple buffering, frame dropping
- **Threading** - Producer-consumer pattern, async processing
- **Performance Optimization** - GPU acceleration, parallel processing

**🏃 Hareket Algılama:**
- **Optical Flow:**
  - *Lucas-Kanade* - Sparse feature tracking
  - *Horn-Schunck* - Dense flow field
  - *Farneback* - Polynomial expansion method

- **Background Subtraction:**
  - *MOG (Mixture of Gaussians)* - Statistical background modeling
  - *MOG2* - Improved shadow detection
  - *KNN* - K-nearest neighbors classifier
  - *GMM* - Gaussian Mixture Models

#### [`04-Nesne-Tespiti/`](04-Nesne-Tespiti/) - Nesne Algılama ve Takip 🎯

**🔍 Feature Detection:**
- **Corner Detectors:**
  - *Harris Corner* - `R = det(M) - k*trace(M)²`
  - *Shi-Tomasi* - Good Features to Track
  - *FAST* (Features from Accelerated Segment Test)

- **Blob Detectors:**
  - *LoG* (Laplacian of Gaussian)
  - *DoG* (Difference of Gaussians)
  - *MSER* (Maximally Stable Extremal Regions)

**🎯 Feature Descriptors:**
- **Classical Descriptors:**
  - *SIFT* (Scale-Invariant Feature Transform) - 128D descriptor
  - *SURF* (Speeded Up Robust Features) - Haar wavelet responses
  - *ORB* (Oriented FAST and Rotated BRIEF) - Binary descriptor

**📐 Geometrik Analiz:**
- **Contour Analysis:**
  - *Contour Detection* - Suzuki-Abe algorithm
  - *Shape Descriptors* - Hu moments, Fourier descriptors
  - *Convex Hull* - Graham scan, gift wrapping

- **Template Matching:**
  - *Normalized Cross Correlation* - Translation invariant
  - *Multi-scale Matching* - Pyramid approach

**🎣 Object Tracking:**
- **Single Object Trackers:**
  - *KCF* (Kernelized Correlation Filters)
  - *CSRT* (Channel and Spatial Reliability Tracker)

---

### ⚡ **İLERİ SEVİYE (Level 5-6)**

#### [`05-Makine-Ogrenmesi/`](05-Makine-Ogrenmesi/) - ML ile Görüntü İşleme 🤖

**🧠 Machine Learning:**
- **Classification**: SVM, Random Forest, K-NN
- **Clustering**: K-Means, Mean Shift, DBSCAN
- **Feature Engineering**: HOG, LBP, GLCM
- **OCR**: Tesseract, CRNN, Attention models

#### [`06-Ileri-Seviye/`](06-Ileri-Seviye/) - Derin Öğrenme 🧬

**🎭 Deep Learning:**
- **DNN Module**: Model loading, inference
- **Face Processing**: Detection, recognition, landmarks
- **Camera Calibration**: Intrinsic, extrinsic parameters
- **Performance**: GPU acceleration, optimization

---

### 🎨 **PRATIK PROJELER**

#### [`07-Projeler/`](07-Projeler/) - Gerçek Dünya Uygulamaları 🏭

**📷 Endüstriyel Projeler:**
- **Fotoğraf Editörü** - Instagram-style filters, HDR
- **Plaka Tanıma** - ALPR systems, OCR
- **Yüz Tanıma** - Real-time recognition
- **Nesne Sayma** - Quality control, defect detection
- **Medikal Analiz** - X-Ray, MRI processing
- **AR/VR** - Augmented reality, gesture recognition

---

## 🛠️ Gereksinimler

### Yazılım Gereksinimleri
```bash
Python 3.7+
OpenCV 4.x
NumPy
Matplotlib
Jupyter Notebook (opsiyonel)
```

### Kurulum
```bash
pip install opencv-python
pip install opencv-contrib-python
pip install numpy matplotlib
pip install jupyter  # notebook'lar için
```

## 📁 Klasör Yapısı

```
opencv-doc/
├── 01-Temeller/          # Kurulum ve temel kavramlar
├── 02-Resim-Isleme/      # Görüntü işleme temelleri
├── 03-Video-Isleme/      # Video işleme ve kamera
├── 04-Nesne-Tespiti/     # Nesne algılama ve takip
├── 05-Makine-Ogrenmesi/  # ML ile görüntü işleme
├── 06-Ileri-Seviye/      # Derin öğrenme ve özel konular
├── 07-Projeler/          # Kapsamlı pratik projeler
├── assets/               # Örnek resim ve video dosyaları
├── utils/                # Yardımcı kodlar ve fonksiyonlar
└── README.md             # Bu dosya
```

## 🎓 Nasıl Öğrenilir?

### 1. **Sıralı İlerleme** 🔢
Bölümleri sırasıyla takip edin. Her bölüm bir öncekinin üzerine kurulur.

### 2. **Pratik Yapın** 💻
Her konu için verilen kod örneklerini çalıştırın ve üzerinde değişiklik yapın.

### 3. **Deneyimleyin** 🧪
Kendi resim ve videolarınızla örnekleri test edin.

### 4. **Proje Yapın** 🚀
Her seviyeyi tamamladıktan sonra o seviyedeki bilgilerle küçük projeler geliştirin.

## 💡 İpuçları

- 📝 **Not Alın**: Önemli kavramları not alıp kendi kelimelerinizle açıklayın
- 🔍 **Araştırın**: Anlamadığınız kavramları daha detaylı araştırın
- 👥 **Paylaşın**: Öğrendiklerinizi başkalarıyla paylaşın
- 🐛 **Hata Yapın**: Hataları öğrenmenin bir parçası olarak görün

## 🆘 Yardım

Herhangi bir sorunla karşılaştığınızda:
1. İlgili bölümün README dosyasını kontrol edin
2. Kod örneklerini dikkatlice inceleyin
3. Hata mesajlarını analiz edin
4. OpenCV resmi dokümantasyonuna başvurun

## 🌟 Katkıda Bulunun

Bu dokümantasyonu geliştirmek için:
- Hata bildirimleri
- Yeni örnek önerileri
- İyileştirme önerileri
- Yeni proje fikirleri

**Hoş geldiniz!** Görüntü işleme dünyasına adım attığınız için tebrikler! 🎉

---

## 🚀 Hemen Başlayın!

**İlk adım:** [`01-Temeller/`](01-Temeller/) klasörüne gidin ve OpenCV'yi kurmaya başlayın!

**Tavsiye:** Jupyter Notebook kullanarak interaktif öğrenme deneyimi yaşayın.

---

*Hazırlayan: Eren Terzi*