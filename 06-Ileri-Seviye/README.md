# 🤖 Makine Öğrenmesi - OpenCV ML

Bu bölüm OpenCV'de makine öğrenmesi algoritmalarının implementasyonunu ve pratik uygulamalarını kapsar. Temel kavramlardan başlayarak, gerçek dünya problemlerini çözen kapsamlı projeler geliştireceğiz.

## 📚 İçerik Yapısı

### 🎯 Temel Modüller

#### 1. [Temel Kavramlar](01-temel-kavramlar.py)
- **OpenCV ML modülü** tanıtımı
- **Veri hazırlama** ve normalizasyon
- **Model eğitimi** ve değerlendirme
- **Cross-validation** implementasyonu
- **Performans metrikleri** hesaplama

**Öğreneceğiniz Teknikler:**
- `cv2.ml` modülü kullanımı
- Veri preprocessing teknikleri
- Train/Test split stratejileri
- Model validation yöntemleri

#### 2. [k-NN Sınıflandırma](02-knn-siniflandirma.py)  
- **k-Nearest Neighbors** algoritması
- **El yazısı rakam tanıma** sistemi
- **k değeri optimizasyonu**
- **İnteraktif karakter tanıma** demo
- **Karar sınır görselleştirme**

**Öğreneceğiniz Teknikler:**
- Lazy learning algoritması implementasyonu
- Distance metrics karşılaştırması
- Hyperparameter tuning
- Real-time prediction sistemi

#### 3. [SVM Nesne Tanıma](03-svm-nesne-tanima.py)
- **Support Vector Machine** algoritması
- **Kernel fonksiyonları** (Linear, RBF, Polynomial)
- **HOG + SVM** ile insan tespiti
- **Hiperparametre optimizasyonu**
- **Grid search** implementasyonu

**Öğreneceğiniz Teknikler:**
- SVM kernel seçimi ve optimizasyonu
- Feature scaling önemini anlama
- Support vector analizi
- Real-time object detection

#### 4. [ANN Karakter Tanıma](04-ann-karakter-tanima.py)
- **Artificial Neural Networks** temel prensipleri
- **Multi-layer perceptron** yapısı
- **Backpropagation** algoritması
- **MNIST** veri seti ile çalışma
- **Network architecture** tasarımı

**Öğreneceğiniz Teknikler:**
- Neural network mimarisi tasarımı
- Activation function seçimi
- Learning rate optimization
- Overfitting prevention techniques

#### 5. [Karar Ağaçları ve Random Forest](05-karar-agaclari.py)
- **Decision Trees** algoritması
- **Random Forest** ensemble yöntemi
- **Feature importance** analizi
- **Overfitting analizi** ve önleme
- **Görüntü tabanlı sınıflandırma**

**Öğreneceğiniz Teknikler:**
- Tree-based model optimization
- Ensemble learning benefits
- Feature selection techniques
- Bias-variance trade-off analysis

### 🎯 Pratik Projeler

#### 6. [Alıştırmalar](06-alistirmalar/)
Kapsamlı projeler ve bunların çözümleri:

##### 📸 [Alıştırma 1: Çok Sınıflı Görüntü Sınıflandırma](06-alistirmalar/alistirma-1.md)
**Proje Açıklaması:** Webcam'den toplanan gerçek nesne görüntüleri ile çok sınıflı sınıflandırma sistemi

**Temel Özellikler:**
- 5 farklı nesne sınıfı tanıma
- 32 boyutlu kapsamlı özellik çıkarma
- Tüm ML algoritmalarını karşılaştırma
- Real-time prediction sistemi
- Cross-validation analizi

**Teknik Detaylar:**
- **Veri Toplama:** İnteraktif webcam interface
- **Feature Extraction:** Histogram, istatistiksel, kenar, LBP özellikleri
- **Model Comparison:** k-NN, SVM, ANN, Decision Tree, Random Forest
- **Performance Analysis:** Accuracy, precision, recall, F1-score
- **Visualization:** Confusion matrices, feature importance

**📁 Dosya Yapısı:**
```
06-alistirmalar/
├── alistirma-1.md          # Detaylı proje açıklaması
├── alistirma-1.py          # Template kod (öğrenci için)
└── cozumler/
    └── cozum-1.py          # Tam çözüm implementasyonu
```

## 🛠️ Teknik Gereksinimler

### Temel Kütüphaneler
```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
```

### Opsiyonel (Gelişmiş özellikler için)
```bash
pip install scikit-image    # LBP ve texture analysis
pip install seaborn         # Gelişmiş görselleştirme
pip install pandas          # Veri analizi
```

## 🚀 Hızlı Başlangıç

### 1. Temel Kavramları Öğrenin
```bash
python 01-temel-kavramlar.py
```
- ML algoritma seçimi
- Veri preprocessing
- Model evaluation

### 2. Algoritma Karşılaştırması
```bash
python 02-knn-siniflandirma.py
python 03-svm-nesne-tanima.py
python 04-ann-karakter-tanima.py
python 05-karar-agaclari.py
```

### 3. Kapsamlı Proje Geliştirin
```bash
cd 06-alistirmalar
python cozumler/cozum-1.py
```

## 📊 Öğrenme Hedefleri

### Temel Seviye
- [ ] OpenCV ML modülünü anlama
- [ ] Temel ML algoritmalarını uygulama
- [ ] Model performansını değerlendirme
- [ ] Veri preprocessing teknikleri

### Orta Seviye  
- [ ] Hiperparametre optimizasyonu
- [ ] Cross-validation implementasyonu
- [ ] Feature engineering teknikleri
- [ ] Model comparison ve selection

### İleri Seviye
- [ ] Real-time prediction sistemi
- [ ] Ensemble learning yöntemleri
- [ ] Feature importance analysis
- [ ] Comprehensive project development

## 🎯 Algoritma Karşılaştırması

| Algoritma | Hız | Doğruluk | Yorumlanabilirlik | Overfitting Risk |
|-----------|-----|----------|-------------------|-------------------|
| **k-NN** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **SVM** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **ANN** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Decision Tree** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Random Forest** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### Algoritma Seçim Rehberi

#### k-NN Kullanın Eğer:
- Basit, hızlı prototype geliştiriyorsanız
- Veri seti küçük-orta boyutluysa
- Non-parametric yaklaşım gerekiyorsa
- Lazy learning avantajı istiyorsanız

#### SVM Kullanın Eğer:
- Yüksek boyutlu verilerle çalışıyorsanız
- Robust sınıflandırma gerekiyorsa
- Kernel trick kullanmak istiyorsanız
- Memory efficient çözüm arıyorsanız

#### ANN Kullanın Eğer:
- Kompleks pattern recognition gerekiyorsa
- Büyük veri setiniz varsa
- Non-linear relationships önemliyse
- High accuracy kritikse

#### Decision Tree Kullanın Eğer:
- Model yorumlanabilirliği önemliyse
- Kategoric verilerle çalışıyorsanız
- Hızlı eğitim gerekiyorsa
- Feature importance analizi istiyorsanız

#### Random Forest Kullanın Eğer:
- Overfitting problem yaşıyorsanız
- Robust performance istiyorsanız
- Feature importance gerekiyorsa
- Ensemble benefits arıyorsanız

## 💡 Pratik İpuçları

### Veri Hazırlama
- **Normalizasyon zorunlu:** Özellikle SVM ve ANN için
- **Outlier detection:** Z-score veya IQR yöntemi
- **Feature scaling:** Min-max veya standardization
- **Data augmentation:** Veri artırma teknikleri

### Model Seçimi
- **Baseline model:** Her zaman basit k-NN ile başlayın
- **Complexity progression:** Basit → Kompleks
- **Cross-validation:** Tüm modeller için zorunlu
- **Ensemble combination:** Voting veya stacking

### Performance Tuning
- **Grid search:** Systematic hyperparameter optimization
- **Random search:** Büyük parameter space için
- **Learning curves:** Overfitting detection
- **Validation curves:** Optimal parameter finding

### Debugging Checklist
- [ ] Data leakage kontrolü
- [ ] Feature correlation analysis
- [ ] Class imbalance check
- [ ] Model convergence verification
- [ ] Prediction consistency test

## 🔗 İlgili Kaynaklar

### OpenCV Dokümantasyonu
- [OpenCV ML Tutorial](https://docs.opencv.org/master/d1/d69/tutorial_py_template_matching.html)
- [cv2.ml Module Reference](https://docs.opencv.org/master/dd/ded/group__ml.html)

### Makine Öğrenmesi Temelleri
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)

### Pratik Uygulamalar
- [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- [Computer Vision Recipes](https://github.com/microsoft/computervision-recipes)

## 🎓 Değerlendirme Kriterleri

### Temel Gereksinimler (70%)
- [ ] Tüm algoritmaları çalıştırabilme
- [ ] Temel performans metriklerini anlama
- [ ] Cross-validation uygulayabilme
- [ ] Model karşılaştırması yapabilme

### İleri Özellikler (20%)
- [ ] Hiperparametre optimizasyonu
- [ ] Feature engineering
- [ ] Ensemble methods
- [ ] Real-time implementation

### Proje Çalışması (10%)
- [ ] Original problem çözme
- [ ] Comprehensive analysis
- [ ] Code quality ve dokümantasyon
- [ ] Creative implementation

## 🚨 Yaygın Hatalar ve Çözümleri

### Veri Problemleri
❌ **Normalizasyon yapmamak**  
✅ **Çözüm:** Her zaman StandardScaler veya MinMaxScaler kullan

❌ **Data leakage**  
✅ **Çözüm:** Train/test split'ten sonra preprocessing yap

❌ **Imbalanced dataset**  
✅ **Çözüm:** Stratified sampling ve class weighting

### Model Problemleri
❌ **Overfitting**  
✅ **Çözüm:** Cross-validation, regularization, ensemble

❌ **Underfitting**  
✅ **Çözüm:** Feature engineering, model complexity artırma

❌ **Wrong hyperparameters**  
✅ **Çözüm:** Grid search, validation curves

### Implementation Hataları
❌ **Yanlış data type (float32/int32)**  
✅ **Çözüm:** OpenCV için explicit type casting

❌ **Memory issues**  
✅ **Çözüm:** Batch processing, data streaming

❌ **Performance bottlenecks**  
✅ **Çözüm:** Profiling, algorithm optimization

---

## 🎯 Sonraki Adımlar

Bu bölümü tamamladıktan sonra:

1. **06-Ileri-Seviye** bölümüne geçin
2. **Deep Learning** konseptlerini öğrenin  
3. **07-Projeler** ile gerçek uygulamalar geliştirin
4. **End-to-end** ML pipeline'ları oluşturun

**Good luck with your machine learning journey! 🚀🤖**