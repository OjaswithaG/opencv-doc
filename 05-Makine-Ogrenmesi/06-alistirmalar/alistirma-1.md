# 🎯 Alıştırma 1: Çok Sınıflı Görüntü Sınıflandırma

## 📋 Hedef
Kameradan toplanan farklı nesne görüntülerini kullanarak çok sınıflı bir sınıflandırma sistemi geliştireceksiniz. Bu projede, öğrenmiş olduğunuz tüm makine öğrenmesi algoritmalarını karşılaştıracak ve en iyi performansı veren modeli seçeceksiniz.

## 🎯 Öğrenme Hedefleri
- Gerçek zamanlı veri toplama implementasyonu
- Farklı ML algoritmalarının performans karşılaştırması
- Özellik mühendisliği (feature engineering) uygulaması
- Model değerlendirme ve seçim kriterleri
- Cross-validation ile model güvenilirliği testi

## 🔧 Gereksinimler

### Teknik Gereksinimler
- **OpenCV:** Kamera erişimi ve görüntü işleme
- **NumPy:** Sayısal hesaplamalar
- **Matplotlib:** Sonuç görselleştirme
- **Scikit-learn:** Performans metrikleri ve veri işleme

### Donanım Gereksinimleri
- Webcam veya kamera
- En az 4GB RAM (büyük veri setleri için)

## 📊 Problem Tanımı

### Veri Seti
- **5 farklı nesne sınıfı** (örnek: kitap, kalem, telefon, bardak, anahtar)
- Her sınıf için **en az 50 görüntü örneği**
- **32x32 piksel** boyutunda normalize edilmiş görüntüler
- **Gri tonlama** format

### Özellik Çıkarma
İmplementasyonunuzda aşağıdaki özellik türlerini kullanın:

1. **Histogram Özelikleri:**
   - 16 bin'lik gri seviye histogramı
   - Normalize edilmiş değerler

2. **İstatistiksel Özellikler:**
   - Ortalama parlaklık değeri
   - Standart sapma
   - Minimum ve maksimum piksel değerleri

3. **Kenar Özelikleri:**
   - Canny kenar algılama ile kenar yoğunluğu
   - Sobel gradyan miktarı

4. **Doku Özelikleri:**
   - LBP (Local Binary Pattern) histogramı
   - Gabor filtre cevapları (opsiyonel)

## 🛠️ Uygulama Adımları

### Adım 1: Veri Toplama Sistemi
```python
def create_data_collector():
    """
    Kameradan veri toplama sistemi oluşturun:
    
    Özellikler:
    - Sınıf seçimi (0-4 tuşları)
    - Otomatik örnek kaydetme (SPACE tuşu)
    - Sınıf başına örnek sayısı gösterimi
    - Toplanan örneklerin preview'ı
    """
    pass
```

### Adım 2: Özellik Çıkarma Modülü
```python
def extract_comprehensive_features(image):
    """
    Görüntüden kapsamlı özellik çıkarın:
    
    Returns:
    - histogram_features: 16 boyutlu histogram
    - statistical_features: 4 boyutlu istatistik
    - edge_features: 2 boyutlu kenar bilgisi
    - texture_features: 10 boyutlu doku bilgisi (LBP)
    
    Total: 32 boyutlu özellik vektörü
    """
    pass
```

### Adım 3: Model Karşılaştırma
Aşağıdaki modelleri implement edin ve karşılaştırın:

```python
def compare_ml_models(X_train, X_test, y_train, y_test):
    """
    ML modellerini karşılaştırın:
    
    Modeller:
    1. k-NN (k=3, k=5, k=7)
    2. SVM (Linear, RBF, Polynomial kernels)
    3. ANN (farklı hidden layer konfigürasyonları)
    4. Decision Tree (farklı max_depth değerleri)
    5. Random Forest (farklı n_trees değerleri)
    
    Metrikler:
    - Accuracy
    - Precision, Recall, F1-score (her sınıf için)
    - Training time
    - Prediction time
    - Memory usage
    """
    pass
```

### Adım 4: Cross-Validation
```python
def perform_cross_validation(X, y, models, k_folds=5):
    """
    K-fold cross validation implementasyonu:
    
    - Her model için 5-fold CV
    - Ortalama ve standart sapma hesaplama
    - Model kararlılığı analizi
    """
    pass
```

### Adım 5: Gerçek Zamanlı Tahmin
```python
def real_time_prediction_system(best_model):
    """
    En iyi modeli kullanarak gerçek zamanlı tahmin:
    
    - Kameradan anlık görüntü alma
    - Özellik çıkarma ve tahmin
    - Güven skoru gösterimi
    - Tahmin geçmişi tracking
    """
    pass
```

## 📈 Beklenen Sonuçlar

### Performans Hedefleri
- **Minimum doğruluk:** %80
- **Tahmin süresi:** <50ms per sample
- **Model kararlılığı:** CV std < 0.05

### Analiz Raporları
1. **Model Karşılaştırma Tablosu:**
   ```
   Model               | Accuracy | Precision | Recall | F1-Score | Time (ms)
   -------------------|----------|-----------|--------|----------|----------
   k-NN (k=5)         |   0.XX   |   0.XX    |  0.XX  |   0.XX   |   XX
   SVM (RBF)          |   0.XX   |   0.XX    |  0.XX  |   0.XX   |   XX
   ANN (64-32)        |   0.XX   |   0.XX    |  0.XX  |   0.XX   |   XX
   Decision Tree      |   0.XX   |   0.XX    |  0.XX  |   0.XX   |   XX
   Random Forest      |   0.XX   |   0.XX    |  0.XX  |   0.XX   |   XX
   ```

2. **Confusion Matrix:** Her model için
3. **Feature Importance:** Random Forest ve Decision Tree için
4. **Learning Curves:** Overfitting analizi

## 🏆 Değerlendirme Kriterleri

### Zorunlu Gereksinimler (70 puan)
- [ ] 5 sınıf, sınıf başına 50+ örnek veri toplama **(10p)**
- [ ] Kapsamlı özellik çıkarma (32 boyutlu) **(15p)**
- [ ] 5 farklı ML algoritması implementasyonu **(20p)**
- [ ] Cross-validation ile model değerlendirme **(15p)**
- [ ] Gerçek zamanlı tahmin sistemi **(10p)**

### Ek Özellikler (30 puan)
- [ ] LBP veya diğer ileri doku özellikleri **(5p)**
- [ ] Hiperparametre optimizasyonu (Grid Search) **(10p)**
- [ ] Model ensemble (voting/stacking) **(10p)**
- [ ] Comprehensive error analysis **(5p)**

### Bonus Görevler (Ekstra kredi)
- [ ] **GUI Interface:** Tkinter/PyQt ile kullanıcı arayüzü **(+10p)**
- [ ] **Model Persistence:** Eğitilmiş modelleri kaydetme/yükleme **(+5p)**
- [ ] **Augmentation:** Veri artırma teknikleri **(+10p)**
- [ ] **Deep Learning Comparison:** CNN ile karşılaştırma **(+15p)**

## 💡 İpuçları ve Öneriler

### Veri Toplama
- Her sınıf için çeşitli açılardan fotoğraflar çekin
- Farklı aydınlatma koşullarında örnekler toplayın
- Arka plan çeşitliliği sağlayın
- Data augmentation yapmayı düşünün

### Feature Engineering
- Özellik normalizasyonu yapmayı unutmayın
- PCA ile boyut indirgeme deneyin
- Korelasyon analizi yapın
- Feature selection teknikleri uygulayın

### Model Tuning
```python
# SVM için örnek hiperparametre grid'i
svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

# Random Forest için
rf_params = {
    'n_trees': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples': [1, 2, 5]
}
```

### Hata Ayıklama
- Veri dengesizliği kontrolü yapın
- Class-wise performance analizi
- False positive/negative örnekleri inceleyin
- Özellik önem sıralaması yapın

## 📝 Teslim Edilecekler

### Kod Dosyaları
1. **`main.py`:** Ana program dosyası
2. **`data_collector.py`:** Veri toplama modülü
3. **`feature_extractor.py`:** Özellik çıkarma fonksiyonları
4. **`model_trainer.py`:** ML model eğitim kodları
5. **`evaluator.py`:** Model değerlendirme araçları
6. **`predictor.py`:** Gerçek zamanlı tahmin sistemi

### Dokümantasyon
1. **`README.md`:** Kurulum ve kullanım kılavuzu
2. **`REPORT.md`:** Detaylı analiz raporu
3. **`requirements.txt`:** Gerekli kütüphaneler

### Veri ve Sonuçlar
1. **`dataset/`:** Toplanan veri seti
2. **`models/`:** Eğitilmiş model dosyaları
3. **`results/`:** Performans grafikleri ve confusion matrix'ler

## ⏰ Tahmini Süre
- **Veri toplama:** 2-3 saat
- **Feature engineering:** 3-4 saat  
- **Model implementation:** 4-6 saat
- **Evaluation ve analysis:** 2-3 saat
- **Documentation:** 1-2 saat

**Toplam:** 12-18 saat

## 🔗 Yararlı Kaynaklar
- [OpenCV Feature Detection Tutorial](https://docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [LBP Texture Analysis](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html)
- [Cross-validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

---

**Not:** Bu alıştırma, makine öğrenmesi konularının pratik uygulamasını kapsayan kapsamlı bir projedir. Adım adım ilerleyin ve her aşamada test edin!