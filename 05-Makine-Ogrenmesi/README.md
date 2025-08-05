# 05-Makine Öğrenmesi

Bu bölüm, OpenCV ile makine öğrenmesi tekniklerini öğretir. Temel kavramlardan başlayarak ileri seviye derin öğrenme konularına kadar kapsamlı bir eğitim sunar.

## 📚 İçerik

### Temel Kavramlar
- **01-temel-kavramlar.py**: Makine öğrenmesi temelleri, veri hazırlama, normalizasyon, metrikler
- **02-knn-siniflandirma.py**: k-NN algoritması, digit recognition, interaktif tahmin
- **03-svm-nesne-tanima.py**: SVM algoritması, nesne tanıma uygulamaları
- **04-ann-karakter-tanima.py**: Yapay sinir ağları, karakter tanıma
- **05-karar-agaclari.py**: Karar ağaçları, Random Forest, Boost algoritmaları

### İleri Seviye Konular
- **06-ensemble-methods.py**: Ensemble yöntemleri (Bagging, Boosting, Voting), cross-validation
- **07-deep-learning-opencv.py**: CNN'ler, Transfer Learning, gerçek zamanlı uygulamalar

### Alıştırmalar
- **06-alistirmalar/**: Pratik uygulamalar ve çözümler
  - **alistirma-1.md/.py**: Temel ML alıştırması
  - **alistirma-2.md/.py**: Ensemble Learning ve model optimizasyonu
  - **alistirma-3.md/.py**: Derin öğrenme ve CNN'ler

## 🎯 Öğrenilecek Konular

### Temel Makine Öğrenmesi
- Veri hazırlama ve ön işleme
- Özellik mühendisliği (Feature Engineering)
- Model seçimi ve değerlendirme
- Cross-validation teknikleri
- Performans metrikleri

### Algoritmalar
- **k-NN (k-Nearest Neighbors)**: Sınıflandırma ve regresyon
- **SVM (Support Vector Machines)**: Nesne tanıma ve sınıflandırma
- **ANN (Artificial Neural Networks)**: Karakter tanıma
- **Decision Trees**: Karar ağaçları ve ensemble yöntemler
- **Random Forest**: Ensemble learning

### İleri Seviye Teknikler
- **Ensemble Methods**: Bagging, Boosting, Voting
- **Deep Learning**: CNN'ler, Transfer Learning
- **Model Optimization**: Hiperparametre optimizasyonu
- **Real-time Applications**: Gerçek zamanlı tahmin

## 🛠️ Gereksinimler

### Temel Kütüphaneler
```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### İleri Seviye Kütüphaneler
```bash
pip install tensorflow keras seaborn
```

### Opsiyonel Kütüphaneler
```bash
pip install jupyter notebook pandas plotly
```

## 📖 Kullanım

### Temel Konular
```bash
# Temel kavramları öğren
python 01-temel-kavramlar.py

# k-NN algoritması
python 02-knn-siniflandirma.py

# SVM ile nesne tanıma
python 03-svm-nesne-tanima.py

# Yapay sinir ağları
python 04-ann-karakter-tanima.py

# Karar ağaçları
python 05-karar-agaclari.py
```

### İleri Seviye Konular
```bash
# Ensemble yöntemleri
python 06-ensemble-methods.py

# Derin öğrenme ve CNN'ler
python 07-deep-learning-opencv.py
```

### Alıştırmalar
```bash
# Alıştırmaları çalıştır
cd 06-alistirmalar
python alistirma-1.py
python alistirma-2.py
python alistirma-3.py
```

## 🎓 Öğrenme Yolu

### 1. Temel Seviye (Başlangıç)
1. **01-temel-kavramlar.py**: Makine öğrenmesi temelleri
2. **02-knn-siniflandirma.py**: İlk algoritma deneyimi
3. **alistirma-1**: Temel ML alıştırması

### 2. Orta Seviye (Gelişim)
1. **03-svm-nesne-tanima.py**: SVM algoritması
2. **04-ann-karakter-tanima.py**: Yapay sinir ağları
3. **05-karar-agaclari.py**: Karar ağaçları
4. **alistirma-2**: Ensemble learning

### 3. İleri Seviye (Uzmanlaşma)
1. **06-ensemble-methods.py**: Ensemble yöntemleri
2. **07-deep-learning-opencv.py**: Derin öğrenme
3. **alistirma-3**: CNN'ler ve transfer learning

## 🔬 Pratik Uygulamalar

### Görüntü Sınıflandırma
- Digit recognition (MNIST benzeri)
- Şekil sınıflandırma (daire, dikdörtgen, üçgen)
- Karakter tanıma
- Nesne tanıma

### Gerçek Zamanlı Uygulamalar
- Webcam ile gerçek zamanlı tahmin
- Çizim tanıma sistemi
- Hareket algılama
- Yüz tanıma

### Model Optimizasyonu
- Hiperparametre optimizasyonu
- Cross-validation
- Ensemble learning
- Transfer learning

## 📊 Performans Metrikleri

### Sınıflandırma
- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1-Score
- Confusion Matrix

### Regresyon
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 🎨 Görselleştirme

### Eğitim Grafikleri
- Eğitim/doğrulama kayıp grafikleri
- Doğruluk grafikleri
- Learning curve analizi

### Model Karşılaştırmaları
- ROC eğrileri
- Precision-Recall eğrileri
- Model performans karşılaştırmaları

### Gerçek Zamanlı Görselleştirme
- Webcam arayüzü
- Tahmin sonuçları
- Güven skorları

## 🚀 İleri Seviye Özellikler

### Ensemble Learning
- **Bagging**: Bootstrap Aggregating
- **Boosting**: AdaBoost, Gradient Boosting
- **Voting**: Soft/Hard voting
- **Stacking**: Model stacking

### Deep Learning
- **CNN**: Convolutional Neural Networks
- **Transfer Learning**: Pre-trained modeller
- **Data Augmentation**: Görüntü artırma
- **Model Optimization**: Hiperparametre tuning

### Real-time Applications
- **Webcam Integration**: Gerçek zamanlı video işleme
- **Performance Monitoring**: FPS takibi
- **User Interface**: Kullanıcı dostu arayüz
- **Model Deployment**: Model dağıtımı

## 📈 Başarı Kriterleri

### Kod Kalitesi
- Temiz ve okunabilir kod
- Fonksiyon modülerliği
- Hata yönetimi
- Dokümantasyon

### Model Performansı
- Yüksek doğruluk oranları
- Hızlı eğitim süreleri
- Düşük overfitting
- Genelleme yeteneği

### Görselleştirme
- Anlamlı grafikler
- Profesyonel görünüm
- Açıklayıcı etiketler
- İnteraktif arayüzler

## 🔧 Sorun Giderme

### Yaygın Sorunlar
1. **Memory Error**: Büyük veri setleri için batch processing
2. **Overfitting**: Regularization ve early stopping
3. **Slow Training**: GPU kullanımı ve optimizasyon
4. **Poor Performance**: Feature engineering ve model seçimi

### Optimizasyon İpuçları
- Veri ön işleme kalitesini artırın
- Feature engineering uygulayın
- Model hiperparametrelerini optimize edin
- Ensemble yöntemleri kullanın

## 📚 Ek Kaynaklar

### Dokümantasyon
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)

### Araştırma Makaleleri
- Ensemble Learning Methods
- Deep Learning for Computer Vision
- Transfer Learning Techniques
- Real-time Machine Learning

### Online Kurslar
- Coursera: Machine Learning
- Udacity: Deep Learning Nanodegree
- edX: Computer Vision
- Fast.ai: Practical Deep Learning

## 🤝 Katkıda Bulunma

Bu projeye katkıda bulunmak için:
1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Pull request gönderin

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Sorularınız için:
- GitHub Issues
- Email: opencv-turkiye@example.com
- Discord: OpenCV Türkiye Topluluğu

---

**Not**: Bu bölüm, makine öğrenmesi konusunda kapsamlı bir eğitim sunar. Her dosya, teorik bilgilerle birlikte pratik uygulamalar içerir. Alıştırmaları tamamlayarak öğrendiklerinizi pekiştirebilirsiniz. 