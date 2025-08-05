# Alıştırma 3: Derin Öğrenme ve CNN'ler

## Amaç
Bu alıştırmada, Convolutional Neural Networks (CNN) kullanarak görüntü sınıflandırma sistemi geliştireceksiniz. Transfer learning, data augmentation ve model optimizasyonu tekniklerini öğreneceksiniz.

## Öğrenilecek Konular
- CNN Mimarisi Tasarımı
- Transfer Learning
- Data Augmentation
- Model Optimizasyonu
- Gerçek Zamanlı Tahmin
- Model Deployment

## Görev

### 1. Veri Hazırlama ve Preprocessing
- Sentetik görüntü verisi oluşturun (en az 5 sınıf)
- Data augmentation uygulayın
- Görüntü ön işleme teknikleri kullanın
- Veriyi eğitim, doğrulama ve test setlerine bölün

### 2. CNN Modeli Tasarımı
Aşağıdaki mimarileri oluşturun:
- Basit CNN (3-4 konvolüsyon katmanı)
- Derin CNN (6-8 konvolüsyon katmanı)
- Transfer Learning (VGG16, ResNet50, MobileNet)

### 3. Transfer Learning
- Pre-trained modelleri kullanın
- Fine-tuning uygulayın
- Farklı katmanları dondurun/çözün
- Learning rate scheduling kullanın

### 4. Model Optimizasyonu
- Hyperparameter tuning
- Learning rate optimization
- Batch size optimization
- Regularization techniques

### 5. Gerçek Zamanlı Uygulama
- Webcam ile gerçek zamanlı tahmin
- Model performansını izleme
- FPS optimizasyonu
- Kullanıcı arayüzü geliştirme

## Gereksinimler

### Kütüphaneler
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import time
import os
```

### Fonksiyonlar
1. `veri_olustur()`: Sentetik görüntü verisi oluştur
2. `data_augmentation()`: Data augmentation uygula
3. `basit_cnn_olustur()`: Basit CNN modeli oluştur
4. `derin_cnn_olustur()`: Derin CNN modeli oluştur
5. `transfer_learning_model()`: Transfer learning modeli oluştur
6. `model_egit()`: Model eğitimi
7. `gercek_zamanli_tahmin()`: Gerçek zamanlı tahmin
8. `model_deployment()`: Model deployment

## Beklenen Çıktılar

### 1. Model Performansları
```
Basit CNN: 0.82
Derin CNN: 0.87
Transfer Learning (VGG16): 0.91
Transfer Learning (ResNet50): 0.93
Transfer Learning (MobileNet): 0.89
```

### 2. Görselleştirmeler
- Eğitim/doğrulama grafikleri
- Confusion matrix
- ROC eğrileri
- Model mimarisi görselleştirmesi
- Gerçek zamanlı tahmin arayüzü

### 3. Analiz Raporu
- En iyi performans gösteren model
- Transfer learning avantajları
- Optimizasyon sonuçları
- Deployment önerileri

## Değerlendirme Kriterleri

### Kod Kalitesi (25%)
- Temiz ve modüler kod
- Fonksiyon tasarımı
- Hata yönetimi
- Dokümantasyon

### Model Performansı (35%)
- Doğruluk oranı
- Eğitim süresi
- Transfer learning etkinliği
- Optimizasyon başarısı

### Görselleştirme (20%)
- Eğitim grafikleri
- Model karşılaştırmaları
- Gerçek zamanlı arayüz
- Profesyonel görünüm

### İnovasyon (20%)
- Yaratıcı çözümler
- Performans optimizasyonu
- Kullanıcı deneyimi
- Teknik derinlik

## İpuçları

1. **Veri Çeşitliliği**: Farklı şekiller, renkler, boyutlar ve açılar kullanın
2. **Data Augmentation**: Rotation, zoom, flip, brightness değişiklikleri uygulayın
3. **Transfer Learning**: Pre-trained modellerin son katmanlarını özelleştirin
4. **Learning Rate**: Düşük learning rate ile başlayın ve scheduling kullanın
5. **Regularization**: Dropout ve batch normalization kullanın
6. **Early Stopping**: Overfitting'i önlemek için early stopping kullanın

## Zorluk Seviyesi
- **Başlangıç**: Basit CNN modeli oluşturun
- **Orta**: Transfer learning uygulayın
- **İleri**: Gerçek zamanlı uygulama geliştirin

## Süre
Tahmini süre: 3-4 saat

## Dosya Yapısı
```
alistirma-3/
├── alistirma-3.py
├── cozumler/
│   └── cozum-3.py
├── models/
│   ├── basit_cnn.h5
│   ├── derin_cnn.h5
│   └── transfer_model.h5
└── README.md
```

## Sonraki Adımlar
Bu alıştırmayı tamamladıktan sonra:
- Object Detection modellerini öğrenin
- Semantic Segmentation uygulayın
- Model compression tekniklerini öğrenin
- Edge deployment tekniklerini öğrenin
- Production-ready sistemler geliştirin

## Ek Kaynaklar
- TensorFlow/Keras dokümantasyonu
- CNN mimarileri (LeNet, AlexNet, VGG, ResNet)
- Transfer learning best practices
- Model optimization techniques
- Real-time inference optimization 