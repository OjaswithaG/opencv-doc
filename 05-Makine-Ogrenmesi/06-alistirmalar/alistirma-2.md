# Alıştırma 2: Ensemble Learning ve Model Optimizasyonu

## Amaç
Bu alıştırmada, ensemble learning yöntemlerini kullanarak bir görüntü sınıflandırma sistemi geliştireceksiniz. Farklı algoritmaları birleştirerek daha güçlü bir model oluşturacak ve hiperparametre optimizasyonu yapacaksınız.

## Öğrenilecek Konular
- Ensemble Learning (Bagging, Boosting, Voting)
- Hiperparametre Optimizasyonu
- Model Performans Karşılaştırması
- Cross-Validation Teknikleri
- Feature Engineering

## Görev

### 1. Veri Hazırlama
- Sentetik görüntü verisi oluşturun (en az 3 sınıf)
- Veriyi eğitim, doğrulama ve test setlerine bölün
- Feature engineering uygulayın (histogram, kenar özellikleri, vb.)

### 2. Ensemble Model Oluşturma
Aşağıdaki modelleri eğitin:
- Random Forest
- AdaBoost
- Gradient Boosting
- SVM
- K-NN

### 3. Voting Ensemble
- Soft voting kullanarak ensemble oluşturun
- Farklı ağırlık kombinasyonları deneyin
- En iyi performansı veren kombinasyonu bulun

### 4. Hiperparametre Optimizasyonu
- Grid Search veya Random Search kullanın
- En az 3 farklı algoritma için optimizasyon yapın
- Cross-validation ile sonuçları değerlendirin

### 5. Performans Analizi
- Confusion matrix oluşturun
- Precision, Recall, F1-score hesaplayın
- ROC eğrisi çizin
- Model karşılaştırma grafiği oluşturun

## Gereksinimler

### Kütüphaneler
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
```

### Fonksiyonlar
1. `veri_olustur()`: Sentetik görüntü verisi oluştur
2. `feature_engineering()`: Özellik çıkarımı yap
3. `ensemble_model_olustur()`: Ensemble model oluştur
4. `hiperparametre_optimizasyonu()`: Grid search ile optimizasyon
5. `performans_analizi()`: Detaylı performans analizi
6. `model_karsilastirma()`: Modelleri karşılaştır

## Beklenen Çıktılar

### 1. Model Performansları
```
Random Forest: 0.85
AdaBoost: 0.82
Gradient Boosting: 0.87
SVM: 0.80
K-NN: 0.78
Ensemble (Soft Voting): 0.89
```

### 2. Görselleştirmeler
- Confusion matrix heatmap
- ROC eğrileri
- Model karşılaştırma grafiği
- Hiperparametre optimizasyonu sonuçları

### 3. Analiz Raporu
- En iyi performans gösteren model
- Optimizasyon sonuçları
- Ensemble'in avantajları
- Öneriler ve iyileştirmeler

## Değerlendirme Kriterleri

### Kod Kalitesi (30%)
- Temiz ve okunabilir kod
- Fonksiyon modülerliği
- Hata yönetimi
- Dokümantasyon

### Model Performansı (40%)
- Doğruluk oranı
- Cross-validation skorları
- Ensemble performansı
- Optimizasyon etkinliği

### Görselleştirme (20%)
- Anlamlı grafikler
- Profesyonel görünüm
- Açıklayıcı etiketler

### Analiz Kalitesi (10%)
- Sonuçların yorumlanması
- Önerilerin kalitesi
- Teknik derinlik

## İpuçları

1. **Veri Çeşitliliği**: Farklı şekiller, renkler ve boyutlar kullanın
2. **Feature Engineering**: Histogram, kenar özellikleri, moment özellikleri ekleyin
3. **Ensemble Çeşitliliği**: Farklı algoritmalar kullanarak çeşitlilik sağlayın
4. **Hiperparametre Aralıkları**: Makul aralıklar seçin, çok geniş aralıklar hesaplama süresini artırır
5. **Cross-Validation**: 5-fold CV kullanın
6. **Görselleştirme**: Matplotlib ve Seaborn kullanarak profesyonel grafikler oluşturun

## Zorluk Seviyesi
- **Başlangıç**: Temel ensemble modeli oluşturun
- **Orta**: Hiperparametre optimizasyonu ekleyin
- **İleri**: Feature engineering ve gelişmiş analizler yapın

## Süre
Tahmini süre: 2-3 saat

## Dosya Yapısı
```
alistirma-2/
├── alistirma-2.py
├── cozumler/
│   └── cozum-2.py
└── README.md
```

## Sonraki Adımlar
Bu alıştırmayı tamamladıktan sonra:
- Deep Learning tekniklerini öğrenin
- Transfer Learning uygulayın
- Gerçek dünya veri setleriyle çalışın
- Model deployment tekniklerini öğrenin 