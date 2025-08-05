#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 OpenCV Makine Öğrenmesi - Temel Kavramlar
==========================================

Bu modül OpenCV'de makine öğrenmesi temellerini kapsar:
- cv2.ml modülü kullanımı
- Veri hazırlama ve normalleştirme
- Model eğitimi ve değerlendirme
- Çapraz doğrulama (Cross-validation)
- Performans metrikleri

Yazan: Eren Terzi
Tarih: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import os

class OpenCVMLBasics:
    """OpenCV Makine Öğrenmesi temel sınıfı"""
    
    def __init__(self):
        print("🤖 OpenCV ML Basics initialized")
        
    def create_sample_classification_data(self, n_samples=1000, n_features=2, n_classes=3):
        """Sınıflandırma için örnek veri oluştur"""
        print(f"📊 {n_samples} örnek, {n_features} özellik, {n_classes} sınıf veri oluşturuluyor...")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_redundant=0,
            n_informative=n_features,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        
        return X.astype(np.float32), y.astype(np.int32)
    
    def create_sample_regression_data(self, n_samples=1000, n_features=1):
        """Regresyon için örnek veri oluştur"""
        print(f"📈 {n_samples} örnek, {n_features} özellik regresyon verisi oluşturuluyor...")
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=10,
            random_state=42
        )
        
        return X.astype(np.float32), y.astype(np.float32)
    
    def normalize_features(self, X_train, X_test=None):
        """Özellikleri normalize et"""
        print("🔄 Özellikler normalize ediliyor...")
        
        # Min-Max normalization
        min_vals = np.min(X_train, axis=0)
        max_vals = np.max(X_train, axis=0)
        
        X_train_norm = (X_train - min_vals) / (max_vals - min_vals)
        
        if X_test is not None:
            X_test_norm = (X_test - min_vals) / (max_vals - min_vals)
            return X_train_norm, X_test_norm, min_vals, max_vals
        
        return X_train_norm, min_vals, max_vals
    
    def visualize_2d_data(self, X, y, title="Veri Dağılımı"):
        """2D veriyi görselleştir"""
        if X.shape[1] != 2:
            print("⚠️ Sadece 2D veri görselleştirilebilir")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Her sınıf için farklı renk
        unique_classes = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=[colors[i]], label=f'Sınıf {cls}', alpha=0.7)
        
        plt.xlabel('Özellik 1')
        plt.ylabel('Özellik 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """Performans metriklerini hesapla"""
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n📊 {model_name} Performans Metrikleri:")
        print(f"   Doğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"   Karışıklık Matrisi:")
        print(f"   {cm}")
        
        return accuracy, cm
    
    def cross_validation_demo(self, X, y, model, k_folds=5):
        """Çapraz doğrulama demonstrasyonu"""
        print(f"\n🔄 {k_folds}-fold Çapraz Doğrulama yapılıyor...")
        
        fold_size = len(X) // k_folds
        accuracies = []
        
        for fold in range(k_folds):
            # Test seti için indeksler
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else len(X)
            
            # Train/Test split
            test_indices = list(range(start_idx, end_idx))
            train_indices = list(range(0, start_idx)) + list(range(end_idx, len(X)))
            
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_test_fold = X[test_indices]
            y_test_fold = y[test_indices]
            
            # Model eğitimi
            model.train(X_train_fold, cv2.ml.ROW_SAMPLE, y_train_fold)
            
            # Tahmin
            _, predictions = model.predict(X_test_fold)
            predictions = predictions.flatten().astype(np.int32)
            
            # Doğruluk hesapla
            accuracy = accuracy_score(y_test_fold, predictions)
            accuracies.append(accuracy)
            
            print(f"   Fold {fold + 1}: {accuracy:.4f}")
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"   Ortalama Doğruluk: {mean_acc:.4f} ± {std_acc:.4f}")
        
        return mean_acc, std_acc

def ornek_1_veri_hazirlama():
    """Örnek 1: Veri hazırlama ve görselleştirme"""
    print("\n🎯 Örnek 1: Veri Hazırlama ve Görselleştirme")
    print("=" * 45)
    
    ml_basics = OpenCVMLBasics()
    
    # Sınıflandırma verisi oluştur
    X, y = ml_basics.create_sample_classification_data(n_samples=500, n_classes=3)
    
    print(f"📊 Veri boyutu: {X.shape}")
    print(f"📊 Sınıf dağılımı: {np.bincount(y)}")
    
    # Veriyi görselleştir
    ml_basics.visualize_2d_data(X, y, "Ham Veri Dağılımı")
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔄 Eğitim seti: {X_train.shape[0]} örnek")
    print(f"🔄 Test seti: {X_test.shape[0]} örnek")
    
    # Normalizasyon
    X_train_norm, X_test_norm, min_vals, max_vals = ml_basics.normalize_features(X_train, X_test)
    
    print(f"📏 Normalizasyon tamamlandı")
    print(f"   Min değerler: {min_vals}")
    print(f"   Max değerler: {max_vals}")
    
    # Normalize edilmiş veriyi görselleştir
    ml_basics.visualize_2d_data(X_train_norm, y_train, "Normalize Edilmiş Veri")
    
    return X_train_norm, X_test_norm, y_train, y_test

def ornek_2_opencv_ml_modulleri():
    """Örnek 2: OpenCV ML modüllerinin tanıtımı"""
    print("\n🎯 Örnek 2: OpenCV ML Modülleri")
    print("=" * 35)
    
    print("🔧 Mevcut OpenCV ML Algoritmaları:")
    
    # k-NN
    try:
        knn = cv2.ml.KNearest_create()
        print("✅ k-NN (k-Nearest Neighbors) - Mevcut")
    except:
        print("❌ k-NN - Mevcut değil")
    
    # SVM
    try:
        svm = cv2.ml.SVM_create()
        print("✅ SVM (Support Vector Machine) - Mevcut")
    except:
        print("❌ SVM - Mevcut değil")
    
    # ANN (Artificial Neural Network)
    try:
        ann = cv2.ml.ANN_MLP_create()
        print("✅ ANN (Artificial Neural Network) - Mevcut")
    except:
        print("❌ ANN - Mevcut değil")
    
    # Decision Tree
    try:
        dtree = cv2.ml.DTrees_create()
        print("✅ Decision Trees - Mevcut")
    except:
        print("❌ Decision Trees - Mevcut değil")
    
    # Random Forest
    try:
        rtrees = cv2.ml.RTrees_create()
        print("✅ Random Trees (Random Forest) - Mevcut")
    except:
        print("❌ Random Trees - Mevcut değil")
    
    # Boost
    try:
        boost = cv2.ml.Boost_create()
        print("✅ Boost - Mevcut")
    except:
        print("❌ Boost - Mevcut değil")
    
    # EM (Expectation Maximization)
    try:
        em = cv2.ml.EM_create()
        print("✅ EM (Expectation Maximization) - Mevcut")
    except:
        print("❌ EM - Mevcut değil")

def ornek_3_basit_siniflandirma():
    """Örnek 3: Basit sınıflandırma örneği"""
    print("\n🎯 Örnek 3: Basit Sınıflandırma (k-NN)")
    print("=" * 40)
    
    ml_basics = OpenCVMLBasics()
    
    # Veri hazırla
    X, y = ml_basics.create_sample_classification_data(n_samples=300, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # k-NN modeli oluştur
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(5)
    knn.setIsClassifier(True)
    
    print("🔄 k-NN modeli eğitiliyor...")
    start_time = time.time()
    
    # Model eğitimi
    knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    
    training_time = time.time() - start_time
    print(f"✅ Eğitim tamamlandı ({training_time:.4f} saniye)")
    
    # Tahmin
    print("🔮 Tahminler yapılıyor...")
    start_time = time.time()
    
    _, predictions = knn.predict(X_test)
    predictions = predictions.flatten().astype(np.int32)
    
    prediction_time = time.time() - start_time
    print(f"✅ Tahmin tamamlandı ({prediction_time:.4f} saniye)")
    
    # Sonuçları değerlendir
    accuracy, cm = ml_basics.calculate_metrics(y_test, predictions, "k-NN")
    
    # Çapraz doğrulama
    ml_basics.cross_validation_demo(X, y, knn, k_folds=5)
    
    return knn, accuracy

def ornek_4_model_karsilastirma():
    """Örnek 4: Farklı modelleri karşılaştırma"""
    print("\n🎯 Örnek 4: Model Karşılaştırması")
    print("=" * 35)
    
    ml_basics = OpenCVMLBasics()
    
    # Veri hazırla
    X, y = ml_basics.create_sample_classification_data(n_samples=500, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizasyon
    X_train_norm, X_test_norm, _, _ = ml_basics.normalize_features(X_train, X_test)
    
    models = {}
    results = {}
    
    print("🔄 Modeller eğitiliyor ve test ediliyor...")
    
    # 1. k-NN
    print("\n1️⃣ k-NN")
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(5)
    knn.setIsClassifier(True)
    
    start_time = time.time()
    knn.train(X_train_norm, cv2.ml.ROW_SAMPLE, y_train)
    _, pred_knn = knn.predict(X_test_norm)
    time_knn = time.time() - start_time
    
    pred_knn = pred_knn.flatten().astype(np.int32)
    acc_knn = accuracy_score(y_test, pred_knn)
    
    models['k-NN'] = knn
    results['k-NN'] = {'accuracy': acc_knn, 'time': time_knn}
    
    print(f"   Doğruluk: {acc_knn:.4f}, Süre: {time_knn:.4f}s")
    
    # 2. SVM
    print("\n2️⃣ SVM")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(1.0)
    svm.setGamma(0.1)
    
    start_time = time.time()
    svm.train(X_train_norm, cv2.ml.ROW_SAMPLE, y_train)
    _, pred_svm = svm.predict(X_test_norm)
    time_svm = time.time() - start_time
    
    pred_svm = pred_svm.flatten().astype(np.int32)
    acc_svm = accuracy_score(y_test, pred_svm)
    
    models['SVM'] = svm
    results['SVM'] = {'accuracy': acc_svm, 'time': time_svm}
    
    print(f"   Doğruluk: {acc_svm:.4f}, Süre: {time_svm:.4f}s")
    
    # 3. Decision Tree
    print("\n3️⃣ Decision Tree")
    dtree = cv2.ml.DTrees_create()
    dtree.setMaxDepth(10)
    dtree.setMinSampleCount(5)
    
    start_time = time.time()
    dtree.train(X_train_norm, cv2.ml.ROW_SAMPLE, y_train)
    _, pred_dtree = dtree.predict(X_test_norm)
    time_dtree = time.time() - start_time
    
    pred_dtree = pred_dtree.flatten().astype(np.int32)
    acc_dtree = accuracy_score(y_test, pred_dtree)
    
    models['Decision Tree'] = dtree
    results['Decision Tree'] = {'accuracy': acc_dtree, 'time': time_dtree}
    
    print(f"   Doğruluk: {acc_dtree:.4f}, Süre: {time_dtree:.4f}s")
    
    # Sonuçları özetle
    print("\n📊 KARŞILAŞTIRMA ÖZETİ")
    print("=" * 30)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (model_name, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {model_name}:")
        print(f"   Doğruluk: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Süre: {metrics['time']:.4f} saniye")
    
    return models, results

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🤖 OpenCV Makine Öğrenmesi - Temel Kavramlar")
        print("="*50)
        print("1. 📊 Veri Hazırlama ve Görselleştirme")
        print("2. 🔧 OpenCV ML Modülleri")  
        print("3. 🎯 Basit Sınıflandırma (k-NN)")
        print("4. ⚖️ Model Karşılaştırması")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-4): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_veri_hazirlama()
            elif secim == "2":
                ornek_2_opencv_ml_modulleri()
            elif secim == "3":
                ornek_3_basit_siniflandirma()
            elif secim == "4":
                ornek_4_model_karsilastirma()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🤖 OpenCV Makine Öğrenmesi - Temel Kavramlar")
    print("Bu modül OpenCV'de ML temellerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Matplotlib (pip install matplotlib)")
    print("   - Scikit-learn (pip install scikit-learn)")
    print("\n📝 Notlar:")
    print("   - İlk örnekler basit kavramları gösterir")
    print("   - Veri görselleştirme için matplotlib kullanılır")
    print("   - Gerçek projeler için daha büyük veri setleri gerekir")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. OpenCV ml modülü temel ML algoritmalarını sağlar
# 2. Veri normalizasyonu modellerin performansını artırır  
# 3. Çapraz doğrulama overfitting'i önler
# 4. Model karşılaştırması en iyi algoritma seçimini sağlar
# 5. Gerçek uygulamalarda feature engineering önemlidir