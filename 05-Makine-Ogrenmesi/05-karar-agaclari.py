#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌳 Karar Ağaçları ve Random Forest
=================================

Bu modül Karar Ağaçları ve Random Forest algoritmalarını kapsar:
- OpenCV Decision Trees (DTrees) implementasyonu
- Random Trees (Random Forest) algoritması
- Entropy ve Gini indeksi
- Ağaç budama (Pruning) teknikleri
- Feature importance analizi
- Görüntü tabanlı sınıflandırma

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import time
import os

class DecisionTreeClassifier:
    """OpenCV Decision Tree Sınıflandırıcı"""
    
    def __init__(self):
        self.model = cv2.ml.DTrees_create()
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        
        # Default parametreler
        self.model.setMaxDepth(10)
        self.model.setMinSampleCount(2)
        self.model.setCVFolds(0)  # Cross-validation kapalı
        self.model.setUseSurrogates(False)
        self.model.setTruncatePrunedTree(True)
        
        print("🌳 Decision Tree oluşturuldu")
    
    def set_parameters(self, max_depth=10, min_sample_count=2, max_categories=10):
        """Karar ağacı parametrelerini ayarla"""
        print(f"🔧 Parametreler: max_depth={max_depth}, min_samples={min_sample_count}")
        
        self.model.setMaxDepth(max_depth)
        self.model.setMinSampleCount(min_sample_count)
        self.model.setMaxCategories(max_categories)
    
    def train(self, X_train, y_train, feature_names=None, class_names=None):
        """Modeli eğit"""
        print("🔄 Decision Tree modeli eğitiliyor...")
        
        # Veri tiplerini kontrol et
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        
        self.feature_names = feature_names
        self.class_names = class_names
        
        start_time = time.time()
        
        # Modeli eğit
        success = self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        
        training_time = time.time() - start_time
        
        if success:
            self.is_trained = True
            print(f"✅ Eğitim tamamlandı ({training_time:.4f} saniye)")
            
            # Ağaç bilgilerini göster
            try:
                depth = self.model.getMaxDepth()
                print(f"📊 Ağaç derinliği: {depth}")
            except:
                pass
        else:
            print("❌ Eğitim başarısız!")
            return None
        
        return training_time
    
    def predict(self, X_test):
        """Tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        X_test = X_test.astype(np.float32)
        
        start_time = time.time()
        _, predictions = self.model.predict(X_test)
        prediction_time = time.time() - start_time
        
        predictions = predictions.flatten().astype(np.int32)
        
        return predictions, prediction_time
    
    def get_variable_importance(self):
        """Özellik önemini al"""
        if not self.is_trained:
            return None
        
        try:
            # OpenCV'de variable importance
            importance = self.model.getVarImportance()
            if importance is not None:
                return importance.flatten()
        except:
            pass
        
        return None

class RandomForestClassifier:
    """OpenCV Random Trees (Random Forest) Sınıflandırıcı"""
    
    def __init__(self, n_trees=100):
        self.model = cv2.ml.RTrees_create()
        self.n_trees = n_trees
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
        
        # Default parametreler
        self.model.setMaxDepth(10)
        self.model.setMinSampleCount(2)
        self.model.setActiveVarCount(0)  # sqrt(n_features) otomatik
        self.model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, n_trees, 0))
        
        print(f"🌲 Random Forest oluşturuldu ({n_trees} ağaç)")
    
    def set_parameters(self, max_depth=10, min_sample_count=2, max_categories=10, active_var_count=0):
        """Random Forest parametrelerini ayarla"""
        print(f"🔧 RF Parametreler: max_depth={max_depth}, min_samples={min_sample_count}")
        print(f"   active_vars={active_var_count} (0=auto)")
        
        self.model.setMaxDepth(max_depth)
        self.model.setMinSampleCount(min_sample_count)
        self.model.setMaxCategories(max_categories)
        self.model.setActiveVarCount(active_var_count)
    
    def train(self, X_train, y_train, feature_names=None, class_names=None):
        """Modeli eğit"""
        print(f"🔄 Random Forest modeli eğitiliyor ({self.n_trees} ağaç)...")
        
        # Veri tiplerini kontrol et
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        
        self.feature_names = feature_names
        self.class_names = class_names
        
        start_time = time.time()
        
        # Modeli eğit
        success = self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        
        training_time = time.time() - start_time
        
        if success:
            self.is_trained = True
            print(f"✅ Eğitim tamamlandı ({training_time:.4f} saniye)")
        else:
            print("❌ Eğitim başarısız!")
            return None
        
        return training_time
    
    def predict(self, X_test):
        """Tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        X_test = X_test.astype(np.float32)
        
        start_time = time.time()
        _, predictions = self.model.predict(X_test)
        prediction_time = time.time() - start_time
        
        predictions = predictions.flatten().astype(np.int32)
        
        return predictions, prediction_time
    
    def get_variable_importance(self):
        """Özellik önemini al"""
        if not self.is_trained:
            return None
        
        try:
            importance = self.model.getVarImportance()
            if importance is not None:
                return importance.flatten()
        except:
            pass
        
        return None

def ornek_1_iris_siniflandirma():
    """Örnek 1: Iris veri seti ile temel sınıflandırma"""
    print("\n🌳 Örnek 1: Iris Sınıflandırma")
    print("=" * 30)
    
    # Iris veri setini yükle
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    print(f"📊 Iris veri seti:")
    print(f"   Özellikler: {feature_names}")
    print(f"   Sınıflar: {class_names}")
    print(f"   Veri boyutu: {X.shape}")
    
    # Veriyi görselleştir (ilk 2 özellik)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green']
    for i, class_name in enumerate(class_names):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=class_name, alpha=0.7)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Iris Veri Seti (İlk 2 Özellik)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Decision Tree
    print("\n🌳 Decision Tree test ediliyor...")
    dt = DecisionTreeClassifier()
    dt.set_parameters(max_depth=5, min_sample_count=2)
    
    dt_training_time = dt.train(X_train, y_train, feature_names, class_names)
    dt_predictions, dt_pred_time = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    
    print(f"   Doğruluk: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
    
    # Random Forest
    print("\n🌲 Random Forest test ediliyor...")
    rf = RandomForestClassifier(n_trees=50)
    rf.set_parameters(max_depth=5, min_sample_count=2)
    
    rf_training_time = rf.train(X_train, y_train, feature_names, class_names)
    rf_predictions, rf_pred_time = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    
    print(f"   Doğruluk: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    
    # Sonuçları karşılaştır
    plt.subplot(1, 3, 2)
    models = ['Decision Tree', 'Random Forest']
    accuracies = [dt_accuracy, rf_accuracy]
    times = [dt_training_time, rf_training_time]
    
    bars = plt.bar(models, accuracies, color=['lightblue', 'lightgreen'])
    plt.ylabel('Doğruluk')
    plt.title('Model Karşılaştırması')
    plt.ylim(0, 1)
    
    # Değerleri çubukların üzerine yaz
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{accuracies[i]:.3f}', ha='center', va='bottom')
    
    # Feature importance
    plt.subplot(1, 3, 3)
    dt_importance = dt.get_variable_importance()
    rf_importance = rf.get_variable_importance()
    
    if dt_importance is not None and rf_importance is not None:
        x_pos = np.arange(len(feature_names))
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, dt_importance, width, label='Decision Tree', alpha=0.7)
        bars2 = plt.bar(x_pos + width/2, rf_importance, width, label='Random Forest', alpha=0.7)
        
        plt.xlabel('Özellikler')
        plt.ylabel('Önem')
        plt.title('Özellik Önem Karşılaştırması')
        plt.xticks(x_pos, [name.split()[0] for name in feature_names], rotation=45)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Özellik önem bilgisi\nalınamadı', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Özellik Önem Analizi')
    
    plt.tight_layout()
    plt.show()
    
    return dt, rf, dt_accuracy, rf_accuracy

def ornek_2_wine_siniflandirma():
    """Örnek 2: Wine veri seti ile gelişmiş sınıflandırma"""
    print("\n🌳 Örnek 2: Wine Sınıflandırma")
    print("=" * 30)
    
    # Wine veri setini yükle
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    feature_names = wine.feature_names
    class_names = wine.target_names
    
    print(f"📊 Wine veri seti:")
    print(f"   Özellik sayısı: {len(feature_names)}")
    print(f"   Sınıf sayısı: {len(class_names)}")
    print(f"   Veri boyutu: {X.shape}")
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Farklı parametrelerle test
    configurations = [
        {'max_depth': 3, 'min_samples': 5, 'name': 'Sığ Ağaç'},
        {'max_depth': 10, 'min_samples': 2, 'name': 'Derin Ağaç'},
        {'max_depth': 15, 'min_samples': 1, 'name': 'Çok Derin Ağaç'}
    ]
    
    dt_results = []
    rf_results = []
    
    print("\n🔄 Farklı konfigürasyonlar test ediliyor...")
    
    for config in configurations:
        print(f"\n📐 {config['name']} (depth={config['max_depth']}, min_samples={config['min_samples']})")
        
        # Decision Tree
        dt = DecisionTreeClassifier()
        dt.set_parameters(max_depth=config['max_depth'], min_sample_count=config['min_samples'])
        
        dt_time = dt.train(X_train, y_train, feature_names, class_names)
        dt_pred, _ = dt.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_pred)
        
        dt_results.append({
            'name': config['name'],
            'accuracy': dt_acc,
            'time': dt_time
        })
        
        print(f"   DT Doğruluk: {dt_acc:.4f}")
        
        # Random Forest
        rf = RandomForestClassifier(n_trees=100)
        rf.set_parameters(max_depth=config['max_depth'], min_sample_count=config['min_samples'])
        
        rf_time = rf.train(X_train, y_train, feature_names, class_names)
        rf_pred, _ = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        rf_results.append({
            'name': config['name'],
            'accuracy': rf_acc,
            'time': rf_time
        })
        
        print(f"   RF Doğruluk: {rf_acc:.4f}")
    
    # Sonuçları görselleştir
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Doğruluk karşılaştırması
    names = [r['name'] for r in dt_results]
    dt_accuracies = [r['accuracy'] for r in dt_results]
    rf_accuracies = [r['accuracy'] for r in rf_results]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, dt_accuracies, width, label='Decision Tree', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, rf_accuracies, width, label='Random Forest', alpha=0.7)
    
    ax1.set_xlabel('Konfigürasyon')
    ax1.set_ylabel('Doğruluk')
    ax1.set_title('Wine Sınıflandırma - Doğruluk Karşılaştırması')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Değerleri yaz
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Süre karşılaştırması
    dt_times = [r['time'] for r in dt_results]
    rf_times = [r['time'] for r in rf_results]
    
    bars3 = ax2.bar(x_pos - width/2, dt_times, width, label='Decision Tree', alpha=0.7)
    bars4 = ax2.bar(x_pos + width/2, rf_times, width, label='Random Forest', alpha=0.7)
    
    ax2.set_xlabel('Konfigürasyon')
    ax2.set_ylabel('Eğitim Süresi (saniye)')
    ax2.set_title('Wine Sınıflandırma - Eğitim Süresi')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # En iyi modeli bul
    best_dt = max(dt_results, key=lambda x: x['accuracy'])
    best_rf = max(rf_results, key=lambda x: x['accuracy'])
    
    print(f"\n🏆 En iyi Decision Tree: {best_dt['name']} - {best_dt['accuracy']:.4f}")
    print(f"🏆 En iyi Random Forest: {best_rf['name']} - {best_rf['accuracy']:.4f}")
    
    return dt_results, rf_results

def ornek_3_overfitting_analizi():
    """Örnek 3: Overfitting analizi"""
    print("\n🌳 Örnek 3: Overfitting Analizi")
    print("=" * 30)
    
    # Karmaşık veri seti oluştur
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    print(f"📊 Sentetik veri: {X.shape[0]} örnek, {X.shape[1]} özellik, {len(np.unique(y))} sınıf")
    
    # Train/Validation/Test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"🔄 Veri bölünmesi: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Farklı max_depth değerleri ile test
    depths = [1, 3, 5, 7, 10, 15, 20, None]  # None = sınırsız
    dt_train_scores = []
    dt_val_scores = []
    rf_train_scores = []
    rf_val_scores = []
    
    print("\n🔄 Farklı derinlikler test ediliyor...")
    
    for depth in depths:
        depth_val = 25 if depth is None else depth  # OpenCV için büyük değer
        print(f"   Derinlik: {'Sınırsız' if depth is None else depth}")
        
        # Decision Tree
        dt = DecisionTreeClassifier()
        dt.set_parameters(max_depth=depth_val, min_sample_count=1)
        dt.train(X_train, y_train)
        
        dt_train_pred, _ = dt.predict(X_train)
        dt_val_pred, _ = dt.predict(X_val)
        
        dt_train_acc = accuracy_score(y_train, dt_train_pred)
        dt_val_acc = accuracy_score(y_val, dt_val_pred)
        
        dt_train_scores.append(dt_train_acc)
        dt_val_scores.append(dt_val_acc)
        
        # Random Forest
        rf = RandomForestClassifier(n_trees=50)
        rf.set_parameters(max_depth=depth_val, min_sample_count=1)
        rf.train(X_train, y_train)
        
        rf_train_pred, _ = rf.predict(X_train)
        rf_val_pred, _ = rf.predict(X_val)
        
        rf_train_acc = accuracy_score(y_train, rf_train_pred)
        rf_val_acc = accuracy_score(y_val, rf_val_pred)
        
        rf_train_scores.append(rf_train_acc)
        rf_val_scores.append(rf_val_acc)
        
        print(f"     DT - Train: {dt_train_acc:.3f}, Val: {dt_val_acc:.3f}")
        print(f"     RF - Train: {rf_train_acc:.3f}, Val: {rf_val_acc:.3f}")
    
    # Overfitting grafiklerini çiz
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    depth_labels = ['1', '3', '5', '7', '10', '15', '20', '∞']
    
    # Decision Tree
    ax1.plot(depth_labels, dt_train_scores, 'o-', label='Eğitim Doğruluğu', linewidth=2)
    ax1.plot(depth_labels, dt_val_scores, 's-', label='Doğrulama Doğruluğu', linewidth=2)
    ax1.set_xlabel('Maksimum Derinlik')
    ax1.set_ylabel('Doğruluk')
    ax1.set_title('Decision Tree - Overfitting Analizi')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfitting noktasını işaretle
    overfitting_gap = [train - val for train, val in zip(dt_train_scores, dt_val_scores)]
    max_gap_idx = np.argmax(overfitting_gap)
    ax1.axvline(x=max_gap_idx, color='red', linestyle='--', alpha=0.7, label='Max Overfitting')
    
    # Random Forest
    ax2.plot(depth_labels, rf_train_scores, 'o-', label='Eğitim Doğruluğu', linewidth=2)
    ax2.plot(depth_labels, rf_val_scores, 's-', label='Doğrulama Doğruluğu', linewidth=2)
    ax2.set_xlabel('Maksimum Derinlik')
    ax2.set_ylabel('Doğruluk')
    ax2.set_title('Random Forest - Overfitting Analizi')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # En iyi hiperparametreyi bul
    best_dt_idx = np.argmax(dt_val_scores)
    best_rf_idx = np.argmax(rf_val_scores)
    
    print(f"\n🏆 En iyi Decision Tree derinliği: {depth_labels[best_dt_idx]} (Val Acc: {dt_val_scores[best_dt_idx]:.4f})")
    print(f"🏆 En iyi Random Forest derinliği: {depth_labels[best_rf_idx]} (Val Acc: {rf_val_scores[best_rf_idx]:.4f})")
    
    # Overfitting analizi
    dt_overfitting = dt_train_scores[best_dt_idx] - dt_val_scores[best_dt_idx]
    rf_overfitting = rf_train_scores[best_rf_idx] - rf_val_scores[best_rf_idx]
    
    print(f"\n📊 Overfitting Analizi:")
    print(f"   Decision Tree gap: {dt_overfitting:.4f}")
    print(f"   Random Forest gap: {rf_overfitting:.4f}")
    
    if rf_overfitting < dt_overfitting:
        print("   🌲 Random Forest daha az overfitting gösteriyor!")
    else:
        print("   🌳 Decision Tree daha az overfitting gösteriyor!")
    
    return depths, dt_train_scores, dt_val_scores, rf_train_scores, rf_val_scores

def ornek_4_goruntu_tabanlı_siniflandirma():
    """Örnek 4: Görüntü tabanlı sınıflandırma"""
    print("\n🌳 Örnek 4: Görüntü Tabanlı Sınıflandırma")
    print("=" * 40)
    
    print("📹 Webcam'den görüntü sınıflandırma...")
    print("Farklı nesneleri kameraya gösterin")
    print("SPACE: Örnek kaydet, T: Eğit, P: Tahmin modu, ESC: Çıkış")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    # Veri toplama
    training_data = []
    training_labels = []
    current_class = 0
    max_classes = 3
    
    # Modeller
    dt_model = None
    rf_model = None
    is_prediction_mode = False
    
    def extract_features(frame):
        """Basit özellik çıkarma"""
        # Gri tonlamaya çevir ve yeniden boyutlandır
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        
        # Histogram özelikleri
        hist = cv2.calcHist([resized], [0], None, [16], [0, 256])
        hist_features = hist.flatten() / np.sum(hist)  # Normalize
        
        # İstatistiksel özellikler
        mean_val = np.mean(resized)
        std_val = np.std(resized)
        
        # Kenar özelikleri
        edges = cv2.Canny(resized, 50, 150)
        edge_density = np.sum(edges > 0) / (32 * 32)
        
        # Tüm özellikleri birleştir
        features = np.concatenate([hist_features, [mean_val, std_val, edge_density]])
        
        return features
    
    print(f"🎯 Sınıf {current_class} için örnekler toplayın (SPACE ile kaydet)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # UI bilgilerini göster
        if not is_prediction_mode:
            status_text = f"Sinif {current_class} - Ornekler: {len([l for l in training_labels if l == current_class])}"
            color = (0, 255, 255)
        else:
            status_text = "TAHMIN MODU AKTIF"
            color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        total_samples = len(training_data)
        cv2.putText(frame, f'Toplam ornek: {total_samples}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Model durumu
        model_status = "Egitilmedi"
        if dt_model is not None:
            model_status = "Egitildi"
        cv2.putText(frame, f'Model: {model_status}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Tahmin modu
        if is_prediction_mode and dt_model is not None:
            features = extract_features(frame)
            
            try:
                dt_pred, _ = dt_model.predict(features.reshape(1, -1))
                rf_pred, _ = rf_model.predict(features.reshape(1, -1))
                
                cv2.putText(frame, f'DT Tahmin: Sinif {dt_pred[0]}', (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f'RF Tahmin: Sinif {rf_pred[0]}', (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except:
                cv2.putText(frame, 'Tahmin hatasi!', (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Kontrol bilgileri
        cv2.putText(frame, 'SPACE:Kaydet T:Egit P:Tahmin ESC:Cikis', 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Goruntu Siniflandirma', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' ') and not is_prediction_mode:  # SPACE - Örnek kaydet
            features = extract_features(frame)
            training_data.append(features)
            training_labels.append(current_class)
            
            samples_in_class = len([l for l in training_labels if l == current_class])
            print(f"✅ Sınıf {current_class} - {samples_in_class}. örnek kaydedildi")
            
            # Yeterli örnek varsa bir sonraki sınıfa geç
            if samples_in_class >= 10 and current_class < max_classes - 1:
                current_class += 1
                print(f"🎯 Sınıf {current_class} için örnekler toplayın")
        
        elif key == ord('t') or key == ord('T'):  # T - Eğit
            if len(training_data) < 10:
                print("⚠️ En az 10 örnek gerekli!")
                continue
            
            print("🔄 Modeller eğitiliyor...")
            
            X = np.array(training_data, dtype=np.float32)
            y = np.array(training_labels, dtype=np.int32)
            
            # Decision Tree
            dt_model = DecisionTreeClassifier()
            dt_model.set_parameters(max_depth=10, min_sample_count=2)
            dt_model.train(X, y)
            
            # Random Forest
            rf_model = RandomForestClassifier(n_trees=50)
            rf_model.set_parameters(max_depth=10, min_sample_count=2)
            rf_model.train(X, y)
            
            print("✅ Modeller eğitildi!")
            
        elif key == ord('p') or key == ord('P'):  # P - Tahmin modu
            if dt_model is None:
                print("⚠️ Önce modeli eğitin!")
            else:
                is_prediction_mode = not is_prediction_mode
                mode_text = "AÇIK" if is_prediction_mode else "KAPALI"
                print(f"🔮 Tahmin modu: {mode_text}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final sonuçları
    if len(training_data) > 0:
        print(f"\n📊 Toplanan veri:")
        for class_id in range(max_classes):
            count = len([l for l in training_labels if l == class_id])
            print(f"   Sınıf {class_id}: {count} örnek")
        
        if dt_model is not None:
            print("✅ Modeller başarıyla eğitildi!")
    
    return dt_model, rf_model

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🌳 Karar Ağaçları ve Random Forest Demo")
        print("="*50)
        print("1. 🌸 Iris Sınıflandırma")
        print("2. 🍷 Wine Sınıflandırma")  
        print("3. 📊 Overfitting Analizi")
        print("4. 📹 Görüntü Tabanlı Sınıflandırma")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-4): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_iris_siniflandirma()
            elif secim == "2":
                ornek_2_wine_siniflandirma()
            elif secim == "3":
                ornek_3_overfitting_analizi()
            elif secim == "4":
                ornek_4_goruntu_tabanlı_siniflandirma()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🌳 Karar Ağaçları ve Random Forest - OpenCV ML")
    print("Bu modül ağaç tabanlı makine öğrenmesi algoritmalarını öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Matplotlib (pip install matplotlib)")
    print("   - Scikit-learn (pip install scikit-learn)")
    print("\n📝 Karar Ağaçları Özellikleri:")
    print("   - Kolay yorumlanabilir")
    print("   - Kategorik ve sürekli verilerle çalışır")
    print("   - Overfitting'e eğilimli")
    print("   - Random Forest overfitting'i azaltır")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Decision tree derinliği overfitting'i doğrudan etkiler
# 2. Random Forest bias-variance trade-off'unu iyileştirir  
# 3. Min sample count küçük dalların oluşmasını önler
# 4. Feature importance analizi model yorumlanabilirliği sağlar
# 5. Görüntü sınıflandırmada özellik mühendisliği kritiktir