#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚖️ SVM Nesne Tanıma - Support Vector Machine
===========================================

Bu modül Support Vector Machine algoritmasını detaylı olarak kapsar:
- SVM algoritması temel prensipleri
- Kernel fonksiyonları (Linear, RBF, Polynomial)
- OpenCV SVM implementasyonu
- Görüntü tabanlı nesne sınıflandırma
- HOG + SVM ile insan tespiti
- Hiperparametre optimizasyonu

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import os

class SVMClassifier:
    """OpenCV SVM Sınıflandırıcı"""
    
    def __init__(self, svm_type=cv2.ml.SVM_C_SVC, kernel=cv2.ml.SVM_RBF):
        self.model = cv2.ml.SVM_create()
        self.model.setType(svm_type)
        self.model.setKernel(kernel)
        self.is_trained = False
        self.scaler = StandardScaler()
        self.use_scaling = True
        
        # Default hiperparametreler
        self.model.setC(1.0)
        self.model.setGamma(0.1)
        
    def set_parameters(self, C=1.0, gamma=0.1, degree=3):
        """SVM parametrelerini ayarla"""
        print(f"🔧 SVM parametreleri: C={C}, gamma={gamma}, degree={degree}")
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.model.setDegree(degree)
    
    def set_kernel(self, kernel):
        """Kernel fonksiyonunu değiştir"""
        kernel_names = {
            cv2.ml.SVM_LINEAR: "Linear",
            cv2.ml.SVM_RBF: "RBF (Gaussian)",
            cv2.ml.SVM_POLY: "Polynomial",
            cv2.ml.SVM_SIGMOID: "Sigmoid"
        }
        
        print(f"🔄 Kernel: {kernel_names.get(kernel, 'Unknown')}")
        self.model.setKernel(kernel)
    
    def train(self, X_train, y_train, normalize=True):
        """Modeli eğit"""
        print("🔄 SVM modeli eğitiliyor...")
        
        # Veri tiplerini kontrol et
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        
        # Normalizasyon
        if normalize and self.use_scaling:
            X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        
        start_time = time.time()
        
        # Modeli eğit
        self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"✅ Eğitim tamamlandı ({training_time:.4f} saniye)")
        
        # Support vector sayısını göster
        support_vectors = self.model.getSupportVectors()
        if support_vectors is not None:
            print(f"📊 Support Vector sayısı: {len(support_vectors)}")
        
        return training_time
    
    def predict(self, X_test, normalize=True):
        """Tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        X_test = X_test.astype(np.float32)
        
        # Normalizasyon
        if normalize and self.use_scaling:
            X_test = self.scaler.transform(X_test).astype(np.float32)
        
        start_time = time.time()
        _, predictions = self.model.predict(X_test)
        prediction_time = time.time() - start_time
        
        predictions = predictions.flatten().astype(np.int32)
        
        return predictions, prediction_time

class HOGPersonDetector:
    """HOG + SVM ile insan tespiti"""
    
    def __init__(self):
        # HOG descriptor parametreleri
        self.hog = cv2.HOGDescriptor()
        
        # Önceden eğitilmiş insan tespit modeli
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Custom SVM için
        self.custom_svm = None
        self.feature_size = None
        
    def extract_hog_features(self, images, resize_to=(64, 128)):
        """Görüntülerden HOG özelliklerini çıkar"""
        print(f"🔄 {len(images)} görüntüden HOG özellikleri çıkarılıyor...")
        
        features = []
        
        for i, img in enumerate(images):
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Boyutu standardize et
            img = cv2.resize(img, resize_to)
            
            # HOG özelliklerini çıkar
            hog_features = self.hog.compute(img)
            features.append(hog_features.flatten())
            
            if (i + 1) % 100 == 0:
                print(f"   İşlenen: {i + 1}/{len(images)}")
        
        features = np.array(features, dtype=np.float32)
        self.feature_size = features.shape[1]
        
        print(f"✅ HOG özellikleri çıkarıldı: {features.shape}")
        
        return features
    
    def detect_people_default(self, image):
        """Önceden eğitilmiş model ile insan tespiti"""
        # Parametreler
        hit_threshold = 0.0
        win_stride = (8, 8)
        padding = (32, 32)
        scale = 1.05
        
        # İnsan tespiti
        rectangles, weights = self.hog.detectMultiScale(
            image,
            hitThreshold=hit_threshold,
            winStride=win_stride,
            padding=padding,
            scale=scale,
            finalThreshold=2
        )
        
        return rectangles, weights
    
    def train_custom_detector(self, positive_images, negative_images):
        """Özel insan tespit modeli eğit"""
        print("🔄 Özel HOG+SVM modeli eğitiliyor...")
        
        # Pozitif örneklerden özellik çıkar
        pos_features = self.extract_hog_features(positive_images)
        pos_labels = np.ones(len(pos_features), dtype=np.int32)
        
        # Negatif örneklerden özellik çıkar
        neg_features = self.extract_hog_features(negative_images)
        neg_labels = np.zeros(len(neg_features), dtype=np.int32)
        
        # Veriyi birleştir
        X = np.vstack([pos_features, neg_features])
        y = np.hstack([pos_labels, neg_labels])
        
        print(f"📊 Toplam eğitim verisi: {len(X)} ({len(pos_features)} pozitif, {len(neg_features)} negatif)")
        
        # SVM eğit
        self.custom_svm = SVMClassifier(kernel=cv2.ml.SVM_LINEAR)
        self.custom_svm.set_parameters(C=1.0)
        training_time = self.custom_svm.train(X, y)
        
        return training_time

def ornek_1_temel_svm():
    """Örnek 1: Temel SVM kullanımı"""
    print("\n⚖️ Örnek 1: Temel SVM Kullanımı")
    print("=" * 35)
    
    # 2D sınıflandırma verisi oluştur
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Veriyi görselleştir
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue']
    for i in range(2):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'Sınıf {i}', alpha=0.7)
    plt.xlabel('Özellik 1')
    plt.ylabel('Özellik 2')
    plt.title('Ham Veri')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Farklı kernel'ları test et
    kernels = [
        (cv2.ml.SVM_LINEAR, "Linear"),
        (cv2.ml.SVM_RBF, "RBF"),
        (cv2.ml.SVM_POLY, "Polynomial")
    ]
    
    results = {}
    
    for i, (kernel, name) in enumerate(kernels):
        print(f"\n🔄 {name} Kernel test ediliyor...")
        
        svm = SVMClassifier(kernel=kernel)
        if kernel == cv2.ml.SVM_POLY:
            svm.set_parameters(C=1.0, degree=3)
        else:
            svm.set_parameters(C=1.0, gamma=0.1)
        
        # Eğit
        training_time = svm.train(X_train, y_train)
        
        # Test et
        predictions, prediction_time = svm.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        results[name] = {
            'accuracy': accuracy,
            'time': training_time + prediction_time,
            'model': svm
        }
        
        print(f"   Doğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Karar sınırını görselleştir
        plt.subplot(1, 3, i + 2)
        visualize_svm_decision_boundary(X_train, y_train, svm, title=f'{name} Kernel')
    
    plt.tight_layout()
    plt.show()
    
    # En iyi modeli bul
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 En iyi model: {best_model[0]} (Doğruluk: {best_model[1]['accuracy']:.4f})")
    
    return results

def ornek_2_rakam_siniflandirma():
    """Örnek 2: SVM ile rakam sınıflandırma"""
    print("\n⚖️ Örnek 2: SVM ile Rakam Sınıflandırma")
    print("=" * 40)
    
    # Rakam veri setini yükle
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    print(f"📊 Veri seti: {X.shape[0]} örnek, {X.shape[1]} özellik, {len(np.unique(y))} sınıf")
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SVM modeli oluştur ve eğit
    svm = SVMClassifier(kernel=cv2.ml.SVM_RBF)
    svm.set_parameters(C=10.0, gamma=0.001)
    
    training_time = svm.train(X_train, y_train)
    predictions, prediction_time = svm.predict(X_test)
    
    # Sonuçları değerlendir
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n📊 SVM Rakam Sınıflandırma Sonuçları:")
    print(f"   Doğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Eğitim süresi: {training_time:.4f} saniye")
    print(f"   Tahmin süresi: {prediction_time:.4f} saniye")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('SVM Rakam Sınıflandırma - Confusion Matrix')
    plt.colorbar()
    
    # Eksen etiketleri
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    
    # Sayıları yazma
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.show()
    
    # Yanlış sınıflandırılan örnekleri göster
    wrong_indices = np.where(y_test != predictions)[0]
    
    if len(wrong_indices) > 0:
        print(f"\n❌ {len(wrong_indices)} yanlış sınıflandırma bulundu")
        
        # İlk 6 yanlış örneği göster
        n_show = min(6, len(wrong_indices))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Yanlış Sınıflandırılan Örnekler', fontsize=16)
        
        for i in range(n_show):
            row = i // 3
            col = i % 3
            
            idx = wrong_indices[i]
            image = X_test[idx].reshape(8, 8)
            true_label = y_test[idx]
            pred_label = predictions[idx]
            
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Gerçek: {true_label}, Tahmin: {pred_label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return svm, accuracy

def ornek_3_hiperparametre_optimizasyonu():
    """Örnek 3: SVM hiperparametre optimizasyonu"""
    print("\n⚖️ Örnek 3: SVM Hiperparametre Optimizasyonu")
    print("=" * 45)
    
    # Küçük veri seti ile hızlı test
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Manuel grid search (OpenCV SVM ile)
    C_values = [0.1, 1.0, 10.0, 100.0]
    gamma_values = [0.001, 0.01, 0.1, 1.0]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    print("🔄 Grid search başlıyor...")
    total_combinations = len(C_values) * len(gamma_values)
    current = 0
    
    for C in C_values:
        for gamma in gamma_values:
            current += 1
            print(f"   {current}/{total_combinations}: C={C}, gamma={gamma}")
            
            # SVM modeli oluştur
            svm = SVMClassifier(kernel=cv2.ml.SVM_RBF)
            svm.set_parameters(C=C, gamma=gamma)
            
            # Eğit ve test et
            try:
                svm.train(X_train, y_train)
                predictions, _ = svm.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                results.append({
                    'C': C,
                    'gamma': gamma,
                    'accuracy': accuracy
                })
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'C': C, 'gamma': gamma}
                
                print(f"     Doğruluk: {accuracy:.4f}")
                
            except Exception as e:
                print(f"     Hata: {e}")
    
    # Sonuçları görselleştir
    if results:
        results_array = np.array([[r['C'], r['gamma'], r['accuracy']] for r in results])
        
        # Heatmap için reshape
        C_unique = sorted(list(set([r['C'] for r in results])))
        gamma_unique = sorted(list(set([r['gamma'] for r in results])))
        
        accuracy_matrix = np.zeros((len(C_unique), len(gamma_unique)))
        
        for r in results:
            i = C_unique.index(r['C'])
            j = gamma_unique.index(r['gamma'])
            accuracy_matrix[i, j] = r['accuracy']
        
        plt.figure(figsize=(10, 8))
        plt.imshow(accuracy_matrix, interpolation='nearest', cmap='viridis')
        plt.colorbar(label='Doğruluk')
        plt.title('SVM Hiperparametre Optimizasyonu')
        
        # Eksen etiketleri
        plt.xticks(range(len(gamma_unique)), [f'{g:.3f}' for g in gamma_unique])
        plt.yticks(range(len(C_unique)), [f'{c:.1f}' for c in C_unique])
        plt.xlabel('Gamma')
        plt.ylabel('C')
        
        # En iyi parametreyi işaretle
        if best_params:
            best_i = C_unique.index(best_params['C'])
            best_j = gamma_unique.index(best_params['gamma'])
            plt.plot(best_j, best_i, 'r*', markersize=20, label=f'En iyi: C={best_params["C"]}, γ={best_params["gamma"]}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n🏆 En iyi parametreler: C={best_params.get('C', 'N/A')}, gamma={best_params.get('gamma', 'N/A')}")
    print(f"🏆 En iyi doğruluk: {best_accuracy:.4f}")
    
    return best_params, best_accuracy

def ornek_4_hog_insan_tespiti():
    """Örnek 4: HOG + SVM ile insan tespiti"""
    print("\n⚖️ Örnek 4: HOG + SVM İnsan Tespiti")
    print("=" * 35)
    
    detector = HOGPersonDetector()
    
    print("📹 Webcam'den insan tespiti başlıyor...")
    print("ESC: Çıkış, S: Screenshot")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # İnsan tespiti
        start_time = time.time()
        rectangles, weights = detector.detect_people_default(frame)
        detection_time = time.time() - start_time
        total_time += detection_time
        
        # Tespitleri çiz
        for i, (x, y, w, h) in enumerate(rectangles):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {i+1}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Performans bilgisi
        avg_fps = frame_count / total_time if total_time > 0 else 0
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Kisi sayisi: {len(rectangles)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'ESC: Cikis, S: Screenshot', (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('HOG + SVM Insan Tespiti', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s') or key == ord('S'):  # Screenshot
            filename = f'hog_detection_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot kaydedildi: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        print(f"\n📊 Performans İstatistikleri:")
        print(f"   Toplam frame: {frame_count}")
        print(f"   Ortalama FPS: {frame_count / total_time:.2f}")
        print(f"   Ortalama tespit süresi: {total_time / frame_count * 1000:.2f} ms")

def visualize_svm_decision_boundary(X_train, y_train, svm, title="SVM Decision Boundary", resolution=100):
    """SVM karar sınırını görselleştir"""
    # Veri aralığını belirle
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    
    # Grid oluştur
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Grid noktalarını tahmin et
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    
    try:
        predictions, _ = svm.predict(grid_points)
        predictions = predictions.reshape(xx.shape)
        
        # Karar sınırlarını çiz
        plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.RdYlBu)
        
        # Veri noktalarını çiz
        colors = ['red', 'blue']
        for i in range(2):
            mask = y_train == i
            plt.scatter(X_train[mask, 0], X_train[mask, 1], 
                       c=colors[i], label=f'Sınıf {i}', s=50, edgecolors='black')
        
        plt.xlabel('Özellik 1')
        plt.ylabel('Özellik 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"❌ Karar sınırları çizilemedi: {e}")

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("⚖️ SVM Nesne Tanıma Demo")
        print("="*50)
        print("1. ⚖️ Temel SVM Kullanımı")
        print("2. 🔢 SVM ile Rakam Sınıflandırma")  
        print("3. 🎯 Hiperparametre Optimizasyonu")
        print("4. 👤 HOG + SVM İnsan Tespiti")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-4): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_temel_svm()
            elif secim == "2":
                ornek_2_rakam_siniflandirma()
            elif secim == "3":
                ornek_3_hiperparametre_optimizasyonu()
            elif secim == "4":
                ornek_4_hog_insan_tespiti()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("⚖️ SVM Nesne Tanıma - OpenCV ML")
    print("Bu modül Support Vector Machine algoritmasını detaylı olarak öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Matplotlib (pip install matplotlib)")
    print("   - Scikit-learn (pip install scikit-learn)")
    print("\n📝 SVM Özellikleri:")
    print("   - Yüksek boyutlu verilerle etkili çalışır")
    print("   - Kernel trick ile non-linear sınıflandırma")
    print("   - Outlier'lara dayanıklı")
    print("   - Bellek verimli (sadece support vector'ları saklar)")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. SVM kernel seçimi verinin doğasına bağlıdır
# 2. RBF kernel çoğu durumda iyi performans gösterir
# 3. C parametresi: bias-variance trade-off kontrol eder
# 4. Gamma parametresi: RBF kernel'ın genişliğini kontrol eder
# 5. Feature scaling SVM için kritik öneme sahiptir