#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 k-NN Sınıflandırma - OpenCV Machine Learning
==============================================

Bu modül k-Nearest Neighbors algoritmasını detaylı olarak kapsar:
- k-NN algoritması temel prensipleri
- OpenCV k-NN implementasyonu
- El yazısı rakam tanıma
- Görüntü sınıflandırma
- k değeri optimizasyonu
- Distance metrics karşılaştırması

Yazan: Eren Terzi
Tarih: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import os

class KNNClassifier:
    """OpenCV k-NN Sınıflandırıcı"""
    
    def __init__(self, k=5):
        self.k = k
        self.model = cv2.ml.KNearest_create()
        self.model.setDefaultK(k)
        self.model.setIsClassifier(True)
        self.is_trained = False
        
    def train(self, X_train, y_train):
        """Modeli eğit"""
        print(f"🔄 k-NN modeli eğitiliyor (k={self.k})...")
        
        # Veri tiplerini kontrol et
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        
        start_time = time.time()
        self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        training_time = time.time() - start_time
        
        self.is_trained = True
        print(f"✅ Eğitim tamamlandı ({training_time:.4f} saniye)")
        
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
    
    def predict_with_neighbors(self, X_test):
        """Komşular ile birlikte tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        X_test = X_test.astype(np.float32)
        
        # En yakın komşuları bul
        _, results, neighbors, distances = self.model.findNearest(X_test, self.k)
        
        predictions = results.flatten().astype(np.int32)
        
        return predictions, neighbors, distances
    
    def set_k(self, k):
        """k değerini değiştir"""
        self.k = k
        self.model.setDefaultK(k)
        print(f"🔄 k değeri {k} olarak güncellendi")

class DigitRecognition:
    """El yazısı rakam tanıma sınıfı"""
    
    def __init__(self):
        self.digits_data = None
        self.knn = None
        
    def load_digits_dataset(self):
        """Rakam veri setini yükle"""
        print("📊 Rakam veri seti yükleniyor...")
        
        digits = load_digits()
        X = digits.data  # 8x8 = 64 özellik
        y = digits.target  # 0-9 rakamları
        
        print(f"   Veri boyutu: {X.shape}")
        print(f"   Sınıf sayısı: {len(np.unique(y))} (0-9 rakamları)")
        print(f"   Toplam örnek: {len(X)}")
        
        self.digits_data = {
            'data': X,
            'target': y,
            'images': digits.images
        }
        
        return X, y
    
    def visualize_digits(self, n_samples=10):
        """Örnek rakamları görselleştir"""
        if self.digits_data is None:
            self.load_digits_dataset()
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle('Örnek Rakam Görüntüleri', fontsize=16)
        
        for i in range(n_samples):
            row = i // 5
            col = i % 5
            
            image = self.digits_data['images'][i]
            label = self.digits_data['target'][i]
            
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Rakam: {label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def train_digit_classifier(self, k=5):
        """Rakam sınıflandırıcısını eğit"""
        if self.digits_data is None:
            self.load_digits_dataset()
        
        X = self.digits_data['data']
        y = self.digits_data['target']
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"🔄 Eğitim seti: {X_train.shape[0]}, Test seti: {X_test.shape[0]}")
        
        # k-NN modeli oluştur ve eğit
        self.knn = KNNClassifier(k=k)
        training_time = self.knn.train(X_train, y_train)
        
        # Test et
        predictions, prediction_time = self.knn.predict(X_test)
        
        # Değerlendir
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n📊 Rakam Tanıma Sonuçları:")
        print(f"   Doğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Eğitim süresi: {training_time:.4f} saniye")
        print(f"   Tahmin süresi: {prediction_time:.4f} saniye")
        
        return X_test, y_test, predictions, accuracy
    
    def predict_single_digit(self, digit_image):
        """Tek bir rakam görüntüsünü tahmin et"""
        if self.knn is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        # 8x8 görüntüyü 64 özelliğe çevir
        if digit_image.shape != (8, 8):
            digit_image = cv2.resize(digit_image, (8, 8))
        
        features = digit_image.flatten().reshape(1, -1)
        
        # Tahmin yap (komşular ile)
        predictions, neighbors, distances = self.knn.predict_with_neighbors(features)
        
        predicted_digit = predictions[0]
        
        return predicted_digit, neighbors[0], distances[0]

def ornek_1_temel_knn():
    """Örnek 1: Temel k-NN kullanımı"""
    print("\n🎯 Örnek 1: Temel k-NN Kullanımı")
    print("=" * 35)
    
    # Basit 2D veri oluştur
    np.random.seed(42)
    
    # 3 sınıf için merkez noktalar
    centers = np.array([[2, 2], [6, 6], [2, 6]])
    n_samples_per_class = 50
    
    X = []
    y = []
    
    for i, center in enumerate(centers):
        # Her sınıf için normal dağılımdan örnekler
        samples = np.random.normal(center, 0.8, (n_samples_per_class, 2))
        X.append(samples)
        y.extend([i] * n_samples_per_class)
    
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Veriyi görselleştir
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    for i in range(3):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'Sınıf {i}', alpha=0.7)
    
    plt.xlabel('Özellik 1')
    plt.ylabel('Özellik 2')
    plt.title('k-NN için Örnek Veri Seti')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # k-NN ile sınıflandır
    knn = KNNClassifier(k=5)
    knn.train(X_train, y_train)
    
    predictions, pred_time = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"📊 Temel k-NN Sonuçları:")
    print(f"   Doğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Tahmin süresi: {pred_time:.4f} saniye")
    
    # Karar sınırlarını görselleştir
    visualize_decision_boundary(X_train, y_train, knn)
    
    return knn, accuracy

def ornek_2_rakam_tanima():
    """Örnek 2: El yazısı rakam tanıma"""
    print("\n🎯 Örnek 2: El Yazısı Rakam Tanıma")
    print("=" * 35)
    
    digit_recognizer = DigitRecognition()
    
    # Örnek rakamları göster
    digit_recognizer.visualize_digits(10)
    
    # Modeli eğit ve test et
    X_test, y_test, predictions, accuracy = digit_recognizer.train_digit_classifier(k=3)
    
    # Yanlış sınıflandırılan örnekleri göster
    wrong_predictions = X_test[y_test != predictions]
    wrong_true_labels = y_test[y_test != predictions]
    wrong_pred_labels = predictions[y_test != predictions]
    
    if len(wrong_predictions) > 0:
        print(f"\n❌ Yanlış sınıflandırılan {len(wrong_predictions)} örnek bulundu")
        
        # İlk 6 yanlış örneği göster
        n_show = min(6, len(wrong_predictions))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('Yanlış Sınıflandırılan Örnekler', fontsize=16)
        
        for i in range(n_show):
            row = i // 3
            col = i % 3
            
            # 64 özelliği 8x8 görüntüye çevir
            image = wrong_predictions[i].reshape(8, 8)
            true_label = wrong_true_labels[i]
            pred_label = wrong_pred_labels[i]
            
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Gerçek: {true_label}, Tahmin: {pred_label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return digit_recognizer, accuracy

def ornek_3_k_optimizasyonu():
    """Örnek 3: k değeri optimizasyonu"""
    print("\n🎯 Örnek 3: k Değeri Optimizasyonu")
    print("=" * 35)
    
    # Rakam veri setini yükle
    digit_recognizer = DigitRecognition()
    X, y = digit_recognizer.load_digits_dataset()
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Farklı k değerlerini test et
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    accuracies = []
    times = []
    
    print("🔄 Farklı k değerleri test ediliyor...")
    
    for k in k_values:
        print(f"   k = {k}")
        
        knn = KNNClassifier(k=k)
        training_time = knn.train(X_train, y_train)
        predictions, prediction_time = knn.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        total_time = training_time + prediction_time
        
        accuracies.append(accuracy)
        times.append(total_time)
        
        print(f"     Doğruluk: {accuracy:.4f}, Süre: {total_time:.4f}s")
    
    # Sonuçları görselleştir
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Doğruluk grafiği
    ax1.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('k Değeri')
    ax1.set_ylabel('Doğruluk')
    ax1.set_title('k Değeri vs Doğruluk')
    ax1.grid(True, alpha=0.3)
    
    # En iyi k değerini işaretle
    best_k_idx = np.argmax(accuracies)
    best_k = k_values[best_k_idx]
    best_accuracy = accuracies[best_k_idx]
    
    ax1.plot(best_k, best_accuracy, 'ro', markersize=12, label=f'En iyi k={best_k}')
    ax1.legend()
    
    # Süre grafiği
    ax2.plot(k_values, times, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('k Değeri')
    ax2.set_ylabel('Toplam Süre (saniye)')
    ax2.set_title('k Değeri vs Hesaplama Süresi')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🏆 En iyi k değeri: {best_k} (Doğruluk: {best_accuracy:.4f})")
    
    return best_k, best_accuracy

def ornek_4_interaktif_tahmin():
    """Örnek 4: İnteraktif rakam tahmini"""
    print("\n🎯 Örnek 4: İnteraktif Rakam Tahmini")
    print("=" * 35)
    
    # Model eğit
    digit_recognizer = DigitRecognition()
    digit_recognizer.train_digit_classifier(k=3)
    
    print("\n🎨 İnteraktif Rakam Çizme Modu")
    print("Fare ile 8x8 alanda rakam çizin")
    print("SPACE: Tahmin yap, R: Temizle, ESC: Çıkış")
    
    # 8x8 çizim alanı (büyütülmüş)
    canvas_size = 400
    cell_size = canvas_size // 8
    
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, canvas
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # Çizgi çiz (kalın)
            cv2.circle(canvas, (x, y), 15, 255, -1)
    
    cv2.namedWindow('Rakam Ciz', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Rakam Ciz', mouse_callback)
    
    while True:
        # Canvas'ı göster
        display_canvas = canvas.copy()
        
        # Grid çizgileri ekle
        for i in range(1, 8):
            cv2.line(display_canvas, (i * cell_size, 0), (i * cell_size, canvas_size), 128, 1)
            cv2.line(display_canvas, (0, i * cell_size), (canvas_size, i * cell_size), 128, 1)
        
        cv2.imshow('Rakam Ciz', display_canvas)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE - Tahmin yap
            # 8x8'e küçült
            digit_8x8 = cv2.resize(canvas, (8, 8))
            
            if np.sum(digit_8x8) > 0:  # Boş değilse
                try:
                    predicted_digit, neighbors, distances = digit_recognizer.predict_single_digit(digit_8x8)
                    
                    print(f"\n🔮 Tahmin: {predicted_digit}")
                    print(f"   En yakın komşu mesafeleri: {distances[:3]}")
                    
                    # Çizilen rakamı göster
                    plt.figure(figsize=(10, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(canvas, cmap='gray')
                    plt.title(f'Çizilen Rakam')
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(digit_8x8, cmap='gray')
                    plt.title(f'Tahmin: {predicted_digit}')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                except Exception as e:
                    print(f"❌ Tahmin hatası: {e}")
            else:
                print("⚠️ Önce bir rakam çizin!")
        
        elif key == ord('r') or key == ord('R'):  # R - Temizle
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            print("🧹 Canvas temizlendi")
    
    cv2.destroyAllWindows()

def visualize_decision_boundary(X_train, y_train, knn, resolution=100):
    """Karar sınırlarını görselleştir"""
    print("🎨 Karar sınırları çiziliyor...")
    
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
        predictions, _ = knn.predict(grid_points)
        predictions = predictions.reshape(xx.shape)
        
        # Karar sınırlarını çiz
        plt.figure(figsize=(12, 10))
        
        # Arka plan renkleri
        plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.RdYlBu)
        
        # Veri noktalarını çiz
        colors = ['red', 'blue', 'green']
        for i in range(3):
            mask = y_train == i
            plt.scatter(X_train[mask, 0], X_train[mask, 1], 
                       c=colors[i], label=f'Sınıf {i}', s=50, edgecolors='black')
        
        plt.xlabel('Özellik 1')
        plt.ylabel('Özellik 2')
        plt.title(f'k-NN Karar Sınırları (k={knn.k})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"❌ Karar sınırları çizilemedi: {e}")

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🎯 k-NN Sınıflandırma Demo")
        print("="*50)
        print("1. 🎯 Temel k-NN Kullanımı")
        print("2. 🔢 El Yazısı Rakam Tanıma")  
        print("3. ⚖️ k Değeri Optimizasyonu")
        print("4. 🎨 İnteraktif Rakam Tahmini")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-4): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_temel_knn()
            elif secim == "2":
                ornek_2_rakam_tanima()
            elif secim == "3":
                ornek_3_k_optimizasyonu()
            elif secim == "4":
                ornek_4_interaktif_tahmin()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🎯 k-NN Sınıflandırma - OpenCV ML")
    print("Bu modül k-Nearest Neighbors algoritmasını detaylı olarak öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Matplotlib (pip install matplotlib)")
    print("   - Scikit-learn (pip install scikit-learn)")
    print("\n📝 k-NN Özellikleri:")
    print("   - Lazy learning algoritması (eğitim fazı yok)")
    print("   - k değeri performansı büyük ölçüde etkiler")
    print("   - Veri boyutu arttıkça yavaşlar")
    print("   - Kategorik ve sürekli verilerle çalışır")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. k-NN lazy learning algoritmasıdır (eğitim fazı yoktur)
# 2. k değeri tek sayı olmalı (eşitlik durumunu önlemek için)
# 3. Mesafe metrikleri: Euclidean, Manhattan, Minkowski
# 4. Veri normalizasyonu k-NN için çok önemlidir
# 5. Büyük veri setlerinde KD-tree veya Ball-tree kullanın