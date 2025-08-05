#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ANN Karakter Tanıma - Artificial Neural Networks
==================================================

Bu modül Yapay Sinir Ağları (ANN) ile karakter tanımayı kapsar:
- OpenCV ANN_MLP implementasyonu
- Multi-layer perceptron (MLP) yapısı
- Backpropagation algoritması
- El yazısı karakter tanıma
- Aktivasyon fonksiyonları
- Ağ mimarisi tasarımı

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import time
import os

class ANNClassifier:
    """OpenCV ANN (Multi-Layer Perceptron) Sınıflandırıcı"""
    
    def __init__(self, hidden_layers=[100], activation_func=cv2.ml.ANN_MLP_SIGMOID_SYM):
        self.model = cv2.ml.ANN_MLP_create()
        self.hidden_layers = hidden_layers
        self.activation_func = activation_func
        self.is_trained = False
        self.scaler = StandardScaler()
        self.label_binarizer = LabelBinarizer()
        
        # Activation fonksiyonlarını ayarla
        self.model.setActivationFunction(activation_func)
        
        print(f"🧠 ANN oluşturuldu: Gizli katmanlar={hidden_layers}")
        
    def create_network_architecture(self, input_size, output_size):
        """Ağ mimarisini oluştur"""
        # Katman boyutları: [input, hidden_layers..., output]
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        layer_sizes = np.array(layer_sizes, dtype=np.int32)
        
        print(f"📐 Ağ mimarisi: {' -> '.join(map(str, layer_sizes))}")
        
        # Ağ topolojisini ayarla
        self.model.setLayerSizes(layer_sizes)
        
        return layer_sizes
    
    def set_training_parameters(self, learning_rate=0.1, momentum=0.9, max_iterations=1000):
        """Eğitim parametrelerini ayarla"""
        print(f"🔧 Eğitim parametreleri: lr={learning_rate}, momentum={momentum}, max_iter={max_iterations}")
        
        # Backpropagation parametreleri
        self.model.setBackpropMomentumScale(momentum)
        self.model.setBackpropWeightScale(learning_rate)
        
        # Durdurma kriterleri
        term_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iterations, 1e-6)
        self.model.setTermCriteria(term_criteria)
        
        # Eğitim metodunu ayarla
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    
    def train(self, X_train, y_train, normalize=True):
        """Modeli eğit"""
        print("🔄 ANN modeli eğitiliyor...")
        
        # Veri tiplerini kontrol et
        X_train = X_train.astype(np.float32)
        
        # Normalizasyon
        if normalize:
            X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        
        # Label encoding (one-hot)
        y_train_encoded = self.label_binarizer.fit_transform(y_train).astype(np.float32)
        
        # Eğer sadece 2 sınıf varsa, bir boyut ekle
        if y_train_encoded.ndim == 1:
            y_train_encoded = y_train_encoded.reshape(-1, 1)
        
        print(f"📊 Eğitim verisi: {X_train.shape}, Etiketler: {y_train_encoded.shape}")
        
        # Ağ mimarisini oluştur
        input_size = X_train.shape[1]
        output_size = y_train_encoded.shape[1]
        self.create_network_architecture(input_size, output_size)
        
        start_time = time.time()
        
        # Modeli eğit
        success = self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train_encoded)
        
        training_time = time.time() - start_time
        
        if success:
            self.is_trained = True
            print(f"✅ Eğitim tamamlandı ({training_time:.4f} saniye)")
        else:
            print("❌ Eğitim başarısız!")
            return None
        
        return training_time
    
    def predict(self, X_test, normalize=True):
        """Tahmin yap"""
        if not self.is_trained:
            raise ValueError("Model henüz eğitilmedi!")
        
        X_test = X_test.astype(np.float32)
        
        # Normalizasyon
        if normalize:
            X_test = self.scaler.transform(X_test).astype(np.float32)
        
        start_time = time.time()
        
        # Tahmin yap
        _, raw_predictions = self.model.predict(X_test)
        
        prediction_time = time.time() - start_time
        
        # One-hot encoded tahminleri sınıf etiketlerine çevir
        if raw_predictions.shape[1] == 1:
            # Binary classification
            predictions = (raw_predictions > 0).astype(int).flatten()
        else:
            # Multi-class classification
            predictions = np.argmax(raw_predictions, axis=1)
        
        # Label binarizer ile orijinal etiketlere dönüştür
        predictions = self.label_binarizer.inverse_transform(predictions)
        
        return predictions, prediction_time, raw_predictions

class HandwrittenCharacterRecognition:
    """El yazısı karakter tanıma sınıfı"""
    
    def __init__(self):
        self.ann = None
        self.characters_data = None
        
    def load_mnist_dataset(self, subset_size=None):
        """MNIST veri setini yükle"""
        print("📊 MNIST veri seti yükleniyor...")
        
        try:
            # OpenML'den MNIST yükle
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X = mnist.data.astype(np.float32) / 255.0  # Normalize to [0,1]
            y = mnist.target.astype(int)
            
            if subset_size:
                # Daha hızlı test için alt küme
                indices = np.random.choice(len(X), subset_size, replace=False)
                X = X[indices]
                y = y[indices]
            
            print(f"   Veri boyutu: {X.shape}")
            print(f"   Sınıf sayısı: {len(np.unique(y))} (0-9 rakamları)")
            print(f"   Toplam örnek: {len(X)}")
            
            self.characters_data = {
                'data': X,
                'target': y,
                'images': X.reshape(-1, 28, 28)
            }
            
            return X, y
            
        except Exception as e:
            print(f"❌ MNIST yüklenirken hata: {e}")
            print("🔄 Yerel digits veri seti kullanılıyor...")
            
            # Fallback: sklearn digits
            digits = load_digits()
            X = digits.data.astype(np.float32) / 16.0  # Normalize to [0,1]
            y = digits.target
            
            self.characters_data = {
                'data': X,
                'target': y,
                'images': digits.images
            }
            
            print(f"   Veri boyutu: {X.shape}")
            print(f"   Sınıf sayısı: {len(np.unique(y))} (0-9 rakamları)")
            
            return X, y
    
    def visualize_characters(self, n_samples=10):
        """Örnek karakterleri görselleştir"""
        if self.characters_data is None:
            self.load_mnist_dataset()
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle('Örnek El Yazısı Karakterler', fontsize=16)
        
        for i in range(n_samples):
            row = i // 5
            col = i % 5
            
            image = self.characters_data['images'][i]
            label = self.characters_data['target'][i]
            
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Karakter: {label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def train_character_classifier(self, hidden_layers=[128, 64], max_samples=10000):
        """Karakter sınıflandırıcısını eğit"""
        if self.characters_data is None:
            # Küçük subset ile hızlı test
            self.load_mnist_dataset(subset_size=max_samples)
        
        X = self.characters_data['data']
        y = self.characters_data['target']
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"🔄 Eğitim seti: {X_train.shape[0]}, Test seti: {X_test.shape[0]}")
        
        # ANN modeli oluştur
        self.ann = ANNClassifier(
            hidden_layers=hidden_layers,
            activation_func=cv2.ml.ANN_MLP_SIGMOID_SYM
        )
        
        # Eğitim parametrelerini ayarla  
        self.ann.set_training_parameters(
            learning_rate=0.01,
            momentum=0.9,
            max_iterations=500
        )
        
        # Modeli eğit
        training_time = self.ann.train(X_train, y_train)
        
        if training_time is None:
            return None, None, None, None
        
        # Test et
        predictions, prediction_time, raw_predictions = self.ann.predict(X_test)
        
        # Değerlendir
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n📊 Karakter Tanıma Sonuçları:")
        print(f"   Doğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Eğitim süresi: {training_time:.4f} saniye")
        print(f"   Tahmin süresi: {prediction_time:.4f} saniye")
        
        return X_test, y_test, predictions, accuracy
    
    def predict_single_character(self, char_image):
        """Tek bir karakter görüntüsünü tahmin et"""
        if self.ann is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        # Görüntüyü model girişine uygun forma getir
        if len(char_image.shape) == 2:
            # 2D görüntüyü flatten et
            if char_image.shape == (28, 28):
                features = char_image.flatten() / 255.0
            elif char_image.shape == (8, 8):
                features = char_image.flatten() / 16.0
            else:
                # Resize gerekli
                char_image = cv2.resize(char_image, (28, 28))
                features = char_image.flatten() / 255.0
        else:
            features = char_image.flatten()
        
        features = features.reshape(1, -1)
        
        # Tahmin yap
        predictions, _, raw_predictions = self.ann.predict(features)
        
        predicted_char = predictions[0]
        confidence = np.max(raw_predictions[0])
        
        return predicted_char, confidence, raw_predictions[0]

def ornek_1_temel_ann():
    """Örnek 1: Temel ANN kullanımı"""
    print("\n🧠 Örnek 1: Temel ANN Kullanımı")
    print("=" * 30)
    
    # Basit XOR problemi
    print("🔄 XOR problemi ile ANN testi...")
    
    # XOR veri seti
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)  # XOR çıkışı
    
    print("📊 XOR Veri Seti:")
    for i in range(len(X)):
        print(f"   {X[i]} -> {y[i]}")
    
    # ANN modeli oluştur
    ann = ANNClassifier(hidden_layers=[4, 3], activation_func=cv2.ml.ANN_MLP_SIGMOID_SYM)
    
    # Eğitim parametreleri
    ann.set_training_parameters(learning_rate=0.1, momentum=0.9, max_iterations=10000)
    
    # Modeli eğit
    training_time = ann.train(X, y, normalize=False)
    
    if training_time is None:
        print("❌ XOR problemi çözülemedi!")
        return None
    
    # Test et
    predictions, prediction_time, raw_predictions = ann.predict(X, normalize=False)
    
    print(f"\n📊 XOR Sonuçları:")
    print("Girdi -> Gerçek | Tahmin | Ham Çıkış")
    print("-" * 35)
    
    for i in range(len(X)):
        print(f"{X[i]} -> {y[i]:6d} | {predictions[i]:6d} | {raw_predictions[i][0]:8.4f}")
    
    accuracy = accuracy_score(y, predictions)
    print(f"\nDoğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return ann, accuracy

def ornek_2_rakam_tanima():
    """Örnek 2: ANN ile rakam tanıma"""
    print("\n🧠 Örnek 2: ANN ile Rakam Tanıma")
    print("=" * 30)
    
    char_recognizer = HandwrittenCharacterRecognition()
    
    # Örnek karakterleri göster
    char_recognizer.visualize_characters(10)
    
    # Modeli eğit ve test et
    X_test, y_test, predictions, accuracy = char_recognizer.train_character_classifier(
        hidden_layers=[100, 50],
        max_samples=5000  # Hızlı test için
    )
    
    if accuracy is None:
        print("❌ Model eğitilemedi!")
        return None
    
    # Yanlış sınıflandırılan örnekleri göster
    if X_test is not None and y_test is not None:
        wrong_mask = y_test != predictions
        wrong_predictions = X_test[wrong_mask]
        wrong_true_labels = y_test[wrong_mask]
        wrong_pred_labels = predictions[wrong_mask]
        
        if len(wrong_predictions) > 0:
            print(f"\n❌ Yanlış sınıflandırılan {len(wrong_predictions)} örnek bulundu")
            
            # İlk 6 yanlış örneği göster
            n_show = min(6, len(wrong_predictions))
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle('Yanlış Sınıflandırılan Örnekler', fontsize=16)
            
            for i in range(n_show):
                row = i // 3
                col = i % 3
                
                # Görüntüyü yeniden şekillendir
                if wrong_predictions[i].shape[0] == 784:  # 28x28 MNIST
                    image = wrong_predictions[i].reshape(28, 28)
                else:  # 8x8 digits
                    image = wrong_predictions[i].reshape(8, 8)
                
                true_label = wrong_true_labels[i]
                pred_label = wrong_pred_labels[i]
                
                axes[row, col].imshow(image, cmap='gray')
                axes[row, col].set_title(f'Gerçek: {true_label}, Tahmin: {pred_label}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    return char_recognizer, accuracy

def ornek_3_ann_mimarisi_karsilastirma():
    """Örnek 3: Farklı ANN mimarilerini karşılaştırma"""
    print("\n🧠 Örnek 3: ANN Mimarisi Karşılaştırması")
    print("=" * 40)
    
    # Küçük veri seti ile hızlı test
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16.0
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Farklı mimari konfigürasyonları
    architectures = [
        ([32], "Tek Gizli Katman (32)"),
        ([64], "Tek Gizli Katman (64)"),
        ([32, 16], "İki Gizli Katman (32-16)"),
        ([64, 32], "İki Gizli Katman (64-32)"),
        ([100, 50, 25], "Üç Gizli Katman (100-50-25)")
    ]
    
    results = []
    
    print("🔄 Farklı mimariler test ediliyor...")
    
    for hidden_layers, name in architectures:
        print(f"\n📐 {name} test ediliyor...")
        
        # ANN modeli oluştur
        ann = ANNClassifier(hidden_layers=hidden_layers)
        ann.set_training_parameters(learning_rate=0.01, max_iterations=300)
        
        # Eğit
        start_time = time.time()
        training_time = ann.train(X_train, y_train)
        
        if training_time is None:
            print(f"   ❌ {name} eğitilemedi!")
            continue
        
        # Test et
        predictions, prediction_time, _ = ann.predict(X_test)
        total_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, predictions)
        
        results.append({
            'name': name,
            'hidden_layers': hidden_layers,
            'accuracy': accuracy,
            'time': total_time,
            'params': sum(hidden_layers) + len(hidden_layers) * 64 + 10 * hidden_layers[-1]  # Yaklaşık parametre sayısı
        })
        
        print(f"   Doğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Toplam süre: {total_time:.2f} saniye")
    
    # Sonuçları görselleştir
    if results:
        names = [r['name'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        times = [r['time'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Doğruluk karşılaştırması
        bars1 = ax1.bar(range(len(names)), accuracies, color='skyblue')
        ax1.set_xlabel('Mimari')
        ax1.set_ylabel('Doğruluk')
        ax1.set_title('ANN Mimarisi vs Doğruluk')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([n.split('(')[0] for n in names], rotation=45, ha='right')
        
        # Değerleri çubukların üzerine yaz
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{accuracies[i]:.3f}', ha='center', va='bottom')
        
        # Süre karşılaştırması
        bars2 = ax2.bar(range(len(names)), times, color='lightcoral')
        ax2.set_xlabel('Mimari')
        ax2.set_ylabel('Süre (saniye)')
        ax2.set_title('ANN Mimarisi vs Eğitim Süresi')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels([n.split('(')[0] for n in names], rotation=45, ha='right')
        
        # Değerleri çubukların üzerine yaz
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{times[i]:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # En iyi modeli bul
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\n🏆 En iyi mimari: {best_result['name']}")
        print(f"   Doğruluk: {best_result['accuracy']:.4f}")
        print(f"   Süre: {best_result['time']:.2f} saniye")
    
    return results

def ornek_4_interaktif_karakter_tanima():
    """Örnek 4: İnteraktif karakter tanıma"""
    print("\n🧠 Örnek 4: İnteraktif Karakter Tanıma")
    print("=" * 35)
    
    # Model eğit
    char_recognizer = HandwrittenCharacterRecognition()
    char_recognizer.train_character_classifier(
        hidden_layers=[64, 32],
        max_samples=3000
    )
    
    if char_recognizer.ann is None:
        print("❌ Model eğitilemedi!")
        return
    
    print("\n🎨 İnteraktif Karakter Çizme Modu")
    print("Fare ile 28x28 alanda karakter çizin")
    print("SPACE: Tahmin yap, R: Temizle, ESC: Çıkış")
    
    # 28x28 çizim alanı (büyütülmüş)
    canvas_size = 280  # 28x28 * 10
    cell_size = canvas_size // 28
    
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
            cv2.circle(canvas, (x, y), 8, 255, -1)
    
    cv2.namedWindow('Karakter Ciz', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Karakter Ciz', mouse_callback)
    
    while True:
        # Canvas'ı göster
        display_canvas = canvas.copy()
        
        cv2.imshow('Karakter Ciz', display_canvas)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE - Tahmin yap
            # 28x28'e küçült
            char_28x28 = cv2.resize(canvas, (28, 28))
            
            if np.sum(char_28x28) > 0:  # Boş değilse
                try:
                    predicted_char, confidence, raw_predictions = char_recognizer.predict_single_character(char_28x28)
                    
                    print(f"\n🔮 Tahmin: {predicted_char}")
                    print(f"   Güven skoru: {confidence:.4f}")
                    
                    # En yüksek 3 tahmini göster
                    sorted_indices = np.argsort(raw_predictions)[::-1][:3]
                    print("   En yüksek tahminler:")
                    for i, idx in enumerate(sorted_indices):
                        print(f"     {i+1}. {idx}: {raw_predictions[idx]:.4f}")
                    
                    # Çizilen karakteri göster
                    plt.figure(figsize=(10, 4))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(canvas, cmap='gray')
                    plt.title(f'Çizilen Karakter - {canvas_size}x{canvas_size}')
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(char_28x28, cmap='gray')
                    plt.title(f'Tahmin: {predicted_char} (Güven: {confidence:.3f})')
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                except Exception as e:
                    print(f"❌ Tahmin hatası: {e}")
            else:
                print("⚠️ Önce bir karakter çizin!")
        
        elif key == ord('r') or key == ord('R'):  # R - Temizle
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            print("🧹 Canvas temizlendi")
    
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("🧠 ANN Karakter Tanıma Demo")
        print("="*50)
        print("1. 🧠 Temel ANN Kullanımı (XOR)")
        print("2. 🔢 ANN ile Rakam Tanıma")  
        print("3. 📐 ANN Mimarisi Karşılaştırması")
        print("4. 🎨 İnteraktif Karakter Tanıma")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-4): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_temel_ann()
            elif secim == "2":
                ornek_2_rakam_tanima()
            elif secim == "3":
                ornek_3_ann_mimarisi_karsilastirma()
            elif secim == "4":
                ornek_4_interaktif_karakter_tanima()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-4 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("🧠 ANN Karakter Tanıma - OpenCV ML")
    print("Bu modül Yapay Sinir Ağları ile karakter tanımayı öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Matplotlib (pip install matplotlib)")
    print("   - Scikit-learn (pip install scikit-learn)")
    print("\n📝 ANN Özellikleri:")
    print("   - Non-linear pattern recognition")
    print("   - Backpropagation ile öğrenme")
    print("   - Gizli katman sayısı performansı etkiler")
    print("   - Overfitting'e eğilimli, regularization gerekebilir")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. ANN eğitimi iteratif bir süreçtir, sabır gerektirir
# 2. Learning rate çok yüksekse model yakınsamaz
# 3. Gizli katman sayısı ve nöron sayısı deneyerek bulunur
# 4. Aktivasyon fonksiyonu seçimi kritik öneme sahiptir
# 5. Veri normalizasyonu ANN için zorunludur