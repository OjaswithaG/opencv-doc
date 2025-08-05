"""
Derin Öğrenme ve OpenCV Entegrasyonu
CNN'ler, Transfer Learning ve Gerçek Zamanlı Uygulamalar

Bu dosya, derin öğrenme tekniklerini OpenCV ile entegre ederek
gerçek zamanlı uygulamalar geliştirmeyi gösterir.

Yazar: Eren Terzi
Tarih: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.utils import to_categorical
import time
import os
import warnings
warnings.filterwarnings('ignore')

class DeepLearningOpenCV:
    """Derin öğrenme ve OpenCV entegrasyonu demo sınıfı"""
    
    def __init__(self):
        """Deep learning demo sınıfını başlat"""
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.input_shape = (224, 224, 3)
        
    def veri_olustur(self, n_samples=1000, img_size=(64, 64)):
        """Sentetik görüntü verisi oluştur"""
        print("📊 Sentetik görüntü verisi oluşturuluyor...")
        
        # Basit şekiller oluştur
        X = []
        y = []
        
        for i in range(n_samples):
            # Boş görüntü
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            
            # Rastgele şekil seç
            shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
            
            if shape_type == 'circle':
                # Daire çiz
                center = (np.random.randint(20, img_size[0]-20), 
                         np.random.randint(20, img_size[1]-20))
                radius = np.random.randint(10, 20)
                color = (np.random.randint(0, 255), 
                        np.random.randint(0, 255), 
                        np.random.randint(0, 255))
                cv2.circle(img, center, radius, color, -1)
                
            elif shape_type == 'rectangle':
                # Dikdörtgen çiz
                x1 = np.random.randint(10, img_size[0]-30)
                y1 = np.random.randint(10, img_size[1]-30)
                x2 = x1 + np.random.randint(20, 40)
                y2 = y1 + np.random.randint(20, 40)
                color = (np.random.randint(0, 255), 
                        np.random.randint(0, 255), 
                        np.random.randint(0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                
            else:  # triangle
                # Üçgen çiz
                pts = np.array([
                    [np.random.randint(10, img_size[0]-10), np.random.randint(10, img_size[1]-10)],
                    [np.random.randint(10, img_size[0]-10), np.random.randint(10, img_size[1]-10)],
                    [np.random.randint(10, img_size[0]-10), np.random.randint(10, img_size[1]-10)]
                ], np.int32)
                color = (np.random.randint(0, 255), 
                        np.random.randint(0, 255), 
                        np.random.randint(0, 255))
                cv2.fillPoly(img, [pts], color)
            
            # Gürültü ekle
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            X.append(img)
            y.append(shape_type)
        
        X = np.array(X)
        y = np.array(y)
        
        # Etiketleri encode et
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )
        
        # Görüntüleri normalize et ve boyutlandır
        X_train = self.preprocess_images(X_train)
        X_test = self.preprocess_images(X_test)
        
        print(f"✅ Veri oluşturuldu: {len(X_train)} eğitim, {len(X_test)} test örneği")
        print(f"📈 Sınıf sayısı: {len(self.class_names)}")
        print(f"🏷️ Sınıflar: {list(self.class_names)}")
        
        return X_train, X_test, y_train, y_test
        
    def preprocess_images(self, images):
        """Görüntüleri ön işleme"""
        processed = []
        for img in images:
            # Boyutlandır
            resized = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            # Normalize et
            normalized = resized.astype(np.float32) / 255.0
            processed.append(normalized)
        return np.array(processed)
        
    def basit_cnn_olustur(self):
        """Basit CNN modeli oluştur"""
        print("\n🧠 BASİT CNN MODELİ OLUŞTURULUYOR")
        print("=" * 50)
        
        model = models.Sequential([
            # Giriş katmanı
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # İkinci konvolüsyon bloğu
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Üçüncü konvolüsyon bloğu
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Düzleştirme
            layers.Flatten(),
            
            # Yoğun katmanlar
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Modeli derle
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("📋 Model Özeti:")
        model.summary()
        
        self.model = model
        return model
        
    def transfer_learning_ornegi(self):
        """Transfer learning örneği"""
        print("\n🔄 TRANSFER LEARNING ÖRNEĞİ")
        print("=" * 50)
        
        # Pre-trained model (VGG16)
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Base model'i dondur
        base_model.trainable = False
        
        # Yeni model oluştur
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Modeli derle
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("📋 Transfer Learning Model Özeti:")
        model.summary()
        
        self.model = model
        return model
        
    def model_egit(self, X_train, y_train, X_test, y_test, epochs=10):
        """Modeli eğit"""
        print(f"\n🎯 MODEL EĞİTİMİ ({epochs} epoch)")
        print("=" * 50)
        
        if self.model is None:
            print("❌ Önce model oluşturun!")
            return None
            
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Eğitim
        start_time = time.time()
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        training_time = time.time() - start_time
        
        print(f"⏱️ Eğitim süresi: {training_time:.2f} saniye")
        
        # Test performansı
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"🎯 Test Doğruluğu: {test_accuracy:.4f}")
        print(f"📉 Test Kaybı: {test_loss:.4f}")
        
        return history
        
    def gercek_zamanli_tahmin(self):
        """Gerçek zamanlı tahmin demo"""
        print("\n🎥 GERÇEK ZAMANLI TAHMİN DEMO")
        print("=" * 50)
        
        if self.model is None:
            print("❌ Önce modeli eğitin!")
            return
            
        # Webcam başlat
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
            
        print("📹 Webcam başlatıldı. Çıkmak için 'q' tuşuna basın.")
        print("🎨 Çizim yapmak için mouse'u kullanın.")
        
        # Çizim için değişkenler
        drawing = False
        last_x, last_y = -1, -1
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, last_x, last_y
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                last_x, last_y = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.line(canvas, (last_x, last_y), (x, y), (255, 255, 255), 5)
                    last_x, last_y = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                
        cv2.namedWindow('Deep Learning Demo')
        cv2.setMouseCallback('Deep Learning Demo', mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Canvas'ı frame'e ekle
            display_frame = frame.copy()
            
            # Canvas'tan tahmin yap
            if np.sum(canvas) > 0:  # Canvas boş değilse
                # Canvas'ı model giriş formatına çevir
                canvas_resized = cv2.resize(canvas, (self.input_shape[0], self.input_shape[1]))
                canvas_normalized = canvas_resized.astype(np.float32) / 255.0
                canvas_input = np.expand_dims(canvas_normalized, axis=0)
                
                # Tahmin yap
                prediction = self.model.predict(canvas_input, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                class_name = self.class_names[predicted_class]
                
                # Sonucu göster
                cv2.putText(display_frame, f"Tahmin: {class_name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Guven: {confidence:.3f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Canvas'ı frame'e ekle
            canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
            display_frame = cv2.addWeighted(display_frame, 0.7, canvas_resized, 0.3, 0)
            
            # Kontrolleri göster
            cv2.putText(display_frame, "Cizim yapmak icin mouse kullanin", 
                       (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Temizlemek icin 'c', Cikmak icin 'q'", 
                       (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Deep Learning Demo', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                
        cap.release()
        cv2.destroyAllWindows()
        
    def model_performans_analizi(self, X_test, y_test):
        """Model performans analizi"""
        print("\n📊 MODEL PERFORMANS ANALİZİ")
        print("=" * 50)
        
        if self.model is None:
            print("❌ Model bulunamadı!")
            return
            
        # Tahminler
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Doğruluk
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Genel Doğruluk: {accuracy:.4f}")
        
        # Sınıflandırma raporu
        print("\n📊 Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix görselleştirme
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Değerleri göster
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('Gerçek Sınıf')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.tight_layout()
        plt.show()
        
    def model_kaydet_yukle(self, model_path="deep_learning_model.h5"):
        """Modeli kaydet ve yükle"""
        print(f"\n💾 MODEL KAYDETME/YÜKLEME")
        print("=" * 50)
        
        if self.model is None:
            print("❌ Kaydedilecek model bulunamadı!")
            return
            
        # Modeli kaydet
        self.model.save(model_path)
        print(f"✅ Model kaydedildi: {model_path}")
        
        # Modeli yükle
        loaded_model = tf.keras.models.load_model(model_path)
        print(f"✅ Model yüklendi: {model_path}")
        
        return loaded_model
        
    def gercek_dunya_ornegi(self):
        """Gerçek dünya derin öğrenme örneği"""
        print("\n🌍 GERÇEK DÜNYA ÖRNEĞİ: Görüntü Sınıflandırma Sistemi")
        print("=" * 70)
        
        # Veri oluştur
        X_train, X_test, y_train, y_test = self.veri_olustur(n_samples=500)
        
        # Model oluştur ve eğit
        self.basit_cnn_olustur()
        history = self.model_egit(X_train, y_train, X_test, y_test, epochs=5)
        
        # Performans analizi
        self.model_performans_analizi(X_test, y_test)
        
        # Model kaydet
        self.model_kaydet_yukle()
        
        print("\n🎉 Gerçek dünya örneği tamamlandı!")
        print("📚 Öğrenilen konular:")
        print("  • CNN mimarisi tasarımı")
        print("  • Transfer learning uygulaması")
        print("  • Gerçek zamanlı tahmin")
        print("  • Model performans analizi")
        print("  • Model kaydetme/yükleme")

def demo_menu():
    """Demo menüsü"""
    demo = DeepLearningOpenCV()
    
    while True:
        print("\n" + "="*60)
        print("🧠 DEEP LEARNING VE OPENCV DEMO MENÜSÜ")
        print("="*60)
        print("1. 📊 Veri Oluşturma")
        print("2. 🧠 Basit CNN Modeli")
        print("3. 🔄 Transfer Learning")
        print("4. 🎯 Model Eğitimi")
        print("5. 🎥 Gerçek Zamanlı Tahmin")
        print("6. 📊 Performans Analizi")
        print("7. 💾 Model Kaydetme/Yükleme")
        print("8. 🌍 Gerçek Dünya Örneği")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-8): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                demo.veri_olustur()
            elif secim == "2":
                demo.basit_cnn_olustur()
            elif secim == "3":
                demo.transfer_learning_ornegi()
            elif secim == "4":
                if demo.model is None:
                    print("❌ Önce model oluşturun! (Seçenek 2 veya 3)")
                else:
                    # Veri oluştur (eğer yoksa)
                    if not hasattr(demo, 'X_train') or demo.X_train is None:
                        demo.veri_olustur()
                    demo.model_egit(demo.X_train, demo.y_train, demo.X_test, demo.y_test)
            elif secim == "5":
                if demo.model is None:
                    print("❌ Önce modeli eğitin! (Seçenek 4)")
                else:
                    demo.gercek_zamanli_tahmin()
            elif secim == "6":
                if demo.model is None:
                    print("❌ Önce modeli eğitin! (Seçenek 4)")
                else:
                    demo.model_performans_analizi(demo.X_test, demo.y_test)
            elif secim == "7":
                demo.model_kaydet_yukle()
            elif secim == "8":
                demo.gercek_dunya_ornegi()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-8 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n👋 Program sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    print("🧠 Deep Learning ve OpenCV Entegrasyonu")
    print("=" * 60)
    print("Bu demo, derin öğrenme tekniklerini OpenCV ile entegre eder.")
    print("📚 Öğrenilecek Konular:")
    print("  • CNN (Convolutional Neural Networks)")
    print("  • Transfer Learning")
    print("  • Gerçek Zamanlı Tahmin")
    print("  • Model Performans Analizi")
    print("  • Model Kaydetme/Yükleme")
    print("  • Gerçek Dünya Uygulamaları")
    
    demo_menu() 