"""
Yüz Tanıma ve Duygu Analizi Sistemi
OpenCV ile Yüz Tanıma ve Duygu Analizi Projesi

Bu proje, yüz tanıma ve duygu analizi tekniklerini kullanarak
gerçek zamanlı bir sistem geliştirir.

Yazar: OpenCV Türkiye Topluluğu
Tarih: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

class YuzTanimaSistemi:
    """Yüz tanıma ve duygu analizi sistemi"""
    
    def __init__(self):
        """Sistem başlatma"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Yüz tanıma için LBPH recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Duygu analizi için model
        self.emotion_model = None
        self.emotion_labels = ['Mutlu', 'Uzgun', 'Sinirli', 'Saskin', 'Korkmus', 'Igrenmis', 'Nostral']
        
        # Kullanıcı veritabanı
        self.users = {}
        self.current_user = None
        
    def yuz_tespit(self, frame):
        """Yüz tespit et"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces, gray
        
    def goz_tespit(self, frame, face_roi):
        """Göz tespit et"""
        x, y, w, h = face_roi
        roi_gray = frame[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        return eyes
        
    def gulumseme_tespit(self, frame, face_roi):
        """Gülümseme tespit et"""
        x, y, w, h = face_roi
        roi_gray = frame[y:y+h, x:x+w]
        smiles = self.smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=20
        )
        return smiles
        
    def yuz_ozellikleri_cikar(self, face_roi, gray):
        """Yüz özelliklerini çıkar"""
        x, y, w, h = face_roi
        face_roi_gray = gray[y:y+h, x:x+w]
        
        # Yüz özellikleri
        features = []
        
        # Histogram özellikleri
        hist = cv2.calcHist([face_roi_gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist[:50])  # İlk 50 değer
        
        # LBP özellikleri (basitleştirilmiş)
        lbp = self.lbp_ozellikleri(face_roi_gray)
        features.extend(lbp)
        
        # Geometrik özellikler
        aspect_ratio = w / h
        features.append(aspect_ratio)
        
        return np.array(features)
        
    def lbp_ozellikleri(self, img):
        """LBP (Local Binary Pattern) özellikleri"""
        # Basitleştirilmiş LBP
        height, width = img.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = img[i, j]
                code = 0
                
                # 8 komşu piksel
                neighbors = [
                    img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                    img[i, j+1], img[i+1, j+1], img[i+1, j],
                    img[i+1, j-1], img[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        # LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        
        return hist[:32]  # İlk 32 değer
        
    def duygu_analizi(self, face_roi, gray):
        """Duygu analizi yap"""
        if self.emotion_model is None:
            return "Model yuklenmedi"
            
        features = self.yuz_ozellikleri_cikar(face_roi, gray)
        features = features.reshape(1, -1)
        
        try:
            prediction = self.emotion_model.predict(features)[0]
            confidence = self.emotion_model.predict_proba(features)[0].max()
            return self.emotion_labels[prediction], confidence
        except:
            return "Bilinmiyor", 0.0
            
    def yuz_tanima(self, face_roi, gray):
        """Yüz tanıma yap"""
        if len(self.users) == 0:
            return "Kullanici yok", 0.0
            
        x, y, w, h = face_roi
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_gray = cv2.resize(face_roi_gray, (100, 100))
        
        try:
            label, confidence = self.face_recognizer.predict(face_roi_gray)
            if confidence < 100:  # Eşik değeri
                user_name = list(self.users.keys())[label]
                return user_name, confidence
            else:
                return "Bilinmeyen", confidence
        except:
            return "Hata", 0.0
            
    def kullanici_ekle(self, name, face_images):
        """Yeni kullanıcı ekle"""
        if name in self.users:
            print(f"❌ {name} kullanıcısı zaten mevcut!")
            return False
            
        # Yüz görüntülerini hazırla
        prepared_faces = []
        labels = []
        
        for i, img in enumerate(face_images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            prepared_faces.append(gray)
            labels.append(len(self.users))
            
        # Kullanıcıyı ekle
        self.users[name] = {
            'images': prepared_faces,
            'label': len(self.users)
        }
        
        # Modeli yeniden eğit
        self.model_egit()
        
        print(f"✅ {name} kullanıcısı eklendi!")
        return True
        
    def model_egit(self):
        """Yüz tanıma modelini eğit"""
        if len(self.users) == 0:
            return
            
        faces = []
        labels = []
        
        for user_name, user_data in self.users.items():
            for face_img in user_data['images']:
                faces.append(face_img)
                labels.append(user_data['label'])
                
        if len(faces) > 0:
            self.face_recognizer.train(faces, np.array(labels))
            print("✅ Yüz tanıma modeli eğitildi!")
            
    def duygu_modeli_egit(self, training_data):
        """Duygu analizi modelini eğit"""
        X = []
        y = []
        
        for emotion, features_list in training_data.items():
            for features in features_list:
                X.append(features)
                y.append(emotion)
                
        if len(X) > 0:
            # Label encoding
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Model eğitimi
            self.emotion_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.emotion_model.fit(X, y_encoded)
            
            print("✅ Duygu analizi modeli eğitildi!")
            
    def gercek_zamanli_analiz(self):
        """Gerçek zamanlı yüz analizi"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
            
        print("📹 Gerçek zamanlı yüz analizi başlatıldı!")
        print("🔑 Kontroller:")
        print("  'q' - Çıkış")
        print("  's' - Ekran görüntüsü al")
        print("  'r' - Yüz tanıma modunu değiştir")
        
        face_recognition_mode = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Yüz tespit
            faces, gray = self.yuz_tespit(frame)
            
            for (x, y, w, h) in faces:
                # Yüz çerçevesi çiz
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Yüz tanıma
                if face_recognition_mode and len(self.users) > 0:
                    user_name, confidence = self.yuz_tanima((x, y, w, h), gray)
                    cv2.putText(frame, f"Kullanici: {user_name}", 
                               (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Guven: {confidence:.1f}", 
                               (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Duygu analizi
                emotion, emotion_conf = self.duygu_analizi((x, y, w, h), gray)
                cv2.putText(frame, f"Duygu: {emotion}", 
                           (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Guven: {emotion_conf:.2f}", 
                           (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Göz tespit
                eyes = self.goz_tespit(gray, (x, y, w, h))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
                    
                # Gülümseme tespit
                smiles = self.gulumseme_tespit(gray, (x, y, w, h))
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 1)
                    
            # Mod bilgisi
            mode_text = "Yuz Tanima" if face_recognition_mode else "Sadece Duygu Analizi"
            cv2.putText(frame, f"Mod: {mode_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Yuz Tanima ve Duygu Analizi', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                face_recognition_mode = not face_recognition_mode
                print(f"🔄 Mod değiştirildi: {mode_text}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Ekran görüntüsü kaydedildi: {filename}")
                
        cap.release()
        cv2.destroyAllWindows()
        
    def kullanici_kayit(self):
        """Yeni kullanıcı kayıt sistemi"""
        print("👤 Yeni Kullanıcı Kayıt Sistemi")
        print("=" * 50)
        
        name = input("Kullanıcı adını girin: ").strip()
        if not name:
            print("❌ Geçersiz kullanıcı adı!")
            return
            
        print(f"📸 {name} için yüz görüntüleri alınıyor...")
        print("💡 Farklı açılardan 5 fotoğraf çekin")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
            
        face_images = []
        count = 0
        
        while count < 5:
            ret, frame = cap.read()
            if not ret:
                continue
                
            faces, gray = self.yuz_tespit(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
            cv2.putText(frame, f"Fotoğraf: {count+1}/5", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE - Fotoğraf çek", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "ESC - İptal", 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Kullanıcı Kayıt', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if len(faces) > 0:
                    face_img = frame.copy()
                    face_images.append(face_img)
                    count += 1
                    print(f"✅ Fotoğraf {count} alındı")
                    
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_images) == 5:
            self.kullanici_ekle(name, face_images)
        else:
            print("❌ Yeterli fotoğraf alınamadı!")
            
    def model_kaydet(self, filename="yuz_tanima_modeli.pkl"):
        """Modeli kaydet"""
        model_data = {
            'users': self.users,
            'face_recognizer': self.face_recognizer,
            'emotion_model': self.emotion_model
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model kaydedildi: {filename}")
        
    def model_yukle(self, filename="yuz_tanima_modeli.pkl"):
        """Modeli yükle"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
                
            self.users = model_data['users']
            self.face_recognizer = model_data['face_recognizer']
            self.emotion_model = model_data['emotion_model']
            print(f"✅ Model yüklendi: {filename}")
        else:
            print(f"❌ Model dosyası bulunamadı: {filename}")
            
    def istatistikler(self):
        """Sistem istatistikleri"""
        print("\n📊 SİSTEM İSTATİSTİKLERİ")
        print("=" * 50)
        print(f"👥 Kayıtlı kullanıcı sayısı: {len(self.users)}")
        
        if self.users:
            print("\n📋 Kullanıcı listesi:")
            for i, (name, data) in enumerate(self.users.items()):
                print(f"  {i+1}. {name} ({len(data['images'])} fotoğraf)")
                
        if self.emotion_model:
            print(f"\n😊 Duygu analizi modeli: ✅ Yüklü")
        else:
            print(f"\n😊 Duygu analizi modeli: ❌ Yüklenmedi")
            
        print(f"\n🤖 Yüz tanıma modeli: ✅ Hazır")

def demo_menu():
    """Demo menüsü"""
    sistem = YuzTanimaSistemi()
    
    # Model yüklemeyi dene
    sistem.model_yukle()
    
    while True:
        print("\n" + "="*60)
        print("👤 YÜZ TANIMA VE DUYGU ANALİZİ SİSTEMİ")
        print("="*60)
        print("1. 👤 Yeni Kullanıcı Kayıt")
        print("2. 📹 Gerçek Zamanlı Analiz")
        print("3. 📊 Sistem İstatistikleri")
        print("4. 💾 Model Kaydet")
        print("5. 📂 Model Yükle")
        print("6. 🎯 Demo Modu")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-6): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                sistem.kullanici_kayit()
            elif secim == "2":
                sistem.gercek_zamanli_analiz()
            elif secim == "3":
                sistem.istatistikler()
            elif secim == "4":
                sistem.model_kaydet()
            elif secim == "5":
                filename = input("Model dosya adını girin: ").strip()
                if filename:
                    sistem.model_yukle(filename)
                else:
                    sistem.model_yukle()
            elif secim == "6":
                print("🎯 Demo modu başlatılıyor...")
                print("💡 Bu mod, örnek verilerle sistemi test eder")
                
                # Demo kullanıcıları ekle
                if len(sistem.users) == 0:
                    print("📝 Demo kullanıcıları ekleniyor...")
                    # Burada demo kullanıcıları eklenebilir
                    
                sistem.gercek_zamanli_analiz()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-6 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n👋 Program sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    print("👤 Yüz Tanıma ve Duygu Analizi Sistemi")
    print("=" * 60)
    print("Bu proje, yüz tanıma ve duygu analizi tekniklerini uygular.")
    print("📚 Öğrenilecek Konular:")
    print("  • Haar Cascade ile yüz tespiti")
    print("  • LBPH ile yüz tanıma")
    print("  • Duygu analizi algoritmaları")
    print("  • Gerçek zamanlı video işleme")
    print("  • Model kaydetme/yükleme")
    
    demo_menu() 