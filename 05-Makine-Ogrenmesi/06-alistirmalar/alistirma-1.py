#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Alıştırma 1: Çok Sınıflı Görüntü Sınıflandırma - Template
===========================================================

Bu dosya alıştırma için template kod yapısını içerir.
Öğrenciler bu template'i kullanarak kendi implementasyonlarını yapacaklar.

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import os
import pickle
from datetime import datetime

class ImageClassificationProject:
    """Görüntü sınıflandırma projesi ana sınıfı"""
    
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.class_names = [f"Sinif_{i}" for i in range(n_classes)]
        self.dataset = []
        self.labels = []
        self.models = {}
        self.scaler = StandardScaler()
        
        print(f"🎯 Görüntü Sınıflandırma Projesi başlatıldı ({n_classes} sınıf)")
    
    def set_class_names(self, names):
        """Sınıf isimlerini ayarla"""
        if len(names) != self.n_classes:
            raise ValueError(f"Sınıf sayısı {self.n_classes} olmalı, {len(names)} verildi")
        self.class_names = names
        print(f"📝 Sınıf isimleri güncellendi: {names}")

class DataCollector:
    """Veri toplama sınıfı"""
    
    def __init__(self, project):
        self.project = project
        self.current_class = 0
        self.samples_per_class = [0] * project.n_classes
        
    def collect_data_interactive(self):
        """İnteraktif veri toplama"""
        print("\n📸 İnteraktif Veri Toplama Modu")
        print("=" * 35)
        print("Kontroller:")
        print("  0-4: Sınıf seçimi")
        print("  SPACE: Örnek kaydet")
        print("  S: İstatistikleri göster")
        print("  ESC: Çıkış")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # TODO: UI bilgilerini frame üzerine çiz
            # - Mevcut sınıf
            # - Sınıf başına örnek sayısı
            # - Toplam örnek sayısı
            # - Kontrol bilgileri
            
            # BURAYA KOD YAZIN
            pass
            
            cv2.imshow('Veri Toplama', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key >= ord('0') and key <= ord('4'):  # Sınıf seçimi
                # TODO: Sınıf değiştirme
                # BURAYA KOD YAZIN
                pass
            elif key == ord(' '):  # SPACE - Örnek kaydet
                # TODO: Mevcut frame'i işle ve kaydet
                # BURAYA KOD YAZIN
                pass
            elif key == ord('s') or key == ord('S'):  # İstatistik
                self.show_statistics()
        
        cap.release()
        cv2.destroyAllWindows()
        
        return len(self.project.dataset) > 0
    
    def save_example(self, frame, class_id):
        """Örnek kaydetme"""
        # TODO: Frame'i işleyip veri setine ekle
        # 1. Gri tonlamaya çevir
        # 2. 32x32'ye resize et
        # 3. Özellik çıkar
        # 4. Dataset'e ekle
        
        # BURAYA KOD YAZIN
        pass
    
    def show_statistics(self):
        """Veri istatistiklerini göster"""
        print("\n📊 Veri İstatistikleri:")
        print("-" * 25)
        total = sum(self.samples_per_class)
        for i, count in enumerate(self.samples_per_class):
            print(f"  {self.project.class_names[i]}: {count} örnek")
        print(f"  Toplam: {total} örnek")

class FeatureExtractor:
    """Özellik çıkarma sınıfı"""
    
    @staticmethod
    def extract_comprehensive_features(image):
        """Kapsamlı özellik çıkarma"""
        # TODO: Tüm özellik türlerini implement edin
        
        features = []
        
        # 1. Histogram Özellikleri (16 boyut)
        # BURAYA KOD YAZIN
        
        # 2. İstatistiksel Özellikler (4 boyut)
        # BURAYA KOD YAZIN
        
        # 3. Kenar Özellikleri (2 boyut)
        # BURAYA KOD YAZIN
        
        # 4. Doku Özellikleri - LBP (10 boyut)
        # BURAYA KOD YAZIN
        
        # Toplam 32 boyutlu özellik vektörü döndür
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def extract_histogram_features(image, bins=16):
        """Histogram özelliklerini çıkar"""
        # TODO: Gri seviye histogramı hesapla ve normalize et
        # BURAYA KOD YAZIN
        pass
    
    @staticmethod
    def extract_statistical_features(image):
        """İstatistiksel özellikler"""
        # TODO: Mean, std, min, max hesapla
        # BURAYA KOD YAZIN
        pass
    
    @staticmethod
    def extract_edge_features(image):
        """Kenar özelliklerini çıkar"""
        # TODO: Canny edge detection ve kenar yoğunluğu
        # BURAYA KOD YAZIN
        pass
    
    @staticmethod
    def extract_lbp_features(image, radius=1, n_points=8, n_bins=10):
        """LBP doku özellikleri"""
        # TODO: Local Binary Pattern implementasyonu
        # BURAYA KOD YAZIN
        pass

class ModelTrainer:
    """Model eğitim sınıfı"""
    
    def __init__(self, project):
        self.project = project
        self.results = {}
    
    def prepare_data(self):
        """Veriyi eğitim için hazırla"""
        if len(self.project.dataset) == 0:
            raise ValueError("Veri seti boş!")
        
        # TODO: Veriyi numpy array'e çevir ve train/test split yap
        X = np.array(self.project.dataset)
        y = np.array(self.project.labels)
        
        # BURAYA KOD YAZIN - train_test_split
        
        # TODO: Normalizasyon
        # BURAYA KOD YAZIN
        
        return X_train, X_test, y_train, y_test
    
    def train_knn_models(self, X_train, X_test, y_train, y_test):
        """k-NN modellerini eğit"""
        print("\n🎯 k-NN Modelleri Eğitiliyor...")
        
        k_values = [3, 5, 7]
        
        for k in k_values:
            # TODO: k-NN modeli oluştur ve eğit
            # BURAYA KOD YAZIN
            pass
    
    def train_svm_models(self, X_train, X_test, y_train, y_test):
        """SVM modellerini eğit"""
        print("\n⚖️ SVM Modelleri Eğitiliyor...")
        
        kernels = ['linear', 'rbf', 'poly']
        
        for kernel in kernels:
            # TODO: SVM modeli oluştur ve eğit
            # BURAYA KOD YAZIN
            pass
    
    def train_ann_models(self, X_train, X_test, y_train, y_test):
        """ANN modellerini eğit"""
        print("\n🧠 ANN Modelleri Eğitiliyor...")
        
        architectures = [[32], [64], [32, 16], [64, 32]]
        
        for hidden_layers in architectures:
            # TODO: ANN modeli oluştur ve eğit
            # BURAYA KOD YAZIN
            pass
    
    def train_tree_models(self, X_train, X_test, y_train, y_test):
        """Ağaç modelleri eğit"""
        print("\n🌳 Ağaç Modelleri Eğitiliyor...")
        
        # Decision Tree
        # TODO: Farklı max_depth değerleri ile test et
        # BURAYA KOD YAZIN
        
        # Random Forest
        # TODO: Farklı n_trees değerleri ile test et
        # BURAYA KOD YAZIN
        pass
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Model değerlendirme"""
        # TODO: Model performansını ölç
        # - Accuracy
        # - Classification report
        # - Prediction time
        # BURAYA KOD YAZIN
        pass
    
    def perform_cross_validation(self, X, y):
        """Cross-validation analizi"""
        print("\n🔄 Cross-Validation Analizi...")
        
        # TODO: Her model için 5-fold CV yap
        # BURAYA KOD YAZIN
        pass

class ResultAnalyzer:
    """Sonuç analizi sınıfı"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.results = trainer.results
    
    def create_comparison_table(self):
        """Model karşılaştırma tablosu"""
        print("\n📊 MODEL KARŞILAŞTIRMA TABLOSU")
        print("=" * 80)
        print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Time (ms)':<10}")
        print("-" * 80)
        
        # TODO: Results dictionary'den tablo oluştur
        # BURAYA KOD YAZIN
        pass
    
    def plot_performance_comparison(self):
        """Performans karşılaştırma grafikleri"""
        # TODO: Bar chart ile model performanslarını karşılaştır
        # BURAYA KOD YAZIN
        pass
    
    def plot_confusion_matrices(self):
        """Confusion matrix'leri çiz"""
        # TODO: En iyi modeller için confusion matrix
        # BURAYA KOD YAZIN
        pass
    
    def analyze_feature_importance(self):
        """Özellik önem analizi"""
        # TODO: Random Forest ve Decision Tree için özellik önemleri
        # BURAYA KOD YAZIN
        pass

class RealtimePredictor:
    """Gerçek zamanlı tahmin sınıfı"""
    
    def __init__(self, project, best_model):
        self.project = project
        self.best_model = best_model  
        self.prediction_history = []
    
    def start_realtime_prediction(self):
        """Gerçek zamanlı tahmin başlat"""
        print("\n🔮 Gerçek Zamanlı Tahmin Modu")
        print("ESC: Çıkış")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # TODO: Frame'i işle ve tahmin yap
            # 1. Özellik çıkar
            # 2. Model ile tahmin yap
            # 3. Güven skoru hesapla
            # 4. Sonuçları frame üzerine çiz
            
            # BURAYA KOD YAZIN
            
            cv2.imshow('Gercek Zamanli Tahmin', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Ana program"""
    print("🎯 Çok Sınıflı Görüntü Sınıflandırma Projesi")
    print("=" * 45)
    
    # Proje başlat
    project = ImageClassificationProject(n_classes=5)
    
    # Sınıf isimlerini ayarla
    class_names = ["Kitap", "Kalem", "Telefon", "Bardak", "Anahtar"]
    project.set_class_names(class_names)
    
    while True:
        print("\n" + "="*50)
        print("🎯 ANA MENÜ")
        print("="*50)
        print("1. 📸 Veri Toplama")
        print("2. 🔧 Model Eğitimi") 
        print("3. 📊 Sonuç Analizi")
        print("4. 🔮 Gerçek Zamanlı Tahmin")
        print("5. 💾 Veri/Model Kaydet")
        print("6. 📁 Veri/Model Yükle")
        print("0. ❌ Çıkış")
        
        try:
            choice = input("\nSeçiminizi yapın (0-6): ").strip()
            
            if choice == "0":
                print("👋 Program sonlandırıldı!")
                break
            elif choice == "1":
                # Veri toplama
                collector = DataCollector(project)
                success = collector.collect_data_interactive()
                if success:
                    print("✅ Veri toplama tamamlandı!")
                    collector.show_statistics()
                else:
                    print("❌ Veri toplama başarısız!")
            
            elif choice == "2":
                # Model eğitimi
                if len(project.dataset) == 0:
                    print("❌ Önce veri toplamalısınız!")
                    continue
                
                trainer = ModelTrainer(project)
                
                try:
                    X_train, X_test, y_train, y_test = trainer.prepare_data()
                    
                    # Tüm modelleri eğit
                    trainer.train_knn_models(X_train, X_test, y_train, y_test)
                    trainer.train_svm_models(X_train, X_test, y_train, y_test)
                    trainer.train_ann_models(X_train, X_test, y_train, y_test)
                    trainer.train_tree_models(X_train, X_test, y_train, y_test)
                    
                    # Cross-validation
                    X_all = np.vstack([X_train, X_test])
                    y_all = np.hstack([y_train, y_test])
                    trainer.perform_cross_validation(X_all, y_all)
                    
                    print("✅ Tüm modeller eğitildi!")
                    
                except Exception as e:
                    print(f"❌ Model eğitimi hatası: {e}")
            
            elif choice == "3":
                # Sonuç analizi
                if not hasattr(trainer, 'results') or not trainer.results:
                    print("❌ Önce modelleri eğitmelisiniz!")
                    continue
                
                analyzer = ResultAnalyzer(trainer)
                analyzer.create_comparison_table()
                analyzer.plot_performance_comparison()
                analyzer.plot_confusion_matrices()
                analyzer.analyze_feature_importance()
            
            elif choice == "4":
                # Gerçek zamanlı tahmin
                if not hasattr(trainer, 'results') or not trainer.results:
                    print("❌ Önce modelleri eğitmelisiniz!")
                    continue
                
                # En iyi modeli seç
                # TODO: En yüksek accuracy'e sahip modeli bul
                # BURAYA KOD YAZIN
                best_model = None  # Placeholder
                
                if best_model:
                    predictor = RealtimePredictor(project, best_model)
                    predictor.start_realtime_prediction()
                else:
                    print("❌ En iyi model bulunamadı!")
            
            elif choice == "5":
                # Kaydetme
                # TODO: Veri seti ve modelleri kaydet
                # BURAYA KOD YAZIN
                print("💾 Kaydetme özelliği henüz implementasyona hazır!")
            
            elif choice == "6":
                # Yükleme
                # TODO: Veri seti ve modelleri yükle
                # BURAYA KOD YAZIN
                print("📁 Yükleme özelliği henüz implementasyona hazır!")
            
            else:
                print("❌ Geçersiz seçim! Lütfen 0-6 arasında bir sayı girin.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Program kullanıcı tarafından sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()

# 📝 TEMPLATE NOTLARI:
# Bu dosya öğrenciler için template'dir.
# "BURAYA KOD YAZIN" yazan yerlere implementasyon yapılacak.
# 
# Önemli implementasyon noktaları:
# 1. Özellik çıkarma fonksiyonları
# 2. Model eğitim ve değerlendirme
# 3. Cross-validation implementasyonu
# 4. Gerçek zamanlı tahmin sistemi
# 5. Sonuç görselleştirme ve analiz
#
# Her TODO yorumu bir implementasyon görevi belirtir.