"""
Ensemble Yöntemleri ve İleri Seviye Makine Öğrenmesi
OpenCV ile Ensemble Learning Uygulamaları

Bu dosya, ensemble yöntemlerini ve ileri seviye makine öğrenmesi 
tekniklerini OpenCV kullanarak uygular.

Yazar: OpenCV Türkiye Topluluğu
Tarih: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

class EnsembleLearningDemo:
    """Ensemble Learning yöntemlerini gösteren demo sınıfı"""
    
    def __init__(self):
        """Ensemble learning demo sınıfını başlat"""
        self.models = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def veri_hazirla(self, n_samples=1000, n_features=20, n_classes=3):
        """Ensemble learning için veri seti hazırla"""
        print("📊 Ensemble Learning için veri hazırlanıyor...")
        
        # Karmaşık veri seti oluştur
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=3,
            n_repeated=2,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Veriyi böl
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Özellikleri normalize et
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✅ Veri hazırlandı: {self.X_train.shape[0]} eğitim, {self.X_test.shape[0]} test örneği")
        print(f"📈 Özellik sayısı: {self.X_train.shape[1]}, Sınıf sayısı: {len(np.unique(y))}")
        
    def bagging_ornegi(self):
        """Bagging (Bootstrap Aggregating) örneği"""
        print("\n👜 BAGGING (Bootstrap Aggregating) Örneği")
        print("=" * 50)
        
        # Bagging Classifier
        bagging_clf = BaggingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            n_estimators=5,
            max_samples=0.8,
            bootstrap=True,
            random_state=42
        )
        
        # Modeli eğit
        start_time = time.time()
        bagging_clf.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        # Tahmin yap
        y_pred = bagging_clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"⏱️  Eğitim süresi: {training_time:.3f} saniye")
        print(f"🎯 Doğruluk: {accuracy:.4f}")
        print(f"📊 Sınıflandırma Raporu:")
        print(classification_report(self.y_test, y_pred))
        
        self.models['bagging'] = bagging_clf
        return bagging_clf
        
    def boosting_ornegi(self):
        """Boosting örneği (AdaBoost ve Gradient Boosting)"""
        print("\n🚀 BOOSTING Örneği")
        print("=" * 50)
        
        # AdaBoost
        print("📈 AdaBoost Classifier:")
        adaboost_clf = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        )
        
        start_time = time.time()
        adaboost_clf.fit(self.X_train, self.y_train)
        adaboost_time = time.time() - start_time
        
        y_pred_adaboost = adaboost_clf.predict(self.X_test)
        accuracy_adaboost = accuracy_score(self.y_test, y_pred_adaboost)
        
        print(f"⏱️  Eğitim süresi: {adaboost_time:.3f} saniye")
        print(f"🎯 Doğruluk: {accuracy_adaboost:.4f}")
        
        # Gradient Boosting
        print("\n📈 Gradient Boosting Classifier:")
        gb_clf = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        start_time = time.time()
        gb_clf.fit(self.X_train, self.y_train)
        gb_time = time.time() - start_time
        
        y_pred_gb = gb_clf.predict(self.X_test)
        accuracy_gb = accuracy_score(self.y_test, y_pred_gb)
        
        print(f"⏱️  Eğitim süresi: {gb_time:.3f} saniye")
        print(f"🎯 Doğruluk: {accuracy_gb:.4f}")
        
        self.models['adaboost'] = adaboost_clf
        self.models['gradient_boosting'] = gb_clf
        
        return adaboost_clf, gb_clf
        
    def voting_ornegi(self):
        """Voting (Oylama) ensemble örneği"""
        print("\n🗳️ VOTING ENSEMBLE Örneği")
        print("=" * 50)
        
        # Farklı base classifier'lar
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        
        clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
        clf2 = SVC(probability=True, random_state=42)
        clf3 = LogisticRegression(random_state=42)
        
        # Voting Classifier (Soft Voting)
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', clf1),
                ('svc', clf2),
                ('lr', clf3)
            ],
            voting='soft'  # Soft voting (probability-based)
        )
        
        # Modeli eğit
        start_time = time.time()
        voting_clf.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        # Tahmin yap
        y_pred = voting_clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"⏱️  Eğitim süresi: {training_time:.3f} saniye")
        print(f"🎯 Doğruluk: {accuracy:.4f}")
        print(f"📊 Sınıflandırma Raporu:")
        print(classification_report(self.y_test, y_pred))
        
        self.models['voting'] = voting_clf
        return voting_clf
        
    def cross_validation_karsilastirma(self):
        """Farklı ensemble yöntemlerini cross-validation ile karşılaştır"""
        print("\n🔄 CROSS-VALIDATION KARŞILAŞTIRMASI")
        print("=" * 50)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Bagging': BaggingClassifier(n_estimators=5, random_state=42),
            'Voting': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=20, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=20, random_state=42))
                ],
                voting='soft'
            )
        }
        
        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"{name:20} | Ortalama: {scores.mean():.4f} | Std: {scores.std():.4f}")
            
        return results
        
    def performans_analizi(self):
        """Ensemble modellerin performans analizi"""
        print("\n📊 PERFORMANS ANALİZİ")
        print("=" * 50)
        
        if not self.models:
            print("❌ Önce modelleri eğitin!")
            return
            
        results = {}
        for name, model in self.models.items():
            # Test doğruluğu
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            results[name] = {
                'test_accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"\n{name.upper()}:")
            print(f"  🎯 Test Doğruluğu: {accuracy:.4f}")
            print(f"  📈 CV Ortalama: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
        return results
        
    def ensemble_gorselleştirme(self):
        """Ensemble yöntemlerin görselleştirilmesi"""
        print("\n🎨 ENSEMBLE GÖRSELLEŞTİRME")
        print("=" * 50)
        
        # Basit 2D veri seti oluştur
        X_simple, y_simple = make_classification(
            n_samples=300,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_classes=2,
            random_state=42
        )
        
        X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
            X_simple, y_simple, test_size=0.3, random_state=42
        )
        
        # Farklı ensemble modeller
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=10, random_state=42)
        }
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Ensemble Yöntemlerin Karşılaştırması', fontsize=16)
        
        for idx, (name, model) in enumerate(models.items()):
            # Modeli eğit
            model.fit(X_train_simple, y_train_simple)
            
            # Karar sınırlarını çiz
            x_min, x_max = X_simple[:, 0].min() - 1, X_simple[:, 0].max() + 1
            y_min, y_max = X_simple[:, 1].min() - 1, X_simple[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                np.arange(y_min, y_max, 0.1))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot
            axes[idx].contourf(xx, yy, Z, alpha=0.4)
            axes[idx].scatter(X_train_simple[:, 0], X_train_simple[:, 1], 
                            c=y_train_simple, alpha=0.8, edgecolors='black')
            axes[idx].set_title(f'{name}\nDoğruluk: {accuracy_score(y_test_simple, model.predict(X_test_simple)):.3f}')
            axes[idx].set_xlabel('Özellik 1')
            axes[idx].set_ylabel('Özellik 2')
            
        plt.tight_layout()
        plt.show()
        
    def gercek_dunya_ornegi(self):
        """Gerçek dünya ensemble learning örneği"""
        print("\n🌍 GERÇEK DÜNYA ÖRNEĞİ: Çoklu Sensör Verisi Sınıflandırması")
        print("=" * 70)
        
        # Simüle edilmiş sensör verisi
        np.random.seed(42)
        n_samples = 1000
        
        # Farklı sensörlerden gelen veriler
        sensor1 = np.random.normal(0, 1, n_samples)  # Sıcaklık sensörü
        sensor2 = np.random.normal(0, 1, n_samples)  # Nem sensörü
        sensor3 = np.random.normal(0, 1, n_samples)  # Basınç sensörü
        sensor4 = np.random.normal(0, 1, n_samples)  # Işık sensörü
        
        # Gürültü ekle
        noise = np.random.normal(0, 0.1, n_samples)
        
        # Hedef değişken (anomali tespiti)
        # Karmaşık bir kural oluştur
        target = ((sensor1 > 1.5) & (sensor2 < -0.5)) | \
                 ((sensor3 > 1.0) & (sensor4 < -1.0)) | \
                 ((sensor1 + sensor2 + sensor3 + sensor4) > 3.0)
        
        target = target.astype(int)
        
        # Veriyi birleştir
        X_sensor = np.column_stack([sensor1, sensor2, sensor3, sensor4])
        X_sensor += noise.reshape(-1, 1)
        
        # Veriyi böl
        X_train_sensor, X_test_sensor, y_train_sensor, y_test_sensor = train_test_split(
            X_sensor, target, test_size=0.3, random_state=42
        )
        
        # Ensemble modeller
        ensemble_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'Voting Ensemble': VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
                ],
                voting='soft'
            )
        }
        
        print("📊 Sensör Verisi Analizi:")
        print(f"   - Toplam örnek: {len(X_sensor)}")
        print(f"   - Anomali oranı: {np.mean(target):.2%}")
        print(f"   - Sensör sayısı: {X_sensor.shape[1]}")
        
        results = {}
        for name, model in ensemble_models.items():
            # Modeli eğit
            model.fit(X_train_sensor, y_train_sensor)
            
            # Tahmin yap
            y_pred = model.predict(X_test_sensor)
            accuracy = accuracy_score(y_test_sensor, y_pred)
            
            results[name] = accuracy
            print(f"\n{name}:")
            print(f"  🎯 Doğruluk: {accuracy:.4f}")
            print(f"  📊 Sınıflandırma Raporu:")
            print(classification_report(y_test_sensor, y_pred))
            
        return results

def demo_menu():
    """Demo menüsü"""
    demo = EnsembleLearningDemo()
    
    while True:
        print("\n" + "="*60)
        print("🎯 ENSEMBLE LEARNING DEMO MENÜSÜ")
        print("="*60)
        print("1. 📊 Veri Hazırlama")
        print("2. 👜 Bagging Örneği")
        print("3. 🚀 Boosting Örneği")
        print("4. 🗳️ Voting Ensemble Örneği")
        print("5. 🔄 Cross-Validation Karşılaştırması")
        print("6. 📊 Performans Analizi")
        print("7. 🎨 Ensemble Görselleştirme")
        print("8. 🌍 Gerçek Dünya Örneği")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-8): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                demo.veri_hazirla()
            elif secim == "2":
                if demo.X_train is None:
                    print("❌ Önce veri hazırlayın! (Seçenek 1)")
                else:
                    demo.bagging_ornegi()
            elif secim == "3":
                if demo.X_train is None:
                    print("❌ Önce veri hazırlayın! (Seçenek 1)")
                else:
                    demo.boosting_ornegi()
            elif secim == "4":
                if demo.X_train is None:
                    print("❌ Önce veri hazırlayın! (Seçenek 1)")
                else:
                    demo.voting_ornegi()
            elif secim == "5":
                if demo.X_train is None:
                    print("❌ Önce veri hazırlayın! (Seçenek 1)")
                else:
                    demo.cross_validation_karsilastirma()
            elif secim == "6":
                if not demo.models:
                    print("❌ Önce modelleri eğitin!")
                else:
                    demo.performans_analizi()
            elif secim == "7":
                demo.ensemble_gorselleştirme()
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
    print("🎯 Ensemble Learning ve İleri Seviye Makine Öğrenmesi")
    print("=" * 60)
    print("Bu demo, ensemble yöntemlerini ve ileri seviye ML tekniklerini gösterir.")
    print("📚 Öğrenilecek Konular:")
    print("  • Bagging (Bootstrap Aggregating)")
    print("  • Boosting (AdaBoost, Gradient Boosting)")
    print("  • Voting (Oylama) Ensemble")
    print("  • Cross-Validation Karşılaştırması")
    print("  • Performans Analizi")
    print("  • Gerçek Dünya Uygulamaları")
    
    demo_menu() 