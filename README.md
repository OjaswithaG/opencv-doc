# Proje Hakkında

## OpenCV ile Görüntü ve Video İşleme, Makine Öğrenmesi ve Proje Uygulamaları

Bu depo, OpenCV ve Python kullanarak temel ve ileri seviye görüntü işleme, video işleme, makine öğrenmesi ve gerçek dünya projeleri geliştirmek isteyenler için kapsamlı bir eğitim ve uygulama kaynağıdır.

---

## 🎯 Amaç
- Görüntü ve video işleme temellerini öğretmek
- Makine öğrenmesi ve derin öğrenme algoritmalarını uygulamalı göstermek
- Gerçek dünya problemlerine yönelik projeler geliştirmek
- Türkçe kaynak eksikliğini gidermek ve topluluğa katkı sağlamak

---

## 📚 İçerik ve Klasör Yapısı

- **[01-Temeller/](https://github.com/erent8/opencv-doc/tree/main/01-Temeller)**: OpenCV kurulumu, temel veri yapıları, ilk programlar
- **[02-Resim-Isleme/](https://github.com/erent8/opencv-doc/tree/main/02-Resim-Isleme)**: Geometrik dönüşümler, filtreleme, morfolojik işlemler, histogram, kontrast, gürültü azaltma, kenar algılama, uygulamalı alıştırmalar
- **[03-Video-Isleme/](https://github.com/erent8/opencv-doc/tree/main/03-Video-Isleme)**: Video okuma/yazma, frame işleme, hareket algılama, nesne takibi, arka plan çıkarma, video analizi, uygulamalı alıştırmalar
- **[04-Nesne-Tespiti/](https://github.com/erent8/opencv-doc/tree/main/04-Nesne-Tespiti)**: Klasik ve DNN tabanlı nesne tespiti, yüz tespiti, şekil tespiti, renk tabanlı tespit, QR/barcode okuma, uygulamalı alıştırmalar
- **[05-Makine-Ogrenmesi/](https://github.com/erent8/opencv-doc/tree/main/05-Makine-Ogrenmesi)**: Temel ML kavramları, k-NN, SVM, ANN, karar ağaçları, ensemble yöntemler, derin öğrenme, uygulamalı alıştırmalar
- **[06-Ileri-Seviye/](https://github.com/erent8/opencv-doc/tree/main/06-Ileri-Seviye)**: (Geliştirilecek) Gelişmiş teknikler ve özel uygulamalar
- **[07-Projeler/](https://github.com/erent8/opencv-doc/tree/main/07-Projeler)**: Gerçek dünya projeleri (yüz tanıma, plaka tanıma, hareket algılama, vb.)
- **[assets/](https://github.com/erent8/opencv-doc/tree/main/assets)**, **[examples/](https://github.com/erent8/opencv-doc/tree/main/examples)**, **[utils/](https://github.com/erent8/opencv-doc/tree/main/utils)**: Destekleyici dosyalar, örnekler ve yardımcı fonksiyonlar

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler
- Python 3.8+
- OpenCV, NumPy, Matplotlib, scikit-learn, TensorFlow, Keras, Pillow, Tesseract, tqdm, requests, seaborn, ve diğerleri

### 2. Kurulum
```bash
pip install -r requirements.txt
```
Ekstra: `pip install tensorflow keras mediapipe dlib pytesseract pillow imutils requests tqdm`

**Gereksinim Dosyaları:**
- [requirements.txt](https://github.com/erent8/opencv-doc/blob/main/requirements.txt)
- [requirements-minimal.txt](https://github.com/erent8/opencv-doc/blob/main/requirements-minimal.txt)

### 3. Model Dosyaları
- Büyük model dosyaları (ör. YOLOv3 weights) otomatik indirilir veya `models/` klasörüne manuel eklenir.
- [`utils/model_downloader.py`](https://github.com/erent8/opencv-doc/blob/main/utils/model_downloader.py) ile otomatik indirme yapılabilir.

### 4. Çalıştırma
Her klasördeki Python dosyalarını doğrudan çalıştırabilirsiniz:
```bash
python 01-Temeller/02-ilk-program.py
python 02-Resim-Isleme/01-geometrik-transformasyonlar.py
python 05-Makine-Ogrenmesi/06-alistirmalar/alistirma-2.py
python 07-Projeler/01-yuz-tanima-sistemi.py
```

**Örnek Dosyalar:**
- [İlk Program](https://github.com/erent8/opencv-doc/blob/main/01-Temeller/02-ilk-program.py)
- [Geometrik Dönüşümler](https://github.com/erent8/opencv-doc/blob/main/02-Resim-Isleme/01-geometrik-transformasyonlar.py)
- [Yüz Tanıma Sistemi](https://github.com/erent8/opencv-doc/blob/main/07-Projeler/01-yuz-tanima-sistemi.py)

---

## 🧩 Bölüm Özeti

### 1. Temeller
- OpenCV kurulumu, temel fonksiyonlar, veri yapıları

### 2. Görüntü İşleme
- Geometrik dönüşümler, filtreler, histogram, morfoloji, renk uzayları, kenar bulma
- Pratik alıştırmalar ve çözümler

### 3. Video İşleme
- Video okuma/yazma, gerçek zamanlı işleme, hareket algılama, nesne takibi
- Pratik alıştırmalar ve çözümler

### 4. Nesne Tespiti
- Klasik yöntemler (Haar, HOG, Template Matching)
- DNN tabanlı tespit (YOLO, SSD, Haar Cascade fallback)
- QR/barcode okuma, renk tabanlı tespit

### 5. Makine Öğrenmesi
- Temel ML algoritmaları (k-NN, SVM, ANN, Karar Ağaçları)
- Ensemble yöntemler, derin öğrenme, transfer learning
- Pratik alıştırmalar ve çözümler

### 6. Projeler
- Yüz tanıma ve duygu analizi
- Plaka tanıma
- Hareket algılama ve güvenlik
- (Geliştirilecek: Nesne takibi, segmentasyon, stil transferi, 3D poz tahmini, vb.)

---

## 🛠️ Özellikler
- Modüler ve okunabilir Python kodları
- Sınıf tabanlı ve fonksiyonel örnekler
- Gerçek zamanlı webcam/video işleme
- Otomatik model indirme ve yönetimi
- Türkçe açıklamalar ve kullanıcı dostu menüler
- Hata yönetimi ve performans testleri
- Görselleştirme ve analiz araçları

---

## 👨‍💻 Katkı ve Geliştirme
- [Pull request](https://github.com/erent8/opencv-doc/pulls) ve [issue](https://github.com/erent8/opencv-doc/issues) açarak katkıda bulunabilirsiniz
- Yeni alıştırma, proje veya örnek ekleyebilirsiniz
- Kodunuzu Türkçe açıklamalarla ve temiz şekilde yazmaya özen gösterin
- Büyük model dosyalarını GitHub'a yüklemeyin, otomatik indirme sistemini kullanın
- [Fork](https://github.com/erent8/opencv-doc/fork) yaparak kendi geliştirmelerinizi yapabilirsiniz

---

## 📄 Lisans
MIT Lisansı altında sunulmuştur. Detaylar için [LICENSE](https://github.com/erent8/opencv-doc/blob/main/LICENSE) dosyasına bakınız.

---

## 📞 İletişim ve Topluluk
- Soru, öneri ve katkılarınız için [GitHub Issues](https://github.com/erent8/opencv-doc/issues) bölümünü kullanabilirsiniz
- E-posta: erenterzi@protonmail.com
- X: [@therenn8](https://x.com/therenn8)
- GitHub: [@erent8](https://github.com/erent8)
- Proje Linki: [https://github.com/erent8/opencv-doc](https://github.com/erent8/opencv-doc)

---

## 🌟 Vizyon
Bu proje, Türkçe kaynak eksikliğini gidermek, topluluğa modern ve pratik bir bilgisayarla görme/makine öğrenmesi eğitim seti sunmak ve gerçek dünya uygulamalarına ilham vermek için hazırlanmıştır. Her seviyeden geliştiriciye açık, sürekli güncellenen ve topluluk katkısına açık bir projedir.

---

**Teşekkürler!**

Eren Terzi 
