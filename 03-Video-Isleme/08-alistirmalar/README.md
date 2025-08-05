# 🎬 Video İşleme Alıştırmaları

Bu klasör video işleme konularında pratik yapabileceğiniz kapsamlı alıştırmalar içerir. Her alıştırma gerçek dünya problemlerine odaklanmış olup, öğrendiğiniz teknikleri uygulama fırsatı sunar.

## 📋 Alıştırma Listesi

### 1. 🔒 Video Güvenlik Sistemi (Video Surveillance System)
**Dosya:** `alistirma-1.py` / `alistirma-1.md`
**Seviye:** Orta
**Süre:** 2-3 saat

**Konu:** Hareket algılama tabanlı güvenlik sistemi
- Background subtraction ile hareket tespiti
- Otomatik kayıt sistemi
- Olay tespiti ve alarm sistemi
- Video metadata yönetimi

**Öğrenilen Teknikler:**
- MOG2 background subtraction
- Contour detection ve analiz
- Video yazma ve kayıt
- Event-driven programming
- File sistem yönetimi

### 2. 🎬 Real-time Video Filtreleme ve Analiz
**Dosya:** `alistirma-2.py` / `alistirma-2.md`
**Seviye:** Orta-İleri
**Süre:** 2-3 saat

**Konu:** Real-time video filtreleme ve analiz sistemi
- Multiple filter modes (Blur, Sharpen, Edge, Emboss)
- Real-time histogram analizi
- Frame quality metrics
- Interactive parameter control

**Öğrenilen Teknikler:**
- Convolution filtering
- Real-time histogram hesaplama
- Performance monitoring (FPS tracking)
- Multi-window GUI management
- Interactive user interfaces

### 3. 🎯 İleri Video İşleme ve Nesne Takibi
**Dosya:** `alistirma-3.py` / `alistirma-3.md`
**Seviye:** İleri
**Süre:** 3-4 saat

**Konu:** Çoklu nesne tespiti, takibi ve trajectory analizi
- Color-based object detection
- Multi-object tracking algorithms
- Trajectory recording ve analiz
- Interactive HSV tuning

**Öğrenilen Teknikler:**
- HSV color space manipulation
- Centroid tracking implementation
- Distance matrix calculations
- Trajectory visualization
- Object ID management

## 🚀 Nasıl Başlarım?

### Adım 1: Ön Hazırlık
```bash
# Gerekli kütüphaneleri yükle
pip install opencv-python numpy matplotlib scikit-image

# Test videoları hazırla (opsiyonel)
# Webcam kullanabilir veya kendi videolarınızı kullanabilirsiniz
```

### Adım 2: Alıştırma Seçimi
1. **Markdown dosyasını okuyun** (örn: `alistirma-1.md`)
   - Problem tanımını anlayın
   - Gereksinimleri inceleyin
   - Beklenen çıktıları gözden geçirin

2. **Python dosyasını açın** (örn: `alistirma-1.py`)
   - TODO kısımlarını bulun
   - İlgili fonksiyonları tamamlayın
   - Test edin ve geliştirin

### Adım 3: Test ve Doğrulama
- Webcam ile test edin
- Farklı senaryoları deneyin
- Performansı optimize edin
- Çözümünüzü `cozumler/` klasöründeki örnek ile karşılaştırın

## 📁 Klasör Yapısı

```
08-alistirmalar/
├── README.md                   # Bu dosya
├── alistirma-1.md             # Güvenlik sistemi problemi
├── alistirma-1.py             # Güvenlik sistemi template
├── alistirma-2.md             # Video filtreleme problemi  
├── alistirma-2.py             # Video filtreleme template
├── alistirma-3.md             # Nesne takibi problemi
├── alistirma-3.py             # Nesne takibi template
├── cozumler/                  # Örnek çözümler
│   ├── cozum-1.py            # Güvenlik sistemi çözümü
│   ├── cozum-2.py            # Video filtreleme çözümü
│   └── cozum-3.py            # Nesne takibi çözümü
└── test-veriler/              # Test için örnek dosyalar
    ├── README.md             # Test verileri açıklaması
    └── video_olusturucu.py   # Test videosu oluşturucu
```

## 🎯 Değerlendirme Kriterleri

### Temel Kriterler (70%)
- ✅ Temel fonksiyonalite çalışıyor
- ✅ Kod hatasız çalışıyor  
- ✅ Gerekli çıktıları üretiyor
- ✅ Basic test senaryolarını geçiyor

### İleri Kriterler (20%)
- 🚀 Performans optimizasyonu
- 🎨 Kullanıcı arayüzü iyileştirmeleri
- 📊 Ek analiz ve raporlama
- 🔧 Error handling ve robustluk

### Yaratıcılık (10%)
- 💡 Orijinal özellikler ekleme
- 🎯 Problem çözme yaklaşımı
- 📈 Ek analiz teknikleri
- 🎨 Görselleştirme iyileştirmeleri

## 💡 İpuçları ve Tavsiyeler

### Genel İpuçları
1. **Küçük adımlarla ilerleyin** - Her TODO'yu tek tek tamamlayın
2. **Sık test edin** - Her değişiklikten sonra çalıştırın
3. **Debug print'leri kullanın** - Ara sonuçları kontrol edin
4. **Performansı izleyin** - FPS ve processing time'ı gözlemleyin

### Yaygın Hatalar
- ❌ Frame boyutlarını kontrol etmemek
- ❌ Null pointer exception'ları handle etmemek
- ❌ Memory leak'leri (webcam release etmemek)
- ❌ File path'lerinde işletim sistemi farklılıkları

### Performans İpuçları
- 🚀 Frame'leri resize edin (işlem hızı için)
- 🚀 ROI (Region of Interest) kullanın
- 🚀 Gereksiz işlemleri optimizasyon yapın
- 🚀 Multi-threading kullanmayı deneyin

## 🔧 Sorun Giderme

### Webcam Problemleri
```python
# Webcam test kodu
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam bulunamadı!")
    # Alternatif: Test videosu kullan
    cap = cv2.VideoCapture('test_video.avi')
```

### OpenCV Kurulum Problemleri
```bash
# Tam kurulum
pip install opencv-python
pip install opencv-contrib-python

# Eğer hala sorun varsa
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.0.76
```

### Performans Problemleri
```python
# Frame boyutunu küçült
frame = cv2.resize(frame, (640, 480))

# FPS limit koy
cv2.waitKey(33)  # ~30 FPS
```

## 📚 Ek Kaynaklar

### Dokümantasyon
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

### Video İşleme Referansları
- Computer Vision: Algorithms and Applications (Szeliski)
- Learning OpenCV 4 (Kaehler & Bradski)
- Multiple Object Tracking Papers (MOT Challenge)

### Online Kurslar
- OpenCV Course (PyImageSearch)
- Computer Vision Nanodegree (Udacity)
- Computer Vision Specialization (Coursera)

## 🏆 Başarı Rozeti Sistemi

Alıştırmaları tamamladıkça şu rozetleri kazanabilirsiniz:

- 🥉 **Başlangıç**: İlk alıştırmayı tamamla
- 🥈 **Gelişen**: İki alıştırmayı tamamla  
- 🥇 **Uzman**: Üç alıştırmayı da tamamla
- 🌟 **Usta**: Tüm bonus özelliklerle tamamla
- 🚀 **İnovatör**: Orijinal özellikler ekle

## 🤝 Yardım ve Destek

Sorunlarla karşılaştığınızda:

1. **README dosyalarını** tekrar okuyun
2. **Örnek çözümleri** inceleyin (ama kopyalamayın!)
3. **Debug print'leri** kullanarak sorunu isolate edin  
4. **Küçük test örnekleri** yazın
5. **Online dokümantasyonu** kontrol edin

**Not:** Bu alıştırmalar öğrenme amaçlıdır. Çözümleri direkt kopyalamak yerine anlayarak implement etmeye odaklanın.

---

**Video İşleme Alıştırmaları - OpenCV Türkçe Dokümantasyonu**  
*Hazırlayan: Eren Terzi - 2024*