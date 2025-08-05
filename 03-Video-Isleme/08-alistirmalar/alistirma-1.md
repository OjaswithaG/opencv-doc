# 🔒 Alıştırma 1: Video Güvenlik Sistemi

## 📋 Problem Tanımı

Bir video güvenlik sistemi geliştirin. Bu sistem webcam'den gelen görüntüleri sürekli izleyerek hareket tespit ettiğinde otomatik olarak kayıt yapacak ve alarm verecek.

## 🎯 Gereksinimler

### Temel Özellikler (Zorunlu)
1. **Hareket Algılama**
   - Background subtraction ile hareket tespiti
   - Gürültü filtreleme (minimum hareket alanı)
   - Hareket durumu göstergesi

2. **Otomatik Kayıt Sistemi**
   - Hareket tespit edildiğinde kayıt başlatma
   - Hareket bittiğinde kayıt durdurma
   - Video dosyalarını timestamp ile kaydetme

3. **Alarm Sistemi**
   - Görsel alarm (ekranda uyarı)
   - Hareket tespit edildiğinde frame'i kırmızı çerçeve ile gösterme
   - Alarm sayacı (kaç kez alarm verildi)

4. **Kullanıcı Arayüzü**
   - Canlı video görüntüsü
   - Sistem durumu (aktif/pasif)
   - İstatistikler (toplam kayıt süresi, alarm sayısı)
   - Basit kontroller (q:çıkış, r:reset, s:sistem on/off)

### İleri Özellikler (Bonus)
1. **Gelişmiş Analiz**
   - Hareket yoğunluğu analizi
   - Nesne boyutu filtreleme
   - Multiple motion zones

2. **Kayıt Yönetimi**
   - Video compression
   - Otomatik eski kayıt silme
   - Video thumbnail generation

3. **Configuration**
   - Ayarlanabilir sensitivity
   - Farklı background subtraction algoritmaları
   - Kayıt süresi limitleri

## 📊 Beklenen Çıktılar

### Görsel Çıktılar
- Ana video penceresi (live feed)
- Motion detection mask'i (küçük pencere)
- Sistem durumu bilgileri (overlay text)
- Kayıt durumu göstergesi

### Dosya Çıktıları
- Video kayıtları: `motion_YYYYMMDD_HHMMSS.avi`
- Log dosyası: `security_log.txt`
- Configuration dosyası: `config.ini` (bonus)

### Konsol Çıktıları
```
🔒 Video Güvenlik Sistemi Başlatıldı
📷 Webcam bağlandı: 640x480 @ 30fps
🎯 Background model öğreniliyor...
⚡ Sistem aktif - hareket izleniyor
🚨 HAREKET TESPİT EDİLDİ! (Area: 2450px)
🎬 Kayıt başladı: motion_20241201_143022.avi
⏹️ Kayıt durduruldu: 15.2 saniye
📊 Toplam: 3 alarm, 45.6s kayıt
```

## 🛠️ Teknik Şartname

### Kullanılacak Teknolojiler
- **OpenCV**: Video işleme ve GUI
- **NumPy**: Matematiksel işlemler
- **DateTime**: Timestamp işlemleri
- **OS/Path**: Dosya yönetimi

### Performans Hedefleri
- **FPS**: Minimum 20 FPS real-time işleme
- **Latency**: Hareket tespitinde maksimum 500ms gecikme
- **Memory**: Maksimum 100MB RAM kullanımı
- **Storage**: Video kayıtları H.264 compression ile

### Algoritma Detayları
```python
# Background Subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    detectShadows=True,
    varThreshold=50,
    history=500
)

# Motion Detection Parameters
MIN_MOTION_AREA = 1000  # piksel
LEARNING_RATE = 0.01
IDLE_TIMEOUT = 3.0      # saniye

# Recording Parameters
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'XVID')
VIDEO_FPS = 20
MAX_RECORDING_TIME = 300  # saniye
```

## 📝 İmplementasyon Rehberi

### Adım 1: Temel Yapı
1. Webcam bağlantısı kurma
2. Background subtractor initialize etme
3. Ana video loop oluşturma

### Adım 2: Hareket Algılama
1. Frame'i background subtractor'dan geçirme
2. Noise filtreleme (morphological operations)
3. Contour detection ile hareket alanı hesaplama

### Adım 3: Kayıt Sistemi
1. Motion state tracking (idle, motion detected, recording)
2. VideoWriter ile kayıt başlatma/durdurma
3. Filename generation with timestamp

### Adım 4: UI ve İstatistikler
1. Overlay text ile bilgileri gösterme
2. Motion mask'i küçük pencerede gösterme
3. Keyboard kontrollerini implement etme

### Adım 5: Optimizasyon
1. Performance monitoring
2. Memory management
3. Error handling

## 🧪 Test Senaryoları

### Test 1: Temel Hareket Algılama
- **Senaryo**: Kameranın önünde el sallama
- **Beklenen**: Hareket tespit edilmeli, alarm verilmeli
- **Kontrol**: Motion area > threshold

### Test 2: Kayıt Sistemi
- **Senaryo**: 10 saniye hareket, 5 saniye durma
- **Beklenen**: 10 saniyelik video dosyası oluşmalı
- **Kontrol**: Video dosya boyutu > 0, süre ~10s

### Test 3: Gürültü Filtreleme
- **Senaryo**: Küçük hareketler (parmak sallamak)
- **Beklenen**: Alarm verilmemeli
- **Kontrol**: Motion area < threshold

### Test 4: Çoklu Olay
- **Senaryo**: 3 kez ardışık hareket
- **Beklenen**: 3 ayrı video dosyası veya birleşik kayıt
- **Kontrol**: Dosya sayısı/içeriği doğru

### Test 5: Uzun Süreli Çalışma
- **Senaryo**: 10 dakika sürekli çalıştırma
- **Beklenen**: Memory leak olmaması, kararlı FPS
- **Kontrol**: Memory usage stable, FPS > 15

## 💡 İpuçları ve Püf Noktaları

### Hareket Algılama İpuçları
```python
# Morphological operations ile gürültü temizleme
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Contour area filtreleme
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > MIN_AREA)
```

### Kayıt Sistemi İpuçları
```python
# Unique filename generation
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"motion_{timestamp}.avi"

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
```

### Performance İpuçları
```python
# Frame resize for better performance
frame = cv2.resize(frame, (640, 480))

# Skip frames if processing is slow
frame_skip = 2
if frame_count % frame_skip == 0:
    # Process frame
```

## 📈 Değerlendirme Metrikleri

### Fonksiyonellik (50%)
- ✅ Hareket algılama çalışıyor
- ✅ Kayıt sistemi çalışıyor  
- ✅ UI ve kontroller çalışıyor
- ✅ Dosya outputları doğru

### Performans (20%)
- ⚡ FPS > 15
- 💾 Memory kullanımı reasonable
- 🎯 Motion detection latency < 1s
- 💿 Video file size reasonable

### Kod Kalitesi (20%)
- 🧹 Temiz ve okunabilir kod
- 🛡️ Error handling
- 📝 Yeterli comment
- 🔧 Modular yapı

### Yaratıcılık (10%)
- 💡 Bonus özellikler
- 🎨 UI iyileştirmeleri
- 📊 Ek analizler
- 🔧 Configuration options

## 🚫 Yaygın Hatalar

1. **Webcam release etmemek**
   ```python
   # YANLIŞ
   # Program kapanırken cap.release() çağrılmıyor
   
   # DOĞRU
   try:
       # main loop
   finally:
       cap.release()
       cv2.destroyAllWindows()
   ```

2. **Background learning rate yanlış**
   ```python
   # Çok yüksek learning rate - background çok hızlı adapte oluyor
   # Çok düşük - background yavaş adapte oluyor
   learning_rate = 0.01  # İyi başlangıç değeri
   ```

3. **Motion area threshold yanlış**
   ```python
   # Çok düşük threshold - her küçük gürültüde alarm
   # Çok yüksek - gerçek hareket tespit edilmiyor
   MIN_MOTION_AREA = frame.shape[0] * frame.shape[1] * 0.001  # Frame'in %0.1'i
   ```

## 🎓 Öğrenme Hedefleri

Bu alıştırmayı tamamladığınızda şunları öğrenmiş olacaksınız:

- ✅ Background subtraction teknikleri
- ✅ Real-time video processing
- ✅ Event-driven programming
- ✅ File I/O operations
- ✅ Performance optimization
- ✅ Error handling patterns
- ✅ User interface design (OpenCV)

---

**Başarılar! 🚀 Sorunlarla karşılaştığınızda `cozumler/cozum-1.py` dosyasına bakabilirsiniz.**