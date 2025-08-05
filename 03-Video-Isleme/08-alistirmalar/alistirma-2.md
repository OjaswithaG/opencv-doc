# 🎬 Alıştırma 2: Real-time Video Filtreleme ve Analiz

## 📋 Problem Tanımı

Real-time video akışı üzerinde çeşitli filtreler ve analiz teknikleri uygulayan bir sistem geliştirin. Bu sistem webcam'den gelen görüntüleri gerçek zamanlı olarak işleyerek farklı görsel efektler ve analiz sonuçları gösterecek.

## 🎯 Gereksinimler

### Temel Özellikler (Zorunlu)
1. **Real-time Video Filtreleme**
   - Multiple filter modes (Blur, Sharpen, Edge, Emboss)
   - Smooth filter transitions
   - Real-time filter switching
   - Filter intensity control

2. **Histogram Analizi**
   - Real-time histogram calculation
   - RGB histogram display
   - Histogram equalization
   - Adaptive histogram stretching

3. **Frame Analizi**
   - Brightness/contrast analysis
   - Motion intensity tracking
   - Frame quality metrics
   - Color distribution analysis

4. **Kullanıcı Arayüzü**
   - Multi-window display (original + filtered + histogram)
   - Interactive controls (keyboard shortcuts)  
   - Real-time statistics overlay
   - Performance monitoring (FPS, processing time)

### İleri Özellikler (Bonus)
1. **Gelişmiş Filtreler**
   - Custom convolution kernels
   - Frequency domain filtering
   - Noise reduction algorithms
   - Artistic style filters

2. **Video Kayıt**
   - Filtered video recording
   - Multiple format support
   - Real-time compression
   - Timestamp overlay

3. **Performance Optimization**
   - Multi-threading support
   - GPU acceleration hints
   - Adaptive quality control
   - Memory usage optimization

## 📊 Beklenen Çıktılar

### Görsel Çıktılar
- Ana video penceresi (original feed)
- Filtered video penceresi 
- Real-time histogram display
- Statistics overlay panel

### Performance Metrikleri
- Real-time FPS monitoring
- Filter processing time
- Memory usage tracking
- Quality metrics display

### Konsol Çıktıları
```
🎬 Real-time Video Filtreleme Sistemi
📷 Webcam başlatıldı: 640x480 @ 30fps
🎯 Filter modu: Blur (intensity: 5)
📊 FPS: 28.5 | İşlem süresi: 12.3ms
🔄 Filter değiştirildi: Sharpen
📈 Ortalama parlaklık: 125.6
⚡ Sistem performansı: İyi
```

## 🛠️ Teknik Şartname

### Kullanılacak Teknolojiler
- **OpenCV**: Video işleme ve GUI
- **NumPy**: Matematiksel işlemler ve array operations
- **Matplotlib**: Histogram görselleştirme (opsiyonel)
- **Threading**: Async processing
- **Time**: Performance monitoring

### Performans Hedefleri
- **FPS**: Minimum 25 FPS real-time işleme
- **Latency**: Filter değişimi maksimum 100ms
- **Memory**: Maksimum 150MB RAM kullanımı
- **CPU**: Tek çekirdekte maksimum %60 kullanım

### Filter Algoritmaları
```python
# Predefined Kernels
BLUR_KERNEL = np.ones((5,5), np.float32) / 25
SHARPEN_KERNEL = np.array([[-1,-1,-1],
                          [-1, 9,-1], 
                          [-1,-1,-1]])
EDGE_KERNEL = np.array([[-1,-1,-1],
                       [-1, 8,-1],
                       [-1,-1,-1]])
EMBOSS_KERNEL = np.array([[-2,-1,0],
                         [-1, 1,1],
                         [ 0, 1,2]])

# Filter Parameters
FILTER_MODES = ['Original', 'Blur', 'Sharpen', 'Edge', 'Emboss', 'Custom']
INTENSITY_RANGE = (1, 10)
DEFAULT_INTENSITY = 5
```

## 📝 İmplementasyon Rehberi

### Adım 1: Video Capture Setup
1. Webcam bağlantısı ve konfigürasyon
2. Frame rate ve resolution ayarları
3. Buffer management

### Adım 2: Filter System
1. Kernel-based filtering implementation
2. Filter mode switching mechanism
3. Intensity control system

### Adım 3: Histogram Analysis
1. Real-time histogram calculation
2. RGB channel analysis  
3. Statistics computation

### Adım 4: UI ve Display
1. Multi-window management
2. Real-time overlay information
3. Keyboard control handling

### Adım 5: Performance Monitoring
1. FPS calculation
2. Processing time measurement
3. Resource usage tracking

## 🧪 Test Senaryoları

### Test 1: Filter Performance
- **Senaryo**: Tüm filtreleri sırayla test et
- **Beklenen**: Smooth transitions, stable FPS
- **Kontrol**: FPS > 20, processing time < 50ms

### Test 2: Histogram Accuracy
- **Senaryo**: Bilinen renk kartları ile test
- **Beklenen**: Doğru histogram dağılımı
- **Kontrol**: Peak values correct positions

### Test 3: Long-term Stability
- **Senaryo**: 30 dakika sürekli çalıştırma
- **Beklenen**: Memory leak yok, stable performance
- **Kontrol**: Memory usage stable, FPS consistent

### Test 4: Filter Intensity
- **Senaryo**: Intensity 1'den 10'a kadar test
- **Beklenen**: Progressive effect changes
- **Kontrol**: Visual differences observable

### Test 5: Multi-window Performance
- **Senaryo**: Tüm pencereler açık çalıştırma
- **Beklenen**: All windows update smoothly
- **Kontrol**: No window freezing

## 💡 İpuçları ve Püf Noktaları

### Filter İpuçları
```python
# Efficient kernel application
def apply_kernel_filter(frame, kernel, intensity=1.0):
    # Scale kernel by intensity
    scaled_kernel = kernel * intensity
    # Apply filter
    filtered = cv2.filter2D(frame, -1, scaled_kernel)
    return filtered

# Smooth transitions between filters
def blend_filters(frame1, frame2, alpha=0.5):
    return cv2.addWeighted(frame1, alpha, frame2, 1-alpha, 0)
```

### Histogram İpuçları
```python
# Fast histogram calculation
def fast_histogram(frame):
    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
    return hist_b, hist_g, hist_r

# Real-time histogram display
def draw_histogram(hist_b, hist_g, hist_r, height=400, width=512):
    hist_image = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw histogram lines
    # ... implementation
    return hist_image
```

### Performance İpuçları
```python
# FPS calculation
class FPS_Counter:
    def __init__(self, window_size=30):
        self.times = []
        self.window_size = window_size
    
    def update(self):
        self.times.append(time.time())
        if len(self.times) > self.window_size:
            self.times.pop(0)
    
    def get_fps(self):
        if len(self.times) < 2:
            return 0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])
```

## 📈 Değerlendirme Metrikleri

### Fonksiyonellik (40%)
- ✅ Filter system çalışıyor
- ✅ Histogram analizi aktif
- ✅ Real-time processing
- ✅ UI kontrolleri responsive

### Performans (30%)
- ⚡ FPS > 20
- 💾 Memory kullanımı reasonable
- 🎯 Filter switching < 200ms
- 💿 CPU usage < 70%

### Kod Kalitesi (20%)
- 🧹 Temiz ve modular kod
- 🛡️ Error handling
- 📝 Documentation
- 🔧 Configurable parameters

### Yaratıcılık (10%)
- 💡 Custom filters
- 🎨 UI improvements
- 📊 Additional analytics
- 🔧 Advanced features

## 🚫 Yaygın Hatalar

1. **Memory Management**
   ```python
   # YANLIŞ - memory leak
   histograms = []
   while True:
       hist = calculate_histogram(frame)
       histograms.append(hist)  # Sürekli birikir!
   
   # DOĞRU - circular buffer
   class HistogramBuffer:
       def __init__(self, size=10):
           self.buffer = []
           self.size = size
   ```

2. **Filter Application**
   ```python
   # YANLIŞ - veri tipi problemi
   filtered = cv2.filter2D(frame, -1, kernel)  # uint8 overflow
   
   # DOĞRU - proper type handling
   frame_float = frame.astype(np.float32)
   filtered = cv2.filter2D(frame_float, -1, kernel)
   filtered = np.clip(filtered, 0, 255).astype(np.uint8)
   ```

3. **Performance Issues**
   ```python
   # YANLIŞ - her frame için yeniden hesaplama
   for frame in video:
       kernel = create_kernel()  # Expensive!
   
   # DOĞRU - kernel caching
   kernels = {
       'blur': create_blur_kernel(),
       'sharpen': create_sharpen_kernel()
   }
   ```

## 🎓 Öğrenme Hedefleri

Bu alıştırmayı tamamladığınızda şunları öğrenmiş olacaksınız:

- ✅ Real-time video processing
- ✅ Convolution filtering techniques
- ✅ Histogram analysis methods
- ✅ Multi-window GUI management
- ✅ Performance optimization strategies
- ✅ Memory management patterns
- ✅ Interactive user interfaces

---

**Başarılar! 🚀 Sorunlarla karşılaştığınızda `cozumler/cozum-2.py` dosyasına bakabilirsiniz.**