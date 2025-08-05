# 🎯 Alıştırma 3: İleri Video İşleme ve Nesne Takibi

## 📋 Problem Tanımı

İleri seviye video işleme teknikleri kullanarak multi-object tracking sistemi geliştirin. Bu sistem webcam'den gelen görüntülerde çoklu nesne tespiti, takibi ve trajectory analizi yapacak.

## 🎯 Gereksinimler

### Temel Özellikler (Zorunlu)
1. **Çoklu Nesne Tespiti**
   - Color-based object detection
   - Multiple color ranges configuration
   - Object filtering (size, shape)
   - Real-time detection feedback

2. **Nesne Takip Sistemi**
   - Multi-object tracking algorithms
   - Trajectory recording ve visualization
   - Object ID assignment ve management
   - Tracking consistency across frames

3. **Trajectory Analizi**
   - Movement pattern analysis
   - Speed calculation
   - Direction tracking
   - Path prediction

4. **Interaktif Konfigürasyon**
   - Dynamic HSV range adjustment
   - Detection parameter tuning
   - Real-time threshold control
   - Object selection interface

### İleri Özellikler (Bonus)
1. **Gelişmiş Tracking**
   - Kalman filter implementation
   - Hungarian algorithm for assignment
   - Occlusion handling
   - Multi-camera fusion

2. **Akıllı Analiz**
   - Behavioral pattern recognition
   - Anomaly detection
   - Statistical analysis
   - Heatmap generation

3. **Performance Optimization**
   - Region of Interest (ROI)
   - Multi-threading support
   - Adaptive frame rate
   - Memory optimization

## 📊 Beklenen Çıktılar

### Görsel Çıktılar
- Ana video stream (with detections)
- HSV color space tuning interface
- Trajectory visualization overlay
- Real-time statistics panel
- Heatmap visualization (bonus)

### Data Outputs
- Object trajectory data (JSON/CSV format)
- Performance metrics log
- Detection confidence scores
- Movement statistics

### Konsol Çıktıları
```
🎯 Multi-Object Tracking Sistemi
📷 Kamera başlatıldı: 640x480 @ 30fps
🎨 HSV Aralıkları:
   Kırmızı: H(0-10), S(50-255), V(50-255)
   Mavi: H(100-130), S(50-255), V(50-255)
🔍 Tespit: 3 nesne bulundu
📍 ID:1 - Kırmızı top (x:245, y:156) Hız:12.5px/s
📍 ID:2 - Mavi kutu (x:421, y:298) Hız:8.3px/s
📈 Toplam: 5 trajectory kaydedildi
```

## 🛠️ Teknik Şartname

### Kullanılacak Teknolojiler
- **OpenCV**: Video processing ve computer vision
- **NumPy**: Mathematical operations ve array handling
- **Collections**: Data structures (deque, defaultdict)
- **JSON**: Configuration ve data export
- **Math**: Geometric calculations

### Performans Hedefleri
- **FPS**: Minimum 20 FPS with tracking
- **Detection Latency**: Maksimum 50ms per frame
- **Memory**: Maksimum 200MB RAM usage
- **Tracking Accuracy**: %85+ object consistency

### Detection Parameters
```python
# Color Detection HSV Ranges
COLOR_RANGES = {
    'red': {'lower': (0, 50, 50), 'upper': (10, 255, 255)},
    'blue': {'lower': (100, 50, 50), 'upper': 130, 255, 255)},
    'green': {'lower': (40, 50, 50), 'upper': (80, 255, 255)},
    'yellow': {'lower': (20, 50, 50), 'upper': (30, 255, 255)}
}

# Object Detection Parameters
MIN_OBJECT_AREA = 500
MAX_OBJECT_AREA = 50000
MIN_SOLIDITY = 0.3
TRACKING_DISTANCE_THRESHOLD = 50
MAX_DISAPPEARED_FRAMES = 10
```

## 📝 İmplementasyon Rehberi

### Adım 1: Color Detection Setup
1. HSV color space conversion
2. Color range configuration system
3. Morphological operations for noise reduction
4. Contour detection ve filtering

### Adım 2: Object Tracking Implementation
1. Object detection ve feature extraction
2. ID assignment algorithm
3. Frame-to-frame tracking
4. Trajectory data management

### Adım 3: Trajectory Analysis
1. Path recording system
2. Movement statistics calculation
3. Velocity ve direction analysis
4. Prediction algorithms

### Adım 4: Interactive Interface
1. HSV trackbar interface
2. Real-time parameter adjustment
3. Object selection tools
4. Configuration save/load

### Adım 5: Visualization ve Analytics
1. Trajectory overlay rendering
2. Statistics display system
3. Heatmap generation
4. Export functionality

## 🧪 Test Senaryoları

### Test 1: Single Object Tracking
- **Senaryo**: Tek renkli nesneyi hareket ettir
- **Beklenen**: Consistent ID, smooth trajectory
- **Kontrol**: Tracking accuracy > %90

### Test 2: Multi-Object Scenario
- **Senaryo**: 3+ farklı renkli nesne aynı anda
- **Beklenen**: Unique IDs, no ID switching
- **Kontrol**: Each object tracked separately

### Test 3: Occlusion Handling
- **Senaryo**: Nesnelerin birbiriyle çakışması
- **Beklenen**: ID consistency after occlusion
- **Kontrol**: Object reappearance detection

### Test 4: Speed Calculation
- **Senaryo**: Bilinen hızda nesne hareketi
- **Beklenen**: Accurate speed measurement
- **Kontrol**: ±%10 accuracy margin

### Test 5: Long-term Tracking
- **Senaryo**: 10 dakika sürekli tracking
- **Beklenen**: Memory stable, no performance degradation
- **Kontrol**: FPS consistent, memory usage stable

## 💡 İpuçları ve Püf Noktaları

### Color Detection İpuçları
```python
# HSV color space'de renk tespiti
def detect_color_objects(frame, color_ranges):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_objects = []
    
    for color_name, color_range in color_ranges.items():
        # Color mask oluştur
        mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if is_valid_object(contour):
                detected_objects.append({
                    'color': color_name,
                    'contour': contour,
                    'centroid': calculate_centroid(contour)
                })
    
    return detected_objects
```

### Object Tracking İpuçları
```python
# Basit centroid tracking algoritması
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detected_centroids):
        # Tracking logic implementation
        # Distance calculation
        # Hungarian algorithm for assignment
        pass
```

### Trajectory Analysis İpuçları
```python
# Trajectory ve speed calculation
def calculate_movement_stats(trajectory):
    if len(trajectory) < 2:
        return {'speed': 0, 'distance': 0, 'direction': 0}
    
    # Total distance
    total_distance = 0
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i-1][0]
        dy = trajectory[i][1] - trajectory[i-1][1]
        total_distance += np.sqrt(dx**2 + dy**2)
    
    # Average speed (pixel/frame)
    time_frames = len(trajectory) - 1
    avg_speed = total_distance / time_frames if time_frames > 0 else 0
    
    # Direction (degrees)
    start_point = trajectory[0]
    end_point = trajectory[-1]
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    direction = np.degrees(np.arctan2(dy, dx))
    
    return {
        'speed': avg_speed,
        'distance': total_distance,
        'direction': direction
    }
```

## 📈 Değerlendirme Metrikleri

### Fonksiyonellik (35%)
- ✅ Color detection çalışıyor
- ✅ Multi-object tracking aktif
- ✅ Trajectory recording
- ✅ Interactive configuration

### Doğruluk (25%)
- 🎯 Detection accuracy > %80
- 🔍 Tracking consistency > %85
- 📏 Speed calculation accuracy
- 🎨 Color discrimination

### Performans (25%)
- ⚡ FPS > 15 with tracking
- 💾 Memory usage reasonable
- 🚀 Real-time processing
- 📊 Smooth visualization

### Kod Kalitesi (15%)
- 🧹 Clean, modular architecture
- 🛡️ Robust error handling
- 📝 Comprehensive documentation
- 🔧 Configurable parameters

## 🚫 Yaygın Hatalar

1. **ID Switching Problemi**
   ```python
   # YANLIŞ - insufficient distance threshold
   def assign_ids(current_objects, previous_objects):
       # Çok küçük distance threshold ID karışımına neden olur
       DISTANCE_THRESHOLD = 10  # Çok küçük!
   
   # DOĞRU - adaptive threshold
   def assign_ids(current_objects, previous_objects):
       # Object boyutuna göre adaptive threshold
       adaptive_threshold = max(30, object_size * 0.5)
   ```

2. **Memory Leak**
   ```python
   # YANLIŞ - unlimited trajectory storage
   class ObjectTracker:
       def __init__(self):
           self.trajectories = {}  # Sürekli büyür!
   
   # DOĞRU - bounded trajectory storage
   class ObjectTracker:
       def __init__(self, max_trajectory_length=100):
           self.trajectories = defaultdict(lambda: deque(maxlen=max_trajectory_length))
   ```

3. **HSV Range Problemi**
   ```python
   # YANLIŞ - fixed HSV ranges
   red_lower = (0, 50, 50)    # Lighting'a göre değişmeli
   red_upper = (10, 255, 255)
   
   # DOĞRU - adaptive HSV with user control
   def create_hsv_trackbars():
       cv2.createTrackbar('H Min', 'HSV Control', 0, 179, lambda x: None)
       cv2.createTrackbar('S Min', 'HSV Control', 50, 255, lambda x: None)
       # ... other trackbars
   ```

## 🎓 Öğrenme Hedefleri

Bu alıştırmayı tamamladığınızda şunları öğrenmiş olacaksınız:

- ✅ Color-based object detection techniques
- ✅ Multi-object tracking algorithms
- ✅ Trajectory analysis methods
- ✅ HSV color space manipulation
- ✅ Centroid tracking implementation
- ✅ Interactive parameter tuning
- ✅ Performance optimization strategies
- ✅ Data visualization techniques

## 🚀 Bonus Görevler

Temel alıştırmayı tamamladıysanız bunları deneyin:

- [ ] Kalman filter implementation
- [ ] Deep learning object detection integration
- [ ] Multi-camera tracking fusion
- [ ] Real-time heatmap generation
- [ ] Behavioral pattern analysis
- [ ] Anomaly detection system
- [ ] 3D trajectory reconstruction
- [ ] Export to video with overlays

---

**Başarılar! 🚀 Sorunlarla karşılaştığınızda `cozumler/cozum-3.py` dosyasına bakabilirsiniz.**