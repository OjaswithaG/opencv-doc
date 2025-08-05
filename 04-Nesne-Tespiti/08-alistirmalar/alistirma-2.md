# 🛡️ Alıştırma 2: Güvenlik İzleme Sistemi

**Seviye:** ⭐⭐⭐⭐ (İleri)  
**Süre:** 3-4 saat  
**Hedef:** Real-time güvenlik kamerası sistemi geliştirme

## 🎯 Alıştırma Hedefi

Bu alıştırmada **profesyonel seviyede bir güvenlik izleme sistemi** oluşturacaksınız. Sistem gerçek zamanlı olarak güvenlik tehditleri tespit edecek, alarm üretecek ve tüm olayları kayıt altına alacak.

## 🛠️ Teknik Gereksinimler

### Kullanılacak Teknolojiler
- **DNN Object Detection** (YOLO/SSD for person, vehicle detection)
- **Face Recognition** (Known vs Unknown person classification)
- **Motion Detection** (Background subtraction, optical flow)
- **Behavior Analysis** (Loitering, intrusion detection)
- **Alert System** (Visual/audio alerts, logging)

### Gerekli Kütüphaneler
```python
import cv2
import numpy as np
import time
import datetime
import json
import os
from collections import deque, defaultdict
import threading
```

## 📋 Sistem Özellikleri

### 🔧 Temel Özellikler (Zorunlu)

#### 1️⃣ Multi-Zone Monitoring
- [ ] **Detection zones** - kullanıcı tanımlı izleme alanları
- [ ] **Zone types**: Public, Restricted, High-Security
- [ ] **Zone-specific rules** - her alan için farklı kurallar
- [ ] **Visual zone indicators** - alanları ekranda gösterir

#### 2️⃣ Person Detection & Classification
- [ ] **Person detection** - DNN ile insan tespiti
- [ ] **Face recognition** - bilinen/bilinmeyen kişi sınıflandırma
- [ ] **Person tracking** - kişileri ID ile takip
- [ ] **Behavior analysis** - şüpheli davranış tespiti

#### 3️⃣ Security Events
- [ ] **Intrusion detection** - yasak alana giriş
- [ ] **Loitering detection** - uzun süre aynı yerde kalma
- [ ] **Unauthorized access** - bilinmeyen kişi tespiti  
- [ ] **Motion alerts** - hareket tabanlı uyarılar

#### 4️⃣ Alert & Logging System
- [ ] **Visual alerts** - ekranda uyarı mesajları
- [ ] **Audio alerts** - sesli uyarı sistemi (optional)
- [ ] **Event logging** - tüm olayları JSON'a kaydet
- [ ] **Screenshot capture** - olay anında fotoğraf çek

### 🌟 Gelişmiş Özellikler (Bonus)

#### 5️⃣ Advanced Analytics
- [ ] **Crowd detection** - kalabalık tespiti
- [ ] **Vehicle detection** - araç tespiti ve plaka okuma
- [ ] **Object left behind** - terk edilmiş nesne tespiti
- [ ] **Direction analysis** - hareket yönü analizi

#### 6️⃣ System Integration
- [ ] **Database integration** - SQLite ile event storage
- [ ] **Web dashboard** - web arayüzü ile monitoring
- [ ] **Email notifications** - otomatik email gönderimi
- [ ] **Mobile alerts** - telefon bildirimleri

## 🏗️ Sistem Mimarisi

### Sınıf Yapısı
```python
class SecurityMonitoringSystem:
    def __init__(self):
        # Detection modules
        self.person_detector = PersonDetector()
        self.face_recognizer = FaceRecognizer()
        self.motion_detector = MotionDetector()
        self.behavior_analyzer = BehaviorAnalyzer()
        
        # Security zones
        self.zones = {}
        self.zone_rules = {}
        
        # Alert system
        self.alert_manager = AlertManager()
        self.event_logger = EventLogger()
        
        # Tracking
        self.person_tracker = PersonTracker()
        self.track_history = deque(maxlen=1000)

class SecurityZone:
    def __init__(self, name, points, zone_type, rules):
        self.name = name
        self.polygon = points
        self.zone_type = zone_type  # 'public', 'restricted', 'high_security'
        self.rules = rules
        self.violations = []

class SecurityEvent:
    def __init__(self, event_type, zone, person_id, timestamp, confidence):
        self.event_type = event_type
        self.zone = zone
        self.person_id = person_id
        self.timestamp = timestamp
        self.confidence = confidence
        self.screenshot_path = None
```

### Ana Program Akışı
```
1. System Initialization
   ├── Load detection models
   ├── Setup security zones
   ├── Initialize alert system
   └── Load known faces database

2. Main Security Loop
   ├── Frame capture & preprocessing
   ├── Person detection & tracking
   ├── Face recognition
   ├── Zone violation check
   ├── Behavior analysis
   ├── Alert generation
   └── Event logging

3. Alert Processing (Separate Thread)
   ├── Alert prioritization
   ├── Notification dispatch
   ├── Screenshot capture
   └── Database update
```

## 📊 Güvenlik Kuralları

### 🚨 Event Types
```python
SECURITY_EVENTS = {
    'INTRUSION': {
        'priority': 'HIGH',
        'description': 'Unauthorized person in restricted zone',
        'action': ['alert', 'screenshot', 'log']
    },
    'LOITERING': {
        'priority': 'MEDIUM', 
        'description': 'Person staying too long in area',
        'action': ['alert', 'log']
    },
    'UNKNOWN_PERSON': {
        'priority': 'MEDIUM',
        'description': 'Unrecognized person detected',
        'action': ['alert', 'screenshot', 'log']
    },
    'MOTION_DETECTED': {
        'priority': 'LOW',
        'description': 'Motion detected in monitored area',
        'action': ['log']
    }
}
```

### 🏛️ Zone Configuration
```python
ZONE_RULES = {
    'public': {
        'max_occupancy': 50,
        'loitering_time': 300,  # 5 minutes
        'unknown_person_alert': False
    },
    'restricted': {
        'max_occupancy': 5,
        'loitering_time': 60,   # 1 minute
        'unknown_person_alert': True,
        'authorized_faces': ['admin', 'security', 'employee']
    },
    'high_security': {
        'max_occupancy': 1,
        'loitering_time': 30,   # 30 seconds
        'unknown_person_alert': True,
        'authorized_faces': ['admin', 'security'],
        'motion_sensitivity': 'high'
    }
}
```

## 🎮 Kontrol Sistemi

### Klavye Kontrolleri
```
ESC     : Sistem kapatma
SPACE   : Pause/Resume monitoring
1-9     : Zone selection/editing
a       : Alert system on/off
s       : Screenshot capture
r       : Reset all alerts
l       : Live/Playback mode toggle
z       : Zone configuration mode
f       : Face registration mode
```

### Mouse Kontrolleri
```
Left Click      : Zone point definition
Right Click     : Context menu (zone operations)
Double Click    : Person tracking focus
Drag           : Zone area selection
Ctrl+Click     : Multi-zone selection
```

## 🧪 Test Senaryoları

### 📝 Test Case 1: Basic Monitoring
1. **Sistem başlat ve 3 zone tanımla** (public, restricted, high-security)
2. **Public zone'da normal hareket** - alert olmamalı
3. **Restricted zone'a giriş** - intrusion alert
4. **Event log'un doğru kayıt edildiğini kontrol et**

### 📝 Test Case 2: Face Recognition
1. **Bilinen yüzleri sisteme kaydet** (5-10 kişi)
2. **Bilinmeyen kişi restricted zone'a girsin** - unknown person alert
3. **Bilinen kişi aynı zone'a girsin** - alert olmamalı
4. **Recognition accuracy'yi değerlendir**

### 📝 Test Case 3: Loitering Detection
1. **Public zone'da 5 dakika dur** - loitering alert
2. **Restricted zone'da 1 dakika dur** - daha erken alert
3. **High-security zone'da 30 saniye dur** - immediate alert
4. **Timer accuracy'yi kontrol et**

### 📝 Test Case 4: Multi-Person Tracking
1. **3-4 kişi aynı anda farklı zone'lara girsin**
2. **Her kişinin doğru track edildiğini kontrol et**
3. **ID collision'ın olmadığını doğrula**
4. **Performance impact'ini ölç**

### 📝 Test Case 5: System Stress Test
1. **1 saat boyunca continuous monitoring**
2. **Memory usage ve CPU utilization'ı izle**
3. **Alert system responsiveness'ı test et**
4. **Log file integrity'yi kontrol et**

## 💻 İmplementasyon Adımları

### 🏃‍♂️ Hızlı Başlangıç (1 saat)
1. **Temel sistem yapısını kur**
2. **Person detection'ı implement et** (YOLO/SSD)
3. **Basit zone definition** (mouse ile polygon çizimi)
4. **Basic intrusion detection**

### 🚀 Orta Seviye (2-3 saat)
1. **Face recognition ekle**
2. **Person tracking system**
3. **Loitering detection**
4. **Alert system** (visual alerts)
5. **Event logging** (JSON format)

### 🔥 İleri Seviye (4 saat)
1. **Advanced behavior analysis**
2. **Database integration**
3. **Screenshot capture system**
4. **Performance optimization**
5. **Robust error handling**

## 🎯 Değerlendirme Kriterleri

### ✅ Temel Kriterler (70 puan)
- [ ] **Person detection çalışıyor** (15p)
- [ ] **Zone definition ve intrusion detection** (20p)
- [ ] **Alert system functional** (15p)
- [ ] **Event logging implemented** (10p)
- [ ] **UI controls working** (10p)

### 🌟 Gelişmiş Kriterler (30 puan)
- [ ] **Face recognition integrated** (10p)
- [ ] **Person tracking system** (10p)
- [ ] **Loitering detection** (5p)
- [ ] **Performance optimization** (5p)

### 🏆 Bonus Kriterler (25 puan ek)
- [ ] **Database integration** (8p)
- [ ] **Screenshot capture** (5p)
- [ ] **Advanced behavior analysis** (7p)
- [ ] **System robustness** (5p)

## 💡 İpuçları ve Trickler

### 🎯 Zone Management
```python
def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside
```

### ⚡ Performance Optimization
```python
# Tip: Detection'ı her frame'de değil, belirli intervals'da yap
class DetectionScheduler:
    def __init__(self):
        self.person_detection_interval = 3  # Her 3 frame'de bir
        self.face_recognition_interval = 10  # Her 10 frame'de bir
        self.frame_count = 0
    
    def should_detect_persons(self):
        return self.frame_count % self.person_detection_interval == 0
    
    def should_recognize_faces(self):
        return self.frame_count % self.face_recognition_interval == 0
```

### 🎨 Visual Feedback
```python
# Tip: Alert severity'ye göre farklı renkler
ALERT_COLORS = {
    'HIGH': (0, 0, 255),     # Red - Critical
    'MEDIUM': (0, 165, 255), # Orange - Warning  
    'LOW': (0, 255, 255)     # Yellow - Info
}

def draw_alert(frame, alert):
    color = ALERT_COLORS[alert.priority]
    cv2.rectangle(frame, (10, 10), (400, 60), color, -1)
    cv2.putText(frame, alert.message, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

### 📊 Event Logging
```python
# Tip: Structured logging with timestamps
class EventLogger:
    def __init__(self, log_file="security_events.json"):
        self.log_file = log_file
        self.events = []
    
    def log_event(self, event_type, zone, person_id, confidence=None):
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'zone': zone,
            'person_id': person_id,
            'confidence': confidence
        }
        self.events.append(event)
        
        # Immediate write for critical events
        if event_type in ['INTRUSION', 'UNKNOWN_PERSON']:
            self.save_events()
```

## 🚨 Yaygın Hatalar ve Çözümleri

### ❌ Problem: False Positive Alerts
**Çözüm**: Confidence threshold'ları ayarla, temporal filtering ekle
```python
# Consecutive frame detection gereksinisi
class AlertFilter:
    def __init__(self, min_consecutive_frames=3):
        self.min_frames = min_consecutive_frames
        self.detection_history = defaultdict(list)
    
    def should_alert(self, event_type, person_id):
        history = self.detection_history[f"{event_type}_{person_id}"]
        return len(history) >= self.min_frames
```

### ❌ Problem: Person ID Confusion
**Çözüm**: Better tracking algorithm, appearance features
```python
# Appearance-based re-identification
def calculate_appearance_similarity(person1, person2):
    # Color histogram comparison
    hist1 = cv2.calcHist([person1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([person2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
```

### ❌ Problem: Performance Degradation
**Çözüm**: Multi-threading, frame skipping, resolution scaling
```python
# Separate thread for heavy operations
import threading
from queue import Queue

class SecuritySystem:
    def __init__(self):
        self.frame_queue = Queue(maxsize=10)
        self.alert_queue = Queue()
        
        # Background processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.start()
```

## 📚 Referans Kodlar

### Zone Definition Template
```python
class ZoneManager:
    def __init__(self):
        self.zones = {}
        self.current_zone_points = []
        self.drawing_zone = False
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing_zone:
                self.current_zone_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_zone_points) >= 3:
                # Complete zone
                zone_name = f"Zone_{len(self.zones)+1}"
                self.zones[zone_name] = self.current_zone_points.copy()
                self.current_zone_points = []
                self.drawing_zone = False
    
    def draw_zones(self, frame):
        for zone_name, points in self.zones.items():
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cv2.putText(frame, zone_name, points[0], 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

### Alert System Template
```python
class AlertManager:
    def __init__(self):
        self.active_alerts = []
        self.alert_history = deque(maxlen=100)
        self.sound_enabled = True
    
    def create_alert(self, event_type, message, priority, zone=None):
        alert = {
            'timestamp': time.time(),
            'event_type': event_type,
            'message': message,
            'priority': priority,
            'zone': zone,
            'acknowledged': False
        }
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Play sound for high priority
        if priority == 'HIGH' and self.sound_enabled:
            self.play_alert_sound()
    
    def play_alert_sound(self):
        # Implement sound playing (optional)
        pass
```

## 🎊 Tebrikler!

Bu alıştırmayı tamamladığınızda:
- ✅ **Real-time security system** geliştirme becerisi
- ✅ **Multi-threading** ve **performance optimization** deneyimi
- ✅ **Event-driven programming** anlayışı
- ✅ **Professional system architecture** tasarım becerisi

**🚀 Sıradaki Adım**: [Alıştırma 3: Envanter Yönetim Sistemi](alistirma-3.md)

---
**⏰ Tahmini Süre**: 3-4 saat  
**🎯 Zorluk**: İleri seviye  
**🏆 Kazanım**: Güvenlik sistemi tasarımı becerisi