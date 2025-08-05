# 🎯 Alıştırma 3: Kenar Algılama ve Analiz

**Zorluk:** ⭐⭐⭐⭐ (Uzman)  
**Süre:** 90-120 dakika  
**Konular:** Kenar algılama, şekil tanıma, performans analizi  

Bu alıştırmada kenar algılama algoritmalarını karşılaştırıp performans analizi yapacaksınız.

## 📚 Gerekli Kütüphaneler

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict
```

## 🎯 Görevler

Bu alıştırmada şu görevleri tamamlamanız gerekiyor:

1. ✅ Kenar test resmini yükleyin
2. 🔍 4 farklı kenar algılama yöntemi uygulayın
3. 🎯 Otomatik threshold hesaplama implementasyonu
4. 📊 Kenar kalitesi metrikleri hesaplayın
5. 🔢 Geometrik şekil tespiti ve sayma
6. ⏱️ Performans analizi (hız, bellek kullanımı)
7. 🏆 En iyi yöntem seçimi algoritması
8. 📈 Comprehensive benchmark raporu

---

## 📁 GÖREV 1: Kenar Test Resmi Yükleme

Test için özel kenar algılama resmi oluşturun:

```python
# Test resmi oluştur
test_resmi = kenar_test_resmi_olustur()  # Yardımcı fonksiyon
print(f"✅ Test resmi hazır: {test_resmi.shape}")
```

**Not:** Yardımcı fonksiyon dökümanın sonunda tanımlıdır.

---

## 🔍 GÖREV 2: Multi-method Kenar Algılama

4 farklı kenar algılama yöntemi uygulayın:

### 2.1 Canny Kenar Algılama

```python
def canny_kenar_algilama(resim, low_thresh=None, high_thresh=None):
    """
    TODO: Canny kenar algılama
    
    Eğer threshold verilmezse otomatik hesaplayın:
    1. Resmin median değerini bulun
    2. sigma = 0.33 kullanarak threshold'ları hesaplayın
    3. lower = max(0, (1.0 - sigma) * median)
    4. upper = min(255, (1.0 + sigma) * median)
    """
    
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # TODO: Otomatik threshold hesaplama
    if low_thresh is None or high_thresh is None:
        # BURAYA OTOMATIK THRESHOLD KODUNU YAZIN
        low_thresh = 50   # BUNU DEĞİŞTİRİN!
        high_thresh = 150 # BUNU DEĞİŞTİRİN!
    
    # TODO: Canny uygulayın
    edges = np.zeros_like(gray)  # BUNU DEĞİŞTİRİN!
    
    return edges, (low_thresh, high_thresh)
```

### 2.2 Sobel Kenar Algılama

```python
def sobel_kenar_algilama(resim, threshold=50):
    """TODO: Sobel kenar algılama ve threshold"""
    
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # TODO: Sobel X ve Y gradyanlarını hesaplayın
    # TODO: Magnitude hesaplayın
    # TODO: Threshold uygulayın
    
    edges = np.zeros_like(gray)  # BUNU DEĞİŞTİRİN!
    
    return edges
```

### 2.3 Laplacian Kenar Algılama

```python
def laplacian_kenar_algilama(resim, threshold=30):
    """TODO: Laplacian kenar algılama"""
    
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # TODO: Önce Gaussian blur (gürültü azaltma)
    # TODO: Laplacian uygulayın
    # TODO: Absolute değer alın ve threshold
    
    edges = np.zeros_like(gray)  # BUNU DEĞİŞTİRİN!
    
    return edges
```

### 2.4 Scharr Kenar Algılama

```python
def scharr_kenar_algilama(resim, threshold=50):
    """TODO: Scharr kenar algılama"""
    
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # TODO: Scharr X ve Y
    # TODO: Magnitude hesaplayın
    # TODO: Threshold uygulayın
    
    edges = np.zeros_like(gray)  # BUNU DEĞİŞTİRİN!
    
    return edges
```

**İpuçları:**
- Canny: `cv2.Canny(gray, low_thresh, high_thresh)`
- Sobel: `cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=3)`
- Laplacian: `cv2.Laplacian(gray, cv2.CV_64F)`
- Scharr: `cv2.Scharr(gray, cv2.CV_64F, dx, dy)`
- Magnitude: `np.sqrt(gradx**2 + grady**2)`

### 2.5 Yöntemleri Test Etme

```python
# Yöntemleri uygula ve süreleri ölç
methods = {
    'Canny': canny_kenar_algilama,
    'Sobel': sobel_kenar_algilama,
    'Laplacian': laplacian_kenar_algilama,
    'Scharr': scharr_kenar_algilama
}

results = {}
timing_results = {}

for method_name, method_func in methods.items():
    print(f"  🔄 {method_name} işleniyor...")
    
    start_time = time.time()
    
    if method_name == 'Canny':
        result, params = method_func(test_resmi)
        print(f"     Otomatik threshold: {params}")
    else:
        result = method_func(test_resmi)
    
    end_time = time.time()
    
    results[method_name] = result
    timing_results[method_name] = end_time - start_time
    
    print(f"     ✅ Tamamlandı ({timing_results[method_name]*1000:.1f}ms)")
```

---

## 📊 GÖREV 3: Kenar Kalitesi Analizi

Kenar kalitesi metrikleri hesaplayın:

```python
def kenar_kalite_metrikleri(edge_image):
    """
    TODO: Kenar kalitesi metrikleri hesaplayın
    
    Metrikler:
    1. Toplam kenar piksel sayısı
    2. Kenar yoğunluğu (kenar piksel / toplam piksel)
    3. Bağlantılı kenar bileşenleri sayısı
    4. Ortalama kenar kalınlığı
    5. Kenar sürekliliği (continuity) skoru
    """
    
    # TODO: Metrikleri hesaplayın
    edge_count = 0          # BUNU HESAPLAYIN!
    edge_density = 0.0      # BUNU HESAPLAYIN!
    connected_components = 0 # BUNU HESAPLAYIN!
    avg_thickness = 0.0     # BUNU HESAPLAYIN!
    continuity_score = 0.0  # BUNU HESAPLAYIN!
    
    return {
        'edge_count': edge_count,
        'edge_density': edge_density,
        'connected_components': connected_components,
        'avg_thickness': avg_thickness,
        'continuity_score': continuity_score
    }

quality_metrics = {}
for method_name, result in results.items():
    quality_metrics[method_name] = kenar_kalite_metrikleri(result)
```

**İpuçları:**
- Kenar sayısı: `np.sum(edge_image > 0)`
- Yoğunluk: `edge_count / total_pixels`
- Bağlı bileşenler: `cv2.connectedComponents()`
- Kalınlık: erosion ile tahmini
- Süreklilik: dilation ile tahmini

---

## 🔢 GÖREV 4: Geometrik Şekil Tespiti

Kenar görüntüsünden şekilleri tespit edin:

```python
def sekil_tespit_ve_sayma(edge_image):
    """
    TODO: Kenar görüntüsünden şekilleri tespit edin
    
    Algoritma:
    1. Contour bulma (cv2.findContours)
    2. Her contour için:
       - Alan hesaplama
       - Çevre hesaplama  
       - Approx polygon (cv2.approxPolyDP)
       - Şekil sınıflandırma (köşe sayısına göre)
    3. İstatistik çıkarma
    """
    
    # TODO: Contour bulma
    # TODO: Şekil sınıflandırma
    # TODO: İstatistik hesaplama
    
    shape_stats = {
        'circles': 0,      # BUNU HESAPLAYIN!
        'triangles': 0,    # BUNU HESAPLAYIN!
        'rectangles': 0,   # BUNU HESAPLAYIN!
        'other': 0,        # BUNU HESAPLAYIN!
        'total_area': 0.0, # BUNU HESAPLAYIN!
        'avg_area': 0.0    # BUNU HESAPLAYIN!
    }
    
    return shape_stats

shape_detection = {}
for method_name, result in results.items():
    shape_detection[method_name] = sekil_tespit_ve_sayma(result)
    print(f"  {method_name}: {shape_detection[method_name]['circles']} çember, "
          f"{shape_detection[method_name]['rectangles']} dikdörtgen tespit edildi")
```

**İpuçları:**
- Contour: `cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
- Approx: `cv2.approxPolyDP(contour, epsilon, True)`
- Alan: `cv2.contourArea(contour)`
- Çevre: `cv2.arcLength(contour, True)`
- Sınıflandırma: köşe sayısına göre

---

## ⏱️ GÖREV 5: Performans Benchmarking

Comprehensive performance benchmark yapın:

```python
def benchmark_analizi(test_resmi, methods, iterations=5):
    """
    TODO: Comprehensive performance benchmark
    
    Ölçülecekler:
    1. Ortalama işlem süresi
    2. Bellek kullanımı
    3. Doğruluk skoru (reference ile karşılaştırma)
    4. Stabilite (sonuçların tutarlılığı)
    """
    
    benchmark_results = {}
    
    for method_name, method_func in methods.items():
        print(f"  🔄 {method_name} benchmark...")
        
        times = []
        results_list = []
        
        # TODO: Birden fazla iterasyon çalıştırın
        for i in range(iterations):
            # TODO: Süre ölçümü
            # TODO: Sonucu kaydet
            pass
        
        # TODO: İstatistikleri hesaplayın
        avg_time = 0.0        # BUNU HESAPLAYIN!
        std_time = 0.0        # BUNU HESAPLAYIN!
        stability_score = 0.0 # BUNU HESAPLAYIN!
        
        benchmark_results[method_name] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'stability': stability_score
        }
    
    return benchmark_results

benchmark_results = benchmark_analizi(test_resmi, methods)
```

**İpuçları:**
- Ortalama: `np.mean(times)`
- Standart sapma: `np.std(times)`
- Stabilite: sonuçlar arasındaki fark

---

## 🏆 GÖREV 6: En İyi Yöntem Seçimi

Multi-criteria decision making algoritması:

```python
def en_iyi_yontem_sec(quality_metrics, timing_results, shape_detection, weights=None):
    """
    TODO: Multi-criteria decision making
    
    Kriterler:
    1. Hız (düşük = iyi)
    2. Kenar kalitesi (yüksek = iyi) 
    3. Şekil tespit doğruluğu (yüksek = iyi)
    4. Kenar sürekliliği (yüksek = iyi)
    
    Ağırlıklar (toplam = 1.0):
    - Hız: 0.3
    - Kalite: 0.4  
    - Doğruluk: 0.2
    - Süreklilik: 0.1
    """
    
    if weights is None:
        weights = {'speed': 0.3, 'quality': 0.4, 'accuracy': 0.2, 'continuity': 0.1}
    
    scores = {}
    
    # TODO: Her yöntem için skorları normalize edin ve ağırlıklı toplam hesaplayın
    for method_name in quality_metrics.keys():
        # TODO: Scoring algoritması
        total_score = 0.0  # BUNU HESAPLAYIN!
        scores[method_name] = total_score
    
    # En iyi yöntemi bul
    best_method = max(scores.keys(), key=lambda x: scores[x])
    return best_method, scores

best_method, method_scores = en_iyi_yontem_sec(quality_metrics, timing_results, shape_detection)
print(f"🏆 En iyi yöntem: {best_method}")
```

**İpuçları:**
- Min-max normalizasyon kullanın
- Hız için ters normalizasyon (düşük = iyi)
- Ağırlıklı toplam: `sum(weight * normalized_score)`

---

## 📈 GÖREV 7: Comprehensive Görselleştirme

4x3 subplot ile kapsamlı görselleştirme:

```python
# TODO: 4x3 subplot ile kapsamlı görselleştirme
plt.figure(figsize=(20, 15))

# Orijinal resim
plt.subplot(4, 3, 1)
plt.imshow(cv2.cvtColor(test_resmi, cv2.COLOR_BGR2RGB))
plt.title('Orijinal Test Resmi')
plt.axis('off')

# Kenar algılama sonuçları
for i, (method_name, result) in enumerate(results.items()):
    plt.subplot(4, 3, i+2)
    # TODO: Kenar sonuçlarını gösterin
    plt.title(f'{method_name}\n({timing_results[method_name]*1000:.1f}ms)')
    plt.axis('off')

# Performans grafikleri
plt.subplot(4, 3, 6)
# TODO: Süre karşılaştırma bar chart
plt.title('İşlem Süreleri')

plt.subplot(4, 3, 7)
# TODO: Kenar sayısı karşılaştırması
plt.title('Kenar Piksel Sayısı')

plt.subplot(4, 3, 8)
# TODO: Şekil tespit doğruluğu
plt.title('Şekil Tespit Sonuçları')

plt.subplot(4, 3, 9)
# TODO: Kalite skorları radar chart
plt.title('Kalite Skorları')

plt.subplot(4, 3, 10)
# TODO: ROC curve veya benzer performans eğrisi
plt.title('Performans Karşılaştırması')

plt.subplot(4, 3, 11)
# TODO: En iyi yöntem vurgulaması
plt.title(f'En İyi: {best_method}')

plt.subplot(4, 3, 12)
# TODO: Özet istatistikler tablosu
plt.title('Özet İstatistikler')

plt.tight_layout()
plt.show()
```

---

## 📋 GÖREV 8: Detaylı Rapor

Detaylı performans raporu oluşturun:

```python
print("\n📋 DETAYLI PERFORMANS RAPORU")
print("=" * 50)

for method_name in methods.keys():
    print(f"\n🔍 {method_name.upper()}")
    print("-" * 20)
    print(f"⏱️ İşlem Süresi: {timing_results[method_name]*1000:.2f}ms")
    print(f"📊 Kenar Sayısı: {quality_metrics[method_name]['edge_count']}")
    print(f"📈 Kenar Yoğunluğu: {quality_metrics[method_name]['edge_density']:.3f}")
    print(f"🔢 Tespit Edilen Şekiller:")
    print(f"   • Çember: {shape_detection[method_name]['circles']}")
    print(f"   • Dikdörtgen: {shape_detection[method_name]['rectangles']}")
    print(f"   • Üçgen: {shape_detection[method_name]['triangles']}")
    print(f"🏆 Toplam Skor: {method_scores[method_name]:.3f}")

print(f"\n🎯 SONUÇ: {best_method} en iyi performansı gösterdi!")
```

---

## 🔨 Yardımcı Fonksiyonlar

### Kenar Test Resmi Oluşturucu

```python
def kenar_test_resmi_olustur():
    """Kenar algılama testi için özel resim oluştur"""
    resim = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Arka plan gradient
    for i in range(400):
        for j in range(400):
            value = int(100 + 30 * np.sin(i/60) * np.cos(j/60))
            resim[i, j] = [value, value+10, value+5]
    
    # Net kenarları olan şekiller
    # Dikdörtgenler
    cv2.rectangle(resim, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(resim, (70, 70), (130, 130), (0, 0, 0), -1)
    
    # Çemberler
    cv2.circle(resim, (300, 100), 50, (200, 200, 200), -1)
    cv2.circle(resim, (300, 100), 25, (100, 100, 100), -1)
    
    # Üçgenler
    triangle1 = np.array([[100, 250], [150, 350], [50, 350]], np.int32)
    cv2.fillPoly(resim, [triangle1], (180, 180, 180))
    
    triangle2 = np.array([[300, 250], [350, 350], [250, 350]], np.int32)
    cv2.fillPoly(resim, [triangle2], (160, 160, 160))
    
    # İnce çizgiler (kenar algılama zorluğu için)
    for i in range(200, 380, 20):
        cv2.line(resim, (i, 200), (i, 220), (255, 255, 255), 1)
    
    # Çapraz çizgiler
    cv2.line(resim, (250, 50), (350, 150), (255, 255, 255), 2)
    cv2.line(resim, (250, 150), (350, 50), (255, 255, 255), 2)
    
    return resim
```

---

## ✅ Kontrol Listesi

Alıştırmayı tamamladıktan sonra kontrol edin:

- [ ] 4 farklı kenar algılama yöntemi uygulandı
- [ ] Otomatik threshold hesaplama çalışıyor
- [ ] Kenar kalitesi metrikleri hesaplandı
- [ ] Geometrik şekil tespiti yapılıyor
- [ ] Performance benchmark tamamlandı
- [ ] En iyi yöntem seçim algoritması çalışıyor
- [ ] Comprehensive görselleştirme yapıldı
- [ ] Detaylı performans raporu oluşturuldu

## 💡 İpuçları

### Genel İpuçları
- Her yöntemi ayrı ayrı test edin
- Threshold değerlerini optimize edin
- Sonuçları görsel olarak karşılaştırın
- Performans ölçümlerini doğrulayın

### Teknik İpuçları
- Contour işlemleri için doğru veri tipi kullanın
- Threshold değerlerini resim içeriğine göre ayarlayın
- Performance measurement için time.time() kullanın
- Shape classification için geometrik özellikler hesaplayın
- Scoring algoritmasında normalizasyon yapmayı unutmayın

### Hata Durumları
- Boş contour listesini kontrol edin
- Sıfıra bölme durumlarını handel edin
- Veri tipi uyumsuzluklarına dikkat edin

## 🚀 Bonus Görevler

Temel alıştırmayı tamamladıysanız bunları deneyin:

- [ ] Real-time kenar algılama (webcam)
- [ ] Custom kenar algılama algoritması
- [ ] Machine learning tabanlı kenar kalitesi
- [ ] Multi-scale kenar analizi
- [ ] Paralel processing optimizasyonu
- [ ] GPU acceleration

## 🎯 Çözüm

Tamamladıktan sonra çözümle karşılaştırın:
```bash
python cozumler/cozum-3.py
```

---

*Bu alıştırma Eren Terzi tarafından hazırlanmıştır.*