"""
✅ Çözüm 3: Kenar Algılama ve Analiz
===================================

Bu dosyada Alıştırma 3'ün tam çözümü bulunmaktadır.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict

def cozum_3():
    """Alıştırma 3'ün tam çözümü"""
    
    print("✅ Çözüm 3: Kenar Algılama ve Analiz")
    print("=" * 45)
    
    # GÖREV 1: Test resmini yükle
    print("\n📁 GÖREV 1: Kenar test resmi yükleme")
    
    # Test resmi oluştur
    test_resmi = kenar_test_resmi_olustur()
    print(f"✅ Test resmi hazır: {test_resmi.shape}")
    
    # GÖREV 2: 4 farklı kenar algılama yöntemi
    print("\n🔍 GÖREV 2: Multi-method kenar algılama")
    
    def canny_kenar_algilama(resim, low_thresh=None, high_thresh=None):
        """Canny kenar algılama - otomatik threshold"""
        
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        
        # Otomatik threshold hesaplama
        if low_thresh is None or high_thresh is None:
            # Resmin median değerini bul
            median = np.median(gray)
            
            # Sigma = 0.33 kullanarak threshold'ları hesapla
            sigma = 0.33
            low_thresh = int(max(0, (1.0 - sigma) * median))
            high_thresh = int(min(255, (1.0 + sigma) * median))
        
        # Canny uygula
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        
        return edges, (low_thresh, high_thresh)
    
    def sobel_kenar_algilama(resim, threshold=50):
        """Sobel kenar algılama"""
        
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        
        # Sobel X ve Y gradyanlarını hesapla
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude hesapla
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize et
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # Threshold uygula
        _, edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        
        return edges
    
    def laplacian_kenar_algilama(resim, threshold=30):
        """Laplacian kenar algılama"""
        
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        
        # Önce Gaussian blur (gürültü azaltma)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Laplacian uygula
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Absolute değer al
        laplacian_abs = np.absolute(laplacian)
        
        # Normalize et
        laplacian_norm = np.uint8(laplacian_abs / laplacian_abs.max() * 255)
        
        # Threshold uygula
        _, edges = cv2.threshold(laplacian_norm, threshold, 255, cv2.THRESH_BINARY)
        
        return edges
    
    def scharr_kenar_algilama(resim, threshold=50):
        """Scharr kenar algılama"""
        
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        
        # Scharr X ve Y
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        
        # Magnitude hesapla
        magnitude = np.sqrt(scharrx**2 + scharry**2)
        
        # Normalize et
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        # Threshold uygula
        _, edges = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        
        return edges
    
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
    
    # GÖREV 3: Kenar kalitesi metrikleri
    print("\n📊 GÖREV 3: Kenar kalitesi analizi")
    
    def kenar_kalite_metrikleri(edge_image):
        """Kenar kalitesi metrikleri hesapla"""
        
        # 1. Toplam kenar piksel sayısı
        edge_count = np.sum(edge_image > 0)
        
        # 2. Kenar yoğunluğu
        total_pixels = edge_image.shape[0] * edge_image.shape[1]
        edge_density = edge_count / total_pixels
        
        # 3. Bağlantılı kenar bileşenleri sayısı
        num_labels, _ = cv2.connectedComponents(edge_image)
        connected_components = num_labels - 1  # Background'u çıkar
        
        # 4. Ortalama kenar kalınlığı (erosion ile tahmini)
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(edge_image, kernel, iterations=1)
        eroded_count = np.sum(eroded > 0)
        avg_thickness = edge_count / max(eroded_count, 1)
        
        # 5. Kenar sürekliliği (continuity) skoru
        # Dilation ile kenarları genişlet ve orijinalle karşılaştır
        dilated = cv2.dilate(edge_image, kernel, iterations=1)
        dilated_count = np.sum(dilated > 0)
        continuity_score = edge_count / max(dilated_count, 1)
        
        return {
            'edge_count': int(edge_count),
            'edge_density': float(edge_density),
            'connected_components': int(connected_components),
            'avg_thickness': float(avg_thickness),
            'continuity_score': float(continuity_score)
        }
    
    quality_metrics = {}
    for method_name, result in results.items():
        quality_metrics[method_name] = kenar_kalite_metrikleri(result)
    
    # GÖREV 4: Geometrik şekil tespiti
    print("\n🔢 GÖREV 4: Geometrik şekil tespiti")
    
    def sekil_tespit_ve_sayma(edge_image):
        """Kenar görüntüsünden şekilleri tespit et"""
        
        # Contour bulma
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_stats = {
            'circles': 0,
            'triangles': 0,
            'rectangles': 0,
            'other': 0,
            'total_area': 0.0,
            'avg_area': 0.0
        }
        
        total_area = 0.0
        valid_contours = 0
        
        for contour in contours:
            # Çok küçük contour'ları filtrele
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum alan threshold'u
                continue
            
            total_area += area
            valid_contours += 1
            
            # Çevre hesapla
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Approx polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Şekil sınıflandırma (köşe sayısına göre)
            vertices = len(approx)
            
            if vertices == 3:
                shape_stats['triangles'] += 1
            elif vertices == 4:
                # Kare/dikdörtgen kontrolü
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.8 <= aspect_ratio <= 1.2:
                    shape_stats['rectangles'] += 1  # Kare
                else:
                    shape_stats['rectangles'] += 1  # Dikdörtgen
            elif vertices > 8:
                # Çok fazla köşe varsa muhtemelen çember
                # Ek kontrolle: alan/çevre oranı
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    shape_stats['circles'] += 1
                else:
                    shape_stats['other'] += 1
            else:
                shape_stats['other'] += 1
        
        shape_stats['total_area'] = total_area
        shape_stats['avg_area'] = total_area / max(valid_contours, 1)
        
        return shape_stats
    
    shape_detection = {}
    for method_name, result in results.items():
        shape_detection[method_name] = sekil_tespit_ve_sayma(result)
        print(f"  {method_name}: {shape_detection[method_name]['circles']} çember, "
              f"{shape_detection[method_name]['rectangles']} dikdörtgen tespit edildi")
    
    # GÖREV 5: Performans benchmarking
    print("\n⏱️ GÖREV 5: Performans benchmarking")
    
    def benchmark_analizi(test_resmi, methods, iterations=5):
        """Comprehensive performance benchmark"""
        
        benchmark_results = {}
        
        for method_name, method_func in methods.items():
            print(f"  🔄 {method_name} benchmark...")
            
            times = []
            results_list = []
            
            # Birden fazla iterasyon çalıştır
            for i in range(iterations):
                start_time = time.time()
                
                if method_name == 'Canny':
                    result, _ = method_func(test_resmi)
                else:
                    result = method_func(test_resmi)
                
                end_time = time.time()
                
                times.append(end_time - start_time)
                results_list.append(result)
            
            # İstatistikleri hesapla
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Stabilite skoru - sonuçların tutarlılığı
            if len(results_list) > 1:
                differences = []
                for i in range(1, len(results_list)):
                    diff = np.mean(np.abs(results_list[0].astype(np.float32) - 
                                         results_list[i].astype(np.float32)))
                    differences.append(diff)
                stability_score = 1.0 - (np.mean(differences) / 255.0)
            else:
                stability_score = 1.0
            
            benchmark_results[method_name] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'stability': max(0.0, stability_score)
            }
        
        return benchmark_results
    
    benchmark_results = benchmark_analizi(test_resmi, methods)
    
    # GÖREV 6: En iyi yöntem seçimi
    print("\n🏆 GÖREV 6: En iyi yöntem seçimi")
    
    def en_iyi_yontem_sec(quality_metrics, timing_results, shape_detection, weights=None):
        """Multi-criteria decision making"""
        
        if weights is None:
            weights = {'speed': 0.3, 'quality': 0.4, 'accuracy': 0.2, 'continuity': 0.1}
        
        scores = {}
        
        # Normalize etmek için min-max değerlerini bul
        speed_values = list(timing_results.values())
        quality_values = [m['edge_density'] for m in quality_metrics.values()]
        accuracy_values = [s['circles'] + s['rectangles'] + s['triangles'] for s in shape_detection.values()]
        continuity_values = [m['continuity_score'] for m in quality_metrics.values()]
        
        speed_min, speed_max = min(speed_values), max(speed_values)
        quality_min, quality_max = min(quality_values), max(quality_values)
        accuracy_min, accuracy_max = min(accuracy_values), max(accuracy_values)
        continuity_min, continuity_max = min(continuity_values), max(continuity_values)
        
        # Her yöntem için skorları normalize et ve ağırlıklı toplam hesapla
        for method_name in quality_metrics.keys():
            # Speed score (düşük = iyi, bu yüzden ters çevir)
            if speed_max != speed_min:
                speed_score = 1.0 - (timing_results[method_name] - speed_min) / (speed_max - speed_min)
            else:
                speed_score = 1.0
            
            # Quality score (yüksek = iyi)
            if quality_max != quality_min:
                quality_score = (quality_metrics[method_name]['edge_density'] - quality_min) / (quality_max - quality_min)
            else:
                quality_score = 1.0
            
            # Accuracy score (yüksek = iyi)
            accuracy = shape_detection[method_name]['circles'] + shape_detection[method_name]['rectangles'] + shape_detection[method_name]['triangles']
            if accuracy_max != accuracy_min:
                accuracy_score = (accuracy - accuracy_min) / (accuracy_max - accuracy_min)
            else:
                accuracy_score = 1.0
            
            # Continuity score (yüksek = iyi)
            if continuity_max != continuity_min:
                continuity_score = (quality_metrics[method_name]['continuity_score'] - continuity_min) / (continuity_max - continuity_min)
            else:
                continuity_score = 1.0
            
            # Ağırlıklı toplam
            total_score = (weights['speed'] * speed_score + 
                          weights['quality'] * quality_score + 
                          weights['accuracy'] * accuracy_score + 
                          weights['continuity'] * continuity_score)
            
            scores[method_name] = total_score
        
        # En iyi yöntemi bul
        best_method = max(scores.keys(), key=lambda x: scores[x])
        return best_method, scores
    
    best_method, method_scores = en_iyi_yontem_sec(quality_metrics, timing_results, shape_detection)
    print(f"🏆 En iyi yöntem: {best_method}")
    
    # GÖREV 7: Comprehensive görselleştirme
    print("\n📈 GÖREV 7: Sonuçları görselleştirme")
    
    # 4x3 subplot ile kapsamlı görselleştirme
    plt.figure(figsize=(20, 15))
    
    # Orijinal resim
    plt.subplot(4, 3, 1)
    plt.imshow(cv2.cvtColor(test_resmi, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Test Resmi')
    plt.axis('off')
    
    # Kenar algılama sonuçları
    for i, (method_name, result) in enumerate(results.items()):
        plt.subplot(4, 3, i+2)
        plt.imshow(result, cmap='gray')
        plt.title(f'{method_name}\n({timing_results[method_name]*1000:.1f}ms)')
        plt.axis('off')
    
    # Performans grafikleri
    plt.subplot(4, 3, 6)
    methods_list = list(timing_results.keys())
    times_list = [timing_results[m]*1000 for m in methods_list]
    plt.bar(methods_list, times_list)
    plt.title('İşlem Süreleri (ms)')
    plt.xticks(rotation=45)
    
    plt.subplot(4, 3, 7)
    edge_counts = [quality_metrics[m]['edge_count'] for m in methods_list]
    plt.bar(methods_list, edge_counts)
    plt.title('Kenar Piksel Sayısı')
    plt.xticks(rotation=45)
    
    plt.subplot(4, 3, 8)
    circles = [shape_detection[m]['circles'] for m in methods_list]
    rectangles = [shape_detection[m]['rectangles'] for m in methods_list]
    triangles = [shape_detection[m]['triangles'] for m in methods_list]
    
    x = np.arange(len(methods_list))
    width = 0.25
    
    plt.bar(x - width, circles, width, label='Çember')
    plt.bar(x, rectangles, width, label='Dikdörtgen')
    plt.bar(x + width, triangles, width, label='Üçgen')
    
    plt.title('Şekil Tespit Sonuçları')
    plt.xticks(x, methods_list, rotation=45)
    plt.legend()
    
    plt.subplot(4, 3, 9)
    scores_list = [method_scores[m] for m in methods_list]
    bars = plt.bar(methods_list, scores_list)
    # En iyi yöntemi vurgula
    best_idx = methods_list.index(best_method)
    bars[best_idx].set_color('gold')
    plt.title('Kalite Skorları')
    plt.xticks(rotation=45)
    
    plt.subplot(4, 3, 10)
    densities = [quality_metrics[m]['edge_density'] for m in methods_list]
    continuities = [quality_metrics[m]['continuity_score'] for m in methods_list]
    plt.scatter(densities, continuities, s=100)
    for i, method in enumerate(methods_list):
        plt.annotate(method, (densities[i], continuities[i]))
    plt.xlabel('Kenar Yoğunluğu')
    plt.ylabel('Süreklilik Skoru')
    plt.title('Performans Karşılaştırması')
    
    plt.subplot(4, 3, 11)
    plt.imshow(results[best_method], cmap='gray')
    plt.title(f'En İyi: {best_method}\nSkor: {method_scores[best_method]:.3f}')
    plt.axis('off')
    
    plt.subplot(4, 3, 12)
    # Özet istatistikler tablosu
    plt.axis('off')
    table_data = []
    for method in methods_list:
        row = [method, 
               f"{timing_results[method]*1000:.1f}ms",
               f"{quality_metrics[method]['edge_count']}",
               f"{shape_detection[method]['circles']+shape_detection[method]['rectangles']+shape_detection[method]['triangles']}",
               f"{method_scores[method]:.3f}"]
        table_data.append(row)
    
    plt.table(cellText=table_data,
              colLabels=['Yöntem', 'Süre', 'Kenar', 'Şekil', 'Skor'],
              cellLoc='center',
              loc='center')
    plt.title('Özet İstatistikler')
    
    plt.tight_layout()
    plt.show()
    
    # GÖREV 8: Detaylı rapor
    print("\n📋 DETAYLI PERFORMANS RAPORU")
    print("=" * 50)
    
    for method_name in methods.keys():
        print(f"\n🔍 {method_name.upper()}")
        print("-" * 20)
        print(f"⏱️ İşlem Süresi: {timing_results[method_name]*1000:.2f}ms")
        print(f"📊 Kenar Sayısı: {quality_metrics[method_name]['edge_count']}")
        print(f"📈 Kenar Yoğunluğu: {quality_metrics[method_name]['edge_density']:.3f}")
        print(f"🔗 Bağlı Bileşen: {quality_metrics[method_name]['connected_components']}")
        print(f"📏 Ortalama Kalınlık: {quality_metrics[method_name]['avg_thickness']:.2f}")
        print(f"🔄 Süreklilik: {quality_metrics[method_name]['continuity_score']:.3f}")
        print(f"🔢 Tespit Edilen Şekiller:")
        print(f"   • Çember: {shape_detection[method_name]['circles']}")
        print(f"   • Dikdörtgen: {shape_detection[method_name]['rectangles']}")
        print(f"   • Üçgen: {shape_detection[method_name]['triangles']}")
        print(f"   • Diğer: {shape_detection[method_name]['other']}")
        print(f"🏆 Toplam Skor: {method_scores[method_name]:.3f}")
    
    print(f"\n🎯 SONUÇ: {best_method} en iyi performansı gösterdi!")
    print(f"   • En yüksek skor: {method_scores[best_method]:.3f}")
    print(f"   • İşlem süresi: {timing_results[best_method]*1000:.1f}ms")
    print(f"   • Tespit edilen şekil sayısı: {shape_detection[best_method]['circles']+shape_detection[best_method]['rectangles']+shape_detection[best_method]['triangles']}")
    
    print("\n🎉 Çözüm 3 tamamlandı!")

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

if __name__ == "__main__":
    print("✅ OpenCV Alıştırma 3 - ÇÖZÜM")
    print("Bu dosyada tüm görevlerin çözümleri bulunmaktadır.\n")
    
    cozum_3()