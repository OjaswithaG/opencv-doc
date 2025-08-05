"""
🔍 OpenCV Kenar Algılama Algoritmaları
======================================

Bu dosyada kenar algılama (edge detection) tekniklerini öğreneceksiniz:
- Sobel operatörü (X ve Y yönünde gradyan)
- Canny kenar algılama (en popüler yöntem)
- Laplacian of Gaussian (LoG)
- Scharr operatörü (gelişmiş Sobel)
- Prewitt operatörü
- Roberts cross-gradient
- Zerocrossing edge detection
- Advanced kenar algılama teknikleri

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ornek_resim_olustur():
    """Kenar algılama testleri için örnek resimler oluştur"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 1. Geometrik şekiller resmi
    geometrik_resim = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Arka plan gradyanı
    for i in range(400):
        for j in range(400):
            value = int(100 + 50 * np.sin(i/80) * np.cos(j/80))
            geometrik_resim[i, j] = [value, value+10, value+5]
    
    # Net kenarları olan şekiller
    cv2.rectangle(geometrik_resim, (50, 50), (180, 180), (255, 255, 255), -1)
    cv2.rectangle(geometrik_resim, (70, 70), (160, 160), (0, 0, 0), -1)
    cv2.circle(geometrik_resim, (300, 100), 60, (200, 200, 200), -1)
    cv2.circle(geometrik_resim, (300, 100), 30, (100, 100, 100), -1)
    
    # Üçgen
    triangle = np.array([[100, 250], [200, 250], [150, 350]], np.int32)
    cv2.fillPoly(geometrik_resim, [triangle], (180, 180, 180))
    
    # İnce çizgiler
    cv2.line(geometrik_resim, (250, 200), (350, 300), (255, 255, 255), 2)
    cv2.line(geometrik_resim, (250, 300), (350, 200), (255, 255, 255), 2)
    
    geometrik_dosya = examples_dir / "geometric_edges.jpg"
    cv2.imwrite(str(geometrik_dosya), geometrik_resim)
    
    # 2. Doğal görüntü benzeri resim
    dogal_resim = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Tekstür benzeri arka plan
    for i in range(300):
        for j in range(300):
            noise = np.random.normal(0, 20)
            base = 120 + 30 * np.sin((i+j)/40) + noise
            dogal_resim[i, j] = [np.clip(base, 50, 200)] * 3
    
    # Doğal kenarlar (yumuşak geçişler)
    cv2.ellipse(dogal_resim, (150, 150), (80, 50), 30, 0, 360, (180, 180, 180), -1)
    
    # Gaussian blur ile yumuşak kenarlar
    dogal_resim = cv2.GaussianBlur(dogal_resim, (3, 3), 0)
    
    dogal_dosya = examples_dir / "natural_edges.jpg"
    cv2.imwrite(str(dogal_dosya), dogal_resim)
    
    # 3. Metin ve detaylı resim
    metin_resim = np.full((250, 400, 3), 240, dtype=np.uint8)
    
    # Metin ekle
    cv2.putText(metin_resim, 'EDGE DETECTION', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 0), 2)
    cv2.putText(metin_resim, 'OpenCV Tutorial', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (100, 100, 100), 2)
    cv2.putText(metin_resim, 'Fine Details', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (50, 50, 50), 1)
    
    # İnce detaylar
    for i in range(20, 380, 10):
        cv2.line(metin_resim, (i, 180), (i, 190), (0, 0, 0), 1)
    
    # Çerçeve
    cv2.rectangle(metin_resim, (10, 10), (390, 240), (0, 0, 0), 2)
    
    metin_dosya = examples_dir / "text_edges.jpg"
    cv2.imwrite(str(metin_resim), metin_resim)
    
    # 4. Gürültülü resim
    gurultulu_resim = geometrik_resim.copy().astype(np.float32)
    gurultu = np.random.normal(0, 20, geometrik_resim.shape)
    gurultulu_resim += gurultu
    gurultulu_resim = np.clip(gurultulu_resim, 0, 255).astype(np.uint8)
    
    gurultulu_dosya = examples_dir / "noisy_edges.jpg"
    cv2.imwrite(str(gurultulu_dosya), gurultulu_resim)
    
    print(f"✅ Kenar algılama test resimleri oluşturuldu:")
    print(f"   - Geometrik şekiller: {geometrik_dosya}")
    print(f"   - Doğal görüntü: {dogal_dosya}")
    print(f"   - Metin ve detaylar: {metin_dosya}")
    print(f"   - Gürültülü resim: {gurultulu_dosya}")
    
    return str(geometrik_dosya), str(dogal_dosya), str(metin_dosya), str(gurultulu_dosya)

def sobel_kenar_algilama(resim):
    """Sobel operatörü ile kenar algılama"""
    print("\n🔄 Sobel Kenar Algılama")
    print("=" * 25)
    
    # Gri seviyeye çevir
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Farklı Sobel uygulamaları
    # X yönünde Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    sobel_x = np.uint8(sobel_x)
    
    # Y yönünde Sobel
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.absolute(sobel_y)
    sobel_y = np.uint8(sobel_y)
    
    # Kombinasyon (magnitude)
    sobel_combined = np.sqrt(sobel_x.astype(np.float32)**2 + sobel_y.astype(np.float32)**2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    
    # Farklı kernel boyutları
    sobel_k5 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    sobel_k5 = np.uint8(np.absolute(sobel_k5))
    
    sobel_k7 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=7)
    sobel_k7 = np.uint8(np.absolute(sobel_k7))
    
    # Manuel Sobel kernel implementasyonu
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    manual_sobel_x = cv2.filter2D(gray, cv2.CV_64F, sobel_x_kernel)
    manual_sobel_y = cv2.filter2D(gray, cv2.CV_64F, sobel_y_kernel)
    manual_sobel_combined = np.sqrt(manual_sobel_x**2 + manual_sobel_y**2)
    manual_sobel_combined = np.uint8(np.clip(manual_sobel_combined, 0, 255))
    
    # Gradient yönü hesaplama
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    gradient_direction_degrees = np.degrees(gradient_direction) % 360
    
    # Eşikleme uygulama
    sobel_threshold = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)[1]
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Ana sonuçlar
    plt.subplot(3, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal (Gri)')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel X (Dikey Kenarlar)')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel Y (Yatay Kenarlar)')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Kombinasyon')
    plt.axis('off')
    
    # Farklı kernel boyutları
    plt.subplot(3, 4, 5)
    plt.imshow(sobel_k5, cmap='gray')
    plt.title('Sobel Kernel 5x5')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(sobel_k7, cmap='gray')
    plt.title('Sobel Kernel 7x7')
    plt.axis('off')
    
    # Manuel vs OpenCV
    plt.subplot(3, 4, 7)
    plt.imshow(manual_sobel_combined, cmap='gray')
    plt.title('Manuel Sobel')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(sobel_threshold, cmap='gray')
    plt.title('Sobel + Threshold')
    plt.axis('off')
    
    # Gradient yönü
    plt.subplot(3, 4, 9)
    plt.imshow(gradient_direction_degrees, cmap='hsv')
    plt.title('Gradient Yönü')
    plt.colorbar()
    plt.axis('off')
    
    # Kernel görselleştirme
    plt.subplot(3, 4, 10)
    plt.imshow(sobel_x_kernel, cmap='RdBu', interpolation='nearest')
    plt.title('Sobel X Kernel')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{sobel_x_kernel[i,j]}', ha='center', va='center', 
                    color='white' if abs(sobel_x_kernel[i,j]) > 1 else 'black')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(sobel_y_kernel, cmap='RdBu', interpolation='nearest')
    plt.title('Sobel Y Kernel')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{sobel_y_kernel[i,j]}', ha='center', va='center',
                    color='white' if abs(sobel_y_kernel[i,j]) > 1 else 'black')
    plt.axis('off')
    
    # Sobel analizi
    plt.subplot(3, 4, 12)
    # Histogram
    plt.hist(sobel_combined.flatten(), bins=50, alpha=0.7, color='blue', label='Sobel Magnitude')
    plt.axvline(x=50, color='red', linestyle='--', label='Threshold=50')
    plt.title('Sobel Magnitude Histogramı')
    plt.xlabel('Kenar Gücü')
    plt.ylabel('Piksel Sayısı')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı analiz
    print("\n🔍 Sobel Kenar Algılama Analizi:")
    
    # Kenar piksel istatistikleri
    edge_pixels = np.sum(sobel_threshold == 255)
    total_pixels = sobel_threshold.size
    edge_percentage = (edge_pixels / total_pixels) * 100
    
    print(f"Toplam piksel: {total_pixels}")
    print(f"Kenar piksel: {edge_pixels}")
    print(f"Kenar oranı: %{edge_percentage:.2f}")
    
    # Gradient büyüklük istatistikleri
    print(f"Max gradient büyüklük: {np.max(sobel_combined)}")
    print(f"Ortalama gradient: {np.mean(sobel_combined):.1f}")
    print(f"Gradient std sapma: {np.std(sobel_combined):.1f}")
    
    print("\n📝 Sobel Kenar Algılama İpuçları:")
    print("   • X yönü dikey kenarları, Y yönü yatay kenarları bulur")
    print("   • Magnitude = √(Gx² + Gy²) ile kombinasyon")
    print("   • Büyük kernel = daha yumuşak kenarlar")
    print("   • Gürültüye orta derecede hassas")

def canny_kenar_algilama(resim):
    """Canny kenar algılama algoritması"""
    print("\n⚡ Canny Kenar Algılama")
    print("=" * 25)
    
    # Gri seviyeye çevir
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Farklı Canny parametreleri
    canny_low = cv2.Canny(gray, 50, 100)
    canny_medium = cv2.Canny(gray, 100, 200)
    canny_high = cv2.Canny(gray, 150, 300)
    
    # Önceden blur uygulama
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_blurred = cv2.Canny(blurred, 100, 200)
    
    # Manuel Canny implementasyonu (adım adım)
    def manual_canny_steps(image):
        # Adım 1: Gaussian blur
        step1_blur = cv2.GaussianBlur(image, (5, 5), 1.4)
        
        # Adım 2: Sobel gradyanları
        sobel_x = cv2.Sobel(step1_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(step1_blur, cv2.CV_64F, 0, 1, ksize=3)
        
        # Adım 3: Gradient magnitude ve yönü
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x)
        
        # Adım 4: Non-maximum suppression (basitleştirilmiş)
        angle = direction * 180 / np.pi
        angle[angle < 0] += 180
        
        suppressed = np.zeros_like(magnitude)
        for i in range(1, magnitude.shape[0]-1):
            for j in range(1, magnitude.shape[1]-1):
                try:
                    q = 255
                    r = 255
                    
                    # Açı 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = magnitude[i, j+1]
                        r = magnitude[i, j-1]
                    # Açı 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = magnitude[i+1, j-1]
                        r = magnitude[i-1, j+1]
                    # Açı 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = magnitude[i+1, j]
                        r = magnitude[i-1, j]
                    # Açı 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = magnitude[i-1, j-1]
                        r = magnitude[i+1, j+1]
                        
                    if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                        suppressed[i,j] = magnitude[i,j]
                    else:
                        suppressed[i,j] = 0
                        
                except IndexError:
                    pass
        
        # Adım 5: Double threshold (basitleştirilmiş)
        high_threshold = 100
        low_threshold = 50
        
        thresholded = np.zeros_like(suppressed)
        strong = 255
        weak = 75
        
        strong_i, strong_j = np.where(suppressed >= high_threshold)
        weak_i, weak_j = np.where((suppressed <= high_threshold) & (suppressed >= low_threshold))
        
        thresholded[strong_i, strong_j] = strong
        thresholded[weak_i, weak_j] = weak
        
        return step1_blur, magnitude, suppressed, thresholded
    
    blur_step, magnitude_step, suppressed_step, threshold_step = manual_canny_steps(gray)
    
    # Otomatik threshold hesaplama
    def auto_canny(image, sigma=0.33):
        # Median'ı bulup otomatik threshold hesapla
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)
    
    auto_canny_result = auto_canny(gray)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 20))
    
    # Ana Canny sonuçları
    plt.subplot(4, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal (Gri)')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(canny_low, cmap='gray')
    plt.title('Canny (50, 100)')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(canny_medium, cmap='gray')
    plt.title('Canny (100, 200)')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(canny_high, cmap='gray')
    plt.title('Canny (150, 300)')
    plt.axis('off')
    
    # Blur etkisi
    plt.subplot(4, 4, 5)
    plt.imshow(blurred, cmap='gray')
    plt.title('Gaussian Blur Uygulanmış')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(canny_blurred, cmap='gray')
    plt.title('Blur + Canny')
    plt.axis('off')
    
    # Otomatik Canny
    plt.subplot(4, 4, 7)
    plt.imshow(auto_canny_result, cmap='gray')
    plt.title('Otomatik Threshold Canny')
    plt.axis('off')
    
    # Manuel Canny adımları
    plt.subplot(4, 4, 8)
    plt.imshow(blur_step, cmap='gray')
    plt.title('1. Gaussian Blur')
    plt.axis('off')
    
    plt.subplot(4, 4, 9)
    plt.imshow(magnitude_step, cmap='gray')
    plt.title('2. Gradient Magnitude')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(suppressed_step, cmap='gray')
    plt.title('3. Non-max Suppression')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(threshold_step, cmap='gray')
    plt.title('4. Double Threshold')
    plt.axis('off')
    
    # Threshold karşılaştırması
    plt.subplot(4, 4, 12)
    # Farklı threshold'lar için kenar sayısı
    thresholds = [(30, 80), (50, 100), (100, 200), (150, 300), (200, 400)]
    edge_counts = []
    
    for low, high in thresholds:
        canny_test = cv2.Canny(gray, low, high)
        edge_count = np.sum(canny_test == 255)
        edge_counts.append(edge_count)
    
    threshold_labels = [f'({low},{high})' for low, high in thresholds]
    plt.bar(range(len(thresholds)), edge_counts, alpha=0.7)
    plt.title('Threshold vs Kenar Sayısı')
    plt.xlabel('Threshold (Low, High)')
    plt.ylabel('Kenar Piksel Sayısı')
    plt.xticks(range(len(thresholds)), threshold_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Canny vs Sobel karşılaştırması
    plt.subplot(4, 4, 13)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Karşılaştırma')
    plt.axis('off')
    
    # ROI analizi
    plt.subplot(4, 4, 14)
    roi_x, roi_y, roi_w, roi_h = 50, 50, 100, 100
    roi_original = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_canny = canny_medium[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    roi_comparison = np.hstack([roi_original, roi_canny])
    plt.imshow(roi_comparison, cmap='gray')
    plt.title('ROI: Orijinal | Canny')
    plt.axvline(x=roi_w, color='yellow', linewidth=2)
    plt.axis('off')
    
    # Parametre optimizasyonu
    plt.subplot(4, 4, 15)
    # Farklı sigma değerleri için otomatik Canny
    sigmas = [0.1, 0.2, 0.33, 0.5, 0.7]
    sigma_edge_counts = []
    
    for sigma in sigmas:
        auto_result = auto_canny(gray, sigma)
        sigma_edge_counts.append(np.sum(auto_result == 255))
    
    plt.plot(sigmas, sigma_edge_counts, 'bo-', linewidth=2, markersize=8)
    plt.title('Sigma vs Kenar Sayısı (Otomatik)')
    plt.xlabel('Sigma Parametresi')
    plt.ylabel('Kenar Piksel Sayısı')
    plt.grid(True, alpha=0.3)
    
    # Kalite metrikleri
    plt.subplot(4, 4, 16)
    methods = ['Canny Low', 'Canny Med', 'Canny High', 'Auto Canny', 'Sobel']
    results = [canny_low, canny_medium, canny_high, auto_canny_result, 
               cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)[1]]
    
    # Kenar sayısı ve kenar genişliği
    edge_counts_final = []
    edge_thickness = []
    
    for result in results:
        edge_count = np.sum(result == 255)
        edge_counts_final.append(edge_count)
        
        # Basit kenar genişliği tahmini (morfolojik işlemle)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(result, kernel, iterations=1)
        thickness = np.sum(dilated == 255) - edge_count
        edge_thickness.append(max(1, thickness))
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x_pos - width/2, edge_counts_final, width, label='Kenar Sayısı', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, edge_thickness, width, label='Kenar Genişliği', 
                    alpha=0.7, color='orange')
    
    ax1.set_xlabel('Yöntem')
    ax1.set_ylabel('Kenar Sayısı', color='blue')
    ax2.set_ylabel('Kenar Genişliği', color='orange')
    ax1.set_title('Canny Performans Karşılaştırması')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı analiz
    print("\n🔍 Canny Kenar Algılama Analizi:")
    
    # En iyi threshold kombinasyonu
    best_threshold_idx = np.argmin([abs(count - np.mean(edge_counts)) for count in edge_counts])
    best_threshold = thresholds[best_threshold_idx]
    
    print(f"Önerilen threshold: {best_threshold}")
    print(f"Otomatik threshold sonucu: {np.sum(auto_canny_result == 255)} piksel")
    print(f"En dengeli sonuç: Threshold {best_threshold} -> {edge_counts[best_threshold_idx]} piksel")
    
    # Canny adımlarının etkisi
    original_pixels = np.sum(gray > 0)
    after_blur = np.sum(blur_step > 0)
    after_gradient = np.sum(magnitude_step > 10)  # Gradient threshold
    after_suppression = np.sum(suppressed_step > 0)
    final_edges = np.sum(threshold_step == 255)
    
    print(f"\nCanny adımları etkisi:")
    print(f"1. Orijinal piksel: {original_pixels}")
    print(f"2. Blur sonrası: {after_blur}")
    print(f"3. Gradient hesabı: {after_gradient}")
    print(f"4. Non-max suppression: {after_suppression}")
    print(f"5. Final kenarlar: {final_edges}")
    
    print("\n📝 Canny Kenar Algılama İpuçları:")
    print("   • İki threshold: düşük=kenar devamı, yüksek=güçlü kenar")
    print("   • Önerilen oran: yüksek/düşük = 2:1 veya 3:1")
    print("   • Otomatik threshold için sigma=0.33 iyi başlangıç")
    print("   • Önce Gaussian blur uygulayın (gürültü azaltma)")

def laplacian_kenar_algilama(resim):
    """Laplacian kenar algılama"""
    print("\n🌊 Laplacian Kenar Algılama")
    print("=" * 30)
    
    # Gri seviyeye çevir
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Farklı Laplacian uygulamaları
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = np.uint8(laplacian)
    
    # Farklı kernel boyutları
    laplacian_k1 = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)
    laplacian_k1 = np.uint8(np.absolute(laplacian_k1))
    
    laplacian_k5 = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
    laplacian_k5 = np.uint8(np.absolute(laplacian_k5))
    
    # Laplacian of Gaussian (LoG)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    log_result = cv2.Laplacian(blurred, cv2.CV_64F)
    log_result = np.uint8(np.absolute(log_result))
    
    # Manuel Laplacian kernelleri
    laplacian_kernel_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_kernel_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    
    manual_lap_4 = cv2.filter2D(gray, cv2.CV_64F, laplacian_kernel_4)
    manual_lap_4 = np.uint8(np.absolute(manual_lap_4))
    
    manual_lap_8 = cv2.filter2D(gray, cv2.CV_64F, laplacian_kernel_8)
    manual_lap_8 = np.uint8(np.absolute(manual_lap_8))
    
    # Zero-crossing detection
    def zero_crossing_detection(laplacian_image):
        # Zero-crossing bulma (basitleştirilmiş)
        zero_cross = np.zeros_like(laplacian_image)
        
        # Threshold yaklaşımı
        for i in range(1, laplacian_image.shape[0]-1):
            for j in range(1, laplacian_image.shape[1]-1):
                # 3x3 komşuluk kontrol et
                patch = laplacian_image[i-1:i+2, j-1:j+2].astype(np.int16)
                if np.max(patch) > 0 and np.min(patch) < 0:
                    zero_cross[i, j] = 255
        
        return zero_cross
    
    # Signed Laplacian için
    signed_laplacian = cv2.Laplacian(gray, cv2.CV_16S)
    zero_crossings = zero_crossing_detection(signed_laplacian)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 12))
    
    # Ana sonuçlar
    plt.subplot(3, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal (Gri)')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian (ksize=3)')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(laplacian_k1, cmap='gray')
    plt.title('Laplacian (ksize=1)')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(laplacian_k5, cmap='gray')
    plt.title('Laplacian (ksize=5)')
    plt.axis('off')
    
    # LoG ve manuel kerneller
    plt.subplot(3, 4, 5)
    plt.imshow(log_result, cmap='gray')
    plt.title('Laplacian of Gaussian (LoG)')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(manual_lap_4, cmap='gray')
    plt.title('Manuel Laplacian (4-bağlantı)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(manual_lap_8, cmap='gray')
    plt.title('Manuel Laplacian (8-bağlantı)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(zero_crossings, cmap='gray')
    plt.title('Zero-Crossing Detection')
    plt.axis('off')
    
    # Kernel görselleştirme
    plt.subplot(3, 4, 9)
    plt.imshow(laplacian_kernel_4, cmap='RdBu', interpolation='nearest')
    plt.title('4-Connected Laplacian Kernel')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{laplacian_kernel_4[i,j]}', ha='center', va='center',
                    color='white' if abs(laplacian_kernel_4[i,j]) > 2 else 'black')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(laplacian_kernel_8, cmap='RdBu', interpolation='nearest')
    plt.title('8-Connected Laplacian Kernel')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{laplacian_kernel_8[i,j]}', ha='center', va='center',
                    color='white' if abs(laplacian_kernel_8[i,j]) > 4 else 'black')
    plt.axis('off')
    
    # Canny vs Laplacian karşılaştırması
    plt.subplot(3, 4, 11)
    canny_comp = cv2.Canny(gray, 100, 200)
    comparison = np.hstack([laplacian, canny_comp])
    plt.imshow(comparison, cmap='gray')
    plt.title('Laplacian | Canny')
    plt.axvline(x=laplacian.shape[1], color='yellow', linewidth=2)
    plt.axis('off')
    
    # Performans analizi
    plt.subplot(3, 4, 12)
    methods = ['Laplacian', 'LoG', 'Manual 4', 'Manual 8', 'Canny']
    results = [laplacian, log_result, manual_lap_4, manual_lap_8, canny_comp]
    
    edge_counts = [np.sum(result > 50) for result in results]  # Threshold=50
    
    bars = plt.bar(range(len(methods)), edge_counts, alpha=0.7,
                   color=['blue', 'green', 'red', 'orange', 'purple'])
    plt.title('Kenar Piksel Sayısı (Threshold=50)')
    plt.xlabel('Yöntem')
    plt.ylabel('Kenar Piksel Sayısı')
    plt.xticks(range(len(methods)), methods, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı analiz
    print("\n🔍 Laplacian Kenar Algılama Analizi:")
    
    # Laplacian özellikleri
    lap_mean = np.mean(laplacian)
    lap_std = np.std(laplacian)
    lap_max = np.max(laplacian)
    
    print(f"Laplacian istatistikleri:")
    print(f"  Ortalama: {lap_mean:.1f}")
    print(f"  Std sapma: {lap_std:.1f}")
    print(f"  Maksimum: {lap_max}")
    
    # Zero-crossing sayısı
    zero_cross_count = np.sum(zero_crossings == 255)
    print(f"Zero-crossing noktaları: {zero_cross_count}")
    
    # Karşılaştırma
    print(f"\nYöntem karşılaştırması (kenar piksel sayısı):")
    for method, count in zip(methods, edge_counts):
        print(f"  {method}: {count}")
    
    print("\n📝 Laplacian Kenar Algılama İpuçları:")
    print("   • İkinci türev operatörü (kenarların geçiş noktaları)")
    print("   • Gürültüye çok hassas, önce blur uygulayın (LoG)")
    print("   • Zero-crossing kenar konumunu hassas verir")
    print("   • Tek bir operatörle tüm yönlerde kenar bulur")

def scharr_ve_diger_operatorler(resim):
    """Scharr ve diğer kenar algılama operatörleri"""
    print("\n🔧 Scharr ve Diğer Kenar Operatörleri")
    print("=" * 40)
    
    # Gri seviyeye çevir
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Scharr operatörü
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_combined = np.uint8(np.clip(scharr_combined, 0, 255))
    
    # Prewitt operatörü
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    prewitt_x_result = cv2.filter2D(gray, cv2.CV_64F, prewitt_x)
    prewitt_y_result = cv2.filter2D(gray, cv2.CV_64F, prewitt_y)
    prewitt_combined = np.sqrt(prewitt_x_result**2 + prewitt_y_result**2)
    prewitt_combined = np.uint8(np.clip(prewitt_combined, 0, 255))
    
    # Roberts operatörü
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    
    roberts_x_result = cv2.filter2D(gray, cv2.CV_64F, roberts_x)
    roberts_y_result = cv2.filter2D(gray, cv2.CV_64F, roberts_y)
    roberts_combined = np.sqrt(roberts_x_result**2 + roberts_y_result**2)
    roberts_combined = np.uint8(np.clip(roberts_combined, 0, 255))
    
    # Kirsch operatörü (8 yönlü)
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # N
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # NE
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # E
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # SE
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # S
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # SW
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # W
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])   # NW
    ]
    
    kirsch_results = []
    for kernel in kirsch_kernels:
        kirsch_result = cv2.filter2D(gray, cv2.CV_64F, kernel)
        kirsch_results.append(np.absolute(kirsch_result))
    
    # Maximum response
    kirsch_max = np.maximum.reduce(kirsch_results)
    kirsch_max = np.uint8(np.clip(kirsch_max, 0, 255))
    
    # Sobel karşılaştırması
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Ana operatörler
    plt.subplot(3, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal (Gri)')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(scharr_combined, cmap='gray')
    plt.title('Scharr')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(prewitt_combined, cmap='gray')
    plt.title('Prewitt')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(roberts_combined, cmap='gray')
    plt.title('Roberts')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(kirsch_max, cmap='gray')
    plt.title('Kirsch (8-yönlü)')
    plt.axis('off')
    
    # Kernel görselleştirmeleri
    plt.subplot(3, 4, 7)
    scharr_x_kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    plt.imshow(scharr_x_kernel, cmap='RdBu', interpolation='nearest')
    plt.title('Scharr X Kernel')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{scharr_x_kernel[i,j]}', ha='center', va='center',
                    color='white' if abs(scharr_x_kernel[i,j]) > 5 else 'black')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(prewitt_x, cmap='RdBu', interpolation='nearest')
    plt.title('Prewitt X Kernel')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{prewitt_x[i,j]}', ha='center', va='center',
                    color='white' if abs(prewitt_x[i,j]) > 0 else 'black')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    plt.imshow(roberts_x, cmap='RdBu', interpolation='nearest')
    plt.title('Roberts X Kernel')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{roberts_x[i,j]}', ha='center', va='center',
                    color='white' if abs(roberts_x[i,j]) > 0 else 'black')
    plt.axis('off')
    
    # Performans karşılaştırması
    plt.subplot(3, 4, 10)
    methods = ['Sobel', 'Scharr', 'Prewitt', 'Roberts', 'Kirsch']
    results = [sobel_combined, scharr_combined, prewitt_combined, roberts_combined, kirsch_max]
    
    # Kenar piksel sayısı (threshold=50)
    edge_counts = [np.sum(result > 50) for result in results]
    
    bars = plt.bar(range(len(methods)), edge_counts, alpha=0.7,
                   color=['blue', 'green', 'red', 'orange', 'purple'])
    plt.title('Kenar Sayısı Karşılaştırması')
    plt.xlabel('Operatör')
    plt.ylabel('Kenar Piksel Sayısı')
    plt.xticks(range(len(methods)), methods, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for bar, count in zip(bars, edge_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(edge_counts)*0.01, 
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    # Gradient büyüklük dağılımı
    plt.subplot(3, 4, 11)
    for result, method, color in zip(results, methods, ['blue', 'green', 'red', 'orange', 'purple']):
        plt.hist(result.flatten(), bins=50, alpha=0.3, label=method, color=color, density=True)
    
    plt.title('Gradient Büyüklük Dağılımı')
    plt.xlabel('Gradient Büyüklüğü')
    plt.ylabel('Normalize Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Operatör özellikleri tablosu
    plt.subplot(3, 4, 12)
    properties_text = "Operatör Özellikleri:\n\n"
    properties_text += "Sobel:\n• 3x3 kernel\n• Gaussian smoothing\n• İyi gürültü dayanımı\n\n"
    properties_text += "Scharr:\n• 3x3 optimized kernel\n• Daha iyi rotasyon invariance\n• Sobel'den hassas\n\n"
    properties_text += "Prewitt:\n• 3x3 simple kernel\n• Hızlı hesaplama\n• Orta gürültü dayanımı\n\n"
    properties_text += "Roberts:\n• 2x2 kernel\n• Çok hızlı\n• Gürültüye hassas\n\n"
    properties_text += "Kirsch:\n• 8 yönlü\n• Yön bilgisi\n• Hesaplama yoğun"
    
    plt.text(0.05, 0.95, properties_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=8)
    plt.title('Operatör Karşılaştırması')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı performans analizi
    print("\n🔍 Operatör Performans Analizi:")
    
    # Her operatör için istatistikler
    for method, result in zip(methods, results):
        mean_val = np.mean(result)
        std_val = np.std(result)
        max_val = np.max(result)
        edge_count = np.sum(result > 50)
        
        print(f"{method}:")
        print(f"  Ortalama: {mean_val:.1f}")
        print(f"  Std sapma: {std_val:.1f}")
        print(f"  Maksimum: {max_val}")
        print(f"  Kenar piksel (>50): {edge_count}")
    
    # En iyi operatör önerisi
    # Kenar sayısı ve standart sapma dengelemesi
    scores = []
    for i, result in enumerate(results):
        edge_ratio = np.sum(result > 50) / result.size
        sharpness = np.std(result)
        score = edge_ratio * sharpness  # Basit scoring
        scores.append(score)
    
    best_operator = methods[np.argmax(scores)]
    print(f"\nEn dengeli operatör: {best_operator}")
    
    print("\n📝 Operatör Seçim Rehberi:")
    print("   • Hız önceliği: Roberts")
    print("   • Genel kullanım: Sobel")
    print("   • Hassaslık: Scharr")
    print("   • Yön bilgisi: Kirsch")
    print("   • Basitlik: Prewitt")

def interaktif_kenar_demo():
    """İnteraktif kenar algılama demosu"""
    print("\n🎮 İnteraktif Kenar Algılama Demosu")
    print("=" * 40)
    print("Trackbar'ları kullanarak gerçek zamanlı kenar algılama görün!")
    
    # Test resmi yükle
    geometrik_path, _, _, _ = ornek_resim_olustur()
    resim = cv2.imread(geometrik_path)
    
    if resim is None:
        print("❌ Resim yüklenemedi!")
        return
    
    # Resmi küçült (performans için)
    resim = cv2.resize(resim, (400, 400))
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Pencere oluştur
    window_name = 'Interactive Edge Detection Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Trackbar'lar oluştur
    cv2.createTrackbar('Method', window_name, 0, 5, lambda x: None)
    cv2.createTrackbar('Threshold1', window_name, 100, 300, lambda x: None)
    cv2.createTrackbar('Threshold2', window_name, 200, 300, lambda x: None)
    cv2.createTrackbar('Kernel Size', window_name, 3, 7, lambda x: None)
    cv2.createTrackbar('Blur', window_name, 0, 10, lambda x: None)
    
    method_names = ['Canny', 'Sobel', 'Scharr', 'Laplacian', 'Prewitt', 'Roberts']
    
    print("🎛️ Kontroller:")
    print("   • Method: 0-5 (Canny, Sobel, Scharr, vb.)")
    print("   • Threshold1: 0-300 (Canny low, diğerleri için threshold)")
    print("   • Threshold2: 0-300 (Canny high)")
    print("   • Kernel Size: 3-7 (Sobel, Scharr için)")
    print("   • Blur: 0-10 (Ön blur miktarı)")
    print("   • ESC tuşu ile çıkış")
    
    while True:
        # Trackbar değerlerini oku
        method = cv2.getTrackbarPos('Method', window_name)
        threshold1 = cv2.getTrackbarPos('Threshold1', window_name)
        threshold2 = cv2.getTrackbarPos('Threshold2', window_name)
        kernel_size = cv2.getTrackbarPos('Kernel Size', window_name)
        blur_amount = cv2.getTrackbarPos('Blur', window_name)
        
        # Parametreleri kontrol et
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        if threshold1 < 1:
            threshold1 = 1
        if threshold2 < threshold1:
            threshold2 = threshold1 + 1
        
        # Blur uygula
        if blur_amount > 0:
            processed_gray = cv2.GaussianBlur(gray, (blur_amount*2+1, blur_amount*2+1), 0)
        else:
            processed_gray = gray.copy()
        
        # İşlemi uygula
        try:
            if method == 0:  # Canny
                result = cv2.Canny(processed_gray, threshold1, threshold2)
                info_text = f'Canny: low={threshold1}, high={threshold2}'
                
            elif method == 1:  # Sobel
                sobel_x = cv2.Sobel(processed_gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
                sobel_y = cv2.Sobel(processed_gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
                sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
                result = np.uint8(np.clip(sobel_combined, 0, 255))
                result = cv2.threshold(result, threshold1, 255, cv2.THRESH_BINARY)[1]
                info_text = f'Sobel: kernel={kernel_size}, thresh={threshold1}'
                
            elif method == 2:  # Scharr
                scharr_x = cv2.Scharr(processed_gray, cv2.CV_64F, 1, 0)
                scharr_y = cv2.Scharr(processed_gray, cv2.CV_64F, 0, 1)
                scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
                result = np.uint8(np.clip(scharr_combined, 0, 255))
                result = cv2.threshold(result, threshold1, 255, cv2.THRESH_BINARY)[1]
                info_text = f'Scharr: thresh={threshold1}'
                
            elif method == 3:  # Laplacian
                laplacian = cv2.Laplacian(processed_gray, cv2.CV_64F, ksize=kernel_size)
                result = np.uint8(np.absolute(laplacian))
                result = cv2.threshold(result, threshold1, 255, cv2.THRESH_BINARY)[1]
                info_text = f'Laplacian: kernel={kernel_size}, thresh={threshold1}'
                
            elif method == 4:  # Prewitt
                prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
                px = cv2.filter2D(processed_gray, cv2.CV_64F, prewitt_x)
                py = cv2.filter2D(processed_gray, cv2.CV_64F, prewitt_y)
                prewitt_combined = np.sqrt(px**2 + py**2)
                result = np.uint8(np.clip(prewitt_combined, 0, 255))
                result = cv2.threshold(result, threshold1, 255, cv2.THRESH_BINARY)[1]
                info_text = f'Prewitt: thresh={threshold1}'
                
            elif method == 5:  # Roberts
                roberts_x = np.array([[1, 0], [0, -1]])
                roberts_y = np.array([[0, 1], [-1, 0]])
                rx = cv2.filter2D(processed_gray, cv2.CV_64F, roberts_x)
                ry = cv2.filter2D(processed_gray, cv2.CV_64F, roberts_y)
                roberts_combined = np.sqrt(rx**2 + ry**2)
                result = np.uint8(np.clip(roberts_combined, 0, 255))
                result = cv2.threshold(result, threshold1, 255, cv2.THRESH_BINARY)[1]
                info_text = f'Roberts: thresh={threshold1}'
                
            else:
                result = processed_gray.copy()
                info_text = 'Unknown method'
                
        except Exception as e:
            result = processed_gray.copy()
            info_text = f'Error: {str(e)[:30]}...'
        
        # RGB'ye çevir (renkli metin için)
        if len(result.shape) == 2:
            result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            result_color = result.copy()
        
        # Bilgi metnini ekle
        current_method = method_names[min(method, len(method_names)-1)]
        cv2.putText(result_color, f'Method: {current_method}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_color, info_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if blur_amount > 0:
            cv2.putText(result_color, f'Blur: {blur_amount}', (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(result_color, 'ESC = Exit', (10, result_color.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Kenar sayısını göster
        edge_count = np.sum(result == 255) if len(result.shape) == 2 else np.sum(result[:,:,0] == 255)
        cv2.putText(result_color, f'Edges: {edge_count}', (10, result_color.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Sonucu göster
        cv2.imshow(window_name, result_color)
        
        # ESC tuşu kontrolü
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC tuşu
            break
    
    cv2.destroyAllWindows()
    print("✅ İnteraktif demo tamamlandı!")

def main():
    """Ana program"""
    print("🔍 OpenCV Kenar Algılama Algoritmaları")
    print("Bu program, çeşitli kenar algılama tekniklerini gösterir.\n")
    
    # Test resimleri oluştur
    geometrik_path, dogal_path, metin_path, gurultulu_path = ornek_resim_olustur()
    
    # Ana test resmi olarak geometrik resimi kullan
    resim = cv2.imread(geometrik_path)
    
    if resim is None:
        print("❌ Test resimleri oluşturulamadı!")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("🔍 Kenar Algılama Menüsü")
        print("=" * 50)
        print("1. Sobel Kenar Algılama")
        print("2. Canny Kenar Algılama")
        print("3. Laplacian Kenar Algılama")
        print("4. Scharr ve Diğer Operatörler")
        print("5. İnteraktif Kenar Algılama Demosu")
        print("0. Çıkış")
        
        try:
            secim = input("\nLütfen bir seçenek girin (0-5): ").strip()
            
            if secim == '0':
                print("👋 Görüşmek üzere!")
                break
            elif secim == '1':
                sobel_kenar_algilama(resim)
            elif secim == '2':
                canny_kenar_algilama(resim)
            elif secim == '3':
                laplacian_kenar_algilama(resim)
            elif secim == '4':
                scharr_ve_diger_operatorler(resim)
            elif secim == '5':
                interaktif_kenar_demo()
            else:
                print("❌ Geçersiz seçenek! Lütfen 0-5 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program kullanıcı tarafından sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()