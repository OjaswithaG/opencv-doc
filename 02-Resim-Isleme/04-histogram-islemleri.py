"""
📊 OpenCV Histogram İşlemleri
=============================

Bu dosyada histogram analizi ve düzeltme tekniklerini öğreneceksiniz:
- Histogram hesaplama ve görselleştirme
- Histogram eşitleme (equalization)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Histogram karşılaştırma ve analiz
- Histogram gerdirme (stretching)
- Histogram tabanlı segmentasyon

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ornek_resim_olustur():
    """Histogram analizi için örnek resimler oluştur"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 1. Düşük kontrastlı resim
    dusuk_kontrast = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Gri tonlarda gradient
    for i in range(300):
        for j in range(300):
            value = int(80 + 40 * np.sin(i/50) * np.cos(j/50))
            dusuk_kontrast[i, j] = [value, value, value]
    
    # Şekiller ekle
    cv2.rectangle(dusuk_kontrast, (50, 50), (150, 150), (120, 120, 120), -1)
    cv2.circle(dusuk_kontrast, (200, 200), 50, (140, 140, 140), -1)
    
    dusuk_kontrast_dosya = examples_dir / "low_contrast.jpg"
    cv2.imwrite(str(dusuk_kontrast_dosya), dusuk_kontrast)
    
    # 2. Yüksek kontrastlı resim
    yuksek_kontrast = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Siyah-beyaz desenler
    cv2.rectangle(yuksek_kontrast, (0, 0), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(yuksek_kontrast, (150, 0), (300, 150), (0, 0, 0), -1)
    cv2.rectangle(yuksek_kontrast, (0, 150), (150, 300), (0, 0, 0), -1)
    cv2.rectangle(yuksek_kontrast, (150, 150), (300, 300), (255, 255, 255), -1)
    
    # Gri tonlar ekle
    cv2.circle(yuksek_kontrast, (75, 75), 30, (128, 128, 128), -1)
    cv2.circle(yuksek_kontrast, (225, 225), 30, (64, 64, 64), -1)
    
    yuksek_kontrast_dosya = examples_dir / "high_contrast.jpg"
    cv2.imwrite(str(yuksek_kontrast_dosya), yuksek_kontrast)
    
    # 3. Karanlık resim
    karanlik_resim = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Karanlık gradient
    for i in range(300):
        for j in range(300):
            value = int(20 + 30 * (i + j) / 600)
            karanlik_resim[i, j] = [value, value+5, value+10]
    
    # Karanlık şekiller
    cv2.rectangle(karanlik_resim, (100, 100), (200, 200), (60, 70, 80), -1)
    cv2.circle(karanlik_resim, (150, 50), 25, (40, 45, 50), -1)
    
    karanlik_dosya = examples_dir / "dark_image.jpg"
    cv2.imwrite(str(karanlik_dosya), karanlik_resim)
    
    # 4. Parlak resim
    parlak_resim = np.full((300, 300, 3), 200, dtype=np.uint8)
    
    # Parlak gradient
    for i in range(300):
        for j in range(300):
            value = int(200 + 30 * np.sin(i/30) * np.sin(j/30))
            value = min(255, max(180, value))
            parlak_resim[i, j] = [value-10, value, value-5]
    
    # Parlak şekiller
    cv2.rectangle(parlak_resim, (80, 80), (220, 220), (230, 240, 245), -1)
    cv2.circle(parlak_resim, (150, 150), 40, (250, 255, 255), -1)
    
    parlak_dosya = examples_dir / "bright_image.jpg"
    cv2.imwrite(str(parlak_dosya), parlak_resim)
    
    print(f"✅ Histogram test resimleri oluşturuldu:")
    print(f"   - Düşük kontrast: {dusuk_kontrast_dosya}")
    print(f"   - Yüksek kontrast: {yuksek_kontrast_dosya}")
    print(f"   - Karanlık resim: {karanlik_dosya}")
    print(f"   - Parlak resim: {parlak_dosya}")
    
    return (str(dusuk_kontrast_dosya), str(yuksek_kontrast_dosya), 
            str(karanlik_dosya), str(parlak_dosya))

def histogram_hesaplama_ornekleri(resim):
    """Histogram hesaplama ve görselleştirme örnekleri"""
    print("\n📊 Histogram Hesaplama ve Görselleştirme")
    print("=" * 45)
    
    # RGB histogramları
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(20, 12))
    
    # Orijinal resim
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    # RGB histogramları ayrı ayrı
    plt.subplot(3, 4, 2)
    for i, color in enumerate(colors):
        hist = cv2.calcHist([resim], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.7)
    plt.title('RGB Histogramları')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend(['Blue', 'Green', 'Red'])
    plt.grid(True, alpha=0.3)
    
    # Gri seviye histogramı
    plt.subplot(3, 4, 3)
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist_gray, color='black')
    plt.title('Gri Seviye Histogramı')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.grid(True, alpha=0.3)
    
    # Kümülatif histogram
    plt.subplot(3, 4, 4)
    hist_cumulative = np.cumsum(hist_gray)
    plt.plot(hist_cumulative, color='purple')
    plt.title('Kümülatif Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Kümülatif Frekans')
    plt.grid(True, alpha=0.3)
    
    # Normalize edilmiş histogram
    plt.subplot(3, 4, 5)
    hist_normalized = hist_gray / hist_gray.sum()
    plt.plot(hist_normalized, color='orange')
    plt.title('Normalize Edilmiş Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Olasılık')
    plt.grid(True, alpha=0.3)
    
    # Maskelenmiş histogram
    plt.subplot(3, 4, 6)
    # Merkezi bölge maskesi
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.circle(mask, (gray.shape[1]//2, gray.shape[0]//2), 
               min(gray.shape)//3, 255, -1)
    
    hist_masked = cv2.calcHist([gray], [0], mask, [256], [0, 256])
    hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    plt.plot(hist_full, color='black', alpha=0.5, label='Tam Resim')
    plt.plot(hist_masked, color='red', alpha=0.8, label='Maskelenmiş')
    plt.title('Maskelenmiş Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2D Histogram (HSV)
    plt.subplot(3, 4, 7)
    hsv = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
    hist_2d = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist_2d, interpolation='nearest', cmap='hot')
    plt.title('2D Histogram (H-S)')
    plt.xlabel('Saturation')
    plt.ylabel('Hue')
    plt.colorbar()
    
    # Kanal analizi
    plt.subplot(3, 4, 8)
    # Her kanalın istatistikleri
    stats_text = ""
    for i, (channel, color) in enumerate(zip(cv2.split(resim), colors)):
        mean_val = np.mean(channel)
        std_val = np.std(channel)
        min_val = np.min(channel)
        max_val = np.max(channel)
        
        stats_text += f"{color.upper()}:\n"
        stats_text += f"  Ortalama: {mean_val:.1f}\n"
        stats_text += f"  Std Dev: {std_val:.1f}\n"
        stats_text += f"  Min-Max: {min_val}-{max_val}\n\n"
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    plt.title('Kanal İstatistikleri')
    plt.axis('off')
    
    # Histogram özellikleri
    plt.subplot(3, 4, 9)
    # Peak, valley analizi
    hist_smooth = cv2.GaussianBlur(hist_gray.astype(np.float32), (5, 1), 0)
    peaks = []
    valleys = []
    
    for i in range(1, len(hist_smooth)-1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            if hist_smooth[i] > np.max(hist_smooth) * 0.1:  # Minimum peak height
                peaks.append(i)
        elif hist_smooth[i] < hist_smooth[i-1] and hist_smooth[i] < hist_smooth[i+1]:
            valleys.append(i)
    
    plt.plot(hist_gray, color='black', alpha=0.7)
    for peak in peaks:
        plt.axvline(x=peak, color='red', linestyle='--', alpha=0.7)
    for valley in valleys:
        plt.axvline(x=valley, color='blue', linestyle='--', alpha=0.7)
    
    plt.title(f'Peak-Valley Analizi\n{len(peaks)} peak, {len(valleys)} valley')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.grid(True, alpha=0.3)
    
    # Histogram entropi
    plt.subplot(3, 4, 10)
    prob = hist_normalized[hist_normalized > 0]
    entropy = -np.sum(prob * np.log2(prob))
    
    # Kontrast metriği
    contrast = np.std(gray.astype(np.float32))
    
    # Parlaklık metriği
    brightness = np.mean(gray)
    
    metrics_text = f"Entropi: {entropy:.2f}\n"
    metrics_text += f"Kontrast: {contrast:.2f}\n"
    metrics_text += f"Parlaklık: {brightness:.1f}\n\n"
    metrics_text += f"Değerlendirme:\n"
    
    if entropy < 6:
        metrics_text += "• Düşük entropi (monoton)\n"
    elif entropy > 7:
        metrics_text += "• Yüksek entropi (detaylı)\n"
    else:
        metrics_text += "• Orta entropi (dengeli)\n"
        
    if contrast < 30:
        metrics_text += "• Düşük kontrast\n"
    elif contrast > 60:
        metrics_text += "• Yüksek kontrast\n"
    else:
        metrics_text += "• Normal kontrast\n"
    
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    plt.title('Görüntü Metrikleri')
    plt.axis('off')
    
    # Dinamik aralık analizi
    plt.subplot(3, 4, 11)
    # Etkili dinamik aralık (5th-95th percentile)
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    dynamic_range = p95 - p5
    
    plt.axvspan(0, p5, alpha=0.3, color='red', label=f'Alt %5 (0-{p5:.0f})')
    plt.axvspan(p5, p95, alpha=0.3, color='green', label=f'Etkili Aralık ({p5:.0f}-{p95:.0f})')
    plt.axvspan(p95, 255, alpha=0.3, color='red', label=f'Üst %5 ({p95:.0f}-255)')
    
    plt.plot(hist_gray, color='black', alpha=0.8)
    plt.title(f'Dinamik Aralık: {dynamic_range:.0f}')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram shape analizi
    plt.subplot(3, 4, 12)
    # Skewness ve Kurtosis
    from scipy import stats
    gray_flat = gray.flatten()
    skewness = stats.skew(gray_flat)
    kurtosis = stats.kurtosis(gray_flat)
    
    shape_text = f"Çarpıklık (Skewness): {skewness:.2f}\n"
    shape_text += f"Basıklık (Kurtosis): {kurtosis:.2f}\n\n"
    
    if skewness < -0.5:
        shape_text += "• Sol çarpık (karanlık ağırlıklı)\n"
    elif skewness > 0.5:
        shape_text += "• Sağ çarpık (aydınlık ağırlıklı)\n"
    else:
        shape_text += "• Simetrik dağılım\n"
        
    if kurtosis < 0:
        shape_text += "• Düz dağılım (platykurtic)\n"
    elif kurtosis > 0:
        shape_text += "• Sivri dağılım (leptokurtic)\n"
    else:
        shape_text += "• Normal dağılım (mesokurtic)\n"
    
    plt.text(0.05, 0.95, shape_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9)
    plt.title('Dağılım Şekli Analizi')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Histogram Hesaplama İpuçları:")
    print("   • calcHist() ile histogram hesaplayın")
    print("   • [kanal], mask, histSize, ranges parametrelerini doğru ayarlayın")
    print("   • 2D histogram H-S kanalları ile renk analizi yapın")
    print("   • Entropi ve kontrast metrikleri görüntü kalitesini gösterir")

def histogram_esitleme_ornekleri(resim):
    """Histogram eşitleme örnekleri"""
    print("\n⚖️ Histogram Eşitleme Örnekleri")
    print("=" * 35)
    
    # Gri seviyeye çevir
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Histogram eşitleme
    equalized = cv2.equalizeHist(gray)
    
    # Manuel histogram eşitleme implementasyonu
    def manual_equalize(img):
        # Histogram hesapla
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        
        # Kümülatif dağılım fonksiyonu
        cdf = hist.cumsum()
        
        # CDF'yi normalize et
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Lookup table oluştur
        lut = np.uint8(cdf_normalized)
        
        # LUT uygula
        return cv2.LUT(img, lut), lut
    
    manual_equalized, lut = manual_equalize(gray)
    
    # Renkli resim için histogram eşitleme
    # Yöntem 1: RGB kanalları ayrı ayrı
    bgr_channels = cv2.split(resim)
    bgr_equalized = []
    for channel in bgr_channels:
        bgr_equalized.append(cv2.equalizeHist(channel))
    rgb_equalized = cv2.merge(bgr_equalized)
    
    # Yöntem 2: HSV'de sadece V kanalı
    hsv = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    hsv_equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Yöntem 3: YUV'de sadece Y kanalı
    yuv = cv2.cvtColor(resim, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    yuv_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Gri seviye eşitleme
    plt.subplot(4, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal (Gri)')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title('OpenCV Eşitleme')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(manual_equalized, cmap='gray')
    plt.title('Manuel Eşitleme')
    plt.axis('off')
    
    # Fark gösterimi
    plt.subplot(4, 4, 4)
    diff = cv2.absdiff(equalized, manual_equalized)
    plt.imshow(diff, cmap='hot')
    plt.title('Fark (OpenCV vs Manuel)')
    plt.colorbar()
    plt.axis('off')
    
    # Histogramlar - öncesi
    plt.subplot(4, 4, 5)
    hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.plot(hist_orig, color='blue', alpha=0.7)
    plt.title('Orijinal Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.grid(True, alpha=0.3)
    
    # Histogramlar - sonrası
    plt.subplot(4, 4, 6)
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    plt.plot(hist_eq, color='red', alpha=0.7)
    plt.title('Eşitlenmiş Histogram')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.grid(True, alpha=0.3)
    
    # CDF karşılaştırması
    plt.subplot(4, 4, 7)
    cdf_orig = np.cumsum(hist_orig)
    cdf_eq = np.cumsum(hist_eq)
    
    plt.plot(cdf_orig/cdf_orig.max(), color='blue', label='Orijinal CDF')
    plt.plot(cdf_eq/cdf_eq.max(), color='red', label='Eşitlenmiş CDF')
    plt.plot(np.linspace(0, 1, 256), color='green', linestyle='--', label='İdeal CDF')
    plt.title('Kümülatif Dağılım Fonksiyonu')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Normalize CDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # LUT görselleştirmesi
    plt.subplot(4, 4, 8)
    plt.plot(lut, color='purple', linewidth=2)
    plt.plot(np.arange(256), color='gray', linestyle='--', alpha=0.5, label='y=x')
    plt.title('Transformation Function (LUT)')
    plt.xlabel('Giriş Değeri')
    plt.ylabel('Çıkış Değeri')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Renkli resim eşitleme yöntemleri
    plt.subplot(4, 4, 9)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal (Renkli)')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(cv2.cvtColor(rgb_equalized, cv2.COLOR_BGR2RGB))
    plt.title('RGB Kanalları Ayrı')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(cv2.cvtColor(hsv_equalized, cv2.COLOR_BGR2RGB))
    plt.title('HSV V-Kanalı')
    plt.axis('off')
    
    plt.subplot(4, 4, 12)
    plt.imshow(cv2.cvtColor(yuv_equalized, cv2.COLOR_BGR2RGB))
    plt.title('YUV Y-Kanalı')
    plt.axis('off')
    
    # Kalite metrikleri
    plt.subplot(4, 4, 13)
    # Kontrast iyileştirme
    contrast_orig = np.std(gray.astype(np.float32))
    contrast_eq = np.std(equalized.astype(np.float32))
    contrast_improvement = (contrast_eq - contrast_orig) / contrast_orig * 100
    
    # Entropi iyileştirme
    hist_orig_norm = hist_orig / hist_orig.sum()
    hist_eq_norm = hist_eq / hist_eq.sum()
    
    entropy_orig = -np.sum(hist_orig_norm[hist_orig_norm > 0] * 
                          np.log2(hist_orig_norm[hist_orig_norm > 0]))
    entropy_eq = -np.sum(hist_eq_norm[hist_eq_norm > 0] * 
                        np.log2(hist_eq_norm[hist_eq_norm > 0]))
    
    metrics_text = f"Kontrast İyileştirmesi:\n"
    metrics_text += f"  Öncesi: {contrast_orig:.1f}\n"
    metrics_text += f"  Sonrası: {contrast_eq:.1f}\n"
    metrics_text += f"  İyileştirme: %{contrast_improvement:.1f}\n\n"
    metrics_text += f"Entropi:\n"
    metrics_text += f"  Öncesi: {entropy_orig:.2f}\n"
    metrics_text += f"  Sonrası: {entropy_eq:.2f}\n"
    
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    plt.title('Kalite Metrikleri')
    plt.axis('off')
    
    # Renkli histogram karşılaştırması
    plt.subplot(4, 4, 14)
    colors = ['blue', 'green', 'red']
    for i, color in enumerate(colors):
        hist_orig_color = cv2.calcHist([resim], [i], None, [256], [0, 256])
        hist_eq_color = cv2.calcHist([hsv_equalized], [i], None, [256], [0, 256])
        plt.plot(hist_orig_color, color=color, alpha=0.5, linewidth=1, label=f'{color.upper()} orig')
        plt.plot(hist_eq_color, color=color, alpha=0.8, linewidth=2, linestyle='--', label=f'{color.upper()} eq')
    
    plt.title('Renkli Histogram Karşılaştırması')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Bölgesel analiz
    plt.subplot(4, 4, 15)
    # Resmi 4 bölgeye ayır ve her birinin histogramını göster
    h, w = gray.shape
    regions = [
        gray[:h//2, :w//2],      # Sol üst
        gray[:h//2, w//2:],      # Sağ üst
        gray[h//2:, :w//2],      # Sol alt
        gray[h//2:, w//2:]       # Sağ alt
    ]
    
    region_names = ['Sol Üst', 'Sağ Üst', 'Sol Alt', 'Sağ Alt']
    colors_region = ['red', 'green', 'blue', 'orange']
    
    for region, name, color in zip(regions, region_names, colors_region):
        hist_region = cv2.calcHist([region], [0], None, [256], [0, 256])
        plt.plot(hist_region, color=color, alpha=0.7, label=name)
    
    plt.title('Bölgesel Histogram Analizi')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Eşitleme etkisi değerlendirmesi
    plt.subplot(4, 4, 16)
    evaluation_text = "Histogram Eşitleme Değerlendirmesi:\n\n"
    
    if contrast_improvement > 20:
        evaluation_text += "✅ Kontrast önemli ölçüde iyileşti\n"
    elif contrast_improvement > 0:
        evaluation_text += "⚠️ Kontrast kısmen iyileşti\n"
    else:
        evaluation_text += "❌ Kontrast iyileşmedi\n"
    
    if entropy_eq > entropy_orig:
        evaluation_text += "✅ Bilgi içeriği arttı\n"
    else:
        evaluation_text += "⚠️ Bilgi içeriği azaldı\n"
    
    # Histogram dağılımı kontrol
    hist_spread = np.count_nonzero(hist_eq > hist_eq.max() * 0.01)
    if hist_spread > 200:
        evaluation_text += "✅ İyi histogram dağılımı\n"
    elif hist_spread > 150:
        evaluation_text += "⚠️ Orta histogram dağılımı\n"
    else:
        evaluation_text += "❌ Kötü histogram dağılımı\n"
        
    evaluation_text += f"\nÖneriler:\n"
    if contrast_improvement < 10:
        evaluation_text += "• CLAHE kullanmayı deneyin\n"
    evaluation_text += "• Renkli resimler için HSV kullanın\n"
    evaluation_text += "• Bölgesel eşitleme düşünün\n"
    
    plt.text(0.05, 0.95, evaluation_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=8)
    plt.title('Eşitleme Değerlendirmesi')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Histogram Eşitleme İpuçları:")
    print("   • equalizeHist() sadece gri seviye resimler için")
    print("   • Renkli resimler için HSV V-kanalını eşitleyin")
    print("   • RGB kanalları ayrı eşitlemek renk değişimi yapar")
    print("   • Düşük kontrastlı resimler için idealdir")

def clahe_ornekleri(resim):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) örnekleri"""
    print("\n🎯 CLAHE Örnekleri")
    print("=" * 25)
    
    # Gri seviyeye çevir
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Farklı CLAHE parametreleri
    clahe_default = cv2.createCLAHE()
    clahe_result_default = clahe_default.apply(gray)
    
    clahe_2_8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_result_2_8 = clahe_2_8.apply(gray)
    
    clahe_4_8 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    clahe_result_4_8 = clahe_4_8.apply(gray)
    
    clahe_2_16 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    clahe_result_2_16 = clahe_2_16.apply(gray)
    
    clahe_2_4 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    clahe_result_2_4 = clahe_2_4.apply(gray)
    
    # Global histogram eşitleme ile karşılaştırma
    global_eq = cv2.equalizeHist(gray)
    
    # Renkli resim için CLAHE
    # HSV V-kanalı
    hsv = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = clahe_2_8.apply(hsv[:,:,2])
    hsv_clahe = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # LAB L-kanalı
    lab = cv2.cvtColor(resim, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe_2_8.apply(lab[:,:,0])
    lab_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Ana karşılaştırma
    plt.subplot(4, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(global_eq, cmap='gray')
    plt.title('Global Histogram Eşitleme')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(clahe_result_2_8, cmap='gray')
    plt.title('CLAHE (2.0, 8x8)')
    plt.axis('off')
    
    # Parametreler etkisi
    plt.subplot(4, 4, 4)
    plt.imshow(clahe_result_default, cmap='gray')
    plt.title('CLAHE Default')
    plt.axis('off')
    
    plt.subplot(4, 4, 5)
    plt.imshow(clahe_result_4_8, cmap='gray')
    plt.title('CLAHE (4.0, 8x8)')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(clahe_result_2_16, cmap='gray')
    plt.title('CLAHE (2.0, 16x16)')
    plt.axis('off')
    
    plt.subplot(4, 4, 7)
    plt.imshow(clahe_result_2_4, cmap='gray')
    plt.title('CLAHE (2.0, 4x4)')
    plt.axis('off')
    
    # Histogram karşılaştırması
    plt.subplot(4, 4, 8)
    hist_orig = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_global = cv2.calcHist([global_eq], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([clahe_result_2_8], [0], None, [256], [0, 256])
    
    plt.plot(hist_orig, color='blue', alpha=0.7, label='Orijinal')
    plt.plot(hist_global, color='red', alpha=0.7, label='Global Eq')
    plt.plot(hist_clahe, color='green', alpha=0.7, label='CLAHE')
    plt.title('Histogram Karşılaştırması')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Renkli CLAHE sonuçları
    plt.subplot(4, 4, 9)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal (Renkli)')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(cv2.cvtColor(hsv_clahe, cv2.COLOR_BGR2RGB))
    plt.title('HSV V-Kanalı CLAHE')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(cv2.cvtColor(lab_clahe, cv2.COLOR_BGR2RGB))
    plt.title('LAB L-Kanalı CLAHE')
    plt.axis('off')
    
    # Clip limit etkisi
    plt.subplot(4, 4, 12)
    clip_limits = [1.0, 2.0, 4.0, 8.0]
    clip_results = []
    
    for clip_limit in clip_limits:
        clahe_temp = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        result_temp = clahe_temp.apply(gray)
        contrast = np.std(result_temp.astype(np.float32))
        clip_results.append(contrast)
    
    plt.bar([str(cl) for cl in clip_limits], clip_results, 
            color=['red', 'green', 'blue', 'orange'], alpha=0.7)
    plt.title('Clip Limit vs Kontrast')
    plt.xlabel('Clip Limit')
    plt.ylabel('Kontrast (Std Dev)')
    plt.grid(True, alpha=0.3)
    
    # Tile size etkisi
    plt.subplot(4, 4, 13)
    tile_sizes = [(4,4), (8,8), (16,16), (32,32)]
    tile_results = []
    processing_times = []
    
    import time
    for tile_size in tile_sizes:
        clahe_temp = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
        
        start_time = time.time()
        result_temp = clahe_temp.apply(gray)
        processing_time = time.time() - start_time
        
        contrast = np.std(result_temp.astype(np.float32))
        tile_results.append(contrast)
        processing_times.append(processing_time * 1000)  # ms
    
    x_pos = np.arange(len(tile_sizes))
    width = 0.35
    
    plt.bar(x_pos - width/2, tile_results, width, label='Kontrast', color='blue', alpha=0.7)
    plt.bar(x_pos + width/2, processing_times, width, label='Süre (ms)', color='red', alpha=0.7)
    
    plt.title('Tile Size Etkisi')
    plt.xlabel('Tile Size')
    plt.xticks(x_pos, [f"{ts[0]}x{ts[1]}" for ts in tile_sizes])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bölgesel analiz
    plt.subplot(4, 4, 14)
    # 8x8 tile'ların her birinin kontrastını hesapla
    h, w = gray.shape
    tile_h, tile_w = h//8, w//8
    
    contrast_map = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            tile = gray[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            contrast_map[i, j] = np.std(tile.astype(np.float32))
    
    plt.imshow(contrast_map, cmap='hot', interpolation='nearest')
    plt.title('Bölgesel Kontrast Haritası')
    plt.colorbar(label='Kontrast')
    
    # Gürültü analizi
    plt.subplot(4, 4, 15)
    # ROI'den gürültü analizi
    roi = gray[gray.shape[0]//4:3*gray.shape[0]//4, 
               gray.shape[1]//4:3*gray.shape[1]//4]
    roi_global = global_eq[gray.shape[0]//4:3*gray.shape[0]//4, 
                          gray.shape[1]//4:3*gray.shape[1]//4]
    roi_clahe = clahe_result_2_8[gray.shape[0]//4:3*gray.shape[0]//4, 
                                gray.shape[1]//4:3*gray.shape[1]//4]
    
    # Laplacian varyansı (gürültü/keskinlik metriği)
    noise_orig = cv2.Laplacian(roi, cv2.CV_64F).var()
    noise_global = cv2.Laplacian(roi_global, cv2.CV_64F).var()
    noise_clahe = cv2.Laplacian(roi_clahe, cv2.CV_64F).var()
    
    methods = ['Orijinal', 'Global Eq', 'CLAHE']
    noise_values = [noise_orig, noise_global, noise_clahe]
    colors = ['blue', 'red', 'green']
    
    bars = plt.bar(methods, noise_values, color=colors, alpha=0.7)
    plt.title('Gürültü/Keskinlik Analizi\n(Laplacian Varyansı)')
    plt.ylabel('Varyans')
    plt.grid(True, alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for bar, value in zip(bars, noise_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(noise_values)*0.01, 
                f'{value:.0f}', ha='center', va='bottom')
    
    # Parametre rehberi
    plt.subplot(4, 4, 16)
    guide_text = "CLAHE Parametre Rehberi:\n\n"
    guide_text += "Clip Limit:\n"
    guide_text += "• 1.0-2.0: Az amplifikasyon\n"
    guide_text += "• 2.0-4.0: Orta amplifikasyon\n"
    guide_text += "• 4.0+: Güçlü amplifikasyon\n\n"
    guide_text += "Tile Size:\n"
    guide_text += "• 4x4: Çok yerel, detaylı\n"
    guide_text += "• 8x8: Dengeli (önerilen)\n"
    guide_text += "• 16x16+: Az yerel, yumuşak\n\n"
    guide_text += "Uygulama Alanları:\n"
    guide_text += "• Tıbbi görüntüleme\n"
    guide_text += "• Uydu görüntüleri\n"
    guide_text += "• Zayıf ışık fotoğrafları"
    
    plt.text(0.05, 0.95, guide_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=8)
    plt.title('Kullanım Rehberi')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 CLAHE İpuçları:")
    print("   • createCLAHE(clipLimit, tileGridSize) parametrelerini ayarlayın")
    print("   • Clip limit yüksek = güçlü kontrast, fazla gürültü")
    print("   • Küçük tile = yerel detay, büyük tile = global etki")
    print("   • Renkli resimler için L veya V kanalını işleyin")

def histogram_karsilastirma_ornekleri():
    """Histogram karşılaştırma ve benzerlik ölçütleri"""
    print("\n📈 Histogram Karşılaştırma ve Benzerlik")
    print("=" * 45)
    
    # Test resimleri oluştur
    low_contrast_path, high_contrast_path, dark_path, bright_path = ornek_resim_olustur()
    
    # Resimleri yükle
    img1 = cv2.imread(low_contrast_path)
    img2 = cv2.imread(high_contrast_path)
    img3 = cv2.imread(dark_path)
    img4 = cv2.imread(bright_path)
    
    images = [img1, img2, img3, img4]
    names = ['Düşük Kontrast', 'Yüksek Kontrast', 'Karanlık', 'Parlak']
    
    # Histogramları hesapla
    histograms = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        # Normalize et
        hist = hist / hist.sum()
        histograms.append(hist)
    
    # Benzerlik matrisi hesapla
    methods = [
        ('Correlation', cv2.HISTCMP_CORREL),
        ('Chi-Square', cv2.HISTCMP_CHISQR),
        ('Intersection', cv2.HISTCMP_INTERSECT),
        ('Bhattacharyya', cv2.HISTCMP_BHATTACHARYYA),
        ('Hellinger', cv2.HISTCMP_HELLINGER),
        ('KL-Divergence', cv2.HISTCMP_KL_DIV)
    ]
    
    n_images = len(images)
    n_methods = len(methods)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Orijinal resimler
    for i, (img, name) in enumerate(zip(images, names)):
        plt.subplot(4, 6, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
    
    # Histogramlar
    plt.subplot(4, 6, 5)
    colors = ['blue', 'red', 'green', 'orange']
    for hist, name, color in zip(histograms, names, colors):
        plt.plot(hist, color=color, alpha=0.7, label=name)
    plt.title('Normalize Histogramlar')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Olasılık')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Kümülatif histogramlar
    plt.subplot(4, 6, 6)
    for hist, name, color in zip(histograms, names, colors):
        cdf = np.cumsum(hist)
        plt.plot(cdf, color=color, alpha=0.7, label=name)
    plt.title('Kümülatif Histogramlar')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Kümülatif Olasılık')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Benzerlik matrisleri
    for method_idx, (method_name, method_flag) in enumerate(methods):
        plt.subplot(4, 6, 7 + method_idx)
        
        similarity_matrix = np.zeros((n_images, n_images))
        
        for i in range(n_images):
            for j in range(n_images):
                # OpenCV histogram karşılaştırma
                hist_i = (histograms[i] * 256).astype(np.float32)
                hist_j = (histograms[j] * 256).astype(np.float32)
                
                similarity = cv2.compareHist(hist_i, hist_j, method_flag)
                similarity_matrix[i, j] = similarity
        
        # Renk haritası ayarla
        if method_name in ['Correlation', 'Intersection']:
            cmap = 'viridis'  # Yüksek değer = benzer
            vmin, vmax = 0, 1
        else:
            cmap = 'viridis_r'  # Düşük değer = benzer
            vmin, vmax = None, None
        
        im = plt.imshow(similarity_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f'{method_name}')
        plt.xticks(range(n_images), [n[:4] for n in names], rotation=45)
        plt.yticks(range(n_images), [n[:4] for n in names])
        
        # Değerleri matriste göster
        for i in range(n_images):
            for j in range(n_images):
                plt.text(j, i, f'{similarity_matrix[i,j]:.2f}', 
                        ha='center', va='center', color='white', fontsize=8)
        
        plt.colorbar(im, shrink=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı benzerlik analizi
    print("\n🔍 Detaylı Benzerlik Analizi:")
    
    # En benzer ve en farklı çiftleri bul
    for method_name, method_flag in methods:
        print(f"\n{method_name}:")
        
        similarities = []
        for i in range(n_images):
            for j in range(i+1, n_images):
                hist_i = (histograms[i] * 256).astype(np.float32)
                hist_j = (histograms[j] * 256).astype(np.float32)
                
                sim = cv2.compareHist(hist_i, hist_j, method_flag)
                similarities.append((sim, names[i], names[j]))
        
        # Sırala
        if method_name in ['Correlation', 'Intersection']:
            similarities.sort(reverse=True)  # Büyük değer = benzer
            print(f"  En benzer: {similarities[0][1]} - {similarities[0][2]} ({similarities[0][0]:.3f})")
            print(f"  En farklı: {similarities[-1][1]} - {similarities[-1][2]} ({similarities[-1][0]:.3f})")
        else:
            similarities.sort()  # Küçük değer = benzer
            print(f"  En benzer: {similarities[0][1]} - {similarities[0][2]} ({similarities[0][0]:.3f})")
            print(f"  En farklı: {similarities[-1][1]} - {similarities[-1][2]} ({similarities[-1][0]:.3f})")
    
    # Manuel benzerlik hesaplamaları
    print("\n🧮 Manuel Benzerlik Hesaplamaları:")
    
    # Euclidean distance
    def euclidean_distance(h1, h2):
        return np.sqrt(np.sum((h1 - h2) ** 2))
    
    # Manhattan distance
    def manhattan_distance(h1, h2):
        return np.sum(np.abs(h1 - h2))
    
    # Cosine similarity
    def cosine_similarity(h1, h2):
        return np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))
    
    # Jensen-Shannon divergence
    def jensen_shannon_divergence(h1, h2):
        # Küçük epsilon ekle
        h1 = h1 + 1e-10
        h2 = h2 + 1e-10
        
        m = 0.5 * (h1 + h2)
        return 0.5 * np.sum(h1 * np.log(h1 / m)) + 0.5 * np.sum(h2 * np.log(h2 / m))
    
    manual_methods = [
        ('Euclidean Distance', euclidean_distance),
        ('Manhattan Distance', manhattan_distance),  
        ('Cosine Similarity', cosine_similarity),
        ('Jensen-Shannon Divergence', jensen_shannon_divergence)
    ]
    
    for method_name, method_func in manual_methods:
        print(f"\n{method_name}:")
        
        similarities = []
        for i in range(n_images):
            for j in range(i+1, n_images):
                sim = method_func(histograms[i], histograms[j])
                similarities.append((sim, names[i], names[j]))
        
        # Sırala
        if method_name == 'Cosine Similarity':
            similarities.sort(reverse=True)  # Büyük değer = benzer
        else:
            similarities.sort()  # Küçük değer = benzer
            
        print(f"  En benzer: {similarities[0][1]} - {similarities[0][2]} ({similarities[0][0]:.3f})")
        print(f"  En farklı: {similarities[-1][1]} - {similarities[-1][2]} ({similarities[-1][0]:.3f})")
    
    print("\n📝 Histogram Karşılaştırma İpuçları:")
    print("   • Correlation: -1 ile 1 arası, 1 = mükemmel benzerlik")
    print("   • Chi-Square: 0'a yakın = benzer")
    print("   • Intersection: 0 ile 1 arası, 1 = mükemmel benzerlik")
    print("   • Bhattacharyya: 0'a yakın = benzer")
    print("   • Normalize edilmiş histogramlar kullanın")

def interaktif_histogram_demo():
    """İnteraktif histogram demo"""
    print("\n🎮 İnteraktif Histogram Demosu")
    print("=" * 35)
    print("Trackbar'ları kullanarak gerçek zamanlı histogram işlemleri görün!")
    
    # Test resmi yükle
    low_contrast_path, _, _, _ = ornek_resim_olustur()
    resim = cv2.imread(low_contrast_path)
    
    if resim is None:
        print("❌ Resim yüklenemedi!")
        return
    
    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    # Pencere oluştur
    window_name = 'Interactive Histogram Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Trackbar'lar oluştur
    cv2.createTrackbar('Method', window_name, 0, 3, lambda x: None)
    cv2.createTrackbar('Clip Limit', window_name, 20, 80, lambda x: None)
    cv2.createTrackbar('Tile Size', window_name, 8, 32, lambda x: None)
    cv2.createTrackbar('Show Histogram', window_name, 0, 1, lambda x: None)
    
    method_names = ['Original', 'Global EQ', 'CLAHE', 'Custom']
    
    print("🎛️ Kontroller:")
    print("   • Method: 0-3 (Original, Global EQ, CLAHE, Custom)")
    print("   • Clip Limit: 0.2-8.0 (CLAHE için)")
    print("   • Tile Size: 8-32 (CLAHE için)")
    print("   • Show Histogram: 0-1 (Histogram göster/gizle)")
    print("   • ESC tuşu ile çıkış")
    
    while True:
        # Trackbar değerlerini oku
        method = cv2.getTrackbarPos('Method', window_name)
        clip_limit = cv2.getTrackbarPos('Clip Limit', window_name) / 10.0
        tile_size = cv2.getTrackbarPos('Tile Size', window_name)
        show_hist = cv2.getTrackbarPos('Show Histogram', window_name)
        
        # Parametreleri kontrol et
        if clip_limit < 0.2:
            clip_limit = 0.2
        if tile_size < 4:
            tile_size = 4
        if tile_size % 2 != 0:
            tile_size += 1
        
        # İşlemi uygula
        try:
            if method == 0:  # Original
                result = gray.copy()
            elif method == 1:  # Global Equalization
                result = cv2.equalizeHist(gray)
            elif method == 2:  # CLAHE
                clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                       tileGridSize=(tile_size, tile_size))
                result = clahe.apply(gray)
            elif method == 3:  # Custom (Gamma correction)
                gamma = clip_limit / 4.0  # Gamma değeri olarak kullan
                lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                               for i in np.arange(0, 256)]).astype("uint8")
                result = cv2.LUT(gray, lut)
            else:
                result = gray.copy()
                
        except Exception as e:
            result = gray.copy()
        
        # RGB'ye çevir
        if show_hist:
            # Histogram penceresi oluştur
            hist_height = 200
            hist_width = 256
            hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
            
            # Histogram hesapla
            hist = cv2.calcHist([result], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Normalize et
            if hist.max() > 0:
                hist = hist / hist.max() * (hist_height - 20)
            
            # Histogram çiz
            for i in range(256):
                cv2.line(hist_img, (i, hist_height), 
                        (i, hist_height - int(hist[i])), (0, 255, 0), 1)
            
            # Ana resim ve histogramı birleştir
            result_3ch = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            # Histogramı resmin altına ekle
            combined = np.vstack([result_3ch, hist_img])
        else:
            combined = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
        # Bilgi metnini ekle
        current_method = method_names[min(method, len(method_names)-1)]
        cv2.putText(combined, f'Method: {current_method}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if method == 2:  # CLAHE
            cv2.putText(combined, f'Clip: {clip_limit:.1f}, Tile: {tile_size}x{tile_size}', 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif method == 3:  # Custom
            gamma = clip_limit / 4.0
            cv2.putText(combined, f'Gamma: {gamma:.2f}', (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(combined, 'ESC = Exit', (10, combined.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Sonucu göster
        cv2.imshow(window_name, combined)
        
        # ESC tuşu kontrolü
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC tuşu
            break
    
    cv2.destroyAllWindows()
    print("✅ İnteraktif demo tamamlandı!")

def main():
    """Ana program"""
    print("📊 OpenCV Histogram İşlemleri")
    print("Bu program, histogram analizi ve düzeltme tekniklerini gösterir.\n")
    
    # Test resimleri oluştur
    low_contrast_path, high_contrast_path, dark_path, bright_path = ornek_resim_olustur()
    
    # Ana test resmi olarak düşük kontrastlı resmi kullan
    resim = cv2.imread(low_contrast_path)
    
    if resim is None:
        print("❌ Test resimleri oluşturulamadı!")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("📊 Histogram İşlemleri Menüsü")
        print("=" * 50)
        print("1. Histogram Hesaplama ve Görselleştirme")
        print("2. Histogram Eşitleme Örnekleri")
        print("3. CLAHE (Adaptive Histogram Equalization)")
        print("4. Histogram Karşılaştırma ve Benzerlik")
        print("5. İnteraktif Histogram Demosu")
        print("0. Çıkış")
        
        try:
            secim = input("\nLütfen bir seçenek girin (0-5): ").strip()
            
            if secim == '0':
                print("👋 Görüşmek üzere!")
                break
            elif secim == '1':
                histogram_hesaplama_ornekleri(resim)
            elif secim == '2':
                histogram_esitleme_ornekleri(resim)
            elif secim == '3':
                clahe_ornekleri(resim)
            elif secim == '4':
                histogram_karsilastirma_ornekleri()
            elif secim == '5':
                interaktif_histogram_demo()
            else:
                print("❌ Geçersiz seçenek! Lütfen 0-5 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program kullanıcı tarafından sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()