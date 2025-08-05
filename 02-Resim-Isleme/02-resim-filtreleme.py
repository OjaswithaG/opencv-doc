"""
🎛️ OpenCV Resim Filtreleme
==========================

Bu dosyada resim filtreleme tekniklerini öğreneceksiniz:
- Gaussian blur (bulanıklaştırma)
- Motion blur (hareket bulanıklaştırması)
- Median filter (medyan filtresi)
- Bilateral filter (bilateral filtre)
- Custom kernel'lar ile özel filtreler
- Keskinleştirme filtreleri

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def ornek_resim_olustur():
    """Filtreleme testleri için örnek resimler oluştur"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 1. Detaylı resim oluştur
    resim = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Gradyan arka plan
    for i in range(400):
        for j in range(400):
            resim[i, j] = [min(255, i//2), min(255, (i+j)//3), max(0, min(255, 255-j//2))]
    
    # Keskin kenarlar
    cv2.rectangle(resim, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(resim, (60, 60), (140, 140), (0, 0, 0), -1)
    
    # Çizgiler (filtreleme etkisini görmek için)
    for i in range(0, 400, 10):
        cv2.line(resim, (200, i), (250, i), (255, 255, 255), 1)
        cv2.line(resim, (i, 200), (i, 250), (255, 255, 255), 1)
    
    # Daireler
    cv2.circle(resim, (300, 100), 40, (255, 0, 0), -1)
    cv2.circle(resim, (320, 120), 20, (0, 255, 0), -1)
    
    # Metin (keskin detaylar)
    cv2.putText(resim, 'FILTER', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                2, (255, 255, 0), 3)
    cv2.putText(resim, 'TEST', (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, (0, 255, 255), 2)
    
    dosya_yolu = examples_dir / "filter_test.jpg"
    cv2.imwrite(str(dosya_yolu), resim)
    
    # 2. Gürültülü resim oluştur
    gurultulu_resim = resim.copy()
    gurultu = np.random.normal(0, 25, resim.shape).astype(np.uint8)
    gurultulu_resim = cv2.add(gurultulu_resim, gurultu)
    
    gurultulu_dosya = examples_dir / "noisy_test.jpg"
    cv2.imwrite(str(gurultulu_dosya), gurultulu_resim)
    
    print(f"✅ Test resimleri oluşturuldu:")
    print(f"   - Temiz resim: {dosya_yolu}")
    print(f"   - Gürültülü resim: {gurultulu_dosya}")
    
    return str(dosya_yolu), str(gurultulu_dosya)

def gaussian_blur_ornekleri(resim):
    """Gaussian blur filtreleme örnekleri"""
    print("\n📊 Gaussian Blur Örnekleri")
    print("=" * 35)
    
    # Farklı Gaussian blur seviyeleri
    blur_hafif = cv2.GaussianBlur(resim, (5, 5), 0)
    blur_orta = cv2.GaussianBlur(resim, (15, 15), 0)
    blur_guclu = cv2.GaussianBlur(resim, (35, 35), 0)
    
    # Farklı sigma değerleri ile
    blur_sigma_1 = cv2.GaussianBlur(resim, (21, 21), 1)
    blur_sigma_5 = cv2.GaussianBlur(resim, (21, 21), 5)
    blur_sigma_10 = cv2.GaussianBlur(resim, (21, 21), 10)
    
    # Asimetrik kernel
    blur_asimetrik = cv2.GaussianBlur(resim, (31, 11), 0)
    
    # Manual Gaussian kernel oluşturma
    kernel_size = 15
    sigma = 3
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = kernel @ kernel.T  # Outer product
    manual_blur = cv2.filter2D(resim, -1, kernel_2d)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(blur_hafif, cv2.COLOR_BGR2RGB))
    plt.title('Hafif Blur (5x5)')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(blur_orta, cv2.COLOR_BGR2RGB))
    plt.title('Orta Blur (15x15)')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(cv2.cvtColor(blur_guclu, cv2.COLOR_BGR2RGB))
    plt.title('Güçlü Blur (35x35)')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(cv2.cvtColor(blur_sigma_1, cv2.COLOR_BGR2RGB))
    plt.title('Sigma=1 (21x21)')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(cv2.cvtColor(blur_sigma_5, cv2.COLOR_BGR2RGB))
    plt.title('Sigma=5 (21x21)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(cv2.cvtColor(blur_sigma_10, cv2.COLOR_BGR2RGB))
    plt.title('Sigma=10 (21x21)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(cv2.cvtColor(blur_asimetrik, cv2.COLOR_BGR2RGB))
    plt.title('Asimetrik Blur (31x11)')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(manual_blur, cv2.COLOR_BGR2RGB))
    plt.title('Manuel Kernel (15x15)')
    plt.axis('off')
    
    # Kernel görselleştirmesi
    plt.subplot(3, 4, 10)
    plt.imshow(kernel_2d, cmap='hot', interpolation='nearest')
    plt.title('Gaussian Kernel (15x15)')
    plt.colorbar()
    
    # Profil karşılaştırması
    plt.subplot(3, 4, 11)
    row = resim.shape[0] // 2
    plt.plot(resim[row, :, 0], 'b-', label='Orijinal', alpha=0.7)
    plt.plot(blur_hafif[row, :, 0], 'g-', label='Hafif blur', alpha=0.7)
    plt.plot(blur_orta[row, :, 0], 'r-', label='Orta blur', alpha=0.7)
    plt.plot(blur_guclu[row, :, 0], 'm-', label='Güçlü blur', alpha=0.7)
    plt.title('Yatay Profil Karşılaştırması')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Parametreler açıklaması
    plt.subplot(3, 4, 12)
    plt.text(0.05, 0.95, 'Gaussian Blur Parametreleri:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '• Kernel Size (ksize):\n  Filtrenin boyutu\n  Tek sayı olmalı (3,5,7...)', 
             fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.50, '• Sigma (σ):\n  Bulanıklık miktarı\n  0 = otomatik hesapla\n  Büyük σ = daha bulanık', 
             fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.15, 'Kullanım Alanları:\n• Gürültü azaltma\n• Ön işleme\n• Artistik efektler', 
             fontsize=9, verticalalignment='top')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Kernel boyutu ve sigma ilişkisi
    print("\n📊 Kernel Boyutu ve Sigma İlişkisi:")
    kernel_sizes = [5, 11, 21, 31]
    sigmas = [1, 3, 7, 10]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, ksize in enumerate(kernel_sizes):
        blur_result = cv2.GaussianBlur(resim, (ksize, ksize), 0)
        axes[0, i].imshow(cv2.cvtColor(blur_result, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f'Kernel Size: {ksize}x{ksize}')
        axes[0, i].axis('off')
        
    for i, sigma in enumerate(sigmas):
        blur_result = cv2.GaussianBlur(resim, (21, 21), sigma)
        axes[1, i].imshow(cv2.cvtColor(blur_result, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f'Sigma: {sigma} (21x21)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Gaussian Blur İpuçları:")
    print("   • Kernel boyutu tek sayı olmalı (3, 5, 7, ...)")
    print("   • Büyük kernel = daha yumuşak blur")
    print("   • Sigma=0 otomatik hesaplama")
    print("   • Gaussian blur lineer ve separable bir filtredir")

def median_filter_ornekleri(resim, gurultulu_resim):
    """Median filter örnekleri"""
    print("\n🔢 Median Filter Örnekleri")
    print("=" * 35)
    
    # Salt & Pepper gürültüsü ekle
    salt_pepper = resim.copy()
    # Salt noise (beyaz noktalar)
    salt_coords = [np.random.randint(0, i - 1, int(0.02 * resim.size/3)) for i in resim.shape[:2]]
    salt_pepper[salt_coords[0], salt_coords[1], :] = 255
    
    # Pepper noise (siyah noktalar)
    pepper_coords = [np.random.randint(0, i - 1, int(0.02 * resim.size/3)) for i in resim.shape[:2]]
    salt_pepper[pepper_coords[0], pepper_coords[1], :] = 0
    
    # Farklı kernel boyutları ile median filter
    median_3 = cv2.medianBlur(salt_pepper, 3)
    median_5 = cv2.medianBlur(salt_pepper, 5)
    median_9 = cv2.medianBlur(salt_pepper, 9)
    median_15 = cv2.medianBlur(salt_pepper, 15)
    
    # Gaussian blur ile karşılaştırma (aynı gürültülü resim üzerinde)
    gaussian_5 = cv2.GaussianBlur(salt_pepper, (5, 5), 0)
    gaussian_9 = cv2.GaussianBlur(salt_pepper, (9, 9), 0)
    
    # Normal gaussian gürültü için karşılaştırma
    median_normal = cv2.medianBlur(gurultulu_resim, 5)
    gaussian_normal = cv2.GaussianBlur(gurultulu_resim, (5, 5), 0)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 15))
    
    # Salt & Pepper gürültü ile median filter
    plt.subplot(4, 4, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(cv2.cvtColor(salt_pepper, cv2.COLOR_BGR2RGB))
    plt.title('Salt & Pepper Gürültü')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(cv2.cvtColor(median_3, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter (3x3)')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(cv2.cvtColor(median_5, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter (5x5)')
    plt.axis('off')
    
    plt.subplot(4, 4, 5)
    plt.imshow(cv2.cvtColor(median_9, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter (9x9)')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(cv2.cvtColor(median_15, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter (15x15)')
    plt.axis('off')
    
    # Gaussian vs Median karşılaştırması
    plt.subplot(4, 4, 7)
    plt.imshow(cv2.cvtColor(gaussian_5, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur (5x5)\nSalt&Pepper üzerinde')
    plt.axis('off')
    
    plt.subplot(4, 4, 8)
    plt.imshow(cv2.cvtColor(gaussian_9, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur (9x9)\nSalt&Pepper üzerinde')
    plt.axis('off')
    
    # Normal gürültü karşılaştırması
    plt.subplot(4, 4, 9)
    plt.imshow(cv2.cvtColor(gurultulu_resim, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Gürültü')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(cv2.cvtColor(median_normal, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter (5x5)\nGaussian gürültü üzerinde')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(cv2.cvtColor(gaussian_normal, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur (5x5)\nGaussian gürültü üzerinde')
    plt.axis('off')
    
    # Profil karşılaştırması
    plt.subplot(4, 4, 12)
    row = salt_pepper.shape[0] // 2
    plt.plot(salt_pepper[row, :, 0], 'r-', label='Salt&Pepper', alpha=0.7, linewidth=1)
    plt.plot(median_5[row, :, 0], 'g-', label='Median (5x5)', alpha=0.8, linewidth=2)
    plt.plot(gaussian_5[row, :, 0], 'b-', label='Gaussian (5x5)', alpha=0.8, linewidth=2)
    plt.title('Yatay Profil Karşılaştırması')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Median filter özellikleri
    plt.subplot(4, 4, 13)
    plt.text(0.05, 0.95, 'Median Filter Özellikleri:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '✓ Outlier\'ları (aykırı değer) kaldırır\n✓ Kenarları korur\n✓ Salt & pepper için ideal', 
             fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.50, 'Gaussian vs Median:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.35, '• Gaussian: Tüm gürültü türleri\n• Median: İmpulsive gürültü\n• Median: Kenar koruyucu', 
             fontsize=9, verticalalignment='top')
    plt.axis('off')
    
    # Kernel boyutu etkisi
    plt.subplot(4, 4, 14)
    kernel_sizes = [3, 5, 9, 15]
    performance = []
    
    for ksize in kernel_sizes:
        result = cv2.medianBlur(salt_pepper, ksize)
        # Basit kalite metriği (gürültü azalma)
        diff = np.mean(np.abs(result.astype(float) - resim.astype(float)))
        performance.append(diff)
    
    plt.plot(kernel_sizes, performance, 'bo-', linewidth=2, markersize=8)
    plt.title('Kernel Boyutu vs Hata')
    plt.xlabel('Kernel Boyutu')
    plt.ylabel('Ortalama Piksel Hatası')
    plt.grid(True, alpha=0.3)
    
    # Hesaplama karmaşıklığı
    plt.subplot(4, 4, 15)
    plt.text(0.05, 0.95, 'Hesaplama Karmaşıklığı:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '• Median: O(k²log(k²))\n• Gaussian: O(k²)', fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.50, 'Median daha yavaş ama\nkenar koruyucu!', fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.25, 'Kullanım:\n• Tıbbi görüntüleme\n• Belge işleme\n• Eski fotoğraf restorasyonu', 
             fontsize=9, verticalalignment='top')
    plt.axis('off')
    
    # Gerçek zamanlı karşılaştırma
    plt.subplot(4, 4, 16)
    # ROI seç ve yakınlaştır
    roi_x, roi_y, roi_w, roi_h = 50, 50, 100, 100
    roi_original = salt_pepper[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_median = median_5[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Yan yana göster
    comparison = np.hstack([roi_original, roi_median])
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title('Yakınlaştırma: Öncesi vs Sonrası')
    plt.axvline(x=roi_w, color='yellow', linewidth=2)
    plt.text(roi_w//2, -5, 'Öncesi', ha='center', color='red', weight='bold')
    plt.text(roi_w + roi_w//2, -5, 'Sonrası', ha='center', color='green', weight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Median Filter İpuçları:")
    print("   • Salt & pepper gürültü için idealdir")
    print("   • Kenarları korur, Gaussian'dan daha iyi")
    print("   • Kernel boyutu tek sayı olmalı")
    print("   • Hesaplama yavaş, büyük kernellerden kaçının")

def bilateral_filter_ornekleri(resim, gurultulu_resim):
    """Bilateral filter örnekleri"""
    print("\n🎭 Bilateral Filter Örnekleri")
    print("=" * 35)
    
    # Farklı bilateral filter parametreleri
    bilateral_1 = cv2.bilateralFilter(gurultulu_resim, 9, 75, 75)
    bilateral_2 = cv2.bilateralFilter(gurultulu_resim, 9, 150, 150)
    bilateral_3 = cv2.bilateralFilter(gurultulu_resim, 15, 50, 50)
    bilateral_4 = cv2.bilateralFilter(gurultulu_resim, 15, 100, 100)
    
    # Farklı d değerleri (kernel boyutu)
    bilateral_d5 = cv2.bilateralFilter(gurultulu_resim, 5, 75, 75)
    bilateral_d15 = cv2.bilateralFilter(gurultulu_resim, 15, 75, 75)
    bilateral_d25 = cv2.bilateralFilter(gurultulu_resim, 25, 75, 75)
    
    # Gaussian blur ile karşılaştırma
    gaussian_comp = cv2.GaussianBlur(gurultulu_resim, (9, 9), 0)
    median_comp = cv2.medianBlur(gurultulu_resim, 9)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 15))
    
    plt.subplot(4, 4, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(cv2.cvtColor(gurultulu_resim, cv2.COLOR_BGR2RGB))
    plt.title('Gürültülü Resim')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(cv2.cvtColor(bilateral_1, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral (d=9, σc=75, σs=75)')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(cv2.cvtColor(bilateral_2, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral (d=9, σc=150, σs=150)')
    plt.axis('off')
    
    plt.subplot(4, 4, 5)
    plt.imshow(cv2.cvtColor(bilateral_3, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral (d=15, σc=50, σs=50)')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(cv2.cvtColor(bilateral_4, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral (d=15, σc=100, σs=100)')
    plt.axis('off')
    
    # Kernel boyutu karşılaştırması
    plt.subplot(4, 4, 7)
    plt.imshow(cv2.cvtColor(bilateral_d5, cv2.COLOR_BGR2RGB))
    plt.title('d=5 (Küçük kernel)')
    plt.axis('off')
    
    plt.subplot(4, 4, 8)
    plt.imshow(cv2.cvtColor(bilateral_d15, cv2.COLOR_BGR2RGB))
    plt.title('d=15 (Orta kernel)')
    plt.axis('off')
    
    plt.subplot(4, 4, 9)
    plt.imshow(cv2.cvtColor(bilateral_d25, cv2.COLOR_BGR2RGB))
    plt.title('d=25 (Büyük kernel)')
    plt.axis('off')
    
    # Filtre karşılaştırması
    plt.subplot(4, 4, 10)
    plt.imshow(cv2.cvtColor(gaussian_comp, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur (9x9)')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(cv2.cvtColor(median_comp, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter (9x9)')
    plt.axis('off')
    
    plt.subplot(4, 4, 12)
    plt.imshow(cv2.cvtColor(bilateral_1, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral Filter')
    plt.axis('off')
    
    # Profil karşılaştırması
    plt.subplot(4, 4, 13)
    row = gurultulu_resim.shape[0] // 2
    plt.plot(gurultulu_resim[row, :, 0], 'r-', alpha=0.7, label='Gürültülü')
    plt.plot(gaussian_comp[row, :, 0], 'b-', alpha=0.8, label='Gaussian')
    plt.plot(median_comp[row, :, 0], 'g-', alpha=0.8, label='Median')
    plt.plot(bilateral_1[row, :, 0], 'm-', alpha=0.8, label='Bilateral', linewidth=2)
    plt.plot(resim[row, :, 0], 'k--', alpha=0.5, label='Orijinal')
    plt.title('Yatay Profil Karşılaştırması')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Parametreler açıklaması
    plt.subplot(4, 4, 14)
    plt.text(0.05, 0.95, 'Bilateral Filter Parametreleri:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, 'd: Kernel çapı (5-25)\nσc (sigmaColor): Renk farkı\nσs (sigmaSpace): Mesafe farkı', 
             fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.45, 'Parametre Etkisi:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.30, '• Büyük σc: Daha fazla renk karıştır\n• Büyük σs: Daha geniş alan\n• Büyük d: Daha yavaş', 
             fontsize=9, verticalalignment='top')
    plt.axis('off')
    
    # Kenar koruma karşılaştırması
    plt.subplot(4, 4, 15)
    # ROI seç (kenar içeren bölge)
    roi_x, roi_y, roi_w, roi_h = 45, 45, 110, 110
    
    roi_original = gurultulu_resim[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_gaussian = gaussian_comp[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_bilateral = bilateral_1[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Üst-alt olarak göster
    comparison = np.vstack([
        np.hstack([roi_original, roi_gaussian]),
        np.hstack([roi_bilateral, np.zeros_like(roi_bilateral)])
    ])
    
    plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    plt.title('Kenar Koruma Karşılaştırması')
    
    # Bölgeleri işaretle
    plt.axhline(y=roi_h, color='yellow', linewidth=2)
    plt.axvline(x=roi_w, color='yellow', linewidth=2)
    
    plt.text(roi_w//2, roi_h-10, 'Gürültülü', ha='center', color='white', weight='bold')
    plt.text(roi_w + roi_w//2, roi_h-10, 'Gaussian', ha='center', color='white', weight='bold')
    plt.text(roi_w//2, roi_h + roi_h//2, 'Bilateral', ha='center', color='white', weight='bold')
    
    plt.axis('off')
    
    # Hesaplama süreleri (yaklaşık)
    plt.subplot(4, 4, 16)
    filters = ['Gaussian', 'Median', 'Bilateral']
    times = [1, 5, 20]  # Yaklaşık relatif süreler
    colors = ['blue', 'green', 'magenta']
    
    bars = plt.bar(filters, times, color=colors, alpha=0.7)
    plt.title('Hesaplama Süreleri (Yaklaşık)')
    plt.ylabel('Relatif Süre')
    plt.grid(True, alpha=0.3)
    
    # Değerleri bar'ların üzerine yaz
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{time}x', ha='center', weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Parametre etkisi analizi
    print("\n🔍 Parametre Etkisi Analizi:")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sigma Color etkisi
    sigma_colors = [25, 75, 150]
    for i, sc in enumerate(sigma_colors):
        result = cv2.bilateralFilter(gurultulu_resim, 9, sc, 75)
        axes[0, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f'σc={sc}, σs=75, d=9')
        axes[0, i].axis('off')
    
    # Sigma Space etkisi  
    sigma_spaces = [25, 75, 150]
    for i, ss in enumerate(sigma_spaces):
        result = cv2.bilateralFilter(gurultulu_resim, 9, 75, ss)
        axes[1, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f'σc=75, σs={ss}, d=9')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Bilateral Filter İpuçları:")
    print("   • Kenar koruyucu gürültü azaltma için ideal")
    print("   • σc renk benzerliği, σs mekan benzerliği")
    print("   • Büyük σ değerleri = daha güçlü filtreleme")
    print("   • Hesaplama yavaş, parametre dengelemesi önemli")
    print("   • Portre fotoğrafları için mükemmel")

def motion_blur_ornekleri(resim):
    """Motion blur (hareket bulanıklaştırması) örnekleri"""
    print("\n💨 Motion Blur Örnekleri")
    print("=" * 35)
    
    # Yatay motion blur kernel
    def motion_blur_kernel(size, angle):
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Açıya göre çizgi çiz
        for i in range(size):
            x = int(center + (i - center) * np.cos(np.radians(angle)))
            y = int(center + (i - center) * np.sin(np.radians(angle)))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        return kernel / np.sum(kernel)
    
    # Farklı yönlerde motion blur
    motion_horizontal = cv2.filter2D(resim, -1, motion_blur_kernel(15, 0))
    motion_vertical = cv2.filter2D(resim, -1, motion_blur_kernel(15, 90))
    motion_diagonal = cv2.filter2D(resim, -1, motion_blur_kernel(15, 45))
    motion_reverse_diagonal = cv2.filter2D(resim, -1, motion_blur_kernel(15, -45))
    
    # Farklı boyutlarda motion blur
    motion_small = cv2.filter2D(resim, -1, motion_blur_kernel(7, 0))
    motion_medium = cv2.filter2D(resim, -1, motion_blur_kernel(15, 0))
    motion_large = cv2.filter2D(resim, -1, motion_blur_kernel(25, 0))
    
    # OpenCV'nin kendi motion blur'ü (alternatif yöntem)
    kernel_h = np.zeros((1, 15))
    kernel_h[0, :] = 1/15
    opencv_motion_h = cv2.filter2D(resim, -1, kernel_h)
    
    kernel_v = np.zeros((15, 1))
    kernel_v[:, 0] = 1/15
    opencv_motion_v = cv2.filter2D(resim, -1, kernel_v)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 15))
    
    plt.subplot(4, 4, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(cv2.cvtColor(motion_horizontal, cv2.COLOR_BGR2RGB))
    plt.title('Yatay Motion Blur (0°)')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(cv2.cvtColor(motion_vertical, cv2.COLOR_BGR2RGB))
    plt.title('Dikey Motion Blur (90°)')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(cv2.cvtColor(motion_diagonal, cv2.COLOR_BGR2RGB))
    plt.title('Çapraz Motion Blur (45°)')
    plt.axis('off')
    
    plt.subplot(4, 4, 5)
    plt.imshow(cv2.cvtColor(motion_reverse_diagonal, cv2.COLOR_BGR2RGB))
    plt.title('Ters Çapraz Motion Blur (-45°)')
    plt.axis('off')
    
    # Boyut karşılaştırması
    plt.subplot(4, 4, 6)
    plt.imshow(cv2.cvtColor(motion_small, cv2.COLOR_BGR2RGB))
    plt.title('Küçük Motion Blur (7px)')
    plt.axis('off')
    
    plt.subplot(4, 4, 7)
    plt.imshow(cv2.cvtColor(motion_medium, cv2.COLOR_BGR2RGB))
    plt.title('Orta Motion Blur (15px)')
    plt.axis('off')
    
    plt.subplot(4, 4, 8)
    plt.imshow(cv2.cvtColor(motion_large, cv2.COLOR_BGR2RGB))
    plt.title('Büyük Motion Blur (25px)')
    plt.axis('off')
    
    # OpenCV alternatif yöntemler
    plt.subplot(4, 4, 9)
    plt.imshow(cv2.cvtColor(opencv_motion_h, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Yatay (1x15 kernel)')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(cv2.cvtColor(opencv_motion_v, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Dikey (15x1 kernel)')
    plt.axis('off')
    
    # Kernel görselleştirmeleri
    plt.subplot(4, 4, 11)
    kernel_display = motion_blur_kernel(15, 0)
    plt.imshow(kernel_display, cmap='hot', interpolation='nearest')
    plt.title('Yatay Motion Kernel')
    plt.colorbar()
    
    plt.subplot(4, 4, 12)
    kernel_display = motion_blur_kernel(15, 45)
    plt.imshow(kernel_display, cmap='hot', interpolation='nearest')
    plt.title('Çapraz Motion Kernel')
    plt.colorbar()
    
    # Açı karşılaştırması (dairesel)
    plt.subplot(4, 4, 13)
    angles = [0, 30, 60, 90, 120, 150]
    results = []
    
    for angle in angles:
        result = cv2.filter2D(resim, -1, motion_blur_kernel(11, angle))
        results.append(result)
    
    # Küçük bir ROI'den örnekler göster
    roi = resim[100:200, 100:200]
    combined = roi.copy()
    
    for i, angle in enumerate(angles[:3]):
        result = cv2.filter2D(roi, -1, motion_blur_kernel(7, angle))
        y_start = i * 33
        y_end = min((i + 1) * 33, 100)
        combined[y_start:y_end] = result[y_start:y_end]
    
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title('Farklı Açılar (0°, 30°, 60°)')
    plt.axis('off')
    
    # Motion blur kullanımları
    plt.subplot(4, 4, 14)
    plt.text(0.05, 0.95, 'Motion Blur Kullanım Alanları:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '🎬 Hareket efekti simülasyonu\n📸 Kamera sarsıntısı düzeltme\n🎨 Artistik efektler', 
             fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.45, 'Kernel Oluşturma:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.30, '• Çizgi şeklinde kernel\n• Hareket yönünde 1\'ler\n• Toplam = 1 (normalizasyon)', 
             fontsize=9, verticalalignment='top')
    plt.axis('off')
    
    # Pratik uygulama örneği
    plt.subplot(4, 4, 15)
    # Farklı hızlarda motion blur
    speeds = [3, 7, 15, 25]
    speed_comparison = []
    
    for speed in speeds:
        if speed <= 25:  # Çok büyük kernellerden kaçın
            blurred = cv2.filter2D(resim, -1, motion_blur_kernel(speed, 30))
            # Küçük bir parça al
            piece = blurred[150:200, 150:200]
            speed_comparison.append(piece)
    
    if speed_comparison:
        # İlk parçayı göster
        plt.imshow(cv2.cvtColor(speed_comparison[0], cv2.COLOR_BGR2RGB))
        plt.title('Değişken Hız Efekti')
        plt.axis('off')
    
    # Deblurring ipuçları
    plt.subplot(4, 4, 16)
    plt.text(0.05, 0.95, 'Motion Blur Giderme:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '• Wiener filtreleme\n• Richardson-Lucy\n• Blind deconvolution', 
             fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.45, 'İpuçları:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.30, '• Kernel boyutu tek olmalı\n• Hareket yönü önemli\n• Normalizasyon gerekli', 
             fontsize=9, verticalalignment='top')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Interaktif açı demonstration
    print("\n🎯 Farklı Motion Blur Açıları:")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    for i, angle in enumerate(angles):
        row = i // 4
        col = i % 4
        
        blurred = cv2.filter2D(resim, -1, motion_blur_kernel(13, angle))
        axes[row, col].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'{angle}°')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Motion Blur İpuçları:")
    print("   • filter2D() ile custom kernel kullanın")
    print("   • Kernel boyutu hareket mesafesini belirler")
    print("   • Açı parametresi hareket yönünü kontrol eder")
    print("   • Normalizasyon önemli (kernel toplamı = 1)")
    print("   • Performans için küçük kerneller tercih edin")

def custom_kernel_ornekleri(resim):
    """Custom kernel örnekleri"""
    print("\n🔧 Custom Kernel Örnekleri")
    print("=" * 35)
    
    # 1. Keskinleştirme kernelleri
    # Basit keskinleştirme
    sharpen_kernel = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])
    
    # Güçlü keskinleştirme
    sharpen_strong = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    
    # Unsharp masking kernel
    unsharp_kernel = np.array([[-1, -4, -6, -4, -1],
                               [-4, -16, -24, -16, -4],
                               [-6, -24, 476, -24, -6],
                               [-4, -16, -24, -16, -4],
                               [-1, -4, -6, -4, -1]]) / 256
    
    # 2. Kenar algılama kernelleri
    # Sobel X
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    # Sobel Y
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    # Laplacian
    laplacian = np.array([[ 0, -1,  0],
                          [-1,  4, -1],
                          [ 0, -1,  0]])
    
    # 3. Emboss kernelleri
    emboss_1 = np.array([[-2, -1,  0],
                         [-1,  1,  1],
                         [ 0,  1,  2]])
    
    emboss_2 = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]])
    
    # 4. Artistic kernelleri
    # Edge enhancement
    edge_enhance = np.array([[ 0,  0,  0],
                             [-1,  1,  0],
                             [ 0,  0,  0]])
    
    # Box filter (ortalama alma)
    box_filter = np.ones((5, 5)) / 25
    
    # Filtreleri uygula
    sharpen_result = cv2.filter2D(resim, -1, sharpen_kernel)
    sharpen_strong_result = cv2.filter2D(resim, -1, sharpen_strong)
    unsharp_result = cv2.filter2D(resim, -1, unsharp_kernel)
    
    sobel_x_result = cv2.filter2D(resim, -1, sobel_x)
    sobel_y_result = cv2.filter2D(resim, -1, sobel_y)
    laplacian_result = cv2.filter2D(resim, -1, laplacian)
    
    emboss_1_result = cv2.filter2D(resim, -1, emboss_1)
    emboss_2_result = cv2.filter2D(resim, -1, emboss_2)
    
    edge_enhance_result = cv2.filter2D(resim, -1, edge_enhance)
    box_filter_result = cv2.filter2D(resim, -1, box_filter)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Orijinal
    plt.subplot(4, 5, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    # Keskinleştirme sonuçları
    plt.subplot(4, 5, 2)
    plt.imshow(cv2.cvtColor(sharpen_result, cv2.COLOR_BGR2RGB))
    plt.title('Basit Keskinleştirme')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(cv2.cvtColor(sharpen_strong_result, cv2.COLOR_BGR2RGB))
    plt.title('Güçlü Keskinleştirme')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(cv2.cvtColor(unsharp_result, cv2.COLOR_BGR2RGB))
    plt.title('Unsharp Masking')
    plt.axis('off')
    
    plt.subplot(4, 5, 5)
    plt.imshow(cv2.cvtColor(box_filter_result, cv2.COLOR_BGR2RGB))
    plt.title('Box Filter (5x5)')
    plt.axis('off')
    
    # Kenar algılama sonuçları
    plt.subplot(4, 5, 6)
    plt.imshow(cv2.cvtColor(sobel_x_result, cv2.COLOR_BGR2RGB))
    plt.title('Sobel X')
    plt.axis('off')
    
    plt.subplot(4, 5, 7)
    plt.imshow(cv2.cvtColor(sobel_y_result, cv2.COLOR_BGR2RGB))
    plt.title('Sobel Y')
    plt.axis('off')
    
    plt.subplot(4, 5, 8)
    plt.imshow(cv2.cvtColor(laplacian_result, cv2.COLOR_BGR2RGB))
    plt.title('Laplacian')
    plt.axis('off')
    
    # Emboss sonuçları
    plt.subplot(4, 5, 9)
    plt.imshow(cv2.cvtColor(emboss_1_result, cv2.COLOR_BGR2RGB))
    plt.title('Emboss 1')
    plt.axis('off')
    
    plt.subplot(4, 5, 10)
    plt.imshow(cv2.cvtColor(emboss_2_result, cv2.COLOR_BGR2RGB))
    plt.title('Emboss 2')
    plt.axis('off')
    
    # Kernel görselleştirmeleri
    kernels = [sharpen_kernel, sharpen_strong, sobel_x, sobel_y, laplacian]
    kernel_names = ['Keskin', 'Güçlü Keskin', 'Sobel X', 'Sobel Y', 'Laplacian']
    
    for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        plt.subplot(4, 5, 11 + i)
        plt.imshow(kernel, cmap='RdBu', interpolation='nearest')
        plt.title(f'{name} Kernel')
        plt.colorbar()
        
        # Kernel değerlerini göster
        for (j, k), val in np.ndenumerate(kernel):
            plt.text(k, j, f'{val:.1f}', ha='center', va='center', 
                    color='white' if abs(val) > kernel.max()/2 else 'black')
    
    # Pratik kernel tasarımı
    plt.subplot(4, 5, 16)
    plt.text(0.05, 0.95, 'Kernel Tasarım İlkeleri:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '• Toplam = 0: Kenar algılama\n• Toplam = 1: Filtreleme\n• Merkez > 0: Keskinleştirme', 
             fontsize=9, verticalalignment='top')
    plt.text(0.05, 0.45, 'Örnek Kernel Türleri:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.30, '• Low-pass: Bulanıklaştırma\n• High-pass: Keskinleştirme\n• Edge: Kenar algılama', 
             fontsize=9, verticalalignment='top')
    plt.axis('off')
    
    # Kombine efektler
    plt.subplot(4, 5, 17)
    # Önce blur sonra sharpen
    blurred = cv2.GaussianBlur(resim, (5, 5), 0)
    blur_then_sharp = cv2.filter2D(blurred, -1, sharpen_kernel)
    plt.imshow(cv2.cvtColor(blur_then_sharp, cv2.COLOR_BGR2RGB))
    plt.title('Blur + Sharpen')
    plt.axis('off')
    
    # Gradyan magnitude (Sobel X + Y)
    plt.subplot(4, 5, 18)
    sobel_combined = np.sqrt(sobel_x_result.astype(float)**2 + sobel_y_result.astype(float)**2)
    sobel_combined = np.clip(sobel_combined, 0, 255).astype(np.uint8)
    plt.imshow(cv2.cvtColor(sobel_combined, cv2.COLOR_BGR2RGB))
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    # Custom artistic filter
    plt.subplot(4, 5, 19)
    # Özel sanatsal kernel
    artistic_kernel = np.array([[ 1,  1,  1,  1,  1],
                                [ 1, -2, -2, -2,  1],
                                [ 1, -2, 12, -2,  1],
                                [ 1, -2, -2, -2,  1],
                                [ 1,  1,  1,  1,  1]]) / 4
    artistic_result = cv2.filter2D(resim, -1, artistic_kernel)
    plt.imshow(cv2.cvtColor(artistic_result, cv2.COLOR_BGR2RGB))
    plt.title('Özel Sanatsal Filtre')
    plt.axis('off')
    
    # Kernel etki analizi
    plt.subplot(4, 5, 20)
    # Farklı kernel boyutlarının etkisi
    kernel_sizes = [3, 5, 7, 9]
    effects = []
    
    for size in kernel_sizes:
        # Basit ortalama kernel
        kernel = np.ones((size, size)) / (size * size)
        result = cv2.filter2D(resim, -1, kernel)
        # Bulanıklık seviyesini ölç (basit metrik)
        laplacian_var = cv2.Laplacian(result, cv2.CV_64F).var()
        effects.append(laplacian_var)
    
    plt.plot(kernel_sizes, effects, 'bo-', linewidth=2, markersize=8)
    plt.title('Kernel Boyutu vs Keskinlik')
    plt.xlabel('Kernel Boyutu')
    plt.ylabel('Laplacian Varyansı')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Popüler kernel koleksiyonu
    print("\n🎨 Popüler Kernel Koleksiyonu:")
    
    # Daha fazla kernel örneği
    kernels_dict = {
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Ridge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Diagonal Edge': np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
        'Vertical Edge': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Horizontal Edge': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'High Pass': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, kernel) in enumerate(kernels_dict.items()):
        if i < 6:
            result = cv2.filter2D(resim, -1, kernel)
            axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[i].set_title(name)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Custom Kernel İpuçları:")
    print("   • filter2D() fonksiyonu ile özel kerneller uygulayın")
    print("   • Kernel boyutu genellikle tek sayı (3x3, 5x5, ...)")
    print("   • Toplam=1 → Filtreleme, Toplam=0 → Kenar algılama")
    print("   • Negatif değerler kenar vurgulama yapar")
    print("   • Normalizasyon yapmayı unutmayın")

def interaktif_filtre_demo():
    """İnteraktif filtre demosu"""
    print("\n🎮 İnteraktif Filtre Demosu")
    print("=" * 35)
    print("Trackbar'ları kullanarak gerçek zamanlı filtreleme görün!")
    
    # Test resmi yükle
    resim_yolu, _ = ornek_resim_olustur()
    resim = cv2.imread(resim_yolu)
    
    if resim is None:
        print("❌ Resim yüklenemedi!")
        return
    
    # Resmi küçült (performans için)
    resim = cv2.resize(resim, (400, 400))
    
    # Pencere oluştur
    window_name = 'Interactive Filter Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Trackbar'lar oluştur
    cv2.createTrackbar('Filter Type', window_name, 0, 5, lambda x: None)
    cv2.createTrackbar('Kernel Size', window_name, 5, 25, lambda x: None)
    cv2.createTrackbar('Sigma', window_name, 10, 100, lambda x: None)
    cv2.createTrackbar('Strength', window_name, 10, 20, lambda x: None)
    
    filter_names = ['Original', 'Gaussian', 'Median', 'Bilateral', 'Sharpen', 'Motion']
    
    print("🎛️ Kontroller:")
    print("   • Filter Type: 0-5 (Gaussian, Median, vb.)")
    print("   • Kernel Size: 1-25")
    print("   • Sigma: 1-100 (Gaussian için)")
    print("   • Strength: 1-20 (Keskinleştirme için)")
    print("   • ESC tuşu ile çıkış")
    
    while True:
        # Trackbar değerlerini oku
        filter_type = cv2.getTrackbarPos('Filter Type', window_name)
        kernel_size = cv2.getTrackbarPos('Kernel Size', window_name)
        sigma = cv2.getTrackbarPos('Sigma', window_name)
        strength = cv2.getTrackbarPos('Strength', window_name)
        
        # Kernel boyutunu tek sayı yap
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        
        # Sigma kontrolü
        if sigma < 1:
            sigma = 1
        
        # Filtreyi uygula
        try:
            if filter_type == 0:  # Original
                result = resim.copy()
            elif filter_type == 1:  # Gaussian
                result = cv2.GaussianBlur(resim, (kernel_size, kernel_size), sigma/10.0)
            elif filter_type == 2:  # Median
                result = cv2.medianBlur(resim, kernel_size)
            elif filter_type == 3:  # Bilateral
                result = cv2.bilateralFilter(resim, kernel_size, sigma, sigma)
            elif filter_type == 4:  # Sharpen
                sharpen_kernel = np.array([[ 0, -1,  0],
                                           [-1,  4+strength/5, -1],
                                           [ 0, -1,  0]])
                result = cv2.filter2D(resim, -1, sharpen_kernel)
            elif filter_type == 5:  # Motion
                motion_kernel = np.zeros((kernel_size, kernel_size))
                motion_kernel[kernel_size//2, :] = 1/kernel_size
                result = cv2.filter2D(resim, -1, motion_kernel)
            else:
                result = resim.copy()
                
        except Exception as e:
            result = resim.copy()
        
        # Bilgi metnini ekle
        current_filter = filter_names[min(filter_type, len(filter_names)-1)]
        cv2.putText(result, f'Filter: {current_filter}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f'Kernel: {kernel_size}x{kernel_size}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, f'Sigma: {sigma/10.0:.1f}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, 'ESC = Exit', (10, 370), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Sonucu göster
        cv2.imshow(window_name, result)
        
        # ESC tuşu kontrolü
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC tuşu
            break
    
    cv2.destroyAllWindows()
    print("✅ İnteraktif demo tamamlandı!")

def main():
    """Ana program"""
    print("🎛️ OpenCV Resim Filtreleme")
    print("Bu program, resim filtreleme tekniklerini gösterir.\n")
    
    # Örnek resimler oluştur
    resim_yolu, gurultulu_resim_yolu = ornek_resim_olustur()
    resim = cv2.imread(resim_yolu)
    gurultulu_resim = cv2.imread(gurultulu_resim_yolu)
    
    if resim is None or gurultulu_resim is None:
        print("❌ Test resimleri oluşturulamadı!")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("🎛️ Resim Filtreleme Menüsü")
        print("=" * 50)
        print("1. Gaussian Blur Örnekleri")
        print("2. Median Filter Örnekleri")
        print("3. Bilateral Filter Örnekleri")
        print("4. Motion Blur Örnekleri")
        print("5. Custom Kernel Örnekleri")
        print("6. İnteraktif Filtre Demosu")
        print("0. Çıkış")
        
        try:
            secim = input("\nLütfen bir seçenek girin (0-6): ").strip()
            
            if secim == '0':
                print("👋 Görüşmek üzere!")
                break
            elif secim == '1':
                gaussian_blur_ornekleri(resim)
            elif secim == '2':
                median_filter_ornekleri(resim, gurultulu_resim)
            elif secim == '3':
                bilateral_filter_ornekleri(resim, gurultulu_resim)
            elif secim == '4':
                motion_blur_ornekleri(resim)
            elif secim == '5':
                custom_kernel_ornekleri(resim)
            elif secim == '6':
                interaktif_filtre_demo()
            else:
                print("❌ Geçersiz seçenek! Lütfen 0-6 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program kullanıcı tarafından sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()