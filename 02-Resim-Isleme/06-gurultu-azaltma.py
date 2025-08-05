"""
🧼 OpenCV Gürültü Azaltma Teknikleri
====================================

Bu dosyada çeşitli gürültü türlerini temizleme tekniklerini öğreneceksiniz:
- Gaussian gürültü ve temizleme
- Salt & Pepper gürültü temizleme
- Speckle gürültü azaltma
- Non-local means denoising
- Bilateral filtering
- Wiener filtering
- Wavelet denoising
- Advanced gürültü azaltma teknikleri

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

def ornek_resim_olustur():
    """Gürültü azaltma testleri için örnek resimler oluştur"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 1. Temiz referans resim
    temiz_resim = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Düzgün gradient ve şekiller
    for i in range(300):
        for j in range(300):
            r = int(128 + 60 * np.sin(i/40) * np.cos(j/40))
            g = int(128 + 40 * np.cos(i/30) * np.sin(j/50))
            b = int(128 + 50 * np.sin((i+j)/45))
            temiz_resim[i, j] = [np.clip(b, 50, 200), np.clip(g, 50, 200), np.clip(r, 50, 200)]
    
    # Net şekiller
    cv2.rectangle(temiz_resim, (50, 50), (150, 150), (200, 100, 50), -1)
    cv2.circle(temiz_resim, (200, 200), 40, (50, 150, 200), -1)
    cv2.ellipse(temiz_resim, (100, 250), (30, 15), 30, 0, 360, (150, 200, 100), -1)
    
    # İnce detaylar
    for i in range(10, 290, 20):
        cv2.line(temiz_resim, (250, i), (280, i), (100, 100, 100), 1)
    
    temiz_dosya = examples_dir / "clean_reference.jpg"
    cv2.imwrite(str(temiz_dosya), temiz_resim)
    
    # 2. Gaussian gürültülü resim
    gaussian_gurultu = np.random.normal(0, 25, temiz_resim.shape).astype(np.int16)
    gaussian_resim = np.clip(temiz_resim.astype(np.int16) + gaussian_gurultu, 0, 255).astype(np.uint8)
    
    gaussian_dosya = examples_dir / "gaussian_noisy.jpg"
    cv2.imwrite(str(gaussian_dosya), gaussian_resim)
    
    # 3. Salt & Pepper gürültülü resim
    salt_pepper_resim = temiz_resim.copy().astype(np.float32)
    
    # Salt noise (beyaz noktalar)
    salt_mask = np.random.random(temiz_resim.shape[:2]) < 0.02
    salt_pepper_resim[salt_mask] = 255
    
    # Pepper noise (siyah noktalar)
    pepper_mask = np.random.random(temiz_resim.shape[:2]) < 0.02
    salt_pepper_resim[pepper_mask] = 0
    
    salt_pepper_resim = salt_pepper_resim.astype(np.uint8)
    salt_pepper_dosya = examples_dir / "salt_pepper_noisy.jpg"
    cv2.imwrite(str(salt_pepper_dosya), salt_pepper_resim)
    
    # 4. Speckle gürültülü resim
    speckle_gurultu = np.random.normal(0, 0.1, temiz_resim.shape)
    speckle_resim = temiz_resim.astype(np.float32) * (1 + speckle_gurultu)
    speckle_resim = np.clip(speckle_resim, 0, 255).astype(np.uint8)
    
    speckle_dosya = examples_dir / "speckle_noisy.jpg"
    cv2.imwrite(str(speckle_dosya), speckle_resim)
    
    # 5. Karma gürültülü resim (multiple noise types)
    karma_resim = temiz_resim.copy().astype(np.float32)
    
    # Gaussian + Salt & Pepper + Speckle
    gaussian_n = np.random.normal(0, 15, temiz_resim.shape)
    karma_resim += gaussian_n
    
    # Salt & Pepper (daha az yoğun)
    salt_m = np.random.random(temiz_resim.shape[:2]) < 0.01
    pepper_m = np.random.random(temiz_resim.shape[:2]) < 0.01
    karma_resim[salt_m] = 255
    karma_resim[pepper_m] = 0
    
    # Speckle
    speckle_n = np.random.normal(0, 0.05, temiz_resim.shape)
    karma_resim *= (1 + speckle_n)
    
    karma_resim = np.clip(karma_resim, 0, 255).astype(np.uint8)
    karma_dosya = examples_dir / "mixed_noisy.jpg"
    cv2.imwrite(str(karma_dosya), karma_resim)
    
    print(f"✅ Gürültü azaltma test resimleri oluşturuldu:")
    print(f"   - Temiz referans: {temiz_dosya}")
    print(f"   - Gaussian gürültü: {gaussian_dosya}")
    print(f"   - Salt & Pepper: {salt_pepper_dosya}")
    print(f"   - Speckle gürültü: {speckle_dosya}")
    print(f"   - Karma gürültü: {karma_dosya}")
    
    return (str(temiz_dosya), str(gaussian_dosya), str(salt_pepper_dosya), 
            str(speckle_dosya), str(karma_dosya))

def gaussian_gurultu_temizleme(temiz_resim, gurultulu_resim):
    """Gaussian gürültü temizleme teknikleri"""
    print("\n🔊 Gaussian Gürültü Temizleme Teknikleri")
    print("=" * 45)
    
    # Farklı Gaussian blur çeşitleri
    gaussian_3x3 = cv2.GaussianBlur(gurultulu_resim, (3, 3), 0)
    gaussian_5x5 = cv2.GaussianBlur(gurultulu_resim, (5, 5), 0)
    gaussian_7x7 = cv2.GaussianBlur(gurultulu_resim, (7, 7), 0)
    gaussian_adaptive = cv2.GaussianBlur(gurultulu_resim, (5, 5), 1.5)
    
    # Bilateral filtering (kenar koruyucu)
    bilateral = cv2.bilateralFilter(gurultulu_resim, 9, 75, 75)
    bilateral_strong = cv2.bilateralFilter(gurultulu_resim, 15, 100, 100)
    
    # Non-local means denoising
    nlm_gray = cv2.cvtColor(gurultulu_resim, cv2.COLOR_BGR2GRAY)
    nlm_result_gray = cv2.fastNlMeansDenoising(nlm_gray, None, 10, 7, 21)
    nlm_result_color = cv2.fastNlMeansDenoisingColored(gurultulu_resim, None, 10, 10, 7, 21)
    
    # Wiener filter simulation (basit implementasyon)
    def simple_wiener_filter(image, noise_variance=100):
        # Fourier transform
        f_transform = np.fft.fft2(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        f_shift = np.fft.fftshift(f_transform)
        
        # Power spectrum
        power_spectrum = np.abs(f_shift) ** 2
        
        # Wiener filter
        wiener_filter = power_spectrum / (power_spectrum + noise_variance)
        
        # Apply filter
        filtered = f_shift * wiener_filter
        
        # Inverse transform
        f_ishift = np.fft.ifftshift(filtered)
        result = np.fft.ifft2(f_ishift)
        result = np.abs(result)
        
        # Normalize and convert back to color
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    wiener_result = simple_wiener_filter(gurultulu_resim)
    
    # Adaptive filtering (variance based)
    def adaptive_gaussian_filter(image):
        # Her piksel çevresindeki varyansı hesapla
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_variance = local_sq_mean - local_mean**2
        
        # Yüksek varyans = az blur, düşük varyans = fazla blur
        result = image.copy()
        
        # Düşük varyans bölgeleri (düzgün alanlar) - fazla blur
        low_var_mask = local_variance < 200
        low_var_blur = cv2.GaussianBlur(image, (7, 7), 0)
        result[low_var_mask] = low_var_blur[low_var_mask]
        
        # Yüksek varyans bölgeleri (kenarlar) - az blur
        high_var_mask = local_variance > 800
        high_var_blur = cv2.GaussianBlur(image, (3, 3), 0)
        result[high_var_mask] = high_var_blur[high_var_mask]
        
        return result
    
    adaptive_result = adaptive_gaussian_filter(gurultulu_resim)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Ana görsel karşılaştırma
    results = [
        (temiz_resim, "Temiz Referans"),
        (gurultulu_resim, "Gaussian Gürültülü"),
        (gaussian_3x3, "Gaussian Blur 3x3"),
        (gaussian_5x5, "Gaussian Blur 5x5"),
        (gaussian_7x7, "Gaussian Blur 7x7"),
        (bilateral, "Bilateral Filter"),
        (bilateral_strong, "Strong Bilateral"),
        (nlm_result_color, "Non-Local Means"),
        (wiener_result, "Wiener Filter"),
        (adaptive_result, "Adaptive Filter")
    ]
    
    for i, (result_img, title) in enumerate(results[:10]):
        plt.subplot(3, 4, i+1)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    # PSNR hesaplama ve karşılaştırma
    def calculate_psnr(original, denoised):
        mse = np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    plt.subplot(3, 4, 11)
    method_names = ['Gürültülü', 'Gauss 3x3', 'Gauss 5x5', 'Bilateral', 'NLM', 'Wiener', 'Adaptive']
    test_images = [gurultulu_resim, gaussian_3x3, gaussian_5x5, bilateral, 
                   nlm_result_color, wiener_result, adaptive_result]
    
    psnr_values = []
    for img in test_images:
        psnr = calculate_psnr(temiz_resim, img)
        psnr_values.append(psnr)
    
    bars = plt.bar(range(len(method_names)), psnr_values, 
                   color=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'], alpha=0.7)
    plt.title('PSNR Karşılaştırması (Yüksek = İyi)')
    plt.xlabel('Yöntem')
    plt.ylabel('PSNR (dB)')
    plt.xticks(range(len(method_names)), method_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # PSNR değerlerini bar üzerine yaz
    for bar, psnr in zip(bars, psnr_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{psnr:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Hesaplama süreleri simülasyonu
    plt.subplot(3, 4, 12)
    # Yaklaşık relatif işlem süreleri
    processing_times = [0, 1, 1.2, 8, 15, 3, 5]  # Gaussian en hızlı, NLM en yavaş
    
    time_bars = plt.bar(range(len(method_names)), processing_times, 
                       color=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'], alpha=0.7)
    plt.title('Yaklaşık İşlem Süreleri (Düşük = Hızlı)')
    plt.xlabel('Yöntem')  
    plt.ylabel('Relatif Süre')
    plt.xticks(range(len(method_names)), method_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı analiz
    print("\n🔍 Gaussian Gürültü Temizleme Analizi:")
    
    # En iyi sonucu bul
    best_psnr = max(psnr_values)
    best_method_idx = psnr_values.index(best_psnr)
    best_method = method_names[best_method_idx]
    
    print(f"En yüksek PSNR: {best_psnr:.1f} dB ({best_method})")
    
    # Her yöntem için detaylı istatistikler
    for i, (method, img, psnr) in enumerate(zip(method_names, test_images, psnr_values)):
        ssim_score = calculate_ssim_simple(temiz_resim, img)
        print(f"{method}: PSNR={psnr:.1f} dB, SSIM={ssim_score:.3f}")
    
    print("\n📝 Gaussian Gürültü Temizleme İpuçları:")
    print("   • Gaussian blur: Hızlı ama kenarları bulanıklaştırır")
    print("   • Bilateral filter: Kenar koruyucu, orta hızlı")
    print("   • Non-local means: En iyi kalite, en yavaş")
    print("   • Adaptive filtering: Akıllı, bölgesel uygulama")

def calculate_ssim_simple(img1, img2):
    """Basit SSIM hesaplama"""
    # Gri seviyeye çevir
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # SSIM hesaplama (basitleştirilmiş)
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    
    sigma1_sq = np.var(gray1)
    sigma2_sq = np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
    
    # SSIM formülü
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return ssim

def salt_pepper_temizleme(temiz_resim, salt_pepper_resim):
    """Salt & Pepper gürültü temizleme teknikleri"""
    print("\n🧂 Salt & Pepper Gürültü Temizleme")
    print("=" * 40)
    
    # Median filtering (en etkili yöntem)
    median_3 = cv2.medianBlur(salt_pepper_resim, 3)
    median_5 = cv2.medianBlur(salt_pepper_resim, 5)
    median_7 = cv2.medianBlur(salt_pepper_resim, 7)
    median_9 = cv2.medianBlur(salt_pepper_resim, 9)
    
    # Gaussian blur karşılaştırması
    gaussian_comp = cv2.GaussianBlur(salt_pepper_resim, (5, 5), 0)
    
    # Morfolojik işlemler
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_open = cv2.morphologyEx(salt_pepper_resim, cv2.MORPH_OPEN, kernel)
    morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)
    
    # Adaptive median filter implementasyonu
    def adaptive_median_filter(image, max_window_size=7):
        result = image.copy()
        rows, cols = image.shape[:2]
        
        # Her piksel için
        for i in range(1, rows-max_window_size//2):
            for j in range(1, cols-max_window_size//2):
                for window_size in range(3, max_window_size+1, 2):
                    half_window = window_size // 2
                    
                    # Pencere çıkarımı
                    window = image[i-half_window:i+half_window+1, 
                                 j-half_window:j+half_window+1]
                    
                    if len(window.shape) == 3:
                        window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                    else:
                        window_gray = window
                    
                    z_med = np.median(window_gray)
                    z_min = np.min(window_gray)
                    z_max = np.max(window_gray)
                    z_xy = image[i, j]
                    
                    if len(z_xy.shape) > 0:
                        z_xy_val = np.mean(z_xy) if len(image.shape) == 3 else z_xy
                    else:
                        z_xy_val = z_xy
                    
                    # Stage A
                    A1 = z_med - z_min
                    A2 = z_med - z_max
                    
                    if A1 > 0 and A2 < 0:
                        # Stage B
                        B1 = z_xy_val - z_min
                        B2 = z_xy_val - z_max
                        
                        if B1 > 0 and B2 < 0:
                            break  # Piksel değişmez
                        else:
                            result[i, j] = z_med
                            break
                    else:
                        if window_size < max_window_size:
                            continue  # Pencere boyutunu artır
                        else:
                            result[i, j] = z_med
                            break
        
        return result
    
    # Basit adaptive median (performans için küçük bölgede)
    roi_y, roi_x = 50, 50
    roi_h, roi_w = 100, 100
    roi = salt_pepper_resim[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    adaptive_median_roi = adaptive_median_filter(roi)
    
    adaptive_median_result = salt_pepper_resim.copy()
    adaptive_median_result[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = adaptive_median_roi
    
    # Bilateral filter denemeleri
    bilateral_sp = cv2.bilateralFilter(salt_pepper_resim, 9, 75, 75)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    results = [
        (temiz_resim, "Temiz Referans"),
        (salt_pepper_resim, "Salt & Pepper Gürültü"),
        (median_3, "Median 3x3"),
        (median_5, "Median 5x5"),
        (median_7, "Median 7x7"),
        (median_9, "Median 9x9"),
        (gaussian_comp, "Gaussian 5x5 (Karşılaştırma)"),
        (morph_close, "Morfolojik (Open+Close)"),
        (bilateral_sp, "Bilateral Filter"),
        (adaptive_median_result, "Adaptive Median (ROI)")
    ]
    
    for i, (result_img, title) in enumerate(results):
        plt.subplot(3, 4, i+1)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    # PSNR karşılaştırması
    plt.subplot(3, 4, 11)
    method_names = ['SP Gürültülü', 'Median 3', 'Median 5', 'Median 7', 'Gaussian', 'Morph', 'Bilateral']
    test_images = [salt_pepper_resim, median_3, median_5, median_7, gaussian_comp, morph_close, bilateral_sp]
    
    psnr_values = []
    for img in test_images:
        psnr = calculate_psnr(temiz_resim, img)
        psnr_values.append(psnr)
    
    def calculate_psnr(original, denoised):
        mse = np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    psnr_values = []
    for img in test_images:
        psnr = calculate_psnr(temiz_resim, img)
        psnr_values.append(psnr)
    
    bars = plt.bar(range(len(method_names)), psnr_values, alpha=0.7,
                   color=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'])
    plt.title('Salt & Pepper PSNR Karşılaştırması')
    plt.xlabel('Yöntem')
    plt.ylabel('PSNR (dB)')
    plt.xticks(range(len(method_names)), method_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # En iyi sonucu vurgula
    best_idx = np.argmax(psnr_values)
    bars[best_idx].set_color('gold')
    
    # Kernel boyutu etkisi
    plt.subplot(3, 4, 12)
    kernel_sizes = [3, 5, 7, 9, 11]
    kernel_psnr = []
    kernel_time = []
    
    for k_size in kernel_sizes:
        result = cv2.medianBlur(salt_pepper_resim, k_size)
        psnr = calculate_psnr(temiz_resim, result)
        kernel_psnr.append(psnr)
        kernel_time.append(k_size ** 2)  # Yaklaşık hesaplama karmaşıklığı
    
    # İki eksenlı grafik
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(kernel_sizes, kernel_psnr, 'bo-', label='PSNR')
    line2 = ax2.plot(kernel_sizes, kernel_time, 'ro-', label='Hesaplama Süresi')
    
    ax1.set_xlabel('Kernel Boyutu')
    ax1.set_ylabel('PSNR (dB)', color='blue')
    ax2.set_ylabel('Relatif Süre', color='red')
    ax1.set_title('Kernel Boyutu vs Performans')
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı analiz
    print("\n🔍 Salt & Pepper Gürültü Analizi:")
    
    best_psnr = max(psnr_values)
    best_method = method_names[psnr_values.index(best_psnr)]
    print(f"En iyi sonuç: {best_method} (PSNR: {best_psnr:.1f} dB)")
    
    # Median filter optimum boyut
    optimal_kernel = kernel_sizes[np.argmax(kernel_psnr)]
    print(f"Optimal median kernel boyutu: {optimal_kernel}x{optimal_kernel}")
    
    print("\n📝 Salt & Pepper Temizleme İpuçları:")
    print("   • Median filter en etkili yöntemdir")
    print("   • Kernel boyutu 3-7 arası genellikle yeterli")
    print("   • Gaussian blur etkisiz (gürültüyü yayar)")
    print("   • Adaptive median ağır gürültü için idealdir")

def advanced_denoising_ornekleri(temiz_resim, karma_gurultulu):
    """İleri seviye gürültü azaltma teknikleri"""
    print("\n🚀 İleri Seviye Gürültü Azaltma")
    print("=" * 35)
    
    # Non-local means denoising (farklı parametreler)
    nlm_fast = cv2.fastNlMeansDenoisingColored(karma_gurultulu, None, 3, 3, 7, 21)
    nlm_quality = cv2.fastNlMeansDenoisingColored(karma_gurultulu, None, 10, 10, 7, 21)
    nlm_strong = cv2.fastNlMeansDenoisingColored(karma_gurultulu, None, 15, 15, 9, 25)
    
    # BM3D simulation (basit approximation)
    def simple_bm3d_approximation(image):
        # Multi-scale denoising
        scales = [1.0, 0.5, 0.25]
        results = []
        
        for scale in scales:
            if scale < 1.0:
                # Resize down
                h, w = image.shape[:2]
                small = cv2.resize(image, (int(w*scale), int(h*scale)))
                # Denoise
                denoised_small = cv2.bilateralFilter(small, 9, 75, 75)
                # Resize back
                denoised = cv2.resize(denoised_small, (w, h))
            else:
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            results.append(denoised)
        
        # Weighted combination
        final_result = (0.5 * results[0] + 0.3 * results[1] + 0.2 * results[2])
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    bm3d_approx = simple_bm3d_approximation(karma_gurultulu)
    
    # Wavelet denoising simulation
    def simple_wavelet_denoising(image, threshold=20):
        # DFT tabanlı basit wavelet simülasyonu
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        phase = np.angle(f_shift)
        
        # Soft thresholding
        magnitude_thresh = np.maximum(magnitude - threshold, 0.1 * magnitude)
        
        # Reconstruct
        f_thresh = magnitude_thresh * np.exp(1j * phase)
        f_ishift = np.fft.ifftshift(f_thresh)
        result = np.fft.ifft2(f_ishift)
        result = np.abs(result)
        
        # Normalize and convert back to color
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    wavelet_result = simple_wavelet_denoising(karma_gurultulu)
    
    # Anisotropic diffusion simulation
    def simple_anisotropic_diffusion(image, iterations=10, kappa=50, gamma=0.1):
        # Gradient based smoothing
        result = image.astype(np.float32)
        
        for _ in range(iterations):
            # Gradients
            grad_x = cv2.Sobel(result, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(result, cv2.CV_32F, 0, 1, ksize=3)
            
            # Magnitude
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Diffusion coefficient
            diffusion = np.exp(-(grad_mag / kappa)**2)
            
            # Update
            laplacian = cv2.Laplacian(result, cv2.CV_32F)
            result += gamma * diffusion * laplacian
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    anisotropic_result = simple_anisotropic_diffusion(karma_gurultulu)
    
    # Edge-preserving filter
    edge_preserving = cv2.edgePreservingFilter(karma_gurultulu, flags=1, sigma_s=50, sigma_r=0.4)
    
    # Detail-preserving filter
    detail_preserving = cv2.dtFilter(karma_gurultulu, karma_gurultulu, sigma_s=5, sigma_r=15)
    
    # Kombinasyon yöntemleri
    # 1. NLM + Bilateral
    combo1_step1 = cv2.fastNlMeansDenoisingColored(karma_gurultulu, None, 8, 8, 7, 21)
    combo1 = cv2.bilateralFilter(combo1_step1, 5, 50, 50)
    
    # 2. Multi-pass median + gaussian
    combo2_step1 = cv2.medianBlur(karma_gurultulu, 3)
    combo2 = cv2.GaussianBlur(combo2_step1, (3, 3), 0)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 20))
    
    results = [
        (temiz_resim, "Temiz Referans"),
        (karma_gurultulu, "Karma Gürültülü"),
        (nlm_fast, "NLM Hızlı"),
        (nlm_quality, "NLM Kaliteli"),
        (nlm_strong, "NLM Güçlü"),
        (bm3d_approx, "BM3D Yaklaşık"),
        (wavelet_result, "Wavelet Denoising"),
        (anisotropic_result, "Anisotropic Diffusion"),
        (edge_preserving, "Edge Preserving"),
        (detail_preserving, "Detail Preserving"),
        (combo1, "Combo: NLM + Bilateral"),
        (combo2, "Combo: Median + Gaussian")
    ]
    
    for i, (result_img, title) in enumerate(results):
        plt.subplot(4, 3, i+1)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Performans analizi
    print("\n🔍 İleri Seviye Gürültü Azaltma Analizi:")
    
    def calculate_psnr(original, denoised):
        mse = np.mean((original.astype(np.float32) - denoised.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    # PSNR hesapla
    method_names = ['Gürültülü', 'NLM Fast', 'NLM Quality', 'NLM Strong', 'BM3D Approx', 
                   'Wavelet', 'Anisotropic', 'Edge Preserving', 'Detail Preserving', 
                   'Combo1', 'Combo2']
    test_images = [karma_gurultulu, nlm_fast, nlm_quality, nlm_strong, bm3d_approx,
                   wavelet_result, anisotropic_result, edge_preserving, detail_preserving,
                   combo1, combo2]
    
    psnr_values = []
    for img in test_images:
        psnr = calculate_psnr(temiz_resim, img)
        psnr_values.append(psnr)
    
    # En iyi 3 yöntemi bul
    best_indices = np.argsort(psnr_values)[-3:]
    
    print("En iyi 3 yöntem:")
    for i, idx in enumerate(reversed(best_indices)):
        print(f"{i+1}. {method_names[idx]}: {psnr_values[idx]:.1f} dB")
    
    # Yaklaşık işlem süreleri (relatif)
    processing_times = [0, 8, 15, 20, 12, 6, 25, 5, 8, 25, 3]
    
    plt.figure(figsize=(15, 10))
    
    # PSNR karşılaştırması
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(method_names)), psnr_values, alpha=0.7)
    # En iyi 3'ü vurgula
    for idx in best_indices:
        bars[idx].set_color('gold')
    
    plt.title('PSNR Karşılaştırması (Altın = En İyi 3)')
    plt.xlabel('Yöntem')
    plt.ylabel('PSNR (dB)')
    plt.xticks(range(len(method_names)), method_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # İşlem süresi karşılaştırması
    plt.subplot(2, 2, 2)
    plt.bar(range(len(method_names)), processing_times, alpha=0.7, color='orange')
    plt.title('Yaklaşık İşlem Süreleri')
    plt.xlabel('Yöntem')
    plt.ylabel('Relatif Süre')
    plt.xticks(range(len(method_names)), method_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # PSNR vs Süre scatter plot
    plt.subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, len(method_names)))
    
    for i, (psnr, time, name, color) in enumerate(zip(psnr_values, processing_times, method_names, colors)):
        plt.scatter(time, psnr, c=[color], s=100, alpha=0.7, label=name)
    
    plt.xlabel('İşlem Süresi (Relatif)')
    plt.ylabel('PSNR (dB)')
    plt.title('Kalite vs Hız')
    plt.grid(True, alpha=0.3)
    
    # Pareto frontier
    pareto_indices = []
    for i in range(len(psnr_values)):
        is_pareto = True
        for j in range(len(psnr_values)):
            if i != j:
                if psnr_values[j] >= psnr_values[i] and processing_times[j] <= processing_times[i]:
                    if psnr_values[j] > psnr_values[i] or processing_times[j] < processing_times[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)
    
    # Pareto optimal noktaları vurgula
    for idx in pareto_indices:
        plt.scatter(processing_times[idx], psnr_values[idx], c='red', s=200, marker='*', edgecolors='black')
    
    # Öneriler tablosu
    plt.subplot(2, 2, 4)
    recommendation_text = "🎯 Kullanım Önerileri:\n\n"
    
    # Hız odaklı
    fast_methods = [(i, processing_times[i]) for i in range(len(processing_times))]
    fast_methods.sort(key=lambda x: x[1])
    fastest = fast_methods[1]  # İlk gürültülü olanı atla
    recommendation_text += f"⚡ Hız: {method_names[fastest[0]]}\n"
    
    # Kalite odaklı
    quality_method = method_names[np.argmax(psnr_values)]
    recommendation_text += f"🏆 Kalite: {quality_method}\n"
    
    # Dengeli
    efficiency_scores = [(psnr - min(psnr_values)) / (max(psnr_values) - min(psnr_values)) - 
                        (time - min(processing_times)) / (max(processing_times) - min(processing_times)) 
                        for psnr, time in zip(psnr_values, processing_times)]
    balanced_idx = np.argmax(efficiency_scores)
    recommendation_text += f"⚖️ Dengeli: {method_names[balanced_idx]}\n\n"
    
    recommendation_text += "💡 Genel Tavsiyeler:\n"
    recommendation_text += "• Hafif gürültü: Bilateral Filter\n"
    recommendation_text += "• Ağır gürültü: NLM Quality\n"
    recommendation_text += "• Hız gerekli: Median + Gaussian\n"
    recommendation_text += "• En iyi kalite: Kombinasyon yöntemleri"
    
    plt.text(0.05, 0.95, recommendation_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    plt.title('Akıllı Öneriler')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📝 İleri Seviye Gürültü Azaltma İpuçları:")
    print("   • Non-local means: En iyi genel performans")
    print("   • Kombinasyon yöntemleri: Farklı gürültü türleri için")
    print("   • Edge-preserving: Hızlı ve kenar koruyucu")
    print("   • Parametre ayarlama çok önemli")

def interaktif_gurultu_demo():
    """İnteraktif gürültü azaltma demosu"""
    print("\n🎮 İnteraktif Gürültü Azaltma Demosu")
    print("=" * 40)
    print("Trackbar'ları kullanarak gerçek zamanlı gürültü azaltma görün!")
    
    # Test resimleri yükle
    temiz_path, gaussian_path, salt_pepper_path, speckle_path, karma_path = ornek_resim_olustur()
    
    # Test resmi seç (karma gürültülü)
    resim = cv2.imread(karma_path)
    
    if resim is None:
        print("❌ Resim yüklenemedi!")
        return
    
    # Resmi küçült (performans için)
    resim = cv2.resize(resim, (400, 400))
    
    # Pencere oluştur
    window_name = 'Interactive Denoising Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Trackbar'lar oluştur
    cv2.createTrackbar('Method', window_name, 0, 5, lambda x: None)
    cv2.createTrackbar('Strength', window_name, 10, 50, lambda x: None)
    cv2.createTrackbar('Kernel Size', window_name, 5, 15, lambda x: None)
    cv2.createTrackbar('Preserve Edges', window_name, 1, 1, lambda x: None)
    
    method_names = ['Original', 'Gaussian Blur', 'Bilateral', 'Median', 'NLM', 'Edge Preserving']
    
    print("🎛️ Kontroller:")
    print("   • Method: 0-5 (Gaussian, Bilateral, vb.)")
    print("   • Strength: 1-50 (gürültü azaltma gücü)")
    print("   • Kernel Size: 5-15")
    print("   • Preserve Edges: Kenar koruma açık/kapalı")
    print("   • ESC tuşu ile çıkış")
    
    while True:
        # Trackbar değerlerini oku
        method = cv2.getTrackbarPos('Method', window_name)
        strength = cv2.getTrackbarPos('Strength', window_name)
        kernel_size = cv2.getTrackbarPos('Kernel Size', window_name)
        preserve_edges = cv2.getTrackbarPos('Preserve Edges', window_name)
        
        # Parametreleri kontrol et
        if strength < 1:
            strength = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        
        # İşlemi uygula
        try:
            if method == 0:  # Original
                result = resim.copy()
                info_text = 'Original Image'
                
            elif method == 1:  # Gaussian Blur
                sigma = strength / 10.0
                result = cv2.GaussianBlur(resim, (kernel_size, kernel_size), sigma)
                info_text = f'Gaussian: kernel={kernel_size}, σ={sigma:.1f}'
                
            elif method == 2:  # Bilateral
                sigma_color = strength * 2
                sigma_space = strength * 2
                result = cv2.bilateralFilter(resim, kernel_size, sigma_color, sigma_space)
                info_text = f'Bilateral: d={kernel_size}, σc={sigma_color}, σs={sigma_space}'
                
            elif method == 3:  # Median
                result = cv2.medianBlur(resim, kernel_size)
                info_text = f'Median: kernel={kernel_size}x{kernel_size}'
                
            elif method == 4:  # Non-local means
                h = strength / 2
                result = cv2.fastNlMeansDenoisingColored(resim, None, h, h, 7, 21)
                info_text = f'NLM: h={h:.1f}'
                
            elif method == 5:  # Edge Preserving
                sigma_s = strength
                sigma_r = strength / 100.0
                if preserve_edges:
                    result = cv2.edgePreservingFilter(resim, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
                    info_text = f'Edge Preserving: σs={sigma_s}, σr={sigma_r:.2f}'
                else:
                    result = cv2.GaussianBlur(resim, (kernel_size, kernel_size), strength/10.0)
                    info_text = f'Gaussian (Edge preserve OFF): σ={strength/10.0:.1f}'
            else:
                result = resim.copy()
                info_text = 'Unknown method'
                
        except Exception as e:
            result = resim.copy()
            info_text = f'Error: {str(e)[:30]}...'
        
        # Bilgi metnini ekle
        current_method = method_names[min(method, len(method_names)-1)]
        cv2.putText(result, f'Method: {current_method}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, info_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result, 'ESC = Exit', (10, result.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
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
    print("🧼 OpenCV Gürültü Azaltma Teknikleri")
    print("Bu program, çeşitli gürültü azaltma tekniklerini gösterir.\n")
    
    # Test resimleri oluştur
    temiz_path, gaussian_path, salt_pepper_path, speckle_path, karma_path = ornek_resim_olustur()
    
    # Resimleri yükle
    temiz_resim = cv2.imread(temiz_path)
    gaussian_resim = cv2.imread(gaussian_path)
    salt_pepper_resim = cv2.imread(salt_pepper_path)
    karma_resim = cv2.imread(karma_path)
    
    if any(img is None for img in [temiz_resim, gaussian_resim, salt_pepper_resim, karma_resim]):
        print("❌ Test resimleri oluşturulamadı!")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("🧼 Gürültü Azaltma Menüsü")
        print("=" * 50)
        print("1. Gaussian Gürültü Temizleme")
        print("2. Salt & Pepper Gürültü Temizleme")
        print("3. İleri Seviye Gürültü Azaltma")
        print("4. İnteraktif Gürültü Azaltma Demosu")
        print("0. Çıkış")
        
        try:
            secim = input("\nLütfen bir seçenek girin (0-4): ").strip()
            
            if secim == '0':
                print("👋 Görüşmek üzere!")
                break
            elif secim == '1':
                gaussian_gurultu_temizleme(temiz_resim, gaussian_resim)
            elif secim == '2':
                salt_pepper_temizleme(temiz_resim, salt_pepper_resim)
            elif secim == '3':
                advanced_denoising_ornekleri(temiz_resim, karma_resim)
            elif secim == '4':
                interaktif_gurultu_demo()
            else:
                print("❌ Geçersiz seçenek! Lütfen 0-4 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program kullanıcı tarafından sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()