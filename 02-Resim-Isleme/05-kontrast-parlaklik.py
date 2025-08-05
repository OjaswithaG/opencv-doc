"""
✨ OpenCV Kontrast ve Parlaklık İşlemleri
========================================

Bu dosyada kontrast ve parlaklık ayarlama tekniklerini öğreneceksiniz:
- Lineer transformasyonlar (alpha-beta düzeltme)
- Gamma düzeltme (power-law transformation)
- Logaritmik transformasyonlar
- Otomatik kontrast ayarlama
- Histogram germe (stretching)
- Tone mapping teknikleri

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

def ornek_resim_olustur():
    """Kontrast ve parlaklık testleri için örnek resimler oluştur"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 1. Normal resim
    normal_resim = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Düzgün gradient
    for i in range(300):
        for j in range(300):
            r = int(128 + 50 * np.sin(i/50) * np.cos(j/50))
            g = int(128 + 30 * np.cos(i/30))
            b = int(128 + 40 * np.sin(j/40))
            normal_resim[i, j] = [np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255)]
    
    # Şekiller ekle
    cv2.rectangle(normal_resim, (50, 50), (150, 150), (200, 100, 50), -1)
    cv2.circle(normal_resim, (200, 200), 50, (50, 150, 200), -1)
    cv2.ellipse(normal_resim, (100, 250), (40, 20), 45, 0, 360, (150, 200, 100), -1)
    
    normal_dosya = examples_dir / "normal_image.jpg"
    cv2.imwrite(str(normal_dosya), normal_resim)
    
    # 2. Düşük kontrastlı resim
    dusuk_kontrast = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Dar aralıkta gradient
    for i in range(300):
        for j in range(300):
            base_value = 128
            variation = int(20 * np.sin(i/80) * np.cos(j/80))
            value = base_value + variation
            dusuk_kontrast[i, j] = [value, value+5, value+10]
    
    # Düşük kontrastlı şekiller
    cv2.rectangle(dusuk_kontrast, (80, 80), (220, 220), (140, 145, 150), -1)
    cv2.circle(dusuk_kontrast, (150, 150), 40, (120, 125, 130), -1)
    
    dusuk_kontrast_dosya = examples_dir / "low_contrast_advanced.jpg"
    cv2.imwrite(str(dusuk_kontrast_dosya), dusuk_kontrast)
    
    # 3. Karanlık resim
    karanlik_resim = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Karanlık gradient
    for i in range(300):
        for j in range(300):
            r = int(30 + 25 * (i + j) / 600)
            g = int(25 + 20 * i / 300)
            b = int(35 + 15 * j / 300)
            karanlik_resim[i, j] = [b, g, r]
    
    # Karanlık detaylar
    cv2.rectangle(karanlik_resim, (100, 100), (200, 200), (50, 60, 70), -1)
    cv2.circle(karanlik_resim, (150, 80), 25, (40, 45, 55), -1)
    
    karanlik_dosya = examples_dir / "dark_advanced.jpg"
    cv2.imwrite(str(karanlik_dosya), karanlik_resim)
    
    # 4. Aşırı parlak resim
    parlak_resim = np.full((300, 300, 3), 220, dtype=np.uint8)
    
    # Parlak gradient
    for i in range(300):
        for j in range(300):
            r = int(220 + 20 * np.sin(i/40))
            g = int(225 + 15 * np.cos(j/30))
            b = int(215 + 25 * np.sin((i+j)/50))
            parlak_resim[i, j] = [np.clip(b, 180, 255), np.clip(g, 180, 255), np.clip(r, 180, 255)]
    
    # Aşırı parlak bölgeler
    cv2.rectangle(parlak_resim, (50, 50), (150, 150), (245, 250, 255), -1)
    cv2.circle(parlak_resim, (200, 200), 40, (250, 245, 240), -1)
    
    parlak_dosya = examples_dir / "overexposed.jpg"
    cv2.imwrite(str(parlak_dosya), parlak_resim)
    
    print(f"✅ Kontrast/Parlaklık test resimleri oluşturuldu:")
    print(f"   - Normal resim: {normal_dosya}")
    print(f"   - Düşük kontrast: {dusuk_kontrast_dosya}")
    print(f"   - Karanlık resim: {karanlik_dosya}")
    print(f"   - Aşırı parlak: {parlak_dosya}")
    
    return str(normal_dosya), str(dusuk_kontrast_dosya), str(karanlik_dosya), str(parlak_dosya)

def lineer_transformasyon_ornekleri(resim):
    """Lineer transformasyon (alpha-beta düzeltme) örnekleri"""
    print("\n📈 Lineer Transformasyon (Alpha-Beta) Örnekleri")
    print("=" * 50)
    
    # Farklı alpha (kontrast) ve beta (parlaklık) değerleri
    transformations = [
        (1.0, 0, "Orijinal"),
        (1.5, 0, "Kontrast +50%"),
        (0.5, 0, "Kontrast -50%"),
        (1.0, 50, "Parlaklık +50"),
        (1.0, -50, "Parlaklık -50"),
        (1.5, 30, "Kontrast +50%, Parlaklık +30"),
        (0.7, -20, "Kontrast -30%, Parlaklık -20"),
        (2.0, -100, "Yüksek Kontrast, Düşük Parlaklık")
    ]
    
    results = []
    for alpha, beta, desc in transformations:
        # Y = alpha * X + beta transformasyonu
        transformed = cv2.convertScaleAbs(resim, alpha=alpha, beta=beta)
        results.append((transformed, desc, alpha, beta))
    
    # Manuel transformasyon implementasyonu
    def manual_transform(img, alpha, beta):
        # Float'a çevir, transform uygula, clip yap
        result = img.astype(np.float32)
        result = alpha * result + beta
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
    
    manual_result = manual_transform(resim, 1.5, 30)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Ana transformasyonlar
    for i, (transformed, desc, alpha, beta) in enumerate(results):
        plt.subplot(3, 4, i+1)
        if len(transformed.shape) == 3:
            plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(transformed, cmap='gray')
        plt.title(f'{desc}\nα={alpha}, β={beta}')
        plt.axis('off')
    
    # Manuel vs OpenCV karşılaştırması
    plt.subplot(3, 4, 9)
    opencv_result = cv2.convertScaleAbs(resim, alpha=1.5, beta=30)
    if len(opencv_result.shape) == 3:
        plt.imshow(cv2.cvtColor(opencv_result, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(opencv_result, cmap='gray')
    plt.title('OpenCV convertScaleAbs')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    if len(manual_result.shape) == 3:
        plt.imshow(cv2.cvtColor(manual_result, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(manual_result, cmap='gray')
    plt.title('Manuel Transformasyon')
    plt.axis('off')
    
    # Histogram analizi
    plt.subplot(3, 4, 11)
    if len(resim.shape) == 3:
        gray_orig = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        gray_trans = cv2.cvtColor(results[1][0], cv2.COLOR_BGR2GRAY)  # Kontrast +50%
    else:
        gray_orig = resim
        gray_trans = results[1][0]
    
    hist_orig = cv2.calcHist([gray_orig], [0], None, [256], [0, 256])
    hist_trans = cv2.calcHist([gray_trans], [0], None, [256], [0, 256])
    
    plt.plot(hist_orig.flatten(), color='blue', alpha=0.7, label='Orijinal')
    plt.plot(hist_trans.flatten(), color='red', alpha=0.7, label='Kontrast +50%')
    plt.title('Histogram Karşılaştırması')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Transformasyon fonksiyonu görselleştirmesi
    plt.subplot(3, 4, 12)
    x = np.arange(0, 256)
    
    # Farklı alpha-beta kombinasyonları
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [0, 30, -30]
    colors = ['red', 'blue', 'green', 'orange']
    
    for alpha, color in zip(alphas, colors):
        y = alpha * x  # beta=0 için
        y = np.clip(y, 0, 255)
        plt.plot(y, color=color, alpha=0.7, label=f'α={alpha}')
    
    plt.plot(x, color='gray', linestyle='--', alpha=0.5, label='y=x')
    plt.title('Transformasyon Fonksiyonları')
    plt.xlabel('Giriş Değeri')
    plt.ylabel('Çıkış Değeri')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı analiz
    print("\n🔍 Lineer Transformasyon Analizi:")
    
    # İstatistiksel karşılaştırma
    if len(resim.shape) == 3:
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    else:
        gray = resim
    
    orig_mean = np.mean(gray)
    orig_std = np.std(gray)
    
    print(f"Orijinal - Ortalama: {orig_mean:.1f}, Std: {orig_std:.1f}")
    
    for alpha, beta, desc in transformations[1:4]:  # İlk birkaç örnek
        transformed_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        trans_mean = np.mean(transformed_gray)
        trans_std = np.std(transformed_gray)
        
        print(f"{desc} - Ortalama: {trans_mean:.1f}, Std: {trans_std:.1f}")
        print(f"  Teorik - Ortalama: {alpha * orig_mean + beta:.1f}, Std: {alpha * orig_std:.1f}")
    
    print("\n📝 Lineer Transformasyon İpuçları:")
    print("   • α > 1: Kontrast artırma")
    print("   • α < 1: Kontrast azaltma")
    print("   • β > 0: Parlaklık artırma")
    print("   • β < 0: Parlaklık azaltma")
    print("   • Y = α * X + β formülü")

def gamma_duzeltme_ornekleri(resim):
    """Gamma düzeltme örnekleri"""
    print("\n🌟 Gamma Düzeltme Örnekleri")
    print("=" * 30)
    
    # Farklı gamma değerleri
    gamma_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    # Gamma düzeltme lookup table'ı oluştur
    def build_gamma_lut(gamma):
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                       for i in np.arange(0, 256)]).astype("uint8")
        return lut
    
    # Gamma düzeltme uygula
    def apply_gamma_correction(image, gamma):
        lut = build_gamma_lut(gamma)
        return cv2.LUT(image, lut)
    
    # Sonuçları hesapla
    gamma_results = []
    for gamma in gamma_values:
        corrected = apply_gamma_correction(resim, gamma)
        gamma_results.append((corrected, gamma))
    
    # Adaptif gamma düzeltme
    def adaptive_gamma_correction(image):
        # Ortalama parlaklığa göre gamma hesapla
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean_brightness = np.mean(gray) / 255.0
        
        # Karanlık resim için gamma < 1, parlak resim için gamma > 1
        if mean_brightness < 0.3:
            gamma = 0.7  # Karanlık resimleri aydınlat
        elif mean_brightness > 0.7:
            gamma = 1.3  # Parlak resimleri koyulaştır
        else:
            gamma = 1.0  # Normal
        
        return apply_gamma_correction(image, gamma), gamma
    
    adaptive_result, adaptive_gamma = adaptive_gamma_correction(resim)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Farklı gamma değerleri
    for i, (corrected, gamma) in enumerate(gamma_results):
        plt.subplot(3, 4, i+1)
        if len(corrected.shape) == 3:
            plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(corrected, cmap='gray')
        
        # Gamma etkisini belirt
        if gamma < 1.0:
            effect = "Aydınlatma"
        elif gamma > 1.0:
            effect = "Koyulaştırma"
        else:
            effect = "Orijinal"
        
        plt.title(f'γ = {gamma}\n{effect}')
        plt.axis('off')
    
    # Adaptif gamma
    plt.subplot(3, 4, 9)
    if len(adaptive_result.shape) == 3:
        plt.imshow(cv2.cvtColor(adaptive_result, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(adaptive_result, cmap='gray')
    plt.title(f'Adaptif Gamma\nγ = {adaptive_gamma:.1f}')
    plt.axis('off')
    
    # Gamma fonksiyonları
    plt.subplot(3, 4, 10)
    x = np.linspace(0, 1, 256)
    
    for gamma in [0.5, 1.0, 1.5, 2.0, 2.5]:
        y = x ** (1.0 / gamma)
        plt.plot(y, label=f'γ = {gamma}')
    
    plt.plot(x, color='gray', linestyle='--', alpha=0.5, label='y = x')
    plt.title('Gamma Transformasyon Fonksiyonları')
    plt.xlabel('Normalize Giriş')
    plt.ylabel('Normalize Çıkış')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram karşılaştırması
    plt.subplot(3, 4, 11)
    if len(resim.shape) == 3:
        gray_orig = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        gray_gamma_low = cv2.cvtColor(gamma_results[0][0], cv2.COLOR_BGR2GRAY)  # γ=0.5
        gray_gamma_high = cv2.cvtColor(gamma_results[5][0], cv2.COLOR_BGR2GRAY)  # γ=2.0
    else:
        gray_orig = resim
        gray_gamma_low = gamma_results[0][0]
        gray_gamma_high = gamma_results[5][0]
    
    hist_orig = cv2.calcHist([gray_orig], [0], None, [256], [0, 256])
    hist_low = cv2.calcHist([gray_gamma_low], [0], None, [256], [0, 256])
    hist_high = cv2.calcHist([gray_gamma_high], [0], None, [256], [0, 256])
    
    plt.plot(hist_orig.flatten(), color='blue', alpha=0.7, label='Orijinal')
    plt.plot(hist_low.flatten(), color='green', alpha=0.7, label='γ = 0.5')
    plt.plot(hist_high.flatten(), color='red', alpha=0.7, label='γ = 2.0')
    plt.title('Histogram Karşılaştırması')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # LUT görselleştirmesi
    plt.subplot(3, 4, 12)
    # Farklı gamma'lar için LUT
    for gamma in [0.5, 1.0, 2.0]:
        lut = build_gamma_lut(gamma)
        plt.plot(lut, label=f'γ = {gamma}', linewidth=2)
    
    plt.plot(np.arange(256), color='gray', linestyle='--', alpha=0.5, label='Linear')
    plt.title('Gamma LUT Tabloları')
    plt.xlabel('Giriş Değeri')
    plt.ylabel('Çıkış Değeri')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Gamma efektleri analizi
    print("\n🔍 Gamma Düzeltme Analizi:")
    
    # Her gamma değeri için istatistikler
    if len(resim.shape) == 3:
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    else:
        gray = resim
    
    orig_mean = np.mean(gray)
    print(f"Orijinal resim ortalama parlaklığı: {orig_mean:.1f}")
    
    for gamma in [0.5, 1.0, 1.5, 2.0]:
        corrected = apply_gamma_correction(gray, gamma)
        corrected_mean = np.mean(corrected)
        
        print(f"γ = {gamma}: Ortalama = {corrected_mean:.1f}, Değişim = {corrected_mean - orig_mean:+.1f}")
    
    # Uygun gamma önerileri
    print(f"\nAdaptif gamma önerisi: γ = {adaptive_gamma:.1f}")
    
    if orig_mean < 80:
        print("Önerilern: Karanlık resim - γ = 0.5-0.8 deneyin")
    elif orig_mean > 170:
        print("Öneri: Parlak resim - γ = 1.2-2.0 deneyin")
    else:
        print("Öneri: Normal resim - γ = 0.8-1.2 yeterli")
    
    print("\n📝 Gamma Düzeltme İpuçları:")
    print("   • γ < 1: Karanlık bölgeleri aydınlatır")
    print("   • γ > 1: Parlak bölgeleri koyulaştırır")
    print("   • Y = X^(1/γ) formülü")
    print("   • Monitor kalibrasyonu için yaygın kullanım")

def logaritmik_transformasyon_ornekleri(resim):
    """Logaritmik transformasyon örnekleri"""
    print("\n📊 Logaritmik Transformasyon Örnekleri")
    print("=" * 40)
    
    # Logaritmik transformasyon fonksiyonları
    def log_transform(image, c=1):
        # Y = c * log(1 + X)
        image_float = image.astype(np.float32)
        log_image = c * np.log1p(image_float)  # log1p = log(1 + x)
        log_image = np.clip(log_image, 0, 255)
        return log_image.astype(np.uint8)
    
    def inverse_log_transform(image, c=1):
        # Y = exp(X/c) - 1
        image_float = image.astype(np.float32)
        exp_image = np.exp(image_float / c) - 1
        exp_image = np.clip(exp_image, 0, 255)
        return exp_image.astype(np.uint8)
    
    # Farklı c parametreleri ile logaritmik transformasyon
    c_values = [10, 20, 40, 60, 80, 100]
    log_results = []
    
    for c in c_values:
        log_result = log_transform(resim, c)
        log_results.append((log_result, c))
    
    # Ters logaritmik transformasyonlar
    inv_log_results = []
    for c in c_values:
        inv_log_result = inverse_log_transform(resim, c)
        inv_log_results.append((inv_log_result, c))
    
    # Adaptif logaritmik transformasyon
    def adaptive_log_transform(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Dinamik aralığı kontrol et
        min_val = np.min(gray)
        max_val = np.max(gray)
        dynamic_range = max_val - min_val
        
        # C parametresini dinamik aralığa göre ayarla
        if dynamic_range < 100:
            c = 20  # Düşük dinamik aralık
        elif dynamic_range > 200:
            c = 80  # Yüksek dinamik aralık
        else:
            c = 40  # Normal
        
        return log_transform(image, c), c
    
    adaptive_log, adaptive_c = adaptive_log_transform(resim)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Logaritmik transformasyonlar
    for i, (log_result, c) in enumerate(log_results):
        if i >= 6:
            break
        plt.subplot(4, 4, i+1)
        if len(log_result.shape) == 3:
            plt.imshow(cv2.cvtColor(log_result, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(log_result, cmap='gray')
        plt.title(f'Log Transform\nc = {c}')
        plt.axis('off')
    
    # Orijinal resim
    plt.subplot(4, 4, 7)
    if len(resim.shape) == 3:
        plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(resim, cmap='gray')
    plt.title('Orijinal')
    plt.axis('off')
    
    # Adaptif log
    plt.subplot(4, 4, 8)
    if len(adaptive_log.shape) == 3:
        plt.imshow(cv2.cvtColor(adaptive_log, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(adaptive_log, cmap='gray')
    plt.title(f'Adaptive Log\nc = {adaptive_c}')
    plt.axis('off')
    
    # Ters logaritmik transformasyonlar
    for i, (inv_log_result, c) in enumerate(inv_log_results[:4]):
        plt.subplot(4, 4, 9+i)
        if len(inv_log_result.shape) == 3:
            plt.imshow(cv2.cvtColor(inv_log_result, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(inv_log_result, cmap='gray')
        plt.title(f'Inverse Log\nc = {c}')
        plt.axis('off')
    
    # Transformasyon fonksiyonları
    plt.subplot(4, 4, 13)
    x = np.arange(0, 256)
    
    for c in [20, 40, 80]:
        y_log = c * np.log1p(x)
        y_log = np.clip(y_log, 0, 255)
        plt.plot(y_log, label=f'Log c={c}')
    
    plt.plot(x, color='gray', linestyle='--', alpha=0.5, label='y=x')
    plt.title('Logaritmik Transformasyon Fonksiyonları')
    plt.xlabel('Giriş Değeri')
    plt.ylabel('Çıkış Değeri')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ters logaritmik fonksiyonlar
    plt.subplot(4, 4, 14)
    for c in [20, 40, 80]:
        y_inv_log = np.exp(x / c) - 1
        y_inv_log = np.clip(y_inv_log, 0, 255)
        plt.plot(y_inv_log, label=f'Inverse Log c={c}')
    
    plt.plot(x, color='gray', linestyle='--', alpha=0.5, label='y=x')
    plt.title('Ters Logaritmik Transformasyon')
    plt.xlabel('Giriş Değeri')
    plt.ylabel('Çıkış Değeri')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram karşılaştırması
    plt.subplot(4, 4, 15)
    if len(resim.shape) == 3:
        gray_orig = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        gray_log = cv2.cvtColor(log_results[2][0], cv2.COLOR_BGR2GRAY)  # c=40
        gray_inv_log = cv2.cvtColor(inv_log_results[2][0], cv2.COLOR_BGR2GRAY)  # c=40
    else:
        gray_orig = resim
        gray_log = log_results[2][0]
        gray_inv_log = inv_log_results[2][0]
    
    hist_orig = cv2.calcHist([gray_orig], [0], None, [256], [0, 256])
    hist_log = cv2.calcHist([gray_log], [0], None, [256], [0, 256])
    hist_inv_log = cv2.calcHist([gray_inv_log], [0], None, [256], [0, 256])
    
    plt.plot(hist_orig.flatten(), color='blue', alpha=0.7, label='Orijinal')
    plt.plot(hist_log.flatten(), color='green', alpha=0.7, label='Log')
    plt.plot(hist_inv_log.flatten(), color='red', alpha=0.7, label='Inverse Log')
    plt.title('Histogram Karşılaştırması')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dinamik aralık analizi
    plt.subplot(4, 4, 16)
    
    # Her transformasyon için dinamik aralık
    methods = ['Orijinal', 'Log (c=40)', 'Inv Log (c=40)']
    images_to_analyze = [gray_orig, gray_log, gray_inv_log]
    
    dynamic_ranges = []
    contrasts = []
    
    for img in images_to_analyze:
        dr = np.max(img) - np.min(img)
        contrast = np.std(img.astype(np.float32))
        dynamic_ranges.append(dr)
        contrasts.append(contrast)
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x_pos - width/2, dynamic_ranges, width, label='Dinamik Aralık', alpha=0.7)
    plt.bar(x_pos + width/2, contrasts, width, label='Kontrast (Std)', alpha=0.7)
    
    plt.title('Dinamik Aralık ve Kontrast')
    plt.xlabel('Transformasyon')
    plt.xticks(x_pos, methods, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı analiz
    print("\n🔍 Logaritmik Transformasyon Analizi:")
    
    # Her transformasyon türü için istatistikler
    if len(resim.shape) == 3:
        gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    else:
        gray = resim
    
    orig_stats = {
        'min': np.min(gray),
        'max': np.max(gray),
        'mean': np.mean(gray),
        'std': np.std(gray)
    }
    
    print(f"Orijinal: Min={orig_stats['min']}, Max={orig_stats['max']}, "
          f"Mean={orig_stats['mean']:.1f}, Std={orig_stats['std']:.1f}")
    
    # Log transform analizi
    log_test = log_transform(gray, 40)
    log_stats = {
        'min': np.min(log_test),
        'max': np.max(log_test),
        'mean': np.mean(log_test),
        'std': np.std(log_test)
    }
    
    print(f"Log (c=40): Min={log_stats['min']}, Max={log_stats['max']}, "
          f"Mean={log_stats['mean']:.1f}, Std={log_stats['std']:.1f}")
    
    # Dinamik aralık iyileştirmesi
    orig_dr = orig_stats['max'] - orig_stats['min']
    log_dr = log_stats['max'] - log_stats['min']
    dr_improvement = (log_dr - orig_dr) / orig_dr * 100
    
    print(f"Dinamik aralık değişimi: {dr_improvement:+.1f}%")
    
    # Kullanım önerileri
    print(f"\nAdaptif c parametresi: {adaptive_c}")
    
    if orig_dr < 100:
        print("Öneri: Düşük dinamik aralık - Log transform ile genişletin")
    elif np.mean(gray) < 80:  
        print("Öneri: Karanlık resim - Log transform karanlık detayları çıkarır")
    else:
        print("Öneri: Normal resim - Inverse log ile kontrast artırın")
    
    print("\n📝 Logaritmik Transformasyon İpuçları:")
    print("   • Log: Karanlık detayları vurgular, dinamik aralığı genişletir")
    print("   • Inverse Log: Parlak detayları vurgular, kontrast artırır")
    print("   • Y = c * log(1 + X) formülü")
    print("   • Yüksek dinamik aralıklı resimler için ideal")

def otomatik_kontrast_ornekleri(resim):
    """Otomatik kontrast ayarlama örnekleri"""
    print("\n🤖 Otomatik Kontrast Ayarlama Örnekleri")
    print("=" * 40)
    
    # 1. Histogram Germe (Stretching)
    def histogram_stretching(image, low_percentile=2, high_percentile=98):
        if len(image.shape) == 3:
            # Her kanal için ayrı ayrı
            stretched = np.zeros_like(image)
            for i in range(3):
                channel = image[:,:,i]
                p_low = np.percentile(channel, low_percentile)
                p_high = np.percentile(channel, high_percentile)
                
                # Stretch formula: Y = 255 * (X - min) / (max - min)
                stretched_channel = 255 * (channel - p_low) / (p_high - p_low)
                stretched[:,:,i] = np.clip(stretched_channel, 0, 255)
            return stretched.astype(np.uint8)
        else:
            p_low = np.percentile(image, low_percentile)
            p_high = np.percentile(image, high_percentile)
            stretched = 255 * (image - p_low) / (p_high - p_low)
            return np.clip(stretched, 0, 255).astype(np.uint8)
    
    # 2. Otomatik Levels Ayarlama
    def auto_levels(image):
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                channel = image[:,:,i]
                min_val = np.min(channel)
                max_val = np.max(channel)
                
                # Full range'e genişlet
                if max_val > min_val:
                    result[:,:,i] = 255 * (channel - min_val) / (max_val - min_val)
                else:
                    result[:,:,i] = channel
            return result.astype(np.uint8)
        else:
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                result = 255 * (image - min_val) / (max_val - min_val)
                return result.astype(np.uint8)
            else:
                return image
    
    # 3. Adaptif Kontrast (CLAHE alternatifi)
    def simple_adaptive_contrast(image, clip_limit=3.0):
        if len(image.shape) == 3:
            # LAB color space'e çevir
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
            return clahe.apply(image)
    
    # 4. Otomatik Gamma Düzeltme
    def auto_gamma_correction(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Ortalama parlaklığa göre gamma hesapla
        mean_brightness = np.mean(gray) / 255.0
        
        # Sigmoid benzeri gamma hesaplama
        if mean_brightness < 0.5:
            gamma = 0.7 + (0.5 - mean_brightness) * 0.6  # 0.4 - 1.0 arası
        else:
            gamma = 1.0 + (mean_brightness - 0.5) * 1.0  # 1.0 - 1.5 arası
        
        # Gamma LUT oluştur
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                       for i in np.arange(0, 256)]).astype("uint8")
        
        if len(image.shape) == 3:
            return cv2.LUT(image, lut), gamma
        else:
            return cv2.LUT(image, lut), gamma
    
    # 5. Otomatik Renk Dengeleme
    def auto_color_balance(image):
        if len(image.shape) != 3:
            return image
        
        # Gray World algoritması
        result = image.astype(np.float32)
        
        # Her kanal için ortalama hesapla
        avg_b = np.mean(result[:,:,0])
        avg_g = np.mean(result[:,:,1])
        avg_r = np.mean(result[:,:,2])
        
        # Global ortalama
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        # Her kanalı düzelt
        result[:,:,0] = result[:,:,0] * (avg_gray / avg_b)
        result[:,:,1] = result[:,:,1] * (avg_gray / avg_g)
        result[:,:,2] = result[:,:,2] * (avg_gray / avg_r)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    # Tüm yöntemleri uygula
    hist_stretched = histogram_stretching(resim)
    auto_leveled = auto_levels(resim)
    adaptive_contrast = simple_adaptive_contrast(resim)
    auto_gamma_result, gamma_used = auto_gamma_correction(resim)
    color_balanced = auto_color_balance(resim)
    
    # Kombinasyon yöntemleri
    combo1 = histogram_stretching(auto_gamma_result[0] if isinstance(auto_gamma_result, tuple) else auto_gamma_result)
    combo2 = simple_adaptive_contrast(hist_stretched)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    # Ana sonuçlar
    results = [
        (resim, "Orijinal"),
        (hist_stretched, "Histogram Stretching"),
        (auto_leveled, "Auto Levels"),
        (adaptive_contrast, "Adaptive Contrast"),
        (auto_gamma_result[0] if isinstance(auto_gamma_result, tuple) else auto_gamma_result, f"Auto Gamma (γ={gamma_used:.2f})"),
        (color_balanced, "Color Balance"),
        (combo1, "Combo: Gamma + Stretching"),
        (combo2, "Combo: Stretching + Adaptive")
    ]
    
    for i, (result_img, title) in enumerate(results):
        plt.subplot(3, 4, i+1)
        if len(result_img.shape) == 3:
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(result_img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    # Histogram karşılaştırmaları
    plt.subplot(3, 4, 9)
    if len(resim.shape) == 3:
        gray_orig = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        gray_stretched = cv2.cvtColor(hist_stretched, cv2.COLOR_BGR2GRAY)
        gray_auto_level = cv2.cvtColor(auto_leveled, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = resim
        gray_stretched = hist_stretched
        gray_auto_level = auto_leveled
    
    hist_orig = cv2.calcHist([gray_orig], [0], None, [256], [0, 256])
    hist_str = cv2.calcHist([gray_stretched], [0], None, [256], [0, 256])
    hist_auto = cv2.calcHist([gray_auto_level], [0], None, [256], [0, 256])
    
    plt.plot(hist_orig.flatten(), color='blue', alpha=0.7, label='Orijinal')
    plt.plot(hist_str.flatten(), color='green', alpha=0.7, label='Stretched')
    plt.plot(hist_auto.flatten(), color='red', alpha=0.7, label='Auto Levels')
    plt.title('Histogram Karşılaştırması')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Kalite metrikleri
    plt.subplot(3, 4, 10)
    methods = ['Orijinal', 'Hist Stretch', 'Auto Levels', 'Adaptive', 'Auto Gamma']
    images_to_analyze = [gray_orig, gray_stretched, gray_auto_level, 
                        cv2.cvtColor(adaptive_contrast, cv2.COLOR_BGR2GRAY) if len(adaptive_contrast.shape) == 3 else adaptive_contrast,
                        cv2.cvtColor(auto_gamma_result[0], cv2.COLOR_BGR2GRAY) if len(auto_gamma_result[0].shape) == 3 else auto_gamma_result[0]]
    
    contrasts = []
    dynamic_ranges = []
    
    for img in images_to_analyze:
        contrast = np.std(img.astype(np.float32))
        dr = np.max(img) - np.min(img)
        contrasts.append(contrast)
        dynamic_ranges.append(dr)
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x_pos - width/2, contrasts, width, label='Kontrast', alpha=0.7)
    plt.bar(x_pos + width/2, [dr/2.55 for dr in dynamic_ranges], width, label='Dinamik Aralık/2.55', alpha=0.7)
    
    plt.title('Kalite Metrikleri')
    plt.xlabel('Yöntem')
    plt.xticks(x_pos, [m[:8] for m in methods], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Renk dengesi analizi (sadece renkli resimler için)
    plt.subplot(3, 4, 11)
    if len(resim.shape) == 3:
        # Her kanal için ortalama değerler
        orig_means = [np.mean(resim[:,:,0]), np.mean(resim[:,:,1]), np.mean(resim[:,:,2])]
        balanced_means = [np.mean(color_balanced[:,:,0]), np.mean(color_balanced[:,:,1]), np.mean(color_balanced[:,:,2])]
        
        channels = ['Blue', 'Green', 'Red']
        x_pos = np.arange(len(channels))
        width = 0.35
        
        plt.bar(x_pos - width/2, orig_means, width, label='Orijinal', alpha=0.7)
        plt.bar(x_pos + width/2, balanced_means, width, label='Dengelenmiş', alpha=0.7)
        
        plt.title('Renk Dengesi')
        plt.xlabel('Kanal')
        plt.xticks(x_pos, channels)
        plt.ylabel('Ortalama Değer')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Renk dengesi sadece\nrenkli resimler için', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Renk Dengesi')
        plt.axis('off')
    
    # Parametre öneriler tablosu
    plt.subplot(3, 4, 12)
    suggestions_text = "Otomatik Ayarlama Önerileri:\n\n"
    
    # Orijinal resim analizi
    if len(resim.shape) == 3:
        gray_analysis = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    else:
        gray_analysis = resim
    
    mean_bright = np.mean(gray_analysis)
    std_contrast = np.std(gray_analysis)
    min_val = np.min(gray_analysis)
    max_val = np.max(gray_analysis)
    dynamic_range = max_val - min_val
    
    suggestions_text += f"Mevcut Durum:\n"
    suggestions_text += f"• Parlaklık: {mean_bright:.0f}\n"
    suggestions_text += f"• Kontrast: {std_contrast:.0f}\n"
    suggestions_text += f"• Dinamik Aralık: {dynamic_range}\n\n"
    
    suggestions_text += "Öneriler:\n"
    
    if dynamic_range < 150:
        suggestions_text += "✓ Histogram Stretching\n"
    if std_contrast < 30:
        suggestions_text += "✓ Adaptive Contrast\n"
    if mean_bright < 80 or mean_bright > 170:
        suggestions_text += "✓ Auto Gamma\n"
    if len(resim.shape) == 3:
        suggestions_text += "✓ Color Balance\n"
    
    plt.text(0.05, 0.95, suggestions_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9)
    plt.title('Akıllı Öneriler')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Detaylı performans analizi
    print("\n🔍 Otomatik Kontrast Performans Analizi:")
    
    original_stats = {
        'mean': np.mean(gray_analysis),
        'std': np.std(gray_analysis),
        'min': np.min(gray_analysis),
        'max': np.max(gray_analysis)
    }
    
    print(f"Orijinal: Ortalama={original_stats['mean']:.1f}, Kontrast={original_stats['std']:.1f}, "
          f"Aralık=[{original_stats['min']}-{original_stats['max']}]")
    
    # Her yöntem için sonuçlar
    test_methods = [
        ("Histogram Stretching", gray_stretched),
        ("Auto Levels", gray_auto_level),
        ("Auto Gamma", cv2.cvtColor(auto_gamma_result[0], cv2.COLOR_BGR2GRAY) if len(auto_gamma_result[0].shape) == 3 else auto_gamma_result[0])
    ]
    
    for method_name, result_img in test_methods:
        stats = {
            'mean': np.mean(result_img),
            'std': np.std(result_img),
            'min': np.min(result_img),
            'max': np.max(result_img)
        }
        
        mean_change = stats['mean'] - original_stats['mean']
        contrast_change = stats['std'] - original_stats['std']
        
        print(f"{method_name}: Ortalama={stats['mean']:.1f} ({mean_change:+.1f}), "
              f"Kontrast={stats['std']:.1f} ({contrast_change:+.1f})")
    
    print(f"\nOtomatik gamma değeri: {gamma_used:.2f}")
    
    print("\n📝 Otomatik Kontrast İpuçları:")
    print("   • Histogram stretching: Dinamik aralığı tam kullanır")
    print("   • Auto levels: Min-max değerleri 0-255'e çeker")
    print("   • Adaptive contrast: Yerel detayları korur")
    print("   • Auto gamma: Ortalama parlaklığa göre ayarlar")
    print("   • Kombinasyonlar genellikle daha iyi sonuç verir")

def interaktif_kontrast_demo():
    """İnteraktif kontrast ve parlaklık demosu"""
    print("\n🎮 İnteraktif Kontrast ve Parlaklık Demosu")
    print("=" * 45)
    print("Trackbar'ları kullanarak gerçek zamanlı kontrast/parlaklık ayarları görün!")
    
    # Test resmi yükle
    normal_path, _, _, _ = ornek_resim_olustur()
    resim = cv2.imread(normal_path)
    
    if resim is None:
        print("❌ Resim yüklenemedi!")
        return
    
    # Pencere oluştur
    window_name = 'Interactive Contrast/Brightness Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Trackbar'lar oluştur
    cv2.createTrackbar('Method', window_name, 0, 4, lambda x: None)
    cv2.createTrackbar('Alpha x10', window_name, 10, 30, lambda x: None)
    cv2.createTrackbar('Beta', window_name, 50, 100, lambda x: None)
    cv2.createTrackbar('Gamma x10', window_name, 10, 30, lambda x: None)
    cv2.createTrackbar('Auto Mode', window_name, 0, 1, lambda x: None)
    
    method_names = ['Linear (Alpha-Beta)', 'Gamma', 'Log', 'Histogram Stretch', 'CLAHE']
    
    print("🎛️ Kontroller:")
    print("   • Method: 0-4 (Linear, Gamma, Log, Stretch, CLAHE)")
    print("   • Alpha: 0.1-3.0 (kontrast)")
    print("   • Beta: -50 ile +50 (parlaklık)")
    print("   • Gamma: 0.1-3.0")
    print("   • Auto Mode: Otomatik ayarlama")
    print("   • ESC tuşu ile çıkış")
    
    while True:
        # Trackbar değerlerini oku
        method = cv2.getTrackbarPos('Method', window_name)
        alpha = cv2.getTrackbarPos('Alpha x10', window_name) / 10.0
        beta = cv2.getTrackbarPos('Beta', window_name) - 50
        gamma = cv2.getTrackbarPos('Gamma x10', window_name) / 10.0
        auto_mode = cv2.getTrackbarPos('Auto Mode', window_name)
        
        # Parametreleri sınırla
        if alpha < 0.1:
            alpha = 0.1
        if gamma < 0.1:
            gamma = 0.1
        
        # İşlemi uygula
        try:
            if auto_mode:
                # Otomatik mod
                if method == 0:  # Linear
                    # Otomatik alpha-beta hesapla
                    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
                    std_val = np.std(gray)
                    mean_val = np.mean(gray)
                    
                    auto_alpha = 1.0 + (50 - std_val) / 50  # Düşük kontrast = yüksek alpha
                    auto_beta = (128 - mean_val) * 0.5  # Ortalamayı 128'e yaklaştır
                    
                    result = cv2.convertScaleAbs(resim, alpha=auto_alpha, beta=auto_beta)
                    info_text = f'Auto Linear: α={auto_alpha:.2f}, β={auto_beta:.1f}'
                    
                elif method == 1:  # Gamma
                    gray = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
                    mean_bright = np.mean(gray) / 255.0
                    
                    if mean_bright < 0.4:
                        auto_gamma = 0.7
                    elif mean_bright > 0.6:
                        auto_gamma = 1.3
                    else:
                        auto_gamma = 1.0
                    
                    lut = np.array([((i / 255.0) ** (1.0 / auto_gamma)) * 255 
                                   for i in np.arange(0, 256)]).astype("uint8")
                    result = cv2.LUT(resim, lut)
                    info_text = f'Auto Gamma: γ={auto_gamma:.2f}'
                    
                else:  # Diğer yöntemler için basit otomatik
                    result = resim.copy()
                    info_text = 'Auto mode not available for this method'
            else:
                # Manuel mod
                if method == 0:  # Linear Alpha-Beta
                    result = cv2.convertScaleAbs(resim, alpha=alpha, beta=beta)
                    info_text = f'Linear: α={alpha:.2f}, β={beta}'
                    
                elif method == 1:  # Gamma
                    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 
                                   for i in np.arange(0, 256)]).astype("uint8")
                    result = cv2.LUT(resim, lut)
                    info_text = f'Gamma: γ={gamma:.2f}'
                    
                elif method == 2:  # Log
                    c = alpha * 20  # Alpha'yı c parametresi olarak kullan
                    result_float = c * np.log1p(resim.astype(np.float32))
                    result = np.clip(result_float, 0, 255).astype(np.uint8)
                    info_text = f'Log: c={c:.1f}'
                    
                elif method == 3:  # Histogram Stretch
                    # Alpha'yı percentile olarak kullan
                    low_p = max(1, alpha * 2)
                    high_p = min(99, 100 - alpha * 2)
                    
                    stretched = np.zeros_like(resim)
                    for i in range(3):
                        channel = resim[:,:,i]
                        p_low = np.percentile(channel, low_p)
                        p_high = np.percentile(channel, high_p)
                        
                        if p_high > p_low:
                            stretched_channel = 255 * (channel - p_low) / (p_high - p_low)
                            stretched[:,:,i] = np.clip(stretched_channel, 0, 255)
                        else:
                            stretched[:,:,i] = channel
                    
                    result = stretched.astype(np.uint8)
                    info_text = f'Stretch: {low_p:.1f}%-{high_p:.1f}%'
                    
                elif method == 4:  # CLAHE
                    lab = cv2.cvtColor(resim, cv2.COLOR_BGR2LAB)
                    clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    info_text = f'CLAHE: limit={alpha:.2f}'
                    
                else:
                    result = resim.copy()
                    info_text = 'Unknown method'
                    
        except Exception as e:
            result = resim.copy()
            info_text = f'Error: {str(e)[:30]}...'
        
        # Bilgi metnini ekle
        current_method = method_names[min(method, len(method_names)-1)]
        mode_text = 'AUTO' if auto_mode else 'MANUAL'
        
        cv2.putText(result, f'{mode_text}: {current_method}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, info_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result, 'ESC = Exit, Auto Mode ON/OFF', (10, result.shape[0]-10), 
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
    print("✨ OpenCV Kontrast ve Parlaklık İşlemleri")
    print("Bu program, kontrast ve parlaklık ayarlama tekniklerini gösterir.\n")
    
    # Test resimleri oluştur
    normal_path, low_contrast_path, dark_path, bright_path = ornek_resim_olustur()
    
    # Ana test resmi olarak normal resimi kullan
    resim = cv2.imread(normal_path)
    
    if resim is None:
        print("❌ Test resimleri oluşturulamadı!")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("✨ Kontrast ve Parlaklık Menüsü")
        print("=" * 50)
        print("1. Lineer Transformasyon (Alpha-Beta)")
        print("2. Gamma Düzeltme")
        print("3. Logaritmik Transformasyon")
        print("4. Otomatik Kontrast Ayarlama")
        print("5. İnteraktif Kontrast/Parlaklık Demosu")
        print("0. Çıkış")
        
        try:
            secim = input("\nLütfen bir seçenek girin (0-5): ").strip()
            
            if secim == '0':
                print("👋 Görüşmek üzere!")
                break
            elif secim == '1':
                lineer_transformasyon_ornekleri(resim)
            elif secim == '2':
                gamma_duzeltme_ornekleri(resim)
            elif secim == '3':
                logaritmik_transformasyon_ornekleri(resim)
            elif secim == '4':
                otomatik_kontrast_ornekleri(resim)
            elif secim == '5':
                interaktif_kontrast_demo()
            else:
                print("❌ Geçersiz seçenek! Lütfen 0-5 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program kullanıcı tarafından sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()