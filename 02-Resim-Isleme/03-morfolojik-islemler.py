"""
🔄 OpenCV Morfolojik İşlemler
=============================

Bu dosyada morfolojik görüntü işleme tekniklerini öğreneceksiniz:
- Erozyon (erosion) - nesneleri küçültme
- Dilatasyon (dilation) - nesneleri büyütme
- Açma (opening) - gürültü temizleme
- Kapama (closing) - boşluk doldurma
- Gradient - kenar bulma
- Top-hat ve Black-hat transformasyonları

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def ornek_resim_olustur():
    """Morfolojik işlemler için test resimleri oluştur"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 1. Binary resim (siyah-beyaz)
    binary_resim = np.zeros((300, 300), dtype=np.uint8)
    
    # Büyük dikdörtgen
    cv2.rectangle(binary_resim, (50, 50), (150, 100), 255, -1)
    
    # Küçük kareler (gürültü)
    cv2.rectangle(binary_resim, (20, 20), (25, 25), 255, -1)
    cv2.rectangle(binary_resim, (270, 20), (275, 25), 255, -1)
    cv2.rectangle(binary_resim, (20, 270), (25, 275), 255, -1)
    
    # Daire
    cv2.circle(binary_resim, (200, 150), 40, 255, -1)
    
    # İnce çizgiler
    cv2.line(binary_resim, (80, 120), (120, 160), 255, 2)
    cv2.line(binary_resim, (120, 120), (80, 160), 255, 2)
    
    # Boşlukları olan şekil
    cv2.rectangle(binary_resim, (50, 200), (150, 250), 255, -1)
    cv2.rectangle(binary_resim, (70, 210), (80, 240), 0, -1)  # Boşluk
    cv2.rectangle(binary_resim, (120, 210), (130, 240), 0, -1)  # Boşluk
    
    binary_dosya = examples_dir / "morphology_binary.jpg"
    cv2.imwrite(str(binary_dosya), binary_resim)
    
    # 2. Gürültülü metin resmi
    metin_resim = np.zeros((200, 400), dtype=np.uint8)
    cv2.putText(metin_resim, 'MORPHOLOGY', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                2, 255, 3)
    cv2.putText(metin_resim, 'TEST', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 255, 2)
    
    # Salt & pepper gürültü ekle
    gurultu_maskesi = np.random.random((200, 400))
    metin_resim[gurultu_maskesi < 0.02] = 255  # Salt noise
    metin_resim[gurultu_maskesi > 0.98] = 0    # Pepper noise
    
    metin_dosya = examples_dir / "morphology_text.jpg"
    cv2.imwrite(str(metin_dosya), metin_resim)
    
    # 3. Parçalı şekiller resmi
    parcali_resim = np.zeros((250, 250), dtype=np.uint8)
    
    # Kırık daire
    cv2.circle(parcali_resim, (100, 100), 30, 255, 5)
    cv2.rectangle(parcali_resim, (95, 85), (105, 95), 0, -1)  # Boşluk
    
    # Kırık dikdörtgen
    cv2.rectangle(parcali_resim, (150, 80), (220, 120), 255, -1)
    cv2.rectangle(parcali_resim, (175, 85), (195, 115), 0, -1)  # Delik
    
    # İnce bağlantılar
    cv2.line(parcali_resim, (50, 180), (200, 180), 255, 1)
    cv2.line(parcali_resim, (125, 150), (125, 210), 255, 1)
    
    parcali_dosya = examples_dir / "morphology_broken.jpg"
    cv2.imwrite(str(parcali_dosya), parcali_resim)
    
    print(f"✅ Morfolojik test resimleri oluşturuldu:")
    print(f"   - Binary resim: {binary_dosya}")
    print(f"   - Metin resmi: {metin_dosya}")
    print(f"   - Parçalı resim: {parcali_dosya}")
    
    return str(binary_dosya), str(metin_dosya), str(parcali_dosya)

def erozyon_ornekleri(resim):
    """Erozyon işlemi örnekleri"""
    print("\n🔻 Erozyon (Erosion) Örnekleri")
    print("=" * 35)
    
    # Farklı kernel şekilleri
    kernel_rect_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_rect_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_rect_7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_ellipse_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    kernel_cross_5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    # Özel kernel
    custom_kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
    
    # Erezyon işlemleri
    erosion_rect_3 = cv2.erode(resim, kernel_rect_3, iterations=1)
    erosion_rect_5 = cv2.erode(resim, kernel_rect_5, iterations=1)
    erosion_rect_7 = cv2.erode(resim, kernel_rect_7, iterations=1)
    
    erosion_ellipse_5 = cv2.erode(resim, kernel_ellipse_5, iterations=1)
    erosion_ellipse_9 = cv2.erode(resim, kernel_ellipse_9, iterations=1)
    
    erosion_cross = cv2.erode(resim, kernel_cross_5, iterations=1)
    erosion_custom = cv2.erode(resim, custom_kernel, iterations=1)
    
    # Çoklu iterasyon
    erosion_iter_1 = cv2.erode(resim, kernel_rect_5, iterations=1)
    erosion_iter_2 = cv2.erode(resim, kernel_rect_5, iterations=2)
    erosion_iter_3 = cv2.erode(resim, kernel_rect_5, iterations=3)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 15))
    
    plt.subplot(4, 4, 1)
    plt.imshow(resim, cmap='gray')
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    # Farklı kernel boyutları
    plt.subplot(4, 4, 2)
    plt.imshow(erosion_rect_3, cmap='gray')
    plt.title('Dikdörtgen 3x3')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(erosion_rect_5, cmap='gray')
    plt.title('Dikdörtgen 5x5')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(erosion_rect_7, cmap='gray')
    plt.title('Dikdörtgen 7x7')
    plt.axis('off')
    
    # Farklı kernel şekilleri
    plt.subplot(4, 4, 5)
    plt.imshow(erosion_ellipse_5, cmap='gray')
    plt.title('Elips 5x5')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(erosion_ellipse_9, cmap='gray')
    plt.title('Elips 9x9')
    plt.axis('off')
    
    plt.subplot(4, 4, 7)
    plt.imshow(erosion_cross, cmap='gray')
    plt.title('Artı (+) 5x5')
    plt.axis('off')
    
    plt.subplot(4, 4, 8)
    plt.imshow(erosion_custom, cmap='gray')
    plt.title('Özel Kernel')
    plt.axis('off')
    
    # Çoklu iterasyon
    plt.subplot(4, 4, 9)
    plt.imshow(erosion_iter_1, cmap='gray')
    plt.title('1 İterasyon')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(erosion_iter_2, cmap='gray')
    plt.title('2 İterasyon')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(erosion_iter_3, cmap='gray')
    plt.title('3 İterasyon')
    plt.axis('off')
    
    # Kernel görselleştirmeleri
    plt.subplot(4, 4, 12)
    plt.imshow(kernel_rect_5, cmap='hot', interpolation='nearest')
    plt.title('Dikdörtgen Kernel')
    plt.axis('off')
    
    plt.subplot(4, 4, 13)
    plt.imshow(kernel_ellipse_5, cmap='hot', interpolation='nearest')
    plt.title('Elips Kernel')
    plt.axis('off')
    
    plt.subplot(4, 4, 14)
    plt.imshow(kernel_cross_5, cmap='hot', interpolation='nearest')
    plt.title('Artı Kernel')
    plt.axis('off')
    
    plt.subplot(4, 4, 15)
    plt.imshow(custom_kernel, cmap='hot', interpolation='nearest')
    plt.title('Özel Kernel')
    plt.axis('off')
    
    # Alan değişimi analizi
    plt.subplot(4, 4, 16)
    # Her iterasyonda beyaz piksel sayısını hesapla
    iterations = list(range(1, 6))
    areas = []
    for i in iterations:
        eroded = cv2.erode(resim, kernel_rect_5, iterations=i)
        area = np.sum(eroded == 255)
        areas.append(area)
    
    plt.plot(iterations, areas, 'bo-', linewidth=2, markersize=8)
    plt.title('İterasyon vs Alan')
    plt.xlabel('İterasyon Sayısı')
    plt.ylabel('Beyaz Piksel Sayısı')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Erozyon İpuçları:")
    print("   • Nesneleri küçültür, gürültüyü temizler")
    print("   • Kernel boyutu etkiyi belirler")
    print("   • İterasyon sayısı ile güçlendirilebilir")
    print("   • İnce bağlantıları koparır")

def dilatasyon_ornekleri(resim):
    """Dilatasyon işlemi örnekleri"""
    print("\n🔺 Dilatasyon (Dilation) Örnekleri")
    print("=" * 35)
    
    # Farklı kernel şekilleri
    kernel_rect_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_rect_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_rect_7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_ellipse_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    kernel_cross_5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    # Dilatasyon işlemleri
    dilation_rect_3 = cv2.dilate(resim, kernel_rect_3, iterations=1)
    dilation_rect_5 = cv2.dilate(resim, kernel_rect_5, iterations=1)
    dilation_rect_7 = cv2.dilate(resim, kernel_rect_7, iterations=1)
    
    dilation_ellipse_5 = cv2.dilate(resim, kernel_ellipse_5, iterations=1)
    dilation_ellipse_9 = cv2.dilate(resim, kernel_ellipse_9, iterations=1)
    
    dilation_cross = cv2.dilate(resim, kernel_cross_5, iterations=1)
    
    # Çoklu iterasyon
    dilation_iter_1 = cv2.dilate(resim, kernel_rect_5, iterations=1)
    dilation_iter_2 = cv2.dilate(resim, kernel_rect_5, iterations=2)
    dilation_iter_3 = cv2.dilate(resim, kernel_rect_5, iterations=3)
    
    # Erozyon + Dilatasyon karşılaştırması
    erosion_comp = cv2.erode(resim, kernel_rect_5, iterations=1)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 15))
    
    plt.subplot(4, 4, 1)
    plt.imshow(resim, cmap='gray')
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    # Farklı kernel boyutları
    plt.subplot(4, 4, 2)
    plt.imshow(dilation_rect_3, cmap='gray')
    plt.title('Dikdörtgen 3x3')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(dilation_rect_5, cmap='gray')
    plt.title('Dikdörtgen 5x5')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(dilation_rect_7, cmap='gray')
    plt.title('Dikdörtgen 7x7')
    plt.axis('off')
    
    # Farklı kernel şekilleri
    plt.subplot(4, 4, 5)
    plt.imshow(dilation_ellipse_5, cmap='gray')
    plt.title('Elips 5x5')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(dilation_ellipse_9, cmap='gray')
    plt.title('Elips 9x9')
    plt.axis('off')
    
    plt.subplot(4, 4, 7)
    plt.imshow(dilation_cross, cmap='gray')
    plt.title('Artı (+) 5x5')
    plt.axis('off')
    
    # Erozyon vs Dilatasyon
    plt.subplot(4, 4, 8)
    plt.imshow(erosion_comp, cmap='gray')
    plt.title('Erozyon (Karşılaştırma)')
    plt.axis('off')
    
    # Çoklu iterasyon
    plt.subplot(4, 4, 9)
    plt.imshow(dilation_iter_1, cmap='gray')
    plt.title('1 İterasyon')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    plt.imshow(dilation_iter_2, cmap='gray')
    plt.title('2 İterasyon')
    plt.axis('off')
    
    plt.subplot(4, 4, 11)
    plt.imshow(dilation_iter_3, cmap='gray')
    plt.title('3 İterasyon')
    plt.axis('off')
    
    # Yan yana karşılaştırma
    plt.subplot(4, 4, 12)
    comparison = np.hstack([erosion_comp, resim, dilation_rect_5])
    plt.imshow(comparison, cmap='gray')
    plt.title('Erozyon | Orijinal | Dilatasyon')
    plt.axis('off')
    
    # Alan değişimi
    plt.subplot(4, 4, 13)
    iterations = list(range(1, 6))
    areas_erosion = []
    areas_dilation = []
    
    for i in iterations:
        eroded = cv2.erode(resim, kernel_rect_5, iterations=i)
        dilated = cv2.dilate(resim, kernel_rect_5, iterations=i)
        areas_erosion.append(np.sum(eroded == 255))
        areas_dilation.append(np.sum(dilated == 255))
    
    plt.plot(iterations, areas_erosion, 'ro-', label='Erozyon', linewidth=2)
    plt.plot(iterations, areas_dilation, 'bo-', label='Dilatasyon', linewidth=2)
    plt.title('İterasyon vs Alan Karşılaştırması')
    plt.xlabel('İterasyon Sayısı')
    plt.ylabel('Beyaz Piksel Sayısı')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dual operations (Erezyon sonra dilatasyon)
    plt.subplot(4, 4, 14)
    dual_op = cv2.dilate(cv2.erode(resim, kernel_rect_5, iterations=1), 
                         kernel_rect_5, iterations=1)
    plt.imshow(dual_op, cmap='gray')
    plt.title('Erozyon → Dilatasyon')
    plt.axis('off')
    
    # Reverse dual (Dilatasyon sonra erozyon)
    plt.subplot(4, 4, 15)
    reverse_dual = cv2.erode(cv2.dilate(resim, kernel_rect_5, iterations=1), 
                            kernel_rect_5, iterations=1)
    plt.imshow(reverse_dual, cmap='gray')
    plt.title('Dilatasyon → Erozyon')
    plt.axis('off')
    
    # Yoğunluk profili
    plt.subplot(4, 4, 16)
    row = resim.shape[0] // 2
    plt.plot(resim[row, :], 'k-', label='Orijinal', linewidth=2)
    plt.plot(erosion_comp[row, :], 'r--', label='Erozyon', linewidth=2)
    plt.plot(dilation_rect_5[row, :], 'b--', label='Dilatasyon', linewidth=2)
    plt.title('Yatay Profil Karşılaştırması')
    plt.xlabel('Piksel Pozisyonu')
    plt.ylabel('Yoğunluk')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Dilatasyon İpuçları:")
    print("   • Nesneleri büyütür, boşlukları doldurur")
    print("   • Erozyon işleminin tersidir")
    print("   • Bağlantıları güçlendirir")
    print("   • Kernel şekli sonucu etkiler")

def opening_closing_ornekleri(resim):
    """Açma (Opening) ve Kapama (Closing) örnekleri"""
    print("\n🔄 Açma ve Kapama İşlemleri")
    print("=" * 35)
    
    kernel_rect_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_rect_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_cross_5 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    # Opening (Erozyon sonra Dilatasyon)
    opening_rect_3 = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel_rect_3)
    opening_rect_5 = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel_rect_5)
    opening_ellipse = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel_ellipse_5)
    opening_cross = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel_cross_5)
    
    # Closing (Dilatasyon sonra Erozyon)
    closing_rect_3 = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel_rect_3)
    closing_rect_5 = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel_rect_5)
    closing_ellipse = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel_ellipse_5)
    closing_cross = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel_cross_5)
    
    # Manuel olarak opening ve closing
    manual_opening = cv2.dilate(cv2.erode(resim, kernel_rect_5, iterations=1), 
                               kernel_rect_5, iterations=1)
    manual_closing = cv2.erode(cv2.dilate(resim, kernel_rect_5, iterations=1), 
                              kernel_rect_5, iterations=1)
    
    # Ardışık işlemler
    open_then_close = cv2.morphologyEx(opening_rect_5, cv2.MORPH_CLOSE, kernel_rect_5)
    close_then_open = cv2.morphologyEx(closing_rect_5, cv2.MORPH_OPEN, kernel_rect_5)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    plt.subplot(4, 5, 1)
    plt.imshow(resim, cmap='gray')
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    # Opening sonuçları
    plt.subplot(4, 5, 2)
    plt.imshow(opening_rect_3, cmap='gray')
    plt.title('Opening 3x3')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(opening_rect_5, cmap='gray')
    plt.title('Opening 5x5')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(opening_ellipse, cmap='gray')
    plt.title('Opening Elips')
    plt.axis('off')
    
    plt.subplot(4, 5, 5)
    plt.imshow(opening_cross, cmap='gray')
    plt.title('Opening Artı')
    plt.axis('off')
    
    # Closing sonuçları
    plt.subplot(4, 5, 6)
    plt.imshow(closing_rect_3, cmap='gray')
    plt.title('Closing 3x3')
    plt.axis('off')
    
    plt.subplot(4, 5, 7)
    plt.imshow(closing_rect_5, cmap='gray')
    plt.title('Closing 5x5')
    plt.axis('off')
    
    plt.subplot(4, 5, 8)
    plt.imshow(closing_ellipse, cmap='gray')
    plt.title('Closing Elips')
    plt.axis('off')
    
    plt.subplot(4, 5, 9)
    plt.imshow(closing_cross, cmap='gray')
    plt.title('Closing Artı')
    plt.axis('off')
    
    # Karşılaştırma
    plt.subplot(4, 5, 10)
    comparison = np.hstack([opening_rect_5, closing_rect_5])
    plt.imshow(comparison, cmap='gray')
    plt.title('Opening | Closing')
    plt.axis('off')
    
    # Manuel vs Otomatik
    plt.subplot(4, 5, 11)
    plt.imshow(manual_opening, cmap='gray')
    plt.title('Manuel Opening')
    plt.axis('off')
    
    plt.subplot(4, 5, 12)
    plt.imshow(manual_closing, cmap='gray')
    plt.title('Manuel Closing')
    plt.axis('off')
    
    # Fark gösterimi
    plt.subplot(4, 5, 13)
    diff_opening = cv2.absdiff(opening_rect_5, manual_opening)
    plt.imshow(diff_opening, cmap='gray')
    plt.title('Opening Farkı')
    plt.axis('off')
    
    plt.subplot(4, 5, 14)
    diff_closing = cv2.absdiff(closing_rect_5, manual_closing)
    plt.imshow(diff_closing, cmap='gray')
    plt.title('Closing Farkı')
    plt.axis('off')
    
    # Ardışık işlemler
    plt.subplot(4, 5, 15)
    plt.imshow(open_then_close, cmap='gray')
    plt.title('Opening → Closing')
    plt.axis('off')
    
    plt.subplot(4, 5, 16)
    plt.imshow(close_then_open, cmap='gray')
    plt.title('Closing → Opening')
    plt.axis('off')
    
    # Etkiler analizi
    plt.subplot(4, 5, 17)
    # Gürültü temizleme etkisi
    noise_mask = opening_rect_5 != resim
    noise_removed = np.sum(noise_mask)
    
    plt.text(0.1, 0.9, 'Opening Etkileri:', fontsize=12, weight='bold', 
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'• Gürültü temizleme\n• İnce bağlantı koparma\n• Küçük nesne kaldırma', 
             fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f'Temizlenen piksel: {noise_removed}', 
             fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.subplot(4, 5, 18)
    plt.text(0.1, 0.9, 'Closing Etkileri:', fontsize=12, weight='bold', 
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'• Boşluk doldurma\n• Bağlantı güçlendirme\n• Kontur düzgünleştirme', 
             fontsize=10, transform=plt.gca().transAxes)
    
    # Hole filling analizi
    holes_filled = np.sum(closing_rect_5) - np.sum(resim)
    plt.text(0.1, 0.3, f'Doldurulan alan: {holes_filled}', 
             fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    # Kernel boyutu etkisi
    plt.subplot(4, 5, 19)
    kernel_sizes = [3, 5, 7, 9]
    opening_areas = []
    closing_areas = []
    
    for size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        opening = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel)
        opening_areas.append(np.sum(opening == 255))
        closing_areas.append(np.sum(closing == 255))
    
    plt.plot(kernel_sizes, opening_areas, 'ro-', label='Opening', linewidth=2)
    plt.plot(kernel_sizes, closing_areas, 'bo-', label='Closing', linewidth=2)
    plt.title('Kernel Boyutu vs Alan')
    plt.xlabel('Kernel Boyutu')
    plt.ylabel('Beyaz Piksel Sayısı')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Pratik uygulama örneği
    plt.subplot(4, 5, 20)
    # Önce closing (boşlukları doldur) sonra opening (gürültüyü temizle)
    practical_result = cv2.morphologyEx(
        cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel_rect_3),
        cv2.MORPH_OPEN, kernel_rect_3
    )
    plt.imshow(practical_result, cmap='gray')
    plt.title('Pratik: Closing→Opening')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Opening ve Closing İpuçları:")
    print("   • Opening = Erozyon + Dilatasyon (gürültü temizler)")
    print("   • Closing = Dilatasyon + Erozyon (boşluk doldurur)")
    print("   • Dual operasyonlardır (birbirinin tersi değil)")
    print("   • Kernel şekli ve boyutu sonucu belirler")

def gradient_tophat_ornekleri(resim):
    """Gradient, Top-hat ve Black-hat örnekleri"""
    print("\n🎩 Gradient ve Hat Transform Örnekleri")
    print("=" * 35)
    
    kernel_rect_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_rect_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Gradient işlemleri
    gradient_3 = cv2.morphologyEx(resim, cv2.MORPH_GRADIENT, kernel_rect_3)
    gradient_5 = cv2.morphologyEx(resim, cv2.MORPH_GRADIENT, kernel_rect_5)
    gradient_ellipse = cv2.morphologyEx(resim, cv2.MORPH_GRADIENT, kernel_ellipse_5)
    
    # Top-hat (White-hat) - Orijinal - Opening
    tophat_3 = cv2.morphologyEx(resim, cv2.MORPH_TOPHAT, kernel_rect_3)
    tophat_5 = cv2.morphologyEx(resim, cv2.MORPH_TOPHAT, kernel_rect_5)
    tophat_ellipse = cv2.morphologyEx(resim, cv2.MORPH_TOPHAT, kernel_ellipse_5)
    
    # Black-hat - Closing - Orijinal
    blackhat_3 = cv2.morphologyEx(resim, cv2.MORPH_BLACKHAT, kernel_rect_3)
    blackhat_5 = cv2.morphologyEx(resim, cv2.MORPH_BLACKHAT, kernel_rect_5)
    blackhat_ellipse = cv2.morphologyEx(resim, cv2.MORPH_BLACKHAT, kernel_ellipse_5)
    
    # Manuel gradient hesaplama
    dilated = cv2.dilate(resim, kernel_rect_5, iterations=1)
    eroded = cv2.erode(resim, kernel_rect_5, iterations=1)
    manual_gradient = cv2.subtract(dilated, eroded)
    
    # Manuel top-hat
    opening = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel_rect_5)
    manual_tophat = cv2.subtract(resim, opening)
    
    # Manuel black-hat
    closing = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel_rect_5)
    manual_blackhat = cv2.subtract(closing, resim)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 16))
    
    plt.subplot(4, 5, 1)
    plt.imshow(resim, cmap='gray')
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    # Gradient sonuçları
    plt.subplot(4, 5, 2)
    plt.imshow(gradient_3, cmap='gray')
    plt.title('Gradient 3x3')
    plt.axis('off')
    
    plt.subplot(4, 5, 3)
    plt.imshow(gradient_5, cmap='gray')
    plt.title('Gradient 5x5')
    plt.axis('off')
    
    plt.subplot(4, 5, 4)
    plt.imshow(gradient_ellipse, cmap='gray')
    plt.title('Gradient Elips')
    plt.axis('off')
    
    plt.subplot(4, 5, 5)
    plt.imshow(manual_gradient, cmap='gray')
    plt.title('Manuel Gradient')
    plt.axis('off')
    
    # Top-hat sonuçları
    plt.subplot(4, 5, 6)
    plt.imshow(tophat_3, cmap='gray')
    plt.title('Top-hat 3x3')
    plt.axis('off')
    
    plt.subplot(4, 5, 7)
    plt.imshow(tophat_5, cmap='gray')
    plt.title('Top-hat 5x5')
    plt.axis('off')
    
    plt.subplot(4, 5, 8)
    plt.imshow(tophat_ellipse, cmap='gray')
    plt.title('Top-hat Elips')
    plt.axis('off')
    
    plt.subplot(4, 5, 9)
    plt.imshow(manual_tophat, cmap='gray')
    plt.title('Manuel Top-hat')
    plt.axis('off')
    
    # Black-hat sonuçları
    plt.subplot(4, 5, 10)
    plt.imshow(blackhat_3, cmap='gray')
    plt.title('Black-hat 3x3')
    plt.axis('off')
    
    plt.subplot(4, 5, 11)
    plt.imshow(blackhat_5, cmap='gray')
    plt.title('Black-hat 5x5')
    plt.axis('off')
    
    plt.subplot(4, 5, 12)
    plt.imshow(blackhat_ellipse, cmap='gray')
    plt.title('Black-hat Elips')
    plt.axis('off')
    
    plt.subplot(4, 5, 13)
    plt.imshow(manual_blackhat, cmap='gray')
    plt.title('Manuel Black-hat')
    plt.axis('off')
    
    # Kombinasyon örnekleri
    plt.subplot(4, 5, 14)
    # Gradient + Top-hat
    combined = cv2.add(gradient_5, tophat_5)
    plt.imshow(combined, cmap='gray')
    plt.title('Gradient + Top-hat')
    plt.axis('off')
    
    plt.subplot(4, 5, 15)
    # Top-hat + Black-hat
    hat_combined = cv2.add(tophat_5, blackhat_5)
    plt.imshow(hat_combined, cmap='gray')
    plt.title('Top-hat + Black-hat')
    plt.axis('off')
    
    # Kenar algılama karşılaştırması
    plt.subplot(4, 5, 16)
    # Canny kenar algılama
    canny_edges = cv2.Canny(resim, 50, 150)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Kenar Algılama')
    plt.axis('off')
    
    plt.subplot(4, 5, 17)
    # Gradient vs Canny karşılaştırması
    comparison = np.hstack([gradient_5, canny_edges])
    plt.imshow(comparison, cmap='gray')
    plt.title('Gradient | Canny')
    plt.axis('off')
    
    # Aplikasyon örnekleri
    plt.subplot(4, 5, 18)
    # Dokusal analiz için top-hat
    texture_analysis = cv2.add(resim, tophat_5)
    plt.imshow(texture_analysis, cmap='gray')
    plt.title('Dokusal Analiz\n(Orijinal + Top-hat)')
    plt.axis('off')
    
    plt.subplot(4, 5, 19)
    # Arka plan normalizasyonu
    background_norm = cv2.subtract(resim, blackhat_5)
    plt.imshow(background_norm, cmap='gray')
    plt.title('Arka Plan Normalizasyonu\n(Orijinal - Black-hat)')
    plt.axis('off')
    
    # Histogram karşılaştırması
    plt.subplot(4, 5, 20)
    plt.hist(resim.ravel(), bins=50, alpha=0.7, label='Orijinal', color='blue')
    plt.hist(gradient_5.ravel(), bins=50, alpha=0.7, label='Gradient', color='red')
    plt.hist(tophat_5.ravel(), bins=50, alpha=0.7, label='Top-hat', color='green')
    plt.title('Histogram Karşılaştırması')
    plt.xlabel('Piksel Değeri')
    plt.ylabel('Frekans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Praktik uygulama örnekleri
    print("\n🔍 Pratik Uygulamalar:")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Metin vurgulama
    text_enhance = cv2.add(resim, tophat_5)
    axes[0, 0].imshow(text_enhance, cmap='gray')
    axes[0, 0].set_title('Metin Vurgulama')
    axes[0, 0].axis('off')
    
    # 2. Arka plan kaldırma
    bg_removal = cv2.subtract(resim, blackhat_5)
    axes[0, 1].imshow(bg_removal, cmap='gray')
    axes[0, 1].set_title('Arka Plan Kaldırma')
    axes[0, 1].axis('off')
    
    # 3. Kenar vurgulama
    edge_enhance = cv2.add(resim, gradient_5)
    axes[0, 2].imshow(edge_enhance, cmap='gray')
    axes[0, 2].set_title('Kenar Vurgulama')
    axes[0, 2].axis('off')
    
    # 4. Gürültü tespiti
    noise_detection = tophat_3
    axes[1, 0].imshow(noise_detection, cmap='gray')
    axes[1, 0].set_title('Gürültü Tespiti')
    axes[1, 0].axis('off')
    
    # 5. Boşluk tespiti
    gap_detection = blackhat_3
    axes[1, 1].imshow(gap_detection, cmap='gray')
    axes[1, 1].set_title('Boşluk Tespiti')
    axes[1, 1].axis('off')
    
    # 6. Kombine analiz
    combined_analysis = cv2.add(cv2.add(gradient_3, tophat_3), blackhat_3)
    axes[1, 2].imshow(combined_analysis, cmap='gray')
    axes[1, 2].set_title('Kombine Analiz')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Gradient ve Hat Transform İpuçları:")
    print("   • Gradient = Dilatasyon - Erozyon (kenar bulma)")
    print("   • Top-hat = Orijinal - Opening (küçük detay vurgulama)")
    print("   • Black-hat = Closing - Orijinal (boşluk vurgulama)")
    print("   • Dokusal analiz ve arka plan normalizasyonu için ideal")

def morfolojik_filtreleme_ornekleri():
    """Gerçek görüntülerle morfolojik filtreleme örnekleri"""
    print("\n🖼️ Gerçek Görüntülerle Morfolojik Filtreleme")
    print("=" * 50)
    
    # Test resimleri oluştur
    binary_path, text_path, broken_path = ornek_resim_olustur()
    
    # Resimleri yükle
    binary_img = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    text_img = cv2.imread(text_path, cv2.IMREAD_GRAYSCALE)
    broken_img = cv2.imread(broken_path, cv2.IMREAD_GRAYSCALE)
    
    # Kerneller
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Problem çözme örnekleri
    
    # 1. Gürültü temizleme (text image)
    text_cleaned = cv2.morphologyEx(text_img, cv2.MORPH_OPEN, kernel_small)
    text_enhanced = cv2.morphologyEx(text_cleaned, cv2.MORPH_CLOSE, kernel_small)
    
    # 2. Boşluk doldurma (broken image)
    broken_filled = cv2.morphologyEx(broken_img, cv2.MORPH_CLOSE, kernel_medium)
    broken_smooth = cv2.morphologyEx(broken_filled, cv2.MORPH_OPEN, kernel_small)
    
    # 3. Nesne ayırma (binary image)
    binary_separated = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_large)
    
    # 4. Kenar bulma
    edges_binary = cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, kernel_small)
    edges_text = cv2.morphologyEx(text_img, cv2.MORPH_GRADIENT, kernel_small)
    edges_broken = cv2.morphologyEx(broken_img, cv2.MORPH_GRADIENT, kernel_small)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 15))
    
    # Orijinal resimler
    plt.subplot(4, 6, 1)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Binary Orijinal')
    plt.axis('off')
    
    plt.subplot(4, 6, 2)
    plt.imshow(text_img, cmap='gray')
    plt.title('Text Orijinal')
    plt.axis('off')
    
    plt.subplot(4, 6, 3)
    plt.imshow(broken_img, cmap='gray')
    plt.title('Broken Orijinal')
    plt.axis('off')
    
    # Problem çözme sonuçları
    plt.subplot(4, 6, 7)
    plt.imshow(binary_separated, cmap='gray')
    plt.title('Nesne Ayırma')
    plt.axis('off')
    
    plt.subplot(4, 6, 8)
    plt.imshow(text_enhanced, cmap='gray')
    plt.title('Gürültü Temizleme')
    plt.axis('off')
    
    plt.subplot(4, 6, 9)
    plt.imshow(broken_smooth, cmap='gray')
    plt.title('Boşluk Doldurma')
    plt.axis('off')
    
    # Kenar bulma sonuçları
    plt.subplot(4, 6, 13)
    plt.imshow(edges_binary, cmap='gray')
    plt.title('Binary Kenarlar')
    plt.axis('off')
    
    plt.subplot(4, 6, 14)
    plt.imshow(edges_text, cmap='gray')
    plt.title('Text Kenarlar')
    plt.axis('off')
    
    plt.subplot(4, 6, 15)
    plt.imshow(edges_broken, cmap='gray')
    plt.title('Broken Kenarlar')
    plt.axis('off')
    
    # Öncesi-sonrası karşılaştırmaları
    plt.subplot(4, 6, 4)
    before_after_text = np.hstack([text_img, text_enhanced])
    plt.imshow(before_after_text, cmap='gray')
    plt.title('Text: Öncesi | Sonrası')
    plt.axis('off')
    
    plt.subplot(4, 6, 5)
    before_after_broken = np.hstack([broken_img, broken_smooth])
    plt.imshow(before_after_broken, cmap='gray')
    plt.title('Broken: Öncesi | Sonrası')
    plt.axis('off')
    
    plt.subplot(4, 6, 6)
    before_after_binary = np.hstack([binary_img, binary_separated])
    plt.imshow(before_after_binary, cmap='gray')
    plt.title('Binary: Öncesi | Sonrası')
    plt.axis('off')
    
    # İstatistiksel analiz
    plt.subplot(4, 6, 10)
    # Nesne sayısı analizi
    contours_orig, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_proc, _ = cv2.findContours(binary_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plt.bar(['Orijinal', 'İşlenmiş'], [len(contours_orig), len(contours_proc)], 
            color=['red', 'green'], alpha=0.7)
    plt.title('Nesne Sayısı Karşılaştırması')
    plt.ylabel('Nesne Sayısı')
    plt.grid(True, alpha=0.3)
    
    # Gürültü azaltma analizi
    plt.subplot(4, 6, 11)
    noise_reduced = np.sum(text_img) - np.sum(text_enhanced)
    improvement = (noise_reduced / np.sum(text_img)) * 100
    
    plt.pie([improvement, 100-improvement], labels=['Temizlenen', 'Korunan'], 
            colors=['red', 'green'], autopct='%1.1f%%')
    plt.title('Gürültü Temizleme Etkisi')
    
    # Boşluk doldurma analizi
    plt.subplot(4, 6, 12)
    filled_area = np.sum(broken_smooth) - np.sum(broken_img)
    
    plt.text(0.1, 0.8, f'Doldurulan Alan:', fontsize=12, weight='bold', 
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'{filled_area} piksel', fontsize=14, 
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f'İyileşme: %{(filled_area/np.sum(broken_img)*100):.1f}', 
             fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    # Kernel karşılaştırması
    plt.subplot(4, 6, 16)
    kernels = [kernel_small, kernel_medium, kernel_large]
    kernel_names = ['3x3 Rect', '5x5 Rect', '7x7 Ellipse']
    results = []
    
    for kernel in kernels:
        result = cv2.morphologyEx(text_img, cv2.MORPH_OPEN, kernel)
        cleaned_pixels = np.sum(text_img) - np.sum(result)
        results.append(cleaned_pixels)
    
    plt.bar(kernel_names, results, color=['blue', 'orange', 'green'], alpha=0.7)
    plt.title('Kernel Etkisi Karşılaştırması')
    plt.ylabel('Temizlenen Piksel')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Algoritma karşılaştırması
    plt.subplot(4, 6, 17)
    algorithms = ['Opening', 'Closing', 'Gradient', 'Top-hat']
    processing_times = [1, 1, 2, 2]  # Yaklaşık relatif süreler
    
    plt.bar(algorithms, processing_times, color=['red', 'green', 'blue', 'orange'], alpha=0.7)
    plt.title('İşlem Süreleri (Yaklaşık)')
    plt.ylabel('Relatif Süre')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Kullanım alanları
    plt.subplot(4, 6, 18)
    plt.text(0.05, 0.95, 'Kullanım Alanları:', fontsize=12, weight='bold', 
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.80, '📄 Belge işleme\n🔍 Nesne algılama\n🧼 Gürültü temizleme', 
             fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.45, '🩺 Tıbbi görüntüleme\n📊 Şekil analizi\n🎯 Ön işleme', 
             fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return binary_img, text_img, broken_img

def interaktif_morfoloji_demo():
    """İnteraktif morfolojik işlemler demosu"""
    print("\n🎮 İnteraktif Morfoloji Demosu")
    print("=" * 35)
    print("Trackbar'ları kullanarak gerçek zamanlı morfolojik işlemler görün!")
    
    # Test resmi yükle
    binary_path, _, _ = ornek_resim_olustur()
    resim = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    
    if resim is None:
        print("❌ Resim yüklenemedi!")
        return
    
    # Pencere oluştur
    window_name = 'Interactive Morphology Demo'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Trackbar'lar oluştur
    cv2.createTrackbar('Operation', window_name, 0, 6, lambda x: None)
    cv2.createTrackbar('Kernel Shape', window_name, 0, 2, lambda x: None)
    cv2.createTrackbar('Kernel Size', window_name, 3, 15, lambda x: None)
    cv2.createTrackbar('Iterations', window_name, 1, 5, lambda x: None)
    
    operation_names = ['Original', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient', 'Tophat']
    kernel_names = ['Rectangle', 'Ellipse', 'Cross']
    
    print("🎛️ Kontroller:")
    print("   • Operation: 0-6 (Erosion, Dilation, vb.)")
    print("   • Kernel Shape: 0-2 (Rectangle, Ellipse, Cross)")
    print("   • Kernel Size: 3-15")
    print("   • Iterations: 1-5")
    print("   • ESC tuşu ile çıkış")
    
    while True:
        # Trackbar değerlerini oku
        operation = cv2.getTrackbarPos('Operation', window_name)
        kernel_shape = cv2.getTrackbarPos('Kernel Shape', window_name)
        kernel_size = cv2.getTrackbarPos('Kernel Size', window_name)
        iterations = cv2.getTrackbarPos('Iterations', window_name)
        
        # Kernel boyutunu tek sayı yap
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        
        # İterasyon kontrolü
        if iterations < 1:
            iterations = 1
        
        # Kernel oluştur
        if kernel_shape == 0:  # Rectangle
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        elif kernel_shape == 1:  # Ellipse
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:  # Cross
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        
        # İşlemi uygula
        try:
            if operation == 0:  # Original
                result = resim.copy()
            elif operation == 1:  # Erosion
                result = cv2.erode(resim, kernel, iterations=iterations)
            elif operation == 2:  # Dilation
                result = cv2.dilate(resim, kernel, iterations=iterations)
            elif operation == 3:  # Opening
                result = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif operation == 4:  # Closing
                result = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            elif operation == 5:  # Gradient
                result = cv2.morphologyEx(resim, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
            elif operation == 6:  # Top-hat
                result = cv2.morphologyEx(resim, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
            else:
                result = resim.copy()
                
        except Exception as e:
            result = resim.copy()
        
        # RGB'ye çevir (renkli metin için)
        result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Bilgi metnini ekle
        current_op = operation_names[min(operation, len(operation_names)-1)]
        current_kernel = kernel_names[min(kernel_shape, len(kernel_names)-1)]
        
        cv2.putText(result_color, f'Operation: {current_op}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_color, f'Kernel: {current_kernel} {kernel_size}x{kernel_size}', (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_color, f'Iterations: {iterations}', (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_color, 'ESC = Exit', (10, result.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Kernel'i küçük pencerede göster
        if kernel.shape[0] <= 15:  # Sadece küçük kernelları göster
            kernel_display = cv2.resize((kernel * 255).astype(np.uint8), (60, 60), 
                                      interpolation=cv2.INTER_NEAREST)
            kernel_color = cv2.cvtColor(kernel_display, cv2.COLOR_GRAY2BGR)
            
            # Kernel'i ana resmin köşesine koy
            h, w = kernel_color.shape[:2]
            result_color[-h-10:-10, -w-10:-10] = kernel_color
            cv2.putText(result_color, 'Kernel', (result.shape[1]-w-10, result.shape[0]-h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
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
    print("🔄 OpenCV Morfolojik İşlemler")
    print("Bu program, morfolojik görüntü işleme tekniklerini gösterir.\n")
    
    # Test resimleri oluştur
    binary_path, text_path, broken_path = ornek_resim_olustur()
    
    # Binary resmi yükle (ana örnekler için)
    binary_resim = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
    
    if binary_resim is None:
        print("❌ Test resimleri oluşturulamadı!")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("🔄 Morfolojik İşlemler Menüsü")
        print("=" * 50)
        print("1. Erozyon (Erosion) Örnekleri")
        print("2. Dilatasyon (Dilation) Örnekleri")
        print("3. Açma ve Kapama (Opening & Closing) Örnekleri")
        print("4. Gradient ve Hat Transform Örnekleri")
        print("5. Gerçek Görüntülerle Filtreleme")
        print("6. İnteraktif Morfoloji Demosu")
        print("0. Çıkış")
        
        try:
            secim = input("\nLütfen bir seçenek girin (0-6): ").strip()
            
            if secim == '0':
                print("👋 Görüşmek üzere!")
                break
            elif secim == '1':
                erozyon_ornekleri(binary_resim)
            elif secim == '2':
                dilatasyon_ornekleri(binary_resim)
            elif secim == '3':
                opening_closing_ornekleri(binary_resim)
            elif secim == '4':
                gradient_tophat_ornekleri(binary_resim)
            elif secim == '5':
                morfolojik_filtreleme_ornekleri()
            elif secim == '6':
                interaktif_morfoloji_demo()
            else:
                print("❌ Geçersiz seçenek! Lütfen 0-6 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program kullanıcı tarafından sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()