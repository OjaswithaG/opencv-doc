#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌈 OpenCV Renk Uzayları
======================

Bu dosya, OpenCV'de farklı renk uzayları ve dönüşümler konusunu öğretir:
- RGB, BGR, HSV, LAB, YUV renk uzayları
- Renk uzayı dönüşümleri
- Renk filtreleme ve maskeleme
- Pratik uygulamalar

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def renk_uzaylari_aciklama():
    """Renk uzayları hakkında temel bilgi"""
    print("🌈 Renk Uzayları Nedir?")
    print("=" * 35)
    print("""
    Renk uzayı, renkleri sayısal olarak temsil etme yöntemidir.
    Her renk uzayının kendine özgü avantajları vardır:
    
    🔵 BGR (Blue, Green, Red):
        - OpenCV'nin varsayılan formatı
        - Her kanal 0-255 değer alır
        - Monitör ve kameralar için doğal
    
    🔴 RGB (Red, Green, Blue):
        - İnsan algısına daha yakın
        - Web ve grafik tasarımda yaygın
        - Matplotlib'in varsayılan formatı
    
    🌈 HSV (Hue, Saturation, Value):
        - Renk (H), Doygunluk (S), Parlaklık (V)
        - Renk filtreleme için ideal
        - İnsan renk algısına yakın
    
    🔬 LAB (Lightness, A, B):
        - Işık bağımsız renk uzayı
        - Renk karşılaştırma için ideal
        - Profesyonel baskı endüstrisi
    
    📺 YUV/YCrCb:
        - Video kodlama için optimize
        - Parlaklık ve renk bilgisi ayrı
        - Sıkıştırma algoritmalarında kullanılır
    """)

def bgr_rgb_donusumu():
    """BGR ve RGB dönüşüm örnekleri"""
    print("\n🔄 BGR ↔ RGB Dönüşümü")
    print("=" * 30)
    
    # Renkli resim oluştur
    resim = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # BGR formatında kırmızı, yeşil, mavi şeritler
    resim[:, 0:100] = [0, 0, 255]    # Kırmızı (BGR)
    resim[:, 100:200] = [0, 255, 0]  # Yeşil (BGR)
    resim[:, 200:300] = [255, 0, 0]  # Mavi (BGR)
    
    # BGR'den RGB'ye dönüştür
    resim_rgb = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
    
    # Matplotlib ile karşılaştırmalı gösterim
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # BGR (yanlış renklerde görünecek)
    axes[0].imshow(resim)
    axes[0].set_title('BGR Formatında (Yanlış Görünüm)')
    axes[0].axis('off')
    
    # RGB (doğru renklerde görünecek)
    axes[1].imshow(resim_rgb)
    axes[1].set_title('RGB\'ye Dönüştürülmüş (Doğru Görünüm)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ BGR ve RGB formatları karşılaştırıldı!")

def hsv_renk_uzayi():
    """HSV renk uzayı örnekleri"""
    print("\n🌈 HSV Renk Uzayı")
    print("=" * 25)
    
    # HSV renk tekerleği oluştur
    boyut = 300
    hsv_tekerlek = np.zeros((boyut, boyut, 3), dtype=np.uint8)
    
    merkez = boyut // 2
    maksimum_yaricap = merkez - 10
    
    for y in range(boyut):
        for x in range(boyut):
            # Merkeze uzaklık (doygunluk için)
            dx = x - merkez
            dy = y - merkez
            mesafe = np.sqrt(dx*dx + dy*dy)
            
            if mesafe <= maksimum_yaricap:
                # Açı hesapla (renk için)
                aci = np.arctan2(dy, dx)
                hue = int((aci + np.pi) * 180 / (2 * np.pi))
                
                # Doygunluk (merkeze yakınlık)
                saturation = int(255 * mesafe / maksimum_yaricap)
                
                # Parlaklık sabit
                value = 255
                
                hsv_tekerlek[y, x] = [hue, saturation, value]
    
    # HSV'den BGR'ye dönüştür
    bgr_tekerlek = cv2.cvtColor(hsv_tekerlek, cv2.COLOR_HSV2BGR)
    
    # Göster
    cv2.imshow('HSV Renk Tekerleği', bgr_tekerlek)
    print("HSV renk tekerleği gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def renk_filtreleme_hsv():
    """HSV kullanarak renk filtreleme"""
    print("\n🎯 HSV ile Renk Filtreleme")
    print("=" * 35)
    
    # Renkli test resmi oluştur
    test_resim = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Farklı renkler ekle
    cv2.rectangle(test_resim, (50, 50), (150, 150), (0, 0, 255), -1)    # Kırmızı
    cv2.rectangle(test_resim, (200, 50), (300, 150), (0, 255, 0), -1)   # Yeşil
    cv2.rectangle(test_resim, (50, 200), (150, 300), (255, 0, 0), -1)   # Mavi
    cv2.rectangle(test_resim, (200, 200), (300, 300), (0, 255, 255), -1) # Sarı
    
    # HSV'ye dönüştür
    hsv_resim = cv2.cvtColor(test_resim, cv2.COLOR_BGR2HSV)
    
    # Farklı renk aralıkları tanımla
    renk_araliklari = {
        'Kırmızı': [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                    (np.array([170, 120, 70]), np.array([180, 255, 255]))],
        'Yeşil': [(np.array([40, 40, 40]), np.array([80, 255, 255]))],
        'Mavi': [(np.array([100, 150, 0]), np.array([140, 255, 255]))],
        'Sarı': [(np.array([20, 100, 100]), np.array([30, 255, 255]))]
    }
    
    # Her renk için maske oluştur ve göster
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Orijinal resmi göster
    axes[0].imshow(cv2.cvtColor(test_resim, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Orijinal Resim')
    axes[0].axis('off')
    
    for i, (renk_adi, aralıklar) in enumerate(renk_araliklari.items(), 1):
        # Maske oluştur
        maske = np.zeros(hsv_resim.shape[:2], dtype=np.uint8)
        
        for alt_sinir, ust_sinir in aralıklar:
            temp_maske = cv2.inRange(hsv_resim, alt_sinir, ust_sinir)
            maske = cv2.bitwise_or(maske, temp_maske)
        
        # Maskeyi uygula
        filtrelenmis = cv2.bitwise_and(test_resim, test_resim, mask=maske)
        
        # Göster
        axes[i].imshow(cv2.cvtColor(filtrelenmis, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'{renk_adi} Filtresi')
        axes[i].axis('off')
    
    # Son ekseni gizle
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ HSV renk filtreleme örnekleri gösterildi!")

def lab_renk_uzayi():
    """LAB renk uzayı örnekleri"""
    print("\n🔬 LAB Renk Uzayı")
    print("=" * 25)
    
    # Test resmi oluştur
    resim = np.zeros((200, 600, 3), dtype=np.uint8)
    resim[:, 0:200] = [0, 0, 255]    # Kırmızı
    resim[:, 200:400] = [0, 255, 0]  # Yeşil  
    resim[:, 400:600] = [255, 0, 0]  # Mavi
    
    # LAB'a dönüştür
    lab_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2LAB)
    
    # L, A, B kanallarını ayır
    L, A, B = cv2.split(lab_resim)
    
    # Görselleştir
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Orijinal (RGB)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(L, cmap='gray')
    axes[0, 1].set_title('L Kanalı (Parlaklık)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(A, cmap='RdYlGn_r')
    axes[1, 0].set_title('A Kanalı (Yeşil-Kırmızı)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(B, cmap='YlGnBu_r')
    axes[1, 1].set_title('B Kanalı (Mavi-Sarı)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("LAB renk uzayı kanalları:")
    print(f"L (Parlaklık) aralığı: {np.min(L)} - {np.max(L)}")
    print(f"A (Yeşil-Kırmızı) aralığı: {np.min(A)} - {np.max(A)}")
    print(f"B (Mavi-Sarı) aralığı: {np.min(B)} - {np.max(B)}")

def yuv_renk_uzayi():
    """YUV renk uzayı örnekleri"""
    print("\n📺 YUV Renk Uzayı")
    print("=" * 25)
    
    # Test resmi oluştur
    resim = np.zeros((200, 600, 3), dtype=np.uint8)
    
    # Gradyan oluştur
    for i in range(600):
        r = int(255 * i / 600)
        g = int(255 * (1 - i / 600))
        b = 128
        resim[:, i] = [b, g, r]  # BGR formatı
    
    # YUV'ye dönüştür
    yuv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2YUV)
    
    # Y, U, V kanallarını ayır
    Y, U, V = cv2.split(yuv_resim)
    
    # Görselleştir
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Orijinal (RGB)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(Y, cmap='gray')
    axes[0, 1].set_title('Y Kanalı (Parlaklık)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(U, cmap='coolwarm')
    axes[1, 0].set_title('U Kanalı (Mavi Farkı)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(V, cmap='coolwarm')
    axes[1, 1].set_title('V Kanalı (Kırmızı Farkı)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def interaktif_renk_secici():
    """İnteraktif HSV renk seçici"""
    print("\n🎨 İnteraktif HSV Renk Seçici")
    print("=" * 40)
    
    # Test resmi oluştur
    test_resim = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Çeşitli renkler ekle
    cv2.rectangle(test_resim, (50, 50), (150, 100), (0, 0, 255), -1)    # Kırmızı
    cv2.rectangle(test_resim, (200, 50), (350, 100), (0, 255, 0), -1)   # Yeşil
    cv2.rectangle(test_resim, (50, 150), (150, 200), (255, 0, 0), -1)   # Mavi
    cv2.rectangle(test_resim, (200, 150), (350, 200), (0, 255, 255), -1) # Sarı
    cv2.rectangle(test_resim, (125, 225), (225, 275), (255, 0, 255), -1) # Magenta
    
    hsv_resim = cv2.cvtColor(test_resim, cv2.COLOR_BGR2HSV)
    
    # Trackbar'lar için pencere oluştur
    cv2.namedWindow('HSV Renk Secici')
    cv2.namedWindow('Sonuc')
    
    # Trackbar'ları oluştur
    cv2.createTrackbar('H Min', 'HSV Renk Secici', 0, 179, lambda x: None)
    cv2.createTrackbar('H Max', 'HSV Renk Secici', 179, 179, lambda x: None)
    cv2.createTrackbar('S Min', 'HSV Renk Secici', 0, 255, lambda x: None)
    cv2.createTrackbar('S Max', 'HSV Renk Secici', 255, 255, lambda x: None)
    cv2.createTrackbar('V Min', 'HSV Renk Secici', 0, 255, lambda x: None)
    cv2.createTrackbar('V Max', 'HSV Renk Secici', 255, 255, lambda x: None)
    
    print("🎛️ Trackbar'ları kullanarak renk aralığını ayarlayın.")
    print("ESC tuşu ile çıkış yapın.")
    
    while True:
        # Trackbar değerlerini al
        h_min = cv2.getTrackbarPos('H Min', 'HSV Renk Secici')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Renk Secici')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Renk Secici')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Renk Secici')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Renk Secici')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Renk Secici')
        
        # Alt ve üst sınırları tanımla
        alt_sinir = np.array([h_min, s_min, v_min])
        ust_sinir = np.array([h_max, s_max, v_max])
        
        # Maske oluştur
        maske = cv2.inRange(hsv_resim, alt_sinir, ust_sinir)
        
        # Maskeyi uygula
        sonuc = cv2.bitwise_and(test_resim, test_resim, mask=maske)
        
        # Göster
        cv2.imshow('HSV Renk Secici', test_resim)
        cv2.imshow('Sonuc', sonuc)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

def renk_histogrami():
    """Renk histogramı analizi"""
    print("\n📊 Renk Histogramı Analizi")
    print("=" * 35)
    
    # Test resmi oluştur
    resim = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    
    # Renkli bölgeler ekle
    cv2.rectangle(resim, (50, 50), (150, 150), (0, 0, 255), -1)    # Kırmızı
    cv2.rectangle(resim, (200, 50), (300, 150), (0, 255, 0), -1)   # Yeşil
    cv2.rectangle(resim, (125, 200), (225, 300), (255, 0, 0), -1)  # Mavi
    
    # BGR kanallarını ayır
    b, g, r = cv2.split(resim)
    
    # Histogram hesapla
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    
    # Görselleştir
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Orijinal resim
    axes[0, 0].imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Orijinal Resim')
    axes[0, 0].axis('off')
    
    # BGR histogramları
    axes[0, 1].plot(hist_b, color='blue', label='Mavi')
    axes[0, 1].plot(hist_g, color='green', label='Yeşil')
    axes[0, 1].plot(hist_r, color='red', label='Kırmızı')
    axes[0, 1].set_title('BGR Histogramı')
    axes[0, 1].set_xlabel('Piksel Değeri')
    axes[0, 1].set_ylabel('Frekans')
    axes[0, 1].legend()
    
    # HSV histogramı
    hsv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_resim], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv_resim], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_resim], [2], None, [256], [0, 256])
    
    axes[1, 0].plot(hist_h, color='orange', label='Hue')
    axes[1, 0].set_title('Hue Histogramı')
    axes[1, 0].set_xlabel('Hue Değeri (0-179)')
    axes[1, 0].set_ylabel('Frekans')
    
    axes[1, 1].plot(hist_s, color='purple', label='Saturation')
    axes[1, 1].plot(hist_v, color='gray', label='Value')
    axes[1, 1].set_title('Saturation & Value Histogramı')
    axes[1, 1].set_xlabel('Değer (0-255)')
    axes[1, 1].set_ylabel('Frekans')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def renk_uzayi_karsilastirma():
    """Farklı renk uzaylarını karşılaştırma"""
    print("\n⚖️ Renk Uzayları Karşılaştırması")
    print("=" * 40)
    
    # Test resmi oluştur
    resim = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # Gradyan oluştur
    for i in range(400):
        for j in range(200):
            resim[j, i] = [
                int(255 * i / 400),           # Mavi gradyan
                int(255 * j / 200),           # Yeşil gradyan  
                int(255 * (1 - i / 400))      # Kırmızı ters gradyan
            ]
    
    # Farklı renk uzaylarına dönüştür
    rgb_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
    hsv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
    lab_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2LAB)
    yuv_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2YUV)
    
    # Görselleştir
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(rgb_resim)
    axes[0, 0].set_title('RGB (Orijinal)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(hsv_resim)
    axes[0, 1].set_title('HSV')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(lab_resim)
    axes[0, 2].set_title('LAB')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(yuv_resim)
    axes[1, 0].set_title('YUV')
    axes[1, 0].axis('off')
    
    # Gri tonlama karşılaştırması
    gri_normal = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    gri_hsv = hsv_resim[:, :, 2]  # V kanalı
    
    axes[1, 1].imshow(gri_normal, cmap='gray')
    axes[1, 1].set_title('Gri Tonlama (Normal)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(gri_hsv, cmap='gray')
    axes[1, 2].set_title('HSV Value Kanalı')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """Ana program fonksiyonu"""
    print("🌈 OpenCV Renk Uzayları")
    print("Bu program, farklı renk uzaylarını ve dönüşümlerini gösterir.\\n")
    
    # Ana menü
    while True:
        print("\\n" + "="*50)
        print("🌈 OpenCV Renk Uzayları Menüsü")
        print("="*50)
        print("1. Renk Uzayları Açıklama")
        print("2. BGR ↔ RGB Dönüşümü")
        print("3. HSV Renk Uzayı")
        print("4. HSV ile Renk Filtreleme")
        print("5. LAB Renk Uzayı")
        print("6. YUV Renk Uzayı")
        print("7. İnteraktif HSV Renk Seçici")
        print("8. Renk Histogramı Analizi")
        print("9. Renk Uzayları Karşılaştırması")
        print("0. Çıkış")
        
        secim = input("\\nLütfen bir seçenek girin (0-9): ").strip()
        
        if secim == "0":
            print("👋 Görüşürüz!")
            break
        elif secim == "1":
            renk_uzaylari_aciklama()
        elif secim == "2":
            bgr_rgb_donusumu()
        elif secim == "3":
            hsv_renk_uzayi()
        elif secim == "4":
            renk_filtreleme_hsv()
        elif secim == "5":
            lab_renk_uzayi()
        elif secim == "6":
            yuv_renk_uzayi()
        elif secim == "7":
            interaktif_renk_secici()
        elif secim == "8":
            renk_histogrami()
        elif secim == "9":
            renk_uzayi_karsilastirma()
        else:
            print("❌ Geçersiz seçim! Lütfen 0-9 arasında bir sayı girin.")

if __name__ == "__main__":
    main()