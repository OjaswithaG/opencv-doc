#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 İlk OpenCV Programınız
========================

Bu dosya, OpenCV ile ilk programınızı yazmak için hazırlanmıştır.
Adım adım ilerleyerek temel OpenCV işlemlerini öğreneceksiniz.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import sys

def opencv_surumu_kontrol():
    """OpenCV sürüm bilgisini gösterir"""
    print("=" * 50)
    print("🔍 OpenCV Sürüm Kontrolü")
    print("=" * 50)
    print(f"OpenCV Sürümü: {cv2.__version__}")
    print(f"NumPy Sürümü: {np.__version__}")
    print(f"Python Sürümü: {sys.version}")
    print("=" * 50)

def basit_resim_olustur():
    """Basit bir resim oluşturur ve gösterir"""
    print("📸 Basit resim oluşturuluyor...")
    
    # 300x400 piksel, 3 kanallı (BGR) siyah resim
    resim = np.zeros((300, 400, 3), dtype=np.uint8)
    
    print(f"Resim boyutu: {resim.shape}")
    print(f"Resim veri tipi: {resim.dtype}")
    
    # Resmi göster
    cv2.imshow('Siyah Resim', resim)
    print("Resim gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)  # Tuş basılmasını bekle
    cv2.destroyAllWindows()

def renkli_resim_olustur():
    """Renkli bir resim oluşturur"""
    print("🌈 Renkli resim oluşturuluyor...")
    
    # Resim boyutları
    yukseklik, genislik = 300, 400
    resim = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)
    
    # BGR renk değerleri (OpenCV BGR formatı kullanır)
    mavi = (255, 0, 0)
    yesil = (0, 255, 0)
    kirmizi = (0, 0, 255)
    
    # Resmi üç parçaya böl ve renkleri ata
    resim[:100, :] = mavi      # Üst kısım mavi
    resim[100:200, :] = yesil  # Orta kısım yeşil
    resim[200:, :] = kirmizi   # Alt kısım kırmızı
    
    cv2.imshow('Renkli Resim', resim)
    print("Renkli resim gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def metin_ekle():
    """Resme metin ekler"""
    print("✍️ Resme metin ekleniyor...")
    
    # Beyaz arka plan oluştur
    resim = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Metin özellikleri
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_olcegi = 1
    renk = (0, 0, 255)  # Kırmızı (BGR)
    kalinlik = 2
    
    # Metinler ekle
    cv2.putText(resim, 'Merhaba OpenCV!', (100, 150), 
                font, font_olcegi, renk, kalinlik)
    
    cv2.putText(resim, 'Ilk Programim', (150, 200), 
                font, 0.8, (0, 255, 0), 2)
    
    cv2.putText(resim, 'OpenCV 2024', (200, 250), 
                font, 0.6, (255, 0, 0), 2)
    
    cv2.imshow('Metin Eklenmis Resim', resim)
    print("Metinli resim gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def geometrik_sekiller():
    """Temel geometrik şekiller çizer"""
    print("🔷 Geometrik şekiller çiziliyor...")
    
    # Siyah arka plan
    resim = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Dikdörtgen çiz (sol üst köşe, sağ alt köşe, renk, kalınlık)
    cv2.rectangle(resim, (50, 50), (200, 150), (0, 255, 0), 3)
    
    # Dolu dikdörtgen
    cv2.rectangle(resim, (250, 50), (400, 150), (255, 0, 0), -1)
    
    # Çember çiz (merkez, yarıçap, renk, kalınlık)
    cv2.circle(resim, (125, 250), 50, (0, 0, 255), 3)
    
    # Dolu çember
    cv2.circle(resim, (325, 250), 50, (255, 255, 0), -1)
    
    # Çizgi çiz (başlangıç noktası, bitiş noktası, renk, kalınlık)
    cv2.line(resim, (450, 50), (550, 150), (255, 255, 255), 3)
    
    # Elips çiz
    cv2.ellipse(resim, (125, 350), (75, 50), 0, 0, 360, (255, 0, 255), 2)
    
    cv2.imshow('Geometrik Sekiller', resim)
    print("Geometrik şekiller gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mouse_olaylari():
    """Mouse olaylarını yakalar"""
    print("🖱️ Mouse olayları dinleniyor...")
    
    # Global değişkenler
    global resim, cizim_modu, ix, iy
    
    # Beyaz arka plan
    resim = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cizim_modu = False
    ix, iy = -1, -1
    
    def mouse_callback(event, x, y, flags, param):
        global resim, cizim_modu, ix, iy
        
        if event == cv2.EVENT_LBUTTONDOWN:
            cizim_modu = True
            ix, iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if cizim_modu:
                cv2.circle(resim, (x, y), 5, (0, 0, 255), -1)
                
        elif event == cv2.EVENT_LBUTTONUP:
            cizim_modu = False
            cv2.circle(resim, (x, y), 5, (0, 0, 255), -1)
    
    # Mouse callback'ini ayarla
    cv2.namedWindow('Mouse Cizimi')
    cv2.setMouseCallback('Mouse Cizimi', mouse_callback)
    
    print("Mouse ile çizim yapabilirsiniz. ESC tuşu ile çıkış yapın.")
    
    while True:
        cv2.imshow('Mouse Cizimi', resim)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC tuşu
            break
        elif key == ord('c'):  # 'c' tuşu ile temizle
            resim = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    cv2.destroyAllWindows()

def klavye_kontrolleri():
    """Klavye kontrollerini gösterir"""
    print("⌨️ Klavye kontrolleri...")
    
    resim = np.zeros((300, 500, 3), dtype=np.uint8)
    
    # Yardım metnini ekle
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resim, 'Klavye Kontrolleri:', (50, 50), font, 0.7, (255, 255, 255), 2)
    cv2.putText(resim, 'r: Kirmizi', (50, 100), font, 0.5, (0, 0, 255), 1)
    cv2.putText(resim, 'g: Yesil', (50, 130), font, 0.5, (0, 255, 0), 1)
    cv2.putText(resim, 'b: Mavi', (50, 160), font, 0.5, (255, 0, 0), 1)
    cv2.putText(resim, 'ESC: Cikis', (50, 190), font, 0.5, (255, 255, 255), 1)
    
    print("Renk değiştirmek için r, g, b tuşlarını kullanın. ESC ile çıkış.")
    
    while True:
        cv2.imshow('Klavye Kontrolleri', resim)
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Kırmızı
            resim[:] = (0, 0, 255)
            cv2.putText(resim, 'KIRMIZI', (200, 150), font, 1, (255, 255, 255), 2)
        elif key == ord('g'):  # Yeşil
            resim[:] = (0, 255, 0)
            cv2.putText(resim, 'YESIL', (200, 150), font, 1, (0, 0, 0), 2)
        elif key == ord('b'):  # Mavi
            resim[:] = (255, 0, 0)
            cv2.putText(resim, 'MAVI', (200, 150), font, 1, (255, 255, 255), 2)
    
    cv2.destroyAllWindows()

def main():
    """Ana program fonksiyonu"""
    print("🎉 OpenCV'ye Hoş Geldiniz!")
    print("Bu program, OpenCV'nin temel özelliklerini gösterir.\n")
    
    # Programlar listesi
    programlar = [
        ("1", "OpenCV Sürüm Kontrolü", opencv_surumu_kontrol),
        ("2", "Basit Resim Oluşturma", basit_resim_olustur),
        ("3", "Renkli Resim Oluşturma", renkli_resim_olustur),
        ("4", "Metin Ekleme", metin_ekle),
        ("5", "Geometrik Şekiller", geometrik_sekiller),
        ("6", "Mouse Olayları", mouse_olaylari),
        ("7", "Klavye Kontrolleri", klavye_kontrolleri),
        ("0", "Çıkış", None)
    ]
    
    while True:
        print("\n" + "="*50)
        print("🚀 OpenCV İlk Program Menüsü")
        print("="*50)
        
        for numara, baslik, _ in programlar:
            print(f"{numara}. {baslik}")
        
        secim = input("\nLütfen bir seçenek girin (0-7): ").strip()
        
        if secim == "0":
            print("👋 Görüşürüz!")
            break
        
        # Seçilen programı çalıştır
        for numara, baslik, fonksiyon in programlar:
            if secim == numara and fonksiyon:
                print(f"\n🚀 {baslik} çalıştırılıyor...")
                try:
                    fonksiyon()
                except Exception as e:
                    print(f"❌ Hata oluştu: {e}")
                break
        else:
            print("❌ Geçersiz seçim! Lütfen 0-7 arasında bir sayı girin.")

if __name__ == "__main__":
    main()