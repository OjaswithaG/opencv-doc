#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ Çözüm 1: Temel Kurulum ve Kontrol
===================================

Bu dosya, Alıştırma 1'in örnek çözümlerini içerir.
Kendi çözümünüzle karşılaştırarak öğrenebilirsiniz.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import sys

def gorev_1_kurulum_kontrolu():
    """
    ✅ ÇÖZÜM 1: OpenCV Kurulum Kontrolü
    """
    print("🎯 GÖREV 1: Kurulum Kontrolü")
    print("-" * 30)
    
    # Sürüm bilgilerini yazdır
    print(f"OpenCV Sürümü: {cv2.__version__}")
    print(f"NumPy Sürümü: {np.__version__}")
    print(f"Python Sürümü: {sys.version.split()[0]}")
    print("✅ Kurulum başarılı!")

def gorev_2_basit_resim_olusturma():
    """
    ✅ ÇÖZÜM 2: Basit Resim Oluşturma
    """
    print("\\n🎯 GÖREV 2: Basit Resim Oluşturma")
    print("-" * 35)
    
    # 200x300 piksel, 3 kanallı siyah resim
    resim = np.zeros((200, 300, 3), dtype=np.uint8)
    
    # Resim bilgilerini yazdır
    print(f"Resim şekli: {resim.shape}")
    print(f"Veri tipi: {resim.dtype}")
    
    # Resmi göster
    cv2.imshow('Siyah Resim', resim)
    print("Resim gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gorev_3_renkli_resim_olusturma():
    """
    ✅ ÇÖZÜM 3: Renkli Resim Oluşturma
    """
    print("\\n🎯 GÖREV 3: Renkli Resim Oluşturma")
    print("-" * 35)
    
    # 300x400 piksel beyaz resim
    resim = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Sol yarısını kırmızı yap (BGR formatında: 0,0,255)
    resim[:, :200] = [0, 0, 255]
    
    # Sağ yarısını mavi yap (BGR formatında: 255,0,0)
    resim[:, 200:] = [255, 0, 0]
    
    cv2.imshow('Renkli Resim', resim)
    print("Renkli resim gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gorev_4_geometrik_sekiller():
    """
    ✅ ÇÖZÜM 4: Geometrik Şekiller Çizme
    """
    print("\\n🎯 GÖREV 4: Geometrik Şekiller")
    print("-" * 30)
    
    # 400x400 piksel siyah resim
    resim = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Merkezde yeşil dolu daire (yarıçap: 50)
    merkez = (200, 200)
    cv2.circle(resim, merkez, 50, (0, 255, 0), -1)
    
    # Sol üst köşede kırmızı dolu dikdörtgen (50x50)
    cv2.rectangle(resim, (50, 50), (100, 100), (0, 0, 255), -1)
    
    # Sağ alt köşede beyaz çizgi (köşegen)
    cv2.line(resim, (300, 300), (350, 350), (255, 255, 255), 3)
    
    cv2.imshow('Geometrik Sekiller', resim)
    print("Geometrik şekiller gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gorev_5_metin_ekleme():
    """
    ✅ ÇÖZÜM 5: Metin Ekleme
    """
    print("\\n🎯 GÖREV 5: Metin Ekleme")
    print("-" * 25)
    
    # 300x500 piksel beyaz resim
    resim = np.ones((300, 500, 3), dtype=np.uint8) * 255
    
    # Font ayarları
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Üst kısma "OpenCV" (kırmızı)
    cv2.putText(resim, 'OpenCV', (150, 80), font, 2, (0, 0, 255), 3)
    
    # Orta kısma "Alıştırma 1" (mavi)
    cv2.putText(resim, 'Alistirma 1', (120, 150), font, 1.5, (255, 0, 0), 2)
    
    # Alt kısma "Tamamlandı!" (yeşil)  
    cv2.putText(resim, 'Tamamlandi!', (130, 220), font, 1.2, (0, 255, 0), 2)
    
    cv2.imshow('Metin Resim', resim)
    print("Metinli resim gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gorev_6_interaktif_pencere():
    """
    ✅ ÇÖZÜM 6: İnteraktif Pencere
    """
    print("\\n🎯 GÖREV 6: İnteraktif Pencere")
    print("-" * 30)
    print("Kontroller:")
    print("r: Kırmızı")
    print("g: Yeşil") 
    print("b: Mavi")
    print("ESC: Çıkış")
    
    # 250x400 piksel gri resim
    resim = np.ones((250, 400, 3), dtype=np.uint8) * 128
    
    cv2.namedWindow('Interaktif Pencere')
    
    while True:
        cv2.imshow('Interaktif Pencere', resim)
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC tuşu
            print("Çıkış yapılıyor...")
            break
        elif key == ord('r'):
            resim[:] = [0, 0, 255]  # Kırmızı
            print("Renk: Kırmızı")
        elif key == ord('g'):
            resim[:] = [0, 255, 0]  # Yeşil
            print("Renk: Yeşil")
        elif key == ord('b'):
            resim[:] = [255, 0, 0]  # Mavi
            print("Renk: Mavi")
    
    cv2.destroyAllWindows()

def bonus_gorev_sanat_eseri():
    """
    🎨 BONUS ÇÖZÜM: Sanat Eseri Oluşturma
    """
    print("\\n🎨 BONUS GÖREV: Sanat Eseri")
    print("-" * 30)
    
    # 500x600 piksel siyah tuval
    tuval = np.zeros((500, 600, 3), dtype=np.uint8)
    
    # Arka plan gradyanı
    for i in range(600):
        renk_degeri = int(50 + (i / 600) * 100)
        tuval[:, i] = [renk_degeri, 20, 80]
    
    # Güneş
    cv2.circle(tuval, (150, 120), 40, (0, 255, 255), -1)
    
    # Dağlar
    daglar = np.array([[0, 350], [150, 250], [300, 280], [450, 200], [600, 300], [600, 500], [0, 500]], np.int32)
    cv2.fillPoly(tuval, [daglar], (100, 150, 50))
    
    # Ağaçlar
    for x in [100, 250, 400, 520]:
        # Gövde
        cv2.rectangle(tuval, (x-10, 350), (x+10, 400), (50, 100, 20), -1)
        # Yapraklar
        cv2.circle(tuval, (x, 330), 25, (0, 150, 50), -1)
    
    # Bulutlar
    cv2.ellipse(tuval, (400, 100), (60, 30), 0, 0, 360, (200, 200, 200), -1)
    cv2.ellipse(tuval, (450, 80), (40, 20), 0, 0, 360, (220, 220, 220), -1)
    
    # Kuşlar (basit V şekli)
    for i, x in enumerate([300, 350, 320]):
        y = 150 + i * 5
        cv2.line(tuval, (x, y), (x+10, y-5), (100, 100, 100), 2)
        cv2.line(tuval, (x+10, y-5), (x+20, y), (100, 100, 100), 2)
    
    # İmza
    cv2.putText(tuval, 'OpenCV Art 2024', (20, 480), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Sanat Eseri', tuval)
    print("Sanat eseri gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Sanat eserini kaydet
    cv2.imwrite('sanat_eseri.png', tuval)
    print("✅ Sanat eseri 'sanat_eseri.png' olarak kaydedildi!")

def main():
    """Ana çözüm programı"""
    print("✅ OpenCV Alıştırma 1 - ÇÖZÜMLER")
    print("=" * 40)
    print("Bu dosya, alıştırmanın örnek çözümlerini içerir.\\n")
    
    try:
        # Çözümleri sırayla göster
        gorev_1_kurulum_kontrolu()
        gorev_2_basit_resim_olusturma()
        gorev_3_renkli_resim_olusturma()
        gorev_4_geometrik_sekiller()
        gorev_5_metin_ekleme()
        gorev_6_interaktif_pencere()
        
        # Bonus görev
        bonus_cevap = input("\\nBonus çözümü görmek ister misiniz? (e/h): ")
        if bonus_cevap.lower() == 'e':
            bonus_gorev_sanat_eseri()
        
        print("\\n🎉 Tüm çözümler gösterildi!")
        print("\\n📚 Öğrenme Notları:")
        print("   - BGR renk formatına dikkat edin (OpenCV'nin varsayılanı)")
        print("   - Koordinat sistemi: (x, y) ama array indexing: [y, x]")
        print("   - cv2.waitKey(0) sonsuz bekleme, cv2.waitKey(1) 1ms bekleme")
        print("   - Kalınlık -1 ile dolu şekil çizimi")
        print("   - cv2.destroyAllWindows() ile tüm pencereleri kapat")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    main()

# 📝 ÇÖZÜM NOTLARI:
# 1. Bu çözümler örnek niteliğindedir
# 2. Farklı yaklaşımlar da geçerlidir  
# 3. Önemli olan kavramları anlamaktır
# 4. Kendi yaratıcı çözümlerinizi de deneyin
# 5. Parametrelerle oynayarak öğrenin