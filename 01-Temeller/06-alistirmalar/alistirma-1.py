#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Alıştırma 1: Temel Kurulum ve Kontrol
=======================================

Bu alıştırma OpenCV'nin temel kurulum ve kontrol işlemlerini içerir.
Her görevi tamamladıktan sonra sonucu kontrol edin.

Zorluk Seviyesi: 🔰 Başlangıç
Tahmini Süre: 15-20 dakika
"""

import cv2
import numpy as np

def gorev_1_kurulum_kontrolu():
    """
    🎯 GÖREV 1: OpenCV Kurulum Kontrolü
    
    Yapılacaklar:
    1. OpenCV sürümünü yazdırın
    2. NumPy sürümünü yazdırın  
    3. Python sürümünü yazdırın
    4. Kurulumun başarılı olduğunu belirten bir mesaj yazdırın
    
    Beklenen Çıktı:
    OpenCV Sürümü: 4.x.x
    NumPy Sürümü: 1.x.x
    Python Sürümü: 3.x.x
    ✅ Kurulum başarılı!
    """
    print("🎯 GÖREV 1: Kurulum Kontrolü")
    print("-" * 30)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.__version__, np.__version__, sys.version kullanın
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_2_basit_resim_olusturma():
    """
    🎯 GÖREV 2: Basit Resim Oluşturma
    
    Yapılacaklar:
    1. 200x300 piksel boyutunda siyah bir resim oluşturun
    2. Resmin şekil bilgisini yazdırın  
    3. Resmin veri tipini yazdırın
    4. Resmi 'siyah_resim' ismiyle gösterin
    
    Beklenen Çıktı:
    Resim şekli: (200, 300, 3)
    Veri tipi: uint8
    """
    print("\\n🎯 GÖREV 2: Basit Resim Oluşturma")
    print("-" * 35)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: np.zeros() kullanın, dtype=np.uint8 belirtin
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_3_renkli_resim_olusturma():
    """
    🎯 GÖREV 3: Renkli Resim Oluşturma
    
    Yapılacaklar:
    1. 300x400 piksel boyutunda beyaz bir resim oluşturun
    2. Sol yarısını kırmızı (BGR: 0,0,255) yapın
    3. Sağ yarısını mavi (BGR: 255,0,0) yapın
    4. Resmi 'renkli_resim' ismiyle gösterin
    
    Beklenen Sonuç: Sol yarısı kırmızı, sağ yarısı mavi resim
    """
    print("\\n🎯 GÖREV 3: Renkli Resim Oluşturma")
    print("-" * 35)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: np.ones() * 255 ile beyaz resim, indeksleme ile renk atama
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_4_geometrik_sekiller():
    """
    🎯 GÖREV 4: Geometrik Şekiller Çizme
    
    Yapılacaklar:
    1. 400x400 piksel boyutunda siyah bir resim oluşturun
    2. Merkezde yeşil dolu bir daire çizin (yarıçap: 50)
    3. Sol üst köşede kırmızı dolu bir dikdörtgen çizin (50x50)
    4. Sağ alt köşede beyaz bir çizgi çizin (köşegen)
    5. Resmi 'geometrik_sekiller' ismiyle gösterin
    
    Fonksiyonlar: cv2.circle(), cv2.rectangle(), cv2.line()
    """
    print("\\n🎯 GÖREV 4: Geometrik Şekiller")
    print("-" * 30)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.circle(resim, merkez, yarıçap, renk, kalınlık)
    #        kalınlık=-1 dolu şekil çizer
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_5_metin_ekleme():
    """
    🎯 GÖREV 5: Metin Ekleme
    
    Yapılacaklar:
    1. 300x500 piksel boyutunda beyaz bir resim oluşturun
    2. Üst kısma "OpenCV" yazsın (kırmızı renk)
    3. Orta kısma "Alıştırma 1" yazsın (mavi renk)  
    4. Alt kısma "Tamamlandı!" yazsın (yeşil renk)
    5. Resmi 'metin_resim' ismiyle gösterin
    
    Fonksiyon: cv2.putText()
    """
    print("\\n🎯 GÖREV 5: Metin Ekleme")
    print("-" * 25)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.putText(resim, metin, konum, font, ölçek, renk, kalınlık)
    #        font = cv2.FONT_HERSHEY_SIMPLEX
    
    pass  # Bu satırı silin ve kodunuzu yazın

def gorev_6_interaktif_pencere():
    """
    🎯 GÖREV 6: İnteraktif Pencere
    
    Yapılacaklar:
    1. 250x400 piksel boyutunda gri bir resim oluşturun (değer: 128)
    2. Resmi gösterin
    3. Herhangi bir tuşa basılmasını bekleyin
    4. 'r' tuşuna basılırsa resmi kırmızı yapın
    5. 'g' tuşuna basılırsa resmi yeşil yapın
    6. 'b' tuşuna basılırsa resmi mavi yapın
    7. ESC tuşuna basılırsa çıkış yapın
    
    İpucu: while döngüsü ve cv2.waitKey() kullanın
    """
    print("\\n🎯 GÖREV 6: İnteraktif Pencere")
    print("-" * 30)
    print("Kontroller:")
    print("r: Kırmızı")
    print("g: Yeşil") 
    print("b: Mavi")
    print("ESC: Çıkış")
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: while True döngüsü kullanın
    #        key = cv2.waitKey(0) & 0xFF ile tuş kontrolü
    #        key == 27 ESC tuşu kontrolü
    
    pass  # Bu satırı silin ve kodunuzu yazın

def bonus_gorev_sanat_eseri():
    """
    🎨 BONUS GÖREV: Sanat Eseri Oluşturma
    
    Yaratıcılığınızı konuşturun! 
    OpenCV fonksiyonlarını kullanarak özgün bir sanat eseri oluşturun.
    
    Kullanabileceğiniz fonksiyonlar:
    - cv2.circle(), cv2.rectangle(), cv2.line()
    - cv2.ellipse(), cv2.polylines()
    - cv2.putText()
    - Farklı renkler ve şekiller
    
    En az 5 farklı geometrik şekil kullanın!
    """
    print("\\n🎨 BONUS GÖREV: Sanat Eseri")
    print("-" * 30)
    print("Yaratıcılığınızı konuşturun!")
    
    # TODO: Buraya yaratıcı kodunuzu yazın
    # Kendi sanat eserinizi oluşturun!
    
    pass  # Bu satırı silin ve kodunuzu yazın

def main():
    """Ana program - tüm görevleri çalıştırır"""
    print("🎯 OpenCV Alıştırma 1: Temel Kurulum ve Kontrol")
    print("=" * 55)
    print("Bu alıştırmada OpenCV'nin temel kullanımını öğreneceksiniz.\\n")
    
    try:
        # Görevleri sırayla çalıştır
        gorev_1_kurulum_kontrolu()
        gorev_2_basit_resim_olusturma()
        gorev_3_renkli_resim_olusturma()
        gorev_4_geometrik_sekiller()
        gorev_5_metin_ekleme()
        gorev_6_interaktif_pencere()
        
        # Bonus görev (opsiyonel)
        bonus_cevap = input("\\nBonus görevi yapmak ister misiniz? (e/h): ")
        if bonus_cevap.lower() == 'e':
            bonus_gorev_sanat_eseri()
        
        print("\\n🎉 Tebrikler! Alıştırma 1'i tamamladınız!")
        print("✅ Öğrendikleriniz:")
        print("   - OpenCV kurulum kontrolü")
        print("   - Basit resim oluşturma")
        print("   - Geometrik şekiller çizme")
        print("   - Metin ekleme")
        print("   - İnteraktif pencere kontrolü")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        print("💡 İpucu: Hata mesajını dikkatlice okuyun ve kodunuzu kontrol edin.")

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Her görevde 'pass' satırını silip kendi kodunuzu yazın
# 2. Hata alırsanız hata mesajını dikkatlice okuyun
# 3. İpuçlarını takip edin
# 4. Çözüm için cozumler/cozum-1.py dosyasına bakabilirsiniz
# 5. Kendi yaratıcı fikirlerinizi de ekleyebilirsiniz!