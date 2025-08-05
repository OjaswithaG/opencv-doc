#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📸 OpenCV Resim İşlemleri
========================

Bu dosya, OpenCV ile temel resim işlemlerini öğretir:
- Resim okuma, gösterme ve kaydetme
- Farklı dosya formatları ile çalışma
- Resim özelliklerini öğrenme
- Temel manipülasyonlar

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import os
from pathlib import Path

def resim_olustur_ve_kaydet():
    """Örnek resimler oluşturur ve kaydeder"""
    print("🎨 Örnek resimler oluşturuluyor...")
    
    # Examples klasörünü oluştur
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 1. Renkli gradyan resmi
    yukseklik, genislik = 300, 400
    gradyan = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)
    
    for i in range(genislik):
        r = int(255 * i / genislik)
        g = int(255 * (1 - i / genislik))
        b = 128
        gradyan[:, i] = [b, g, r]
    
    cv2.imwrite(str(examples_dir / "gradyan.jpg"), gradyan)
    
    # 2. Geometrik şekiller resmi
    sekiller = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Dikdörtgenler
    cv2.rectangle(sekiller, (50, 50), (150, 150), (0, 255, 0), -1)
    cv2.rectangle(sekiller, (200, 50), (350, 150), (255, 0, 0), 3)
    
    # Çemberler
    cv2.circle(sekiller, (100, 250), 50, (0, 0, 255), -1)
    cv2.circle(sekiller, (300, 250), 50, (255, 255, 0), 5)
    
    # Çizgiler
    cv2.line(sekiller, (50, 350), (350, 350), (255, 255, 255), 3)
    
    cv2.imwrite(str(examples_dir / "sekiller.png"), sekiller)
    
    # 3. Metin içeren resim
    metin_resim = np.ones((200, 500, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(metin_resim, 'OpenCV Resim Islemleri', (50, 80), 
                font, 0.8, (0, 0, 0), 2)
    cv2.putText(metin_resim, 'Ornek Metin Resmi', (100, 130), 
                font, 0.6, (255, 0, 0), 2)
    
    cv2.imwrite(str(examples_dir / "metin.bmp"), metin_resim)
    
    print(f"✅ Örnek resimler '{examples_dir}' klasörüne kaydedildi!")
    return examples_dir

def resim_okuma_ornekleri(examples_dir):
    """Farklı yöntemlerle resim okuma örnekleri"""
    print("\n📖 Resim Okuma Örnekleri")
    print("=" * 40)
    
    # Örnek resim yolu
    resim_yolu = examples_dir / "gradyan.jpg"
    
    if not resim_yolu.exists():
        print("❌ Örnek resim bulunamadı!")
        return None
    
    # 1. Normal okuma (renkli)
    resim_renkli = cv2.imread(str(resim_yolu))
    print(f"Renkli resim şekli: {resim_renkli.shape}")
    
    # 2. Gri tonlama olarak okuma
    resim_gri = cv2.imread(str(resim_yolu), cv2.IMREAD_GRAYSCALE)
    print(f"Gri resim şekli: {resim_gri.shape}")
    
    # 3. Alfa kanallı okuma (varsa)
    resim_alfa = cv2.imread(str(resim_yolu), cv2.IMREAD_UNCHANGED)
    print(f"Alfa kanallı resim şekli: {resim_alfa.shape}")
    
    # Okuma modları açıklaması
    print("\\n📋 Okuma Modları:")
    print("cv2.IMREAD_COLOR (1): Renkli okuma (varsayılan)")
    print("cv2.IMREAD_GRAYSCALE (0): Gri tonlama okuma")  
    print("cv2.IMREAD_UNCHANGED (-1): Orijinal formatta okuma")
    
    return resim_renkli

def resim_bilgileri_goster(resim):
    """Resim hakkında detaylı bilgi gösterir"""
    if resim is None:
        print("❌ Resim yüklenemedi!")
        return
    
    print("\\n📊 Resim Bilgileri")
    print("=" * 30)
    print(f"Şekil (Boyut): {resim.shape}")
    print(f"Veri tipi: {resim.dtype}")
    print(f"Boyut sayısı: {resim.ndim}")
    print(f"Toplam piksel sayısı: {resim.size}")
    
    if len(resim.shape) == 3:
        yukseklik, genislik, kanal = resim.shape
        print(f"Yükseklik: {yukseklik} piksel")
        print(f"Genişlik: {genislik} piksel")
        print(f"Kanal sayısı: {kanal}")
    else:
        yukseklik, genislik = resim.shape
        print(f"Yükseklik: {yukseklik} piksel")
        print(f"Genişlik: {genislik} piksel")
        print("Kanal sayısı: 1 (gri tonlama)")
    
    # İstatistiksel bilgiler
    print(f"\\nMinimum piksel değeri: {np.min(resim)}")
    print(f"Maksimum piksel değeri: {np.max(resim)}")
    print(f"Ortalama piksel değeri: {np.mean(resim):.2f}")
    print(f"Standart sapma: {np.std(resim):.2f}")

def resim_gosterme_ornekleri(resim):
    """Farklı şekillerde resim gösterme"""
    if resim is None:
        return
    
    print("\\n👁️ Resim Gösterme Örnekleri")
    print("=" * 35)
    
    # 1. Normal gösterme
    cv2.imshow('Orijinal Resim', resim)
    print("Orijinal resim gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    
    # 2. Yeniden boyutlandırılmış gösterme
    kucuk_resim = cv2.resize(resim, (200, 150))
    cv2.imshow('Kucuk Resim', kucuk_resim)
    print("Küçültülmüş resim gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    
    # 3. Pencere boyutu ayarlanabilir
    cv2.namedWindow('Ayarlanabilir Pencere', cv2.WINDOW_NORMAL)
    cv2.imshow('Ayarlanabilir Pencere', resim)
    print("Boyutu ayarlanabilir pencere... ESC ile çıkış.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC tuşu
            break
    
    cv2.destroyAllWindows()

def resim_kaydetme_ornekleri(resim, examples_dir):
    """Farklı formatlarda resim kaydetme"""
    if resim is None:
        return
    
    print("\\n💾 Resim Kaydetme Örnekleri")
    print("=" * 35)
    
    # Çıktı klasörü
    output_dir = examples_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 1. JPEG formatında kaydetme (sıkıştırmalı)
    jpeg_yolu = output_dir / "output.jpg"
    kalite = 95  # 0-100 arası (100 en yüksek kalite)
    cv2.imwrite(str(jpeg_yolu), resim, [cv2.IMWRITE_JPEG_QUALITY, kalite])
    print(f"✅ JPEG formatında kaydedildi: {jpeg_yolu}")
    
    # 2. PNG formatında kaydetme (kayıpsız)
    png_yolu = output_dir / "output.png"
    sikistirma = 9  # 0-9 arası (9 en yüksek sıkıştırma)
    cv2.imwrite(str(png_yolu), resim, [cv2.IMWRITE_PNG_COMPRESSION, sikistirma])
    print(f"✅ PNG formatında kaydedildi: {png_yolu}")
    
    # 3. BMP formatında kaydetme (sıkıştırmasız)
    bmp_yolu = output_dir / "output.bmp"
    cv2.imwrite(str(bmp_yolu), resim)
    print(f"✅ BMP formatında kaydedildi: {bmp_yolu}")
    
    # 4. Gri tonlama olarak kaydetme
    gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    gri_yolu = output_dir / "output_gri.jpg"
    cv2.imwrite(str(gri_yolu), gri_resim)
    print(f"✅ Gri tonlama kaydedildi: {gri_yolu}")

def piksel_manipulasyonu(resim):
    """Piksel seviyesinde manipülasyon örnekleri"""
    if resim is None:
        return
    
    print("\\n🎯 Piksel Manipülasyonu")
    print("=" * 30)
    
    # Resmin bir kopyasını oluştur
    manipule_resim = resim.copy()
    
    # 1. Tek piksel değiştirme
    yukseklik, genislik = manipule_resim.shape[:2]
    
    # Merkezdeki pikseli kırmızı yap
    merkez_y, merkez_x = yukseklik // 2, genislik // 2
    manipule_resim[merkez_y, merkez_x] = [0, 0, 255]  # BGR formatında kırmızı
    
    # 2. Bölge manipülasyonu - üst sol köşeyi mavi yap
    manipule_resim[0:50, 0:50] = [255, 0, 0]  # Mavi
    
    # 3. Rastgele noktalar ekle
    for _ in range(100):
        y = np.random.randint(0, yukseklik)
        x = np.random.randint(0, genislik)
        manipule_resim[y, x] = [0, 255, 255]  # Sarı
    
    # 4. Çizgi çiz (manuel piksel işleme)
    for i in range(genislik):
        manipule_resim[yukseklik//4, i] = [255, 255, 255]  # Beyaz çizgi
    
    # Sonucu göster
    cv2.imshow('Orijinal', resim)
    cv2.imshow('Manipule Edilmis', manipule_resim)
    print("Orijinal ve manipüle edilmiş resimler gösteriliyor...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def roi_ornekleri(resim):
    """Region of Interest (İlgilenilen Bölge) örnekleri"""
    if resim is None:
        return
    
    print("\\n🎯 ROI (Region of Interest) Örnekleri")
    print("=" * 45)
    
    yukseklik, genislik = resim.shape[:2]
    
    # 1. Merkezi bölgeyi seç
    merkez_x, merkez_y = genislik // 2, yukseklik // 2
    roi_boyut = 100
    
    roi = resim[merkez_y-roi_boyut//2:merkez_y+roi_boyut//2,
                merkez_x-roi_boyut//2:merkez_x+roi_boyut//2]
    
    cv2.imshow('ROI - Merkez Bölge', roi)
    
    # 2. ROI'yi başka bir yere kopyala
    resim_kopya = resim.copy()
    
    # Sol üst köşeye yapıştır
    resim_kopya[0:roi_boyut, 0:roi_boyut] = roi
    
    # Sağ üst köşeye yapıştır
    resim_kopya[0:roi_boyut, genislik-roi_boyut:genislik] = roi
    
    cv2.imshow('ROI Kopyalanmis Resim', resim_kopya)
    
    # 3. ROI üzerinde işlem yap
    roi_islenmis = roi.copy()
    roi_islenmis = cv2.add(roi_islenmis, np.ones_like(roi) * 50)  # Parlaklık artır
    
    cv2.imshow('ROI - Islenmis', roi_islenmis)
    
    print("ROI örnekleri gösteriliyor... Herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dosya_formatlari_karsilastirma(resim, examples_dir):
    """Farklı dosya formatlarını karşılaştırma"""
    if resim is None:
        return
    
    print("\\n📊 Dosya Formatları Karşılaştırması")
    print("=" * 40)
    
    output_dir = examples_dir / "format_karsilastirma"
    output_dir.mkdir(exist_ok=True)
    
    formatlar = [
        ("jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
        ("png", [cv2.IMWRITE_PNG_COMPRESSION, 5]),
        ("bmp", []),
        ("tiff", [])
    ]
    
    print("Format\\tDosya Boyutu\\tKalite")
    print("-" * 35)
    
    for format_adi, parametreler in formatlar:
        dosya_yolu = output_dir / f"test.{format_adi}"
        
        try:
            # Resmi kaydet
            cv2.imwrite(str(dosya_yolu), resim, parametreler)
            
            # Dosya boyutunu al
            dosya_boyutu = os.path.getsize(dosya_yolu)
            boyut_kb = dosya_boyutu / 1024
            
            # Kalite değerlendirmesi (basit)
            okunan_resim = cv2.imread(str(dosya_yolu))
            if okunan_resim is not None:
                fark = cv2.absdiff(resim, okunan_resim)
                ortalama_fark = np.mean(fark)
                kalite = "Mükemmel" if ortalama_fark < 1 else "İyi" if ortalama_fark < 5 else "Orta"
            else:
                kalite = "N/A"
            
            print(f"{format_adi.upper()}\\t{boyut_kb:.1f} KB\\t\\t{kalite}")
            
        except Exception as e:
            print(f"{format_adi.upper()}\\tHata: {e}")

def interaktif_resim_editor():
    """Basit interaktif resim editörü"""
    print("\\n🎨 İnteraktif Resim Editörü")
    print("=" * 35)
    
    # Beyaz tuval oluştur
    tuval = np.ones((400, 600, 3), dtype=np.uint8) * 255
    aktif_renk = (0, 0, 0)  # Siyah
    fırca_boyutu = 5
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal tuval, aktif_renk, fırca_boyutu
        
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(tuval, (x, y), fırca_boyutu, aktif_renk, -1)
    
    cv2.namedWindow('Resim Editoru')
    cv2.setMouseCallback('Resim Editoru', mouse_callback)
    
    print("🎨 Kontroller:")
    print("Mouse: Çizim yapın")
    print("r: Kırmızı renk")
    print("g: Yeşil renk") 
    print("b: Mavi renk")
    print("k: Siyah renk")
    print("+/-: Fırça boyutu")
    print("c: Temizle")
    print("s: Kaydet")
    print("ESC: Çıkış")
    
    while True:
        cv2.imshow('Resim Editoru', tuval)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            aktif_renk = (0, 0, 255)  # Kırmızı
        elif key == ord('g'):
            aktif_renk = (0, 255, 0)  # Yeşil
        elif key == ord('b'):
            aktif_renk = (255, 0, 0)  # Mavi
        elif key == ord('k'):
            aktif_renk = (0, 0, 0)    # Siyah
        elif key == ord('+') or key == ord('='):
            fırca_boyutu = min(20, fırca_boyutu + 1)
        elif key == ord('-'):
            fırça_boyutu = max(1, fırça_boyutu - 1)
        elif key == ord('c'):
            tuval = np.ones((400, 600, 3), dtype=np.uint8) * 255
        elif key == ord('s'):
            cv2.imwrite('cizimim.png', tuval)
            print("✅ Çizim 'cizimim.png' olarak kaydedildi!")
    
    cv2.destroyAllWindows()

def main():
    """Ana program fonksiyonu"""
    print("🎉 OpenCV Resim İşlemleri")
    print("Bu program, temel resim işlemlerini gösterir.\\n")
    
    # Örnek resimler oluştur
    examples_dir = resim_olustur_ve_kaydet()
    
    # Ana menü
    while True:
        print("\\n" + "="*50)
        print("📸 OpenCV Resim İşlemleri Menüsü")
        print("="*50)
        print("1. Resim Okuma Örnekleri")
        print("2. Resim Bilgilerini Göster")
        print("3. Resim Gösterme Örnekleri")
        print("4. Resim Kaydetme Örnekleri")
        print("5. Piksel Manipülasyonu")
        print("6. ROI (İlgilenilen Bölge) Örnekleri")
        print("7. Dosya Formatları Karşılaştırması")
        print("8. İnteraktif Resim Editörü")
        print("0. Çıkış")
        
        secim = input("\\nLütfen bir seçenek girin (0-8): ").strip()
        
        if secim == "0":
            print("👋 Görüşürüz!")
            break
        elif secim == "1":
            resim = resim_okuma_ornekleri(examples_dir)
        elif secim == "2":
            resim = cv2.imread(str(examples_dir / "gradyan.jpg"))
            resim_bilgileri_goster(resim)
        elif secim == "3":
            resim = cv2.imread(str(examples_dir / "gradyan.jpg"))
            resim_gosterme_ornekleri(resim)
        elif secim == "4":
            resim = cv2.imread(str(examples_dir / "gradyan.jpg"))
            resim_kaydetme_ornekleri(resim, examples_dir)
        elif secim == "5":
            resim = cv2.imread(str(examples_dir / "gradyan.jpg"))
            piksel_manipulasyonu(resim)
        elif secim == "6":
            resim = cv2.imread(str(examples_dir / "gradyan.jpg"))
            roi_ornekleri(resim)
        elif secim == "7":
            resim = cv2.imread(str(examples_dir / "gradyan.jpg"))
            dosya_formatlari_karsilastirma(resim, examples_dir)
        elif secim == "8":
            interaktif_resim_editor()
        else:
            print("❌ Geçersiz seçim! Lütfen 0-8 arasında bir sayı girin.")

if __name__ == "__main__":
    main()