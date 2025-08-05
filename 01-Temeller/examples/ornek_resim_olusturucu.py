#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 OpenCV Örnek Resimleri Oluşturucu
====================================

Bu script, OpenCV temel öğrenme için örnek resimler oluşturur.
Temeller bölümündeki derslerde kullanılacak örnek resimler.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
from pathlib import Path
import math

def create_sample_landscape():
    """Örnek manzara resmi oluşturur"""
    print("🏔️ Manzara resmi oluşturuluyor...")
    
    manzara = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Gökyüzü gradyanı (mavi tonları)
    for y in range(200):
        mavi_deger = 255 - int(y * 0.3)
        manzara[y, :] = [mavi_deger, 150, 100]
    
    # Dağlar (farklı katmanlar)
    # Arka dağlar (koyu)
    dag_noktalari_1 = np.array([[0, 180], [150, 120], [300, 140], [450, 100], [600, 130], [600, 200], [0, 200]], np.int32)
    cv2.fillPoly(manzara, [dag_noktalari_1], (100, 120, 80))
    
    # Orta dağlar (orta ton)
    dag_noktalari_2 = np.array([[0, 220], [100, 160], [250, 180], [400, 140], [550, 170], [600, 160], [600, 250], [0, 250]], np.int32)
    cv2.fillPoly(manzara, [dag_noktalari_2], (120, 140, 100))
    
    # Ön dağlar (açık)
    dag_noktalari_3 = np.array([[0, 280], [80, 220], [200, 240], [350, 200], [500, 230], [600, 220], [600, 300], [0, 300]], np.int32)
    cv2.fillPoly(manzara, [dag_noktalari_3], (140, 160, 120))
    
    # Çimen (yeşil alan)
    manzara[300:400, :] = [0, 140, 60]
    
    # Göl (mavi alan)
    cv2.ellipse(manzara, (300, 340), (200, 30), 0, 0, 360, [180, 100, 50], -1)
    
    # Ağaçlar
    agac_pozisyonlari = [(100, 320), (200, 310), (400, 330), (520, 315)]
    for x, y in agac_pozisyonlari:
        # Gövde
        cv2.rectangle(manzara, (x-5, y), (x+5, y+40), (50, 80, 40), -1)
        # Yapraklar
        cv2.circle(manzara, (x, y), 20, (0, 120, 40), -1)
    
    # Güneş
    cv2.circle(manzara, (500, 80), 30, (0, 200, 255), -1)
    
    # Bulutlar
    cv2.ellipse(manzara, (150, 60), (40, 20), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(manzara, (170, 50), (30, 15), 0, 0, 360, (240, 240, 240), -1)
    
    cv2.ellipse(manzara, (400, 70), (50, 25), 0, 0, 360, (210, 210, 210), -1)
    cv2.ellipse(manzara, (430, 60), (35, 18), 0, 0, 360, (230, 230, 230), -1)
    
    cv2.imwrite('sample-landscape.jpg', manzara)
    return manzara

def create_sample_portrait():
    """Örnek portre resmi oluşturur"""
    print("👤 Portre resmi oluşturuluyor...")
    
    portre = np.ones((500, 400, 3), dtype=np.uint8) * 240
    
    # Basit insan figürü
    # Kafa
    cv2.circle(portre, (200, 150), 60, (255, 220, 177), -1)
    
    # Saç
    cv2.ellipse(portre, (200, 120), (70, 50), 0, 180, 360, (101, 67, 33), -1)
    
    # Gözler
    cv2.circle(portre, (180, 140), 8, (255, 255, 255), -1)  # Sol göz
    cv2.circle(portre, (220, 140), 8, (255, 255, 255), -1)  # Sağ göz
    cv2.circle(portre, (180, 140), 4, (0, 0, 0), -1)        # Sol göz bebeği
    cv2.circle(portre, (220, 140), 4, (0, 0, 0), -1)        # Sağ göz bebeği
    
    # Kaşlar
    cv2.ellipse(portre, (180, 125), (12, 5), 0, 0, 180, (101, 67, 33), 3)
    cv2.ellipse(portre, (220, 125), (12, 5), 0, 0, 180, (101, 67, 33), 3)
    
    # Burun
    cv2.line(portre, (200, 155), (195, 165), (200, 180, 140), 2)
    cv2.line(portre, (195, 165), (200, 170), (200, 180, 140), 2)
    
    # Ağız
    cv2.ellipse(portre, (200, 185), (15, 8), 0, 0, 180, (200, 100, 100), 2)
    
    # Boyun
    cv2.rectangle(portre, (175, 210), (225, 280), (255, 220, 177), -1)
    
    # Gömlek
    cv2.rectangle(portre, (150, 280), (250, 450), (100, 150, 200), -1)
    
    # Gömlek yakası
    cv2.line(portre, (200, 280), (200, 350), (80, 120, 160), 3)
    cv2.line(portre, (200, 280), (170, 320), (80, 120, 160), 3)
    cv2.line(portre, (200, 280), (230, 320), (80, 120, 160), 3)
    
    # Düğmeler
    for y in [320, 360, 400]:
        cv2.circle(portre, (200, y), 4, (60, 100, 140), -1)
    
    cv2.imwrite('sample-portrait.jpg', portre)
    return portre

def create_sample_objects():
    """Nesne tanıma için örnek objeler resmi oluşturur"""
    print("📦 Nesne örnekleri oluşturuluyor...")
    
    nesneler = np.ones((400, 500, 3), dtype=np.uint8) * 250
    
    # Ev
    # Duvar
    cv2.rectangle(nesneler, (50, 150), (150, 250), (139, 69, 19), -1)
    # Çatı
    points = np.array([[50, 150], [100, 100], [150, 150]], np.int32)
    cv2.fillPoly(nesneler, [points], (165, 42, 42))
    # Kapı
    cv2.rectangle(nesneler, (80, 200), (100, 250), (101, 67, 33), -1)
    # Pencere
    cv2.rectangle(nesneler, (110, 170), (140, 200), (173, 216, 230), -1)
    # Pencere çerçevesi
    cv2.rectangle(nesneler, (110, 170), (140, 200), (100, 100, 100), 2)
    cv2.line(nesneler, (125, 170), (125, 200), (100, 100, 100), 1)
    cv2.line(nesneler, (110, 185), (140, 185), (100, 100, 100), 1)
    
    # Araba
    # Gövde
    cv2.rectangle(nesneler, (200, 200), (300, 240), (255, 0, 0), -1)
    # Camlar
    cv2.rectangle(nesneler, (210, 180), (290, 200), (173, 216, 230), -1)
    # Tekerlekler  
    cv2.circle(nesneler, (220, 240), 15, (50, 50, 50), -1)
    cv2.circle(nesneler, (280, 240), 15, (50, 50, 50), -1)
    # Jantlar
    cv2.circle(nesneler, (220, 240), 8, (150, 150, 150), -1)
    cv2.circle(nesneler, (280, 240), 8, (150, 150, 150), -1)
    
    # Ağaç
    # Gövde
    cv2.rectangle(nesneler, (370, 180), (380, 250), (139, 69, 19), -1)
    # Yapraklar (farklı boyutlarda daireler)
    cv2.circle(nesneler, (375, 160), 25, (0, 128, 0), -1)
    cv2.circle(nesneler, (365, 140), 20, (0, 150, 0), -1)
    cv2.circle(nesneler, (385, 145), 18, (0, 140, 0), -1)
    
    # Top
    cv2.circle(nesneler, (100, 320), 30, (0, 255, 255), -1)
    # Top üzerinde desen
    cv2.circle(nesneler, (100, 320), 20, (0, 200, 200), 2)
    
    # Kutu
    cv2.rectangle(nesneler, (200, 280), (260, 340), (160, 82, 45), -1)
    # Kutu kenarları (3D efekti)
    cv2.line(nesneler, (260, 280), (270, 270), (139, 69, 19), 3)
    cv2.line(nesneler, (270, 270), (270, 330), (139, 69, 19), 3)
    cv2.line(nesneler, (270, 330), (260, 340), (139, 69, 19), 3)
    
    # Çiçek
    # Sap
    cv2.line(nesneler, (400, 320), (400, 360), (0, 128, 0), 3)
    # Yapraklar
    cv2.ellipse(nesneler, (395, 340), (8, 15), 45, 0, 360, (0, 150, 0), -1)
    cv2.ellipse(nesneler, (405, 340), (8, 15), -45, 0, 360, (0, 150, 0), -1)
    # Çiçek petalleri
    for angle in range(0, 360, 45):
        x = int(400 + 15 * math.cos(math.radians(angle)))
        y = int(320 + 15 * math.sin(math.radians(angle)))
        cv2.circle(nesneler, (x, y), 8, (255, 100, 150), -1)
    # Çiçek merkezi
    cv2.circle(nesneler, (400, 320), 6, (255, 255, 0), -1)
    
    cv2.imwrite('sample-objects.jpg', nesneler)
    return nesneler

def create_sample_shapes():
    """Geometrik şekiller örnegi oluşturur"""
    print("🔷 Geometrik şekiller oluşturuluyor...")
    
    sekiller = np.ones((400, 500, 3), dtype=np.uint8) * 245
    
    # Başlık
    cv2.putText(sekiller, 'OpenCV Shapes', (150, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # İlk satır şekiller
    # Kare
    cv2.rectangle(sekiller, (50, 80), (130, 160), (255, 0, 0), -1)
    cv2.putText(sekiller, 'Square', (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Daire
    cv2.circle(sekiller, (220, 120), 40, (0, 255, 0), -1)
    cv2.putText(sekiller, 'Circle', (190, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Üçgen
    points = np.array([[340, 80], [300, 160], [380, 160]], np.int32)
    cv2.fillPoly(sekiller, [points], (0, 0, 255))
    cv2.putText(sekiller, 'Triangle', (320, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # İkinci satır şekiller
    # Elips
    cv2.ellipse(sekiller, (90, 260), (50, 30), 0, 0, 360, (255, 255, 0), -1)
    cv2.putText(sekiller, 'Ellipse', (60, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Altıgen
    hex_points = []
    center = (220, 260)
    radius = 40
    for i in range(6):
        angle = i * 60 * math.pi / 180
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        hex_points.append([x, y])
    hex_points = np.array(hex_points, np.int32)
    cv2.fillPoly(sekiller, [hex_points], (255, 0, 255))
    cv2.putText(sekiller, 'Hexagon', (185, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Yıldız
    star_outer = []
    star_inner = []
    center = (360, 260)
    outer_radius = 40
    inner_radius = 20
    
    for i in range(5):
        # Dış noktalar
        angle = i * 72 * math.pi / 180 - math.pi/2
        x = int(center[0] + outer_radius * math.cos(angle))
        y = int(center[1] + outer_radius * math.sin(angle))
        star_outer.append([x, y])
        
        # İç noktalar
        angle = (i * 72 + 36) * math.pi / 180 - math.pi/2
        x = int(center[0] + inner_radius * math.cos(angle))
        y = int(center[1] + inner_radius * math.sin(angle))
        star_inner.append([x, y])
    
    # Yıldız noktalarını birleştir
    star_points = []
    for i in range(5):
        star_points.append(star_outer[i])
        star_points.append(star_inner[i])
    
    star_points = np.array(star_points, np.int32)
    cv2.fillPoly(sekiller, [star_points], (0, 255, 255))
    cv2.putText(sekiller, 'Star', (340, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite('sample-shapes.png', sekiller)
    return sekiller

def create_sample_colors():
    """Renk örnekleri oluşturur"""
    print("🌈 Renk örnekleri oluşturuluyor...")
    
    renkler = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Başlık
    cv2.putText(renkler, 'Color Samples', (120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Temel renkler
    temel_renkler = [
        ('Red', (0, 0, 255), (50, 50)),
        ('Green', (0, 255, 0), (150, 50)),
        ('Blue', (255, 0, 0), (250, 50)),
        ('Yellow', (0, 255, 255), (350, 50)),
    ]
    
    for renk_adi, renk, pozisyon in temel_renkler:
        cv2.rectangle(renkler, pozisyon, (pozisyon[0]+60, pozisyon[1]+60), renk, -1)
        cv2.putText(renkler, renk_adi, (pozisyon[0], pozisyon[1]+80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Karışık renkler
    karisik_renkler = [
        ('Orange', (0, 165, 255), (50, 150)),
        ('Purple', (128, 0, 128), (150, 150)),
        ('Pink', (203, 192, 255), (250, 150)),
        ('Brown', (42, 42, 165), (350, 150)),
    ]
    
    for renk_adi, renk, pozisyon in karisik_renkler:
        cv2.rectangle(renkler, pozisyon, (pozisyon[0]+60, pozisyon[1]+60), renk, -1)
        cv2.putText(renkler, renk_adi, (pozisyon[0], pozisyon[1]+80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Gri tonları
    gri_tonlari = [0, 64, 128, 192, 255]
    for i, gri_deger in enumerate(gri_tonlari):
        x = 50 + i * 70
        cv2.rectangle(renkler, (x, 250), (x+60, 270), (gri_deger, gri_deger, gri_deger), -1)
        cv2.putText(renkler, str(gri_deger), (x+15, 285), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.imwrite('sample-colors.png', renkler)
    return renkler

def create_sample_text():
    """Metin örnekleri oluşturur"""  
    print("📝 Metin örnekleri oluşturuluyor...")
    
    metin = np.ones((400, 600, 3), dtype=np.uint8) * 250
    
    # Farklı font türleri
    fontlar = [
        (cv2.FONT_HERSHEY_SIMPLEX, 'SIMPLEX'),
        (cv2.FONT_HERSHEY_PLAIN, 'PLAIN'),
        (cv2.FONT_HERSHEY_DUPLEX, 'DUPLEX'),
        (cv2.FONT_HERSHEY_COMPLEX, 'COMPLEX'),
        (cv2.FONT_HERSHEY_TRIPLEX, 'TRIPLEX'),
        (cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 'SCRIPT_SIMPLEX'),
        (cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 'SCRIPT_COMPLEX')
    ]
    
    # Başlık
    cv2.putText(metin, 'OpenCV Font Examples', (150, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 150), 2)
    
    # Font örnekleri
    y_pos = 80
    for font, font_adi in fontlar:
        cv2.putText(metin, f'{font_adi}: OpenCV Text', (50, y_pos), 
                   font, 0.8, (0, 0, 0), 1)
        y_pos += 40
    
    # Farklı renkler ve boyutlar
    cv2.putText(metin, 'Buyuk Metin', (50, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    
    cv2.putText(metin, 'Kucuk metin', (300, 340), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(metin, 'Renkli Metin', (50, 370), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    cv2.imwrite('sample-text.jpg', metin)
    return metin

def create_sample_patterns():
    """Çeşitli desenler oluşturur"""
    print("🎨 Desen örnekleri oluşturuluyor...")
    
    desenler = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Çizgili desen
    for i in range(0, 200, 20):
        cv2.line(desenler, (i, 0), (i, 200), (100, 100, 100), 2)
        cv2.line(desenler, (0, i), (200, i), (100, 100, 100), 2)
    
    # Noktalı desen
    for i in range(220, 400, 20):
        for j in range(0, 200, 20):
            cv2.circle(desenler, (i, j), 5, (255, 0, 0), -1)
    
    # Dalga deseni
    for x in range(400):
        y = int(250 + 30 * math.sin(x * math.pi / 50))
        if 0 <= y < 400:
            cv2.circle(desenler, (x, y), 2, (0, 0, 255), -1)
    
    # Spiral desen
    center = (300, 320)
    for angle in range(0, 720, 5):
        radius = angle / 10
        x = int(center[0] + radius * math.cos(math.radians(angle)))
        y = int(center[1] + radius * math.sin(math.radians(angle)))
        if 0 <= x < 400 and 0 <= y < 400:
            cv2.circle(desenler, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imwrite('sample-patterns.png', desenler)
    return desenler

def create_readme():
    """Examples klasörü için README dosyası oluşturur"""
    readme_content = """# 📁 Examples - Örnek Resimler

Bu klasör, OpenCV temelleri öğrenmek için hazırlanmış örnek resimleri içerir.

## 🖼️ İçerik

### 🌅 **Genel Örnekler**
- `sample-landscape.jpg` - Manzara resmi (dağlar, göl, ağaçlar)
- `sample-portrait.jpg` - Portre resmi (insan figürü)
- `sample-objects.jpg` - Nesne örnekleri (ev, araba, ağaç, top, kutu, çiçek)

### 🎨 **Eğitim Amaçlı**
- `sample-shapes.png` - Geometrik şekiller (kare, daire, üçgen, elips, altıgen, yıldız)
- `sample-colors.png` - Renk örnekleri (temel renkler, karışık renkler, gri tonları)
- `sample-text.jpg` - Font örnekleri (farklı OpenCV fontları)
- `sample-patterns.png` - Desen örnekleri (çizgiler, noktalar, dalga, spiral)

## 🎯 Kullanım Amacı

Bu resimler şu konularda kullanılır:
- Temel resim okuma/gösterme
- Renk uzayı öğrenme
- Geometrik şekil tanıma
- Metin işleme örnekleri
- Temel görüntü manipülasyonu

## 💻 Kullanım Örneği

```python
import cv2

# Örnek resim yükle
resim = cv2.imread('examples/sample-landscape.jpg')

# Kontrol et
if resim is not None:
    print("✅ Resim yüklendi!")
    print(f"Boyut: {resim.shape}")
    
    # Göster
    cv2.imshow('Örnek Resim', resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Resim yüklenemedi!")
```

## 📊 Resim Özellikleri

| Resim | Boyut | Format | Açıklama |
|-------|-------|---------|----------|
| sample-landscape.jpg | 400x600 | JPEG | Manzara, doğal objeler |
| sample-portrait.jpg | 500x400 | JPEG | İnsan figürü, portre |
| sample-objects.jpg | 400x500 | JPEG | 6 farklı nesne |
| sample-shapes.png | 400x500 | PNG | 6 geometrik şekil |
| sample-colors.png | 300x400 | PNG | Renk paleti |
| sample-text.jpg | 400x600 | JPEG | Font örnekleri |  
| sample-patterns.png | 400x400 | PNG | Matematik desenler |

## 🔧 Yeniden Oluşturma

Bu resimleri yeniden oluşturmak için:

```bash
python ornek_resim_olusturucu.py
```

## 📚 İlgili Dersler

Bu örnekler şu derslerde kullanılır:
- [02-ilk-program.py](../02-ilk-program.py) - İlk OpenCV programı
- [04-resim-islemleri.py](../04-resim-islemleri.py) - Resim işlemleri
- [05-renk-uzaylari.py](../05-renk-uzaylari.py) - Renk uzayları

## 💡 İpuçları

- Resimler eğitim amaçlı optimize edilmiştir
- Farklı zorluk seviyelerinde örnekler vardır
- Kendi resimlerinizi de ekleyebilirsiniz
- Test resimleri için `../06-alistirmalar/test-resimleri/` klasörüne bakın

---

**🎨 Not:** Tüm resimler programatik olarak oluşturulmuş, telif hakkı sorunu olmayan eğitim materyalleridir.
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("📝 README.md dosyası oluşturuldu")

def main():
    """Ana fonksiyon - Tüm örnek resimleri oluşturur"""
    print("🎨 OpenCV Örnek Resimleri Oluşturucu")
    print("=" * 45)
    print("Bu script, temel OpenCV öğrenimi için örnek resimler oluşturur.\\n")
    
    try:
        # Örnek resimleri oluştur
        create_sample_landscape()
        create_sample_portrait()
        create_sample_objects()
        create_sample_shapes()
        create_sample_colors()
        create_sample_text()
        create_sample_patterns()
        create_readme()
        
        print("\\n🎉 Tüm örnek resimler başarıyla oluşturuldu!")
        print("\\n📊 Oluşturulan Dosyalar:")
        
        # Oluşturulan dosyaları listele
        resim_dosyalari = [
            "sample-landscape.jpg", "sample-portrait.jpg", "sample-objects.jpg",
            "sample-shapes.png", "sample-colors.png", "sample-text.jpg",
            "sample-patterns.png", "README.md", "ornek_resim_olusturucu.py"
        ]
        
        for i, dosya in enumerate(resim_dosyalari, 1):
            print(f"   {i:2d}. {dosya}")
        
        print(f"\\n✅ Toplam {len(resim_dosyalari)} dosya hazır!")
        print("\\n🚀 OpenCV derslerinde kullanabilirsiniz!")
        
        # Özet bilgi
        print("\\n📋 Dosya Türleri Özeti:")
        print("   🌅 Manzara, portre ve nesne örnekleri")
        print("   🎨 Geometrik şekil ve renk örnekleri")
        print("   📝 Metin ve desen örnekleri")
        print("   📚 Detaylı README dokümantasyonu")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()