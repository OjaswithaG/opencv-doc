#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Test Resimleri Oluşturucu
===========================

Bu script, OpenCV alıştırmaları için gerekli test resimlerini oluşturur.
Farklı kalite, kontrast, parlaklık ve özellikler içeren resimler.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
from pathlib import Path
import math

def create_gradient_image():
    """Renk gradyanı resmi oluşturur"""
    print("📸 Gradyan resmi oluşturuluyor...")
    
    gradyan = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Yatay renk gradyanı
    for i in range(400):
        r = int(255 * i / 400)  # 0'dan 255'e kırmızı
        g = int(255 * (1 - i / 400))  # 255'ten 0'a yeşil
        b = 128  # Sabit mavi
        gradyan[:, i] = [b, g, r]  # BGR formatı
    
    cv2.imwrite('gradyan.jpg', gradyan)
    return gradyan

def create_low_contrast_image():
    """Düşük kontrastlı resim oluşturur"""
    print("📸 Düşük kontrastlı resim oluşturuluyor...")
    
    # 80-120 arasında dar bir aralıkta piksel değerleri
    dusuk_kontrast = np.random.randint(80, 120, (250, 350, 3), dtype=np.uint8)
    
    # Hafif desenler ekle
    cv2.rectangle(dusuk_kontrast, (50, 50), (150, 150), (110, 115, 105), -1)
    cv2.circle(dusuk_kontrast, (250, 125), 60, (95, 100, 125), -1)
    
    cv2.imwrite('dusuk-kontrast.jpg', dusuk_kontrast)
    return dusuk_kontrast

def create_high_contrast_image():
    """Yüksek kontrastlı resim oluşturur"""
    print("📸 Yüksek kontrastlı resim oluşturuluyor...")
    
    yuksek_kontrast = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Siyah-beyaz çizgiler
    for i in range(0, 400, 40):
        if (i // 40) % 2 == 0:
            yuksek_kontrast[:, i:i+20] = 255  # Beyaz şerit
        # Diğerleri siyah kalır
    
    # Merkeze renkli daire
    cv2.circle(yuksek_kontrast, (200, 150), 50, (0, 0, 255), -1)
    
    cv2.imwrite('yuksek-kontrast.jpg', yuksek_kontrast)
    return yuksek_kontrast

def create_colorful_shapes():
    """Renkli geometrik şekiller resmi oluşturur"""
    print("📸 Renkli geometrik şekiller oluşturuluyor...")
    
    sekiller = np.zeros((400, 500, 3), dtype=np.uint8)
    
    # Farklı renklerde şekiller
    # Kırmızı dikdörtgen
    cv2.rectangle(sekiller, (50, 50), (150, 150), (0, 0, 255), -1)
    
    # Yeşil daire
    cv2.circle(sekiller, (250, 100), 50, (0, 255, 0), -1)
    
    # Mavi dikdörtgen  
    cv2.rectangle(sekiller, (350, 50), (450, 150), (255, 0, 0), -1)
    
    # Sarı daire
    cv2.circle(sekiller, (100, 250), 40, (0, 255, 255), -1)
    
    # Magenta elips
    cv2.ellipse(sekiller, (250, 250), (60, 40), 0, 0, 360, (255, 0, 255), -1)
    
    # Cyan üçgen
    points = np.array([[350, 200], [400, 300], [450, 200]], np.int32)
    cv2.fillPoly(sekiller, [points], (255, 255, 0))
    
    # Beyaz çizgiler
    cv2.line(sekiller, (50, 350), (450, 350), (255, 255, 255), 5)
    cv2.line(sekiller, (250, 200), (250, 400), (255, 255, 255), 3)
    
    cv2.imwrite('renkli-sekiller.png', sekiller)
    return sekiller

def create_noisy_image():
    """Gürültülü resim oluşturur"""
    print("📸 Gürültülü resim oluşturuluyor...")
    
    # Temiz resim oluştur
    temiz = np.ones((300, 400, 3), dtype=np.uint8) * 128
    cv2.rectangle(temiz, (100, 100), (300, 200), (180, 150, 120), -1)
    cv2.circle(temiz, (200, 150), 60, (120, 180, 160), -1)
    
    # Gaussian gürültüsü ekle
    gurultu = np.random.normal(0, 25, temiz.shape)
    gurultulu = np.clip(temiz.astype(np.float32) + gurultu, 0, 255).astype(np.uint8)
    
    cv2.imwrite('gurultulu.jpg', gurultulu)
    return gurultulu

def create_blurry_image():
    """Bulanık resim oluşturur"""
    print("📸 Bulanık resim oluşturuluyor...")
    
    # Keskin resim oluştur
    keskin = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Keskin detaylar
    for i in range(50, 350, 20):
        cv2.line(keskin, (i, 50), (i, 250), (255, 255, 255), 2)
    
    for i in range(75, 225, 15):
        cv2.line(keskin, (75, i), (325, i), (0, 255, 255), 1)
    
    # Metin ekle
    cv2.putText(keskin, 'SHARP TEXT', (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Bulanıklaştır
    bulanik = cv2.GaussianBlur(keskin, (15, 15), 0)
    
    cv2.imwrite('bulanik.jpg', bulanik)
    return bulanik

def create_bright_image():
    """Aşırı parlak resim oluşturur"""
    print("📸 Aşırı parlak resim oluşturuluyor...")
    
    parlak = np.ones((250, 350, 3), dtype=np.uint8) * 220
    
    # Hafif varyasyon ekle
    for i in range(350):
        for j in range(250):
            varyasyon = int(20 * math.sin(i/20) * math.cos(j/20))
            parlak[j, i] = np.clip(parlak[j, i] + varyasyon, 200, 255)
    
    # Koyu detaylar
    cv2.rectangle(parlak, (50, 50), (100, 100), (180, 180, 180), -1)
    cv2.circle(parlak, (250, 125), 30, (160, 160, 160), -1)
    
    cv2.imwrite('asiri-parlak.jpg', parlak)
    return parlak

def create_dark_image():
    """Çok karanlık resim oluşturur"""
    print("📸 Çok karanlık resim oluşturuluyor...")
    
    karanlik = np.ones((250, 350, 3), dtype=np.uint8) * 30
    
    # Hafif aydınlık alanlar
    cv2.rectangle(karanlik, (100, 100), (250, 150), (60, 60, 60), -1)
    cv2.circle(karanlik, (175, 125), 40, (80, 80, 80), -1)
    
    # Çok hafif detaylar
    cv2.line(karanlik, (50, 200), (300, 200), (50, 50, 50), 2)
    
    cv2.imwrite('karanlik.jpg', karanlik)
    return karanlik

def create_normal_image():
    """Normal kaliteli resim oluşturur"""
    print("📸 Normal kaliteli resim oluşturuluyor...")
    
    normal = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
    
    # Anlamlı objeler ekle
    # Ev çiz
    # Taban
    cv2.rectangle(normal, (100, 150), (250, 220), (139, 69, 19), -1)
    # Çatı
    points = np.array([[100, 150], [175, 100], [250, 150]], np.int32)
    cv2.fillPoly(normal, [points], (160, 82, 45))
    # Kapı
    cv2.rectangle(normal, (160, 180), (190, 220), (101, 67, 33), -1)
    # Pencere
    cv2.rectangle(normal, (120, 160), (150, 190), (173, 216, 230), -1)
    
    # Güneş
    cv2.circle(normal, (320, 80), 25, (0, 255, 255), -1)
    
    # Ağaç
    cv2.rectangle(normal, (50, 180), (60, 220), (139, 69, 19), -1)  # Gövde
    cv2.circle(normal, (55, 160), 25, (0, 128, 0), -1)  # Yapraklar
    
    cv2.imwrite('normal.jpg', normal)
    return normal

def create_test_patterns():
    """Test desenleri oluşturur"""
    print("📸 Test desenleri oluşturuluyor...")
    
    # Satranç tahtası deseni
    satranc = np.zeros((320, 320), dtype=np.uint8)
    kare_boyutu = 40
    
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                satranc[i*kare_boyutu:(i+1)*kare_boyutu, 
                       j*kare_boyutu:(j+1)*kare_boyutu] = 255
    
    cv2.imwrite('satranc-tahtasi.png', satranc)
    
    # Renk testisi
    renk_test = np.zeros((200, 600, 3), dtype=np.uint8)
    
    # RGB şeritleri
    renk_test[:, 0:200] = [0, 0, 255]    # Kırmızı
    renk_test[:, 200:400] = [0, 255, 0]  # Yeşil
    renk_test[:, 400:600] = [255, 0, 0]  # Mavi
    
    cv2.imwrite('renk-testi.png', renk_test)
    
    # Çizgi testisi (kenar algılama için)
    cizgi_test = np.zeros((300, 300), dtype=np.uint8)
    
    # Farklı açılarda çizgiler
    cv2.line(cizgi_test, (50, 50), (250, 50), 255, 3)      # Yatay
    cv2.line(cizgi_test, (50, 50), (50, 250), 255, 3)      # Dikey
    cv2.line(cizgi_test, (50, 250), (250, 50), 255, 3)     # Çapraz
    cv2.line(cizgi_test, (150, 50), (150, 250), 255, 2)    # Orta dikey
    cv2.line(cizgi_test, (50, 150), (250, 150), 255, 2)    # Orta yatay
    
    cv2.imwrite('cizgi-testi.png', cizgi_test)
    
    return satranc, renk_test, cizgi_test

def create_face_detection_sample():
    """Yüz tespiti için basit örnek oluşturur"""
    print("📸 Yüz tespiti örneği oluşturuluyor...")
    
    yuz_ornegi = np.ones((400, 300, 3), dtype=np.uint8) * 240
    
    # Basit yüz çiz
    # Yüz oval
    cv2.ellipse(yuz_ornegi, (150, 200), (80, 100), 0, 0, 360, (255, 220, 177), -1)
    
    # Gözler
    cv2.circle(yuz_ornegi, (120, 170), 15, (255, 255, 255), -1)  # Sol göz
    cv2.circle(yuz_ornegi, (180, 170), 15, (255, 255, 255), -1)  # Sağ göz
    cv2.circle(yuz_ornegi, (120, 170), 8, (0, 0, 0), -1)        # Sol göz bebeği
    cv2.circle(yuz_ornegi, (180, 170), 8, (0, 0, 0), -1)        # Sağ göz bebeği
    
    # Burun
    cv2.line(yuz_ornegi, (150, 190), (150, 210), (200, 180, 140), 3)
    
    # Ağız
    cv2.ellipse(yuz_ornegi, (150, 230), (25, 15), 0, 0, 180, (200, 100, 100), 3)
    
    cv2.imwrite('yuz-ornegi.jpg', yuz_ornegi)
    return yuz_ornegi

def create_document_sample():
    """Belge işleme için örnek oluşturur"""
    print("📸 Belge örneği oluşturuluyor...")
    
    belge = np.ones((400, 300, 3), dtype=np.uint8) * 250
    
    # Metin satırları simülasyonu
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Başlık
    cv2.putText(belge, 'BELGE ORNEGI', (50, 50), font, 0.8, (0, 0, 0), 2)
    
    # Metin satırları (çizgiler ile simüle et)
    for i, y in enumerate(range(80, 350, 25)):
        if i % 3 == 0:  # Her 3. satır daha uzun
            cv2.line(belge, (30, y), (270, y), (100, 100, 100), 2)
        else:
            uzunluk = 200 + (i % 2) * 50
            cv2.line(belge, (30, y), (30 + uzunluk, y), (100, 100, 100), 2)
    
    # Kenar efekti (tarayıcı etkisi)
    gurultu = np.random.normal(0, 5, belge.shape)
    belge = np.clip(belge.astype(np.float32) + gurultu, 0, 255).astype(np.uint8)
    
    cv2.imwrite('belge-ornegi.jpg', belge)
    return belge

def create_coin_detection_sample():
    """Para tespiti için örnek oluşturur"""
    print("📸 Para tespiti örneği oluşturuluyor...")
    
    para_ornegi = np.ones((350, 400, 3), dtype=np.uint8) * 200
    
    # Farklı boyutlarda daireler (paralar)
    paralar = [
        ((100, 100), 40, (180, 165, 130)),  # Büyük para
        ((250, 120), 35, (200, 180, 140)),  # Orta para  
        ((320, 200), 30, (190, 170, 135)),  # Küçük para
        ((150, 250), 38, (185, 175, 145)),  # Büyük para
        ((80, 280), 32, (195, 185, 150))    # Orta para
    ]
    
    for (x, y), radius, color in paralar:
        # Para gövdesi
        cv2.circle(para_ornegi, (x, y), radius, color, -1)
        # Para kenarı
        cv2.circle(para_ornegi, (x, y), radius, (150, 130, 100), 2)
        # Para üzerinde desen
        cv2.circle(para_ornegi, (x, y), radius//3, (160, 140, 110), 2)
    
    cv2.imwrite('para-tespiti.jpg', para_ornegi)
    return para_ornegi

def create_readme():
    """Test resimleri için README dosyası oluşturur"""
    readme_content = """# 🖼️ Test Resimleri

Bu klasör, OpenCV alıştırmaları için özel olarak hazırlanmış test resimlerini içerir.

## 📂 İçerik

### 🎨 **Temel Test Resimleri**
- `gradyan.jpg` - Renk gradyanı (renk uzayı testleri için)
- `normal.jpg` - Normal kaliteli resim (genel testler için)
- `renkli-sekiller.png` - Geometrik şekiller (renk filtreleme için)

### 📊 **Kalite Test Resimleri**  
- `dusuk-kontrast.jpg` - Düşük kontrastlı resim (histogram testleri için)
- `yuksek-kontrast.jpg` - Yüksek kontrastlı resim (kontrast testleri için)
- `asiri-parlak.jpg` - Aşırı parlak resim (parlaklık testleri için)
- `karanlik.jpg` - Çok karanlık resim (parlaklık testleri için)

### 🔧 **İşleme Test Resimleri**
- `gurultulu.jpg` - Gürültülü resim (gürültü azaltma için)
- `bulanik.jpg` - Bulanık resim (keskinleştirme için)

### 🧪 **Özel Test Desenleri**
- `satranc-tahtasi.png` - Satranç tahtası (kamera kalibrasyonu için)
- `renk-testi.png` - RGB renk şeritleri (renk testleri için)
- `cizgi-testi.png` - Çizgi desenleri (kenar algılama için)

### 🎯 **Uygulama Test Resimleri**
- `yuz-ornegi.jpg` - Basit yüz çizimi (yüz tespiti testleri için)
- `belge-ornegi.jpg` - Belge simülasyonu (belge işleme için)
- `para-tespiti.jpg` - Para örnekleri (nesne tespiti için)

## 🚀 Kullanım

### Python'da Test Resmi Yükleme
```python
import cv2

# Test resmi yükle
resim = cv2.imread('test-resimleri/gradyan.jpg')

# Kontrol et
if resim is not None:
    print("✅ Resim başarıyla yüklendi!")
    cv2.imshow('Test Resmi', resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### Alıştırmalarda Kullanım
Bu resimler alıştırmalarda otomatik olarak kullanılır:
- Alıştırma 1: `normal.jpg`, `renkli-sekiller.png`
- Alıştırma 2: `gradyan.jpg`, `dusuk-kontrast.jpg`, `gurultulu.jpg`
- Alıştırma 3: Tüm resimler (batch işlem için)

## 🛠️ Resim Özellikleri

| Resim | Boyut | Format | Özellik |
|-------|-------|---------|---------|
| gradyan.jpg | 300x400 | JPEG | Renk geçişleri |
| dusuk-kontrast.jpg | 250x350 | JPEG | Dar değer aralığı (80-120) |
| renkli-sekiller.png | 400x500 | PNG | 6 farklı renk |
| gurultulu.jpg | 300x400 | JPEG | Gaussian gürültü (σ=25) |
| bulanik.jpg | 300x400 | JPEG | Gaussian blur (15x15) |

## 📝 Notlar

- Tüm resimler programatik olarak oluşturulmuştur
- Telif hakkı sorunu yoktur
- Eğitim amaçlı kullanım için optimize edilmiştir
- İhtiyaç halinde `resim_olusturucu.py` ile yeniden oluşturulabilir

---

**💡 İpucu:** Kendi test resimlerinizi de ekleyebilirsiniz!
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("📝 README.md dosyası oluşturuldu")

def main():
    """Ana fonksiyon - Tüm test resimlerini oluşturur"""
    print("🎨 OpenCV Test Resimleri Oluşturucu")
    print("=" * 45)
    print("Bu script, alıştırmalar için gerekli test resimlerini oluşturur.\\n")
    
    try:
        # Test resimlerini oluştur
        create_gradient_image()
        create_low_contrast_image()
        create_high_contrast_image()
        create_colorful_shapes()
        create_noisy_image()
        create_blurry_image()
        create_bright_image()
        create_dark_image()
        create_normal_image()
        create_test_patterns()
        create_face_detection_sample()
        create_document_sample()
        create_coin_detection_sample()
        create_readme()
        
        print("\\n🎉 Tüm test resimleri başarıyla oluşturuldu!")
        print("\\n📊 Oluşturulan Dosyalar:")
        
        # Oluşturulan dosyaları listele
        resim_dosyalari = [
            "gradyan.jpg", "dusuk-kontrast.jpg", "yuksek-kontrast.jpg",
            "renkli-sekiller.png", "gurultulu.jpg", "bulanik.jpg",
            "asiri-parlak.jpg", "karanlik.jpg", "normal.jpg",
            "satranc-tahtasi.png", "renk-testi.png", "cizgi-testi.png",
            "yuz-ornegi.jpg", "belge-ornegi.jpg", "para-tespiti.jpg",
            "README.md"
        ]
        
        for i, dosya in enumerate(resim_dosyalari, 1):
            print(f"   {i:2d}. {dosya}")
        
        print(f"\\n✅ Toplam {len(resim_dosyalari)} dosya oluşturuldu!")
        print("\\n🚀 Artık alıştırmaları çalıştırabilirsiniz!")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()