"""
🔄 OpenCV Geometrik Transformasyonlar
====================================

Bu dosyada resimler üzerinde geometrik dönüşümler öğreneceksiniz:
- Döndürme (rotation)
- Ölçekleme (scaling) 
- Öteleme (translation)
- Perspektif düzeltme
- Affine transformasyonlar

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def ornek_resim_olustur():
    """Test için örnek resimler oluştur"""
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Basit geometrik şekiller içeren resim
    resim = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Arka plan gradyanı
    for i in range(400):
        resim[i, :] = [i//2, 100, 255-i//2]
    
    # Geometrik şekiller ekle
    cv2.rectangle(resim, (50, 50), (150, 150), (0, 255, 0), -1)
    cv2.circle(resim, (300, 100), 50, (255, 0, 0), -1)
    cv2.ellipse(resim, (200, 300), (80, 40), 45, 0, 360, (0, 255, 255), -1)
    
    # Metin ekle
    cv2.putText(resim, 'TRANSFORM', (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2)
    
    # Koordinat çizgileri
    cv2.line(resim, (200, 0), (200, 400), (128, 128, 128), 1)
    cv2.line(resim, (0, 200), (400, 200), (128, 128, 128), 1)
    
    dosya_yolu = examples_dir / "transform_test.jpg"
    cv2.imwrite(str(dosya_yolu), resim)
    print(f"✅ Test resmi oluşturuldu: {dosya_yolu}")
    
    return str(dosya_yolu)

def dondurme_ornekleri(resim):
    """Resim döndürme örnekleri"""
    print("\n🔄 Resim Döndürme Örnekleri")
    print("=" * 35)
    
    yukseklik, genislik = resim.shape[:2]
    merkez = (genislik // 2, yukseklik // 2)
    
    # 1. Basit döndürme - 45 derece
    rotasyon_matrisi_45 = cv2.getRotationMatrix2D(merkez, 45, 1.0)
    dondurulmus_45 = cv2.warpAffine(resim, rotasyon_matrisi_45, (genislik, yukseklik))
    
    # 2. Ölçekli döndürme - 30 derece, %80 boyut
    rotasyon_matrisi_30 = cv2.getRotationMatrix2D(merkez, 30, 0.8)
    dondurulmus_30 = cv2.warpAffine(resim, rotasyon_matrisi_30, (genislik, yukseklik))
    
    # 3. Sınırları koruyarak döndürme
    rotasyon_matrisi_90 = cv2.getRotationMatrix2D(merkez, 90, 1.0)
    
    # Yeni boyutları hesapla
    cos_val = np.abs(rotasyon_matrisi_90[0, 0])
    sin_val = np.abs(rotasyon_matrisi_90[0, 1])
    yeni_genislik = int((yukseklik * sin_val) + (genislik * cos_val))
    yeni_yukseklik = int((yukseklik * cos_val) + (genislik * sin_val))
    
    # Merkezi ayarla
    rotasyon_matrisi_90[0, 2] += (yeni_genislik / 2) - merkez[0]
    rotasyon_matrisi_90[1, 2] += (yeni_yukseklik / 2) - merkez[1]
    
    dondurulmus_90 = cv2.warpAffine(resim, rotasyon_matrisi_90, 
                                   (yeni_genislik, yeni_yukseklik))
    
    # Sonuçları göster
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(dondurulmus_45, cv2.COLOR_BGR2RGB))
    plt.title('45° Döndürülmüş')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(dondurulmus_30, cv2.COLOR_BGR2RGB))
    plt.title('30° Döndürülmüş + %80 Ölçek')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(dondurulmus_90, cv2.COLOR_BGR2RGB))
    plt.title('90° Döndürülmüş (Sınırlar Korundu)')
    plt.axis('off')
    
    # Rotasyon matrisini göster
    plt.subplot(2, 3, 5)
    plt.text(0.1, 0.8, f'45° Rotasyon Matrisi:\n{rotasyon_matrisi_45}', 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    plt.text(0.1, 0.4, 'Rotasyon Matrisi Formatı:\n[cos(θ) -sin(θ) tx]\n[sin(θ)  cos(θ) ty]', 
             fontsize=10, verticalalignment='top')
    plt.axis('off')
    plt.title('Matris Bilgileri')
    
    plt.subplot(2, 3, 6)
    # Döndürme açısı karşılaştırması
    aclar = [0, 90, 180, 270]
    for i, aci in enumerate(aclar):
        if i < 4:
            rot_mat = cv2.getRotationMatrix2D(merkez, aci, 0.3)
            dondurulmus = cv2.warpAffine(resim, rot_mat, (genislik//3, yukseklik//3))
            
            # Küçük alt pencereler için pozisyonu hesapla
            y_offset = 0.4 if i < 2 else -0.1
            x_offset = 0.1 + (i % 2) * 0.4
            
            plt.text(x_offset, y_offset, f'{aci}°', fontsize=8, 
                    transform=plt.gca().transAxes, ha='center')
    
    plt.text(0.05, 0.9, 'Farklı Açılar:', fontsize=10, weight='bold', 
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.7, '0°, 90°, 180°, 270°\naçılarında döndürme\nörnekleri', 
             fontsize=8, transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Döndürme Örnekleri')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Döndürme İpuçları:")
    print("   • getRotationMatrix2D(merkez, açı, ölçek) kullanın")
    print("   • Pozitif açı saat yönünün tersine döndürür")
    print("   • Ölçek parametresi ile boyut değişikliği yapabilirsiniz")
    print("   • Sınırları korumak için yeni boyutları hesaplayın")

def olcekleme_ornekleri(resim):
    """Resim ölçekleme örnekleri"""
    print("\n📏 Resim Ölçekleme Örnekleri")
    print("=" * 35)
    
    yukseklik, genislik = resim.shape[:2]
    
    # 1. Basit ölçekleme
    buyutulmus = cv2.resize(resim, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    kucultulmus = cv2.resize(resim, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    
    # 2. Belirli boyuta ölçekleme
    yeni_boyut = cv2.resize(resim, (300, 200), interpolation=cv2.INTER_LINEAR)
    
    # 3. Farklı interpolasyon yöntemleri
    yakin_komsuluk = cv2.resize(resim, (200, 200), interpolation=cv2.INTER_NEAREST)
    linear = cv2.resize(resim, (200, 200), interpolation=cv2.INTER_LINEAR)
    cubic = cv2.resize(resim, (200, 200), interpolation=cv2.INTER_CUBIC)
    lanczos = cv2.resize(resim, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    
    # 4. En-boy oranını koruyarak ölçekleme
    def boyut_orani_koru(resim, max_boyut=300):
        h, w = resim.shape[:2]
        if w > h:
            yeni_w = max_boyut
            yeni_h = int(h * (max_boyut / w))
        else:
            yeni_h = max_boyut
            yeni_w = int(w * (max_boyut / h))
        return cv2.resize(resim, (yeni_w, yeni_h), interpolation=cv2.INTER_LINEAR)
    
    oran_korunmus = boyut_orani_koru(resim)
    
    # Sonuçları göster
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title(f'Orijinal\n{genislik}x{yukseklik}')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(buyutulmus, cv2.COLOR_BGR2RGB))
    plt.title(f'%150 Büyütülmüş\n{buyutulmus.shape[1]}x{buyutulmus.shape[0]}')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(kucultulmus, cv2.COLOR_BGR2RGB))
    plt.title(f'%50 Küçültülmüş\n{kucultulmus.shape[1]}x{kucultulmus.shape[0]}')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(cv2.cvtColor(yeni_boyut, cv2.COLOR_BGR2RGB))
    plt.title(f'Sabit Boyut\n{yeni_boyut.shape[1]}x{yeni_boyut.shape[0]}')
    plt.axis('off')
    
    # İnterpolasyon karşılaştırması
    plt.subplot(3, 4, 5)
    plt.imshow(cv2.cvtColor(yakin_komsuluk, cv2.COLOR_BGR2RGB))
    plt.title('NEAREST\n(En yakın komşu)')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(cv2.cvtColor(linear, cv2.COLOR_BGR2RGB))
    plt.title('LINEAR\n(Doğrusal)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(cv2.cvtColor(cubic, cv2.COLOR_BGR2RGB))
    plt.title('CUBIC\n(Kübik)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(cv2.cvtColor(lanczos, cv2.COLOR_BGR2RGB))
    plt.title('LANCZOS4\n(Lanczos)')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(oran_korunmus, cv2.COLOR_BGR2RGB))
    plt.title(f'Oran Korundu\n{oran_korunmus.shape[1]}x{oran_korunmus.shape[0]}')
    plt.axis('off')
    
    # İnterpolasyon açıklaması
    plt.subplot(3, 4, 10)
    plt.text(0.05, 0.95, 'İnterpolasyon Türleri:', fontsize=12, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '• NEAREST: En hızlı, pikselleşmiş\n• LINEAR: Hızlı, düzgün\n• CUBIC: Yavaş, çok düzgün\n• LANCZOS4: En yavaş, en kaliteli', 
             fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.40, 'Kullanım Önerileri:', fontsize=12, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.25, '• Büyütme: CUBIC veya LANCZOS4\n• Küçültme: LINEAR yeterli\n• Hız gerekirse: NEAREST', 
             fontsize=10, verticalalignment='top')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Ölçekleme İpuçları:")
    print("   • fx, fy parametreleri ile oran belirleyin")
    print("   • Büyütmede kalite için CUBIC veya LANCZOS4 kullanın")
    print("   • Küçültmede LINEAR genellikle yeterlidir")
    print("   • En-boy oranını korumak için hesaplama yapın")

def oteleme_ornekleri(resim):
    """Resim öteleme örnekleri"""
    print("\n➡️ Resim Öteleme Örnekleri")
    print("=" * 35)
    
    yukseklik, genislik = resim.shape[:2]
    
    # 1. Basit öteleme
    oteleme_matrisi_1 = np.float32([[1, 0, 50], [0, 1, 30]])  # x=50, y=30 öteleme
    otelenmis_1 = cv2.warpAffine(resim, oteleme_matrisi_1, (genislik, yukseklik))
    
    # 2. Negatif öteleme
    oteleme_matrisi_2 = np.float32([[1, 0, -30], [0, 1, -20]])
    otelenmis_2 = cv2.warpAffine(resim, oteleme_matrisi_2, (genislik, yukseklik))
    
    # 3. Büyük öteleme - sınırlar dışına çıkan kısımlar
    oteleme_matrisi_3 = np.float32([[1, 0, 100], [0, 1, 80]])
    otelenmis_3 = cv2.warpAffine(resim, oteleme_matrisi_3, (genislik, yukseklik))
    
    # 4. Sınırları genişleterek öteleme
    oteleme_matrisi_4 = np.float32([[1, 0, 100], [0, 1, 80]])
    otelenmis_4 = cv2.warpAffine(resim, oteleme_matrisi_4, 
                                (genislik + 150, yukseklik + 120))
    
    # 5. Farklı dolgu modları
    # BORDER_CONSTANT - sabit renk ile doldur
    otelenmis_constant = cv2.warpAffine(resim, oteleme_matrisi_1, (genislik, yukseklik),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0))
    
    # BORDER_REFLECT - yansıtmalı dolgu
    otelenmis_reflect = cv2.warpAffine(resim, oteleme_matrisi_1, (genislik, yukseklik),
                                      borderMode=cv2.BORDER_REFLECT)
    
    # BORDER_WRAP - sarmalı dolgu  
    otelenmis_wrap = cv2.warpAffine(resim, oteleme_matrisi_1, (genislik, yukseklik),
                                   borderMode=cv2.BORDER_WRAP)
    
    # Sonuçları göster
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(otelenmis_1, cv2.COLOR_BGR2RGB))
    plt.title('Öteleme (+50, +30)')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(otelenmis_2, cv2.COLOR_BGR2RGB))
    plt.title('Öteleme (-30, -20)')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(cv2.cvtColor(otelenmis_3, cv2.COLOR_BGR2RGB))
    plt.title('Büyük Öteleme (+100, +80)')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(cv2.cvtColor(otelenmis_4, cv2.COLOR_BGR2RGB))
    plt.title('Genişletilmiş Sınırlar')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(cv2.cvtColor(otelenmis_constant, cv2.COLOR_BGR2RGB))
    plt.title('BORDER_CONSTANT\n(Kırmızı dolgu)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(cv2.cvtColor(otelenmis_reflect, cv2.COLOR_BGR2RGB))
    plt.title('BORDER_REFLECT\n(Yansıtmalı)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(cv2.cvtColor(otelenmis_wrap, cv2.COLOR_BGR2RGB))
    plt.title('BORDER_WRAP\n(Sarmalı)')
    plt.axis('off')
    
    # Öteleme matrisi açıklaması
    plt.subplot(3, 4, 9)
    plt.text(0.05, 0.95, 'Öteleme Matrisi Format:', fontsize=12, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '[[1, 0, tx],\n [0, 1, ty]]', fontsize=14, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.text(0.05, 0.50, 'tx: X ekseni ötelemesi\nty: Y ekseni ötelemesi', 
             fontsize=11, verticalalignment='top')
    plt.text(0.05, 0.25, 'Pozitif değerler:\n• tx: Sağa öteleme\n• ty: Aşağı öteleme', 
             fontsize=10, verticalalignment='top')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Öteleme İpuçları:")
    print("   • Öteleme matrisi: [[1, 0, tx], [0, 1, ty]]")
    print("   • Pozitif tx sağa, pozitif ty aşağı öteleme")
    print("   • Sınır dışı alanlar için borderMode kullanın")
    print("   • Büyük ötelemeler için canvas boyutunu artırın")

def affine_donusum_ornekleri(resim):
    """Affine dönüşüm örnekleri"""
    print("\n🔀 Affine Dönüşüm Örnekleri")
    print("=" * 35)
    
    yukseklik, genislik = resim.shape[:2]
    
    # 1. 3 nokta ile affine dönüşüm
    # Kaynak noktalar
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    # Hedef noktalar (parallelogram şekli)
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    
    # Affine matrisi hesapla
    affine_matrix_1 = cv2.getAffineTransform(pts1, pts2)
    affine_donusum_1 = cv2.warpAffine(resim, affine_matrix_1, (genislik, yukseklik))
    
    # 2. Farklı affine dönüşüm - kayma (shear)
    pts3 = np.float32([[0, 0], [genislik-1, 0], [0, yukseklik-1]])
    pts4 = np.float32([[50, 0], [genislik-1, 50], [0, yukseklik-1]])
    
    affine_matrix_2 = cv2.getAffineTransform(pts3, pts4)
    affine_donusum_2 = cv2.warpAffine(resim, affine_matrix_2, (genislik, yukseklik))
    
    # 3. Kombine affine dönüşüm (döndürme + ölçekleme + öteleme)
    merkez = (genislik//2, yukseklik//2)
    dondurme_matrix = cv2.getRotationMatrix2D(merkez, 30, 0.8)
    # Ek öteleme ekle
    dondurme_matrix[0, 2] += 50
    dondurme_matrix[1, 2] += 30
    
    kombine_donusum = cv2.warpAffine(resim, dondurme_matrix, (genislik, yukseklik))
    
    # 4. Manuel affine matrisi oluşturma
    # Kayma dönüşümü
    kayma_matrisi = np.float32([[1, 0.3, 0],    # x = x + 0.3*y
                                [0.2, 1, 0]])    # y = 0.2*x + y
    kayma_donusum = cv2.warpAffine(resim, kayma_matrisi, (genislik + 100, yukseklik + 100))
    
    # Noktaları görselleştirmek için yardımcı fonksiyon
    def noktalari_ciz(resim, noktalar, renk=(0, 255, 0), yaricap=5):
        resim_kopya = resim.copy()
        for nokta in noktalar:
            cv2.circle(resim_kopya, tuple(nokta.astype(int)), yaricap, renk, -1)
        return resim_kopya
    
    # Noktalı versiyonlar
    resim_noktali = noktalari_ciz(resim, pts1, (0, 255, 0))
    donusum_noktali = noktalari_ciz(affine_donusum_1, pts2, (255, 0, 0))
    
    # Sonuçları göster
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(resim_noktali, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal + Kaynak Noktalar\n(Yeşil)')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(donusum_noktali, cv2.COLOR_BGR2RGB))
    plt.title('Affine Dönüşüm\n(Kırmızı: Hedef noktalar)')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(affine_donusum_2, cv2.COLOR_BGR2RGB))
    plt.title('Kayma Dönüşümü\n(Shear)')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(cv2.cvtColor(kombine_donusum, cv2.COLOR_BGR2RGB))
    plt.title('Kombine Dönüşüm\n(Döndür+Ölçekle+Ötele)')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(cv2.cvtColor(kayma_donusum, cv2.COLOR_BGR2RGB))
    plt.title('Manuel Kayma Matrisi')
    plt.axis('off')
    
    # Matris bilgileri
    plt.subplot(3, 4, 6)
    plt.text(0.05, 0.95, 'İlk Affine Matris:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, f'{affine_matrix_1}', fontsize=8, verticalalignment='top',
             fontfamily='monospace')
    plt.text(0.05, 0.45, 'Kaynak Noktalar:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.30, f'{pts1}', fontsize=8, verticalalignment='top',
             fontfamily='monospace')
    plt.text(0.05, 0.10, 'Hedef Noktalar:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, -0.05, f'{pts2}', fontsize=8, verticalalignment='top',
             fontfamily='monospace')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.text(0.05, 0.95, 'Affine Dönüşüm Özellikleri:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '✓ Paralel çizgiler paralel kalır\n✓ Oranlar korunur\n✓ 3 nokta ile tanımlanır', 
             fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.45, 'Kullanım Alanları:', fontsize=11, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.30, '• Belge düzeltme\n• Perspektif simülasyonu\n• Resim çarpıtma efektleri', 
             fontsize=10, verticalalignment='top')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Affine Dönüşüm İpuçları:")
    print("   • getAffineTransform() ile 3 nokta çifti kullanın")
    print("   • Paralel çizgiler paralel kalır, açılar değişebilir")
    print("   • Döndürme, ölçekleme, öteleme, kayma kombinasyonu")
    print("   • Manuel matris: [[a, b, tx], [c, d, ty]]")

def perspektif_donusum_ornekleri(resim):
    """Perspektif dönüşüm örnekleri"""
    print("\n🏛️ Perspektif Dönüşüm Örnekleri")
    print("=" * 35)
    
    yukseklik, genislik = resim.shape[:2]
    
    # 1. Belge düzeltme simülasyonu
    # Eğimli belge köşe noktaları (perspektif bozulmuş)
    pts1 = np.float32([[50, 80], [350, 50], [380, 350], [20, 320]])
    # Düzeltilmiş dikdörtgen köşeler
    pts2 = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    
    perspektif_matrix_1 = cv2.getPerspectiveTransform(pts1, pts2)
    perspektif_donusum_1 = cv2.warpPerspective(resim, perspektif_matrix_1, (300, 300))
    
    # 2. 3D efekti - trapezoid'den dikdörtgene
    pts3 = np.float32([[0, 0], [genislik, 0], [genislik, yukseklik], [0, yukseklik]])
    pts4 = np.float32([[100, 0], [genislik-100, 0], [genislik, yukseklik], [0, yukseklik]])
    
    perspektif_matrix_2 = cv2.getPerspectiveTransform(pts3, pts4)
    perspektif_donusum_2 = cv2.warpPerspective(resim, perspektif_matrix_2, 
                                               (genislik, yukseklik))
    
    # 3. Ters perspektif - dikdörtgenden trapezoid'e
    pts5 = np.float32([[0, 0], [genislik, 0], [genislik, yukseklik], [0, yukseklik]])
    pts6 = np.float32([[50, 100], [genislik-50, 80], [genislik-20, yukseklik-50], [70, yukseklik-70]])
    
    perspektif_matrix_3 = cv2.getPerspectiveTransform(pts5, pts6)
    perspektif_donusum_3 = cv2.warpPerspective(resim, perspektif_matrix_3, 
                                               (genislik, yukseklik))
    
    # 4. Extreme perspektif - "sonsuzluk noktası" efekti
    pts7 = np.float32([[0, 0], [genislik, 0], [genislik, yukseklik], [0, yukseklik]])
    pts8 = np.float32([[150, 50], [genislik-150, 50], [300, 300], [100, 300]])
    
    perspektif_matrix_4 = cv2.getPerspectiveTransform(pts7, pts8)
    perspektif_donusum_4 = cv2.warpPerspective(resim, perspektif_matrix_4, 
                                               (genislik, yukseklik))
    
    # Noktaları görselleştirmek için yardımcı fonksiyon
    def perspektif_noktalari_ciz(resim, kaynak_pts, hedef_pts):
        resim_kopya = resim.copy()
        
        # Kaynak noktaları yeşil ile işaretle
        for i, nokta in enumerate(kaynak_pts):
            cv2.circle(resim_kopya, tuple(nokta.astype(int)), 8, (0, 255, 0), -1)
            cv2.putText(resim_kopya, str(i+1), tuple(nokta.astype(int) + [10, -10]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Kaynak noktaları çizgi ile birleştir
        cv2.polylines(resim_kopya, [kaynak_pts.astype(int)], True, (0, 255, 0), 2)
        
        return resim_kopya
    
    # Noktalı versiyonlar
    resim_noktali_1 = perspektif_noktalari_ciz(resim, pts1, pts2)
    resim_noktali_2 = perspektif_noktalari_ciz(resim, pts3, pts4)
    
    # Sonuçları göster
    plt.figure(figsize=(20, 15))
    
    plt.subplot(4, 4, 1)
    plt.imshow(cv2.cvtColor(resim_noktali_1, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal + Kaynak Noktalar\n(Eğimli belge)')
    plt.axis('off')
    
    plt.subplot(4, 4, 2)
    plt.imshow(cv2.cvtColor(perspektif_donusum_1, cv2.COLOR_BGR2RGB))
    plt.title('Perspektif Düzeltilmiş\n(Belge düzeltme)')
    plt.axis('off')
    
    plt.subplot(4, 4, 3)
    plt.imshow(cv2.cvtColor(resim_noktali_2, cv2.COLOR_BGR2RGB))
    plt.title('3D Efekt Kaynak')
    plt.axis('off')
    
    plt.subplot(4, 4, 4)
    plt.imshow(cv2.cvtColor(perspektif_donusum_2, cv2.COLOR_BGR2RGB))
    plt.title('3D Efekt Uygulanmış\n(Trapezoid)')
    plt.axis('off')
    
    plt.subplot(4, 4, 5)
    plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Resim')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    plt.imshow(cv2.cvtColor(perspektif_donusum_3, cv2.COLOR_BGR2RGB))
    plt.title('Ters Perspektif\n(Trapezoid efekti)')
    plt.axis('off')
    
    plt.subplot(4, 4, 7)
    plt.imshow(cv2.cvtColor(perspektif_donusum_4, cv2.COLOR_BGR2RGB))
    plt.title('Extreme Perspektif\n("Sonsuzluk noktası")')
    plt.axis('off')
    
    # Matris bilgileri
    plt.subplot(4, 4, 8)
    plt.text(0.05, 0.95, 'Perspektif Matris (3x3):', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, f'{perspektif_matrix_1[0]}\n{perspektif_matrix_1[1]}\n{perspektif_matrix_1[2]}', 
             fontsize=7, verticalalignment='top', fontfamily='monospace')
    plt.text(0.05, 0.35, 'Affine vs Perspektif:', fontsize=10, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.20, '• Affine: 3 nokta, 2x3 matris\n• Perspektif: 4 nokta, 3x3 matris\n• Perspektif: Paralel çizgiler kesişebilir', 
             fontsize=8, verticalalignment='top')
    plt.axis('off')
    
    # Kullanım alanları
    plt.subplot(4, 4, 9)
    plt.text(0.05, 0.95, 'Kullanım Alanları:', fontsize=12, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.80, '📄 Belge tarama ve düzeltme\n🏢 Mimari fotoğraf düzeltme\n🎨 Sanatsal efektler\n📱 QR kod okuma', 
             fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.35, 'Dikkat Edilecekler:', fontsize=12, weight='bold', 
             verticalalignment='top')
    plt.text(0.05, 0.20, '⚠️ Extreme dönüşümler piksel kaybına neden olur\n⚠️ 4 nokta sıralı olmalı\n⚠️ Noktalar çakışmamalı', 
             fontsize=10, verticalalignment='top')
    plt.axis('off')
    
    # Nokta sıralama örneği
    plt.subplot(4, 4, 10)
    plt.text(0.05, 0.95, 'Doğru Nokta Sıralaması:', fontsize=11, weight='bold', 
             verticalalignment='top')
    
    # Basit şema çiz
    plt.plot([0.2, 0.8, 0.8, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2, 0.8], 'b-', linewidth=2)
    plt.text(0.15, 0.85, '1', fontsize=14, weight='bold', color='red')
    plt.text(0.85, 0.85, '2', fontsize=14, weight='bold', color='red')
    plt.text(0.85, 0.15, '3', fontsize=14, weight='bold', color='red')
    plt.text(0.15, 0.15, '4', fontsize=14, weight='bold', color='red')
    
    plt.text(0.05, 0.05, 'Saat yönünde: Sol üst → Sağ üst → Sağ alt → Sol alt', 
             fontsize=8, verticalalignment='bottom')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📝 Perspektif Dönüşüm İpuçları:")
    print("   • getPerspectiveTransform() ile 4 nokta çifti kullanın")
    print("   • Paralel çizgiler kesişebilir (3D efekt)")
    print("   • Belge tarama için ideal")
    print("   • Noktalar saat yönünde sıralanmalı")
    print("   • warpPerspective() fonksiyonunu kullanın")

def interaktif_transform_demo():
    """İnteraktif transformasyon demosu"""
    print("\n🎮 İnteraktif Transformasyon Demosu")
    print("=" * 40)
    print("Trackbar'ları kullanarak gerçek zamanlı dönüşüm görün!")
    
    # Test resmi oluştur veya yükle
    resim_yolu = ornek_resim_olustur()
    resim = cv2.imread(resim_yolu)
    
    if resim is None:
        print("❌ Resim yüklenemedi!")
        return
    
    # Resmi küçült (performans için)
    resim = cv2.resize(resim, (300, 300))
    yukseklik, genislik = resim.shape[:2]
    merkez = (genislik//2, yukseklik//2)
    
    # Pencere oluştur
    window_name = 'Interactive Transform'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Trackbar'lar oluştur
    cv2.createTrackbar('Rotation', window_name, 0, 360, lambda x: None)
    cv2.createTrackbar('Scale %', window_name, 100, 200, lambda x: None)
    cv2.createTrackbar('X Move', window_name, 150, 300, lambda x: None)
    cv2.createTrackbar('Y Move', window_name, 150, 300, lambda x: None)
    cv2.createTrackbar('Shear X', window_name, 50, 100, lambda x: None)
    cv2.createTrackbar('Shear Y', window_name, 50, 100, lambda x: None)
    
    print("🎛️ Kontroller:")
    print("   • Döndürme Açısı: 0-360 derece")
    print("   • Ölçek: %50-200")
    print("   • X/Y Öteleme: -150 ile +150 piksel")
    print("   • Kayma: -50 ile +50")
    print("   • ESC tuşu ile çıkış")
    
    while True:
        # Trackbar değerlerini oku
        aci = cv2.getTrackbarPos('Rotation', window_name)
        olcek = cv2.getTrackbarPos('Scale %', window_name) / 100.0
        x_oteleme = cv2.getTrackbarPos('X Move', window_name) - 150
        y_oteleme = cv2.getTrackbarPos('Y Move', window_name) - 150
        kayma_x = (cv2.getTrackbarPos('Shear X', window_name) - 50) / 100.0
        kayma_y = (cv2.getTrackbarPos('Shear Y', window_name) - 50) / 100.0
        
        # Minimum ölçek kontrolü
        if olcek < 0.1:
            olcek = 0.1
        
        # Rotasyon ve ölçekleme matrisi
        rot_matrix = cv2.getRotationMatrix2D(merkez, aci, olcek)
        
        # Öteleme ekle
        rot_matrix[0, 2] += x_oteleme
        rot_matrix[1, 2] += y_oteleme
        
        # Kayma ekle
        rot_matrix[0, 0] += kayma_x
        rot_matrix[1, 1] += kayma_y
        
        # Dönüşümü uygula
        donusturulmus = cv2.warpAffine(resim, rot_matrix, (genislik + 200, yukseklik + 200),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(50, 50, 50))
        
        # Bilgi metnini ekle
        cv2.putText(donusturulmus, f'Aci: {aci}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(donusturulmus, f'Olcek: {olcek:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(donusturulmus, f'Oteleme: ({x_oteleme}, {y_oteleme})', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(donusturulmus, f'Kayma: ({kayma_x:.2f}, {kayma_y:.2f})', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(donusturulmus, 'ESC = Cikis', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Merkez noktasını işaretle
        cv2.circle(donusturulmus, (merkez[0] + x_oteleme + 100, merkez[1] + y_oteleme + 100), 
                   3, (0, 0, 255), -1)
        
        # Sonucu göster
        cv2.imshow(window_name, donusturulmus)
        
        # ESC tuşu kontrolü
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC tuşu
            break
    
    cv2.destroyAllWindows()
    print("✅ İnteraktif demo tamamlandı!")

def main():
    """Ana program"""
    print("🔄 OpenCV Geometrik Transformasyonlar")
    print("Bu program, geometrik transformasyon tekniklerini gösterir.\n")
    
    # Örnek resim oluştur
    resim_yolu = ornek_resim_olustur()
    resim = cv2.imread(resim_yolu)
    
    if resim is None:
        print("❌ Test resmi oluşturulamadı!")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("🔄 Geometrik Transformasyonlar Menüsü")
        print("=" * 50)
        print("1. Döndürme Örnekleri")
        print("2. Ölçekleme Örnekleri") 
        print("3. Öteleme Örnekleri")
        print("4. Affine Dönüşüm Örnekleri")
        print("5. Perspektif Dönüşüm Örnekleri")
        print("6. İnteraktif Transformasyon Demosu")
        print("0. Çıkış")
        
        try:
            secim = input("\nLütfen bir seçenek girin (0-6): ").strip()
            
            if secim == '0':
                print("👋 Görüşmek üzere!")
                break
            elif secim == '1':
                dondurme_ornekleri(resim)
            elif secim == '2':
                olcekleme_ornekleri(resim)
            elif secim == '3':
                oteleme_ornekleri(resim)
            elif secim == '4':
                affine_donusum_ornekleri(resim)
            elif secim == '5':
                perspektif_donusum_ornekleri(resim)
            elif secim == '6':
                interaktif_transform_demo()
            else:
                print("❌ Geçersiz seçenek! Lütfen 0-6 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program kullanıcı tarafından sonlandırıldı!")
            break
        except Exception as e:
            print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()