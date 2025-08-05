"""
🖼️ Test Resimleri Oluşturucu
============================

Bu script, alıştırmalar için gerekli test resimlerini oluşturur.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
from pathlib import Path

def test_resimleri_olustur():
    """Tüm test resimlerini oluştur"""
    
    print("🖼️ Test Resimleri Oluşturucu")
    print("=" * 35)
    
    # Mevcut dizini al
    current_dir = Path(__file__).parent
    
    # 1. Normal test resmi
    print("\n1️⃣ Normal test resmi oluşturuluyor...")
    normal_resim = normal_test_resmi()
    normal_path = current_dir / "normal.jpg"
    cv2.imwrite(str(normal_path), normal_resim)
    print(f"   ✅ Kaydedildi: {normal_path}")
    
    # 2. Düşük kontrastlı resim
    print("\n2️⃣ Düşük kontrastlı resim oluşturuluyor...")
    dusuk_kontrast_resim = dusuk_kontrast_resmi()
    dusuk_kontrast_path = current_dir / "dusuk_kontrast.jpg"
    cv2.imwrite(str(dusuk_kontrast_path), dusuk_kontrast_resim)
    print(f"   ✅ Kaydedildi: {dusuk_kontrast_path}")
    
    # 3. Gürültülü resim
    print("\n3️⃣ Gürültülü resim oluşturuluyor...")
    gurultulu_resim = gurultulu_test_resmi()
    gurultulu_path = current_dir / "gurultulu.jpg"
    cv2.imwrite(str(gurultulu_path), gurultulu_resim)
    print(f"   ✅ Kaydedildi: {gurultulu_path}")
    
    # 4. Perspektif bozulmuş resim
    print("\n4️⃣ Perspektif bozulmuş resim oluşturuluyor...")
    perspektif_resim = perspektif_test_resmi()
    perspektif_path = current_dir / "perspektif.jpg"
    cv2.imwrite(str(perspektif_path), perspektif_resim)
    print(f"   ✅ Kaydedildi: {perspektif_path}")
    
    # 5. Kenar algılama test resmi
    print("\n5️⃣ Kenar algılama test resmi oluşturuluyor...")
    kenar_resim = kenar_test_resmi()
    kenar_path = current_dir / "kenar_test.jpg"
    cv2.imwrite(str(kenar_path), kenar_resim)
    print(f"   ✅ Kaydedildi: {kenar_path}")
    
    print(f"\n🎉 Tüm test resimleri oluşturuldu!")
    print(f"📁 Konum: {current_dir}")
    print(f"📊 Toplam: 5 resim dosyası")

def normal_test_resmi():
    """Normal test resmi oluştur"""
    resim = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Gradient arka plan
    for i in range(400):
        for j in range(400):
            r = int(120 + 60 * np.sin(i/60) * np.cos(j/60))
            g = int(130 + 50 * np.cos(i/50))
            b = int(140 + 40 * np.sin((i+j)/70))
            resim[i, j] = [np.clip(b, 50, 200), np.clip(g, 50, 200), np.clip(r, 50, 200)]
    
    # Geometrik şekiller
    cv2.rectangle(resim, (50, 50), (180, 180), (255, 255, 255), -1)
    cv2.rectangle(resim, (70, 70), (160, 160), (100, 100, 100), -1)
    
    cv2.circle(resim, (300, 100), 60, (200, 100, 100), -1)
    cv2.circle(resim, (300, 100), 30, (255, 200, 200), -1)
    
    # Üçgen
    triangle = np.array([[100, 250], [200, 250], [150, 350]], np.int32)
    cv2.fillPoly(resim, [triangle], (100, 200, 100))
    
    # Çizgiler
    cv2.line(resim, (250, 200), (350, 300), (255, 255, 255), 3)
    cv2.line(resim, (250, 300), (350, 200), (255, 255, 255), 3)
    
    # Metin
    cv2.putText(resim, 'TEST', (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return resim

def dusuk_kontrast_resmi():
    """Düşük kontrastlı test resmi"""
    resim = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Dar aralıkta gradient
    for i in range(300):
        for j in range(300):
            base_value = 128
            variation = int(25 * np.sin(i/80) * np.cos(j/80))
            value = base_value + variation
            resim[i, j] = [value-5, value, value+5]
    
    # Düşük kontrastlı şekiller
    cv2.rectangle(resim, (50, 50), (200, 200), (150, 155, 160), -1)
    cv2.circle(resim, (150, 150), 40, (120, 125, 130), -1)
    cv2.ellipse(resim, (100, 250), (30, 15), 0, 0, 360, (140, 145, 150), -1)
    
    return resim

def gurultulu_test_resmi():
    """Karma gürültülü test resmi"""
    normal_resim = normal_test_resmi()
    gurultulu = normal_resim.astype(np.float32)
    
    # Gaussian gürültü
    gaussian_noise = np.random.normal(0, 20, normal_resim.shape)
    gurultulu += gaussian_noise
    
    # Salt & Pepper gürültü
    salt_mask = np.random.random(normal_resim.shape[:2]) < 0.02
    pepper_mask = np.random.random(normal_resim.shape[:2]) < 0.02
    
    gurultulu[salt_mask] = 255
    gurultulu[pepper_mask] = 0
    
    # Speckle gürültü
    speckle = np.random.normal(0, 0.1, normal_resim.shape)
    gurultulu *= (1 + speckle)
    
    return np.clip(gurultulu, 0, 255).astype(np.uint8)

def perspektif_test_resmi():
    """Perspektif bozulmuş test resmi"""
    normal_resim = normal_test_resmi()
    
    # Perspektif transformasyon matrisi
    src_points = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
    dst_points = np.float32([[50, 30], [350, 20], [380, 370], [20, 350]])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    perspektif = cv2.warpPerspective(normal_resim, matrix, (400, 400))
    
    return perspektif

def kenar_test_resmi():
    """Kenar algılama için özel test resmi"""
    resim = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Arka plan
    for i in range(400):
        for j in range(400):
            value = int(100 + 30 * np.sin(i/60) * np.cos(j/60))
            resim[i, j] = [value, value+10, value+5]
    
    # Net kenarları olan şekiller
    # Dikdörtgenler
    cv2.rectangle(resim, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.rectangle(resim, (70, 70), (130, 130), (0, 0, 0), -1)
    
    # Çemberler  
    cv2.circle(resim, (300, 100), 50, (200, 200, 200), -1)
    cv2.circle(resim, (300, 100), 25, (100, 100, 100), -1)
    
    # Üçgenler
    triangle1 = np.array([[100, 250], [150, 350], [50, 350]], np.int32)
    cv2.fillPoly(resim, [triangle1], (180, 180, 180))
    
    triangle2 = np.array([[300, 250], [350, 350], [250, 350]], np.int32)
    cv2.fillPoly(resim, [triangle2], (160, 160, 160))
    
    # İnce çizgiler
    for i in range(200, 380, 20):
        cv2.line(resim, (i, 200), (i, 220), (255, 255, 255), 1)
    
    # Çapraz çizgiler
    cv2.line(resim, (250, 50), (350, 150), (255, 255, 255), 2)
    cv2.line(resim, (250, 150), (350, 50), (255, 255, 255), 2)
    
    return resim

def resim_bilgileri_goster():
    """Oluşturulan resimlerin bilgilerini göster"""
    current_dir = Path(__file__).parent
    
    print("\n📊 OLUŞTURULAN RESİMLER")
    print("=" * 40)
    
    resim_listesi = [
        ("normal.jpg", "Normal test resmi - temel işlemler için"),
        ("dusuk_kontrast.jpg", "Düşük kontrastlı - kontrast iyileştirme için"),
        ("gurultulu.jpg", "Karma gürültülü - gürültü temizleme için"),
        ("perspektif.jpg", "Perspektif bozulmuş - geometrik düzeltme için"),
        ("kenar_test.jpg", "Kenar algılama - edge detection için")
    ]
    
    for dosya_adi, aciklama in resim_listesi:
        dosya_yolu = current_dir / dosya_adi
        if dosya_yolu.exists():
            # Dosya boyutunu al
            boyut = dosya_yolu.stat().st_size
            boyut_kb = boyut / 1024
            
            # Resim boyutlarını al
            resim = cv2.imread(str(dosya_yolu))
            if resim is not None:
                h, w = resim.shape[:2]
                print(f"✅ {dosya_adi}")
                print(f"   📐 Boyut: {w}x{h} piksel")
                print(f"   💾 Dosya: {boyut_kb:.1f} KB")
                print(f"   📝 Açıklama: {aciklama}")
            else:
                print(f"❌ {dosya_adi} - Okunamadı")
        else:
            print(f"❌ {dosya_adi} - Bulunamadı")
        print()

if __name__ == "__main__":
    print("🖼️ OpenCV Alıştırmaları - Test Resim Oluşturucu")
    print("Bu script alıştırmalar için gerekli test resimlerini oluşturur.\n")
    
    try:
        test_resimleri_olustur()
        resim_bilgileri_goster()
        
        print("💡 Kullanım:")
        print("   Bu script'i çalıştırdıktan sonra alıştırma dosyalarını çalıştırabilirsiniz.")
        print("   Örnek: python ../alistirma-1.py")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        print("\n🔧 Sorun giderme:")
        print("   • Yazma izinlerinizi kontrol edin")
        print("   • Klasör yolunun doğru olduğundan emin olun")
        print("   • OpenCV ve NumPy yüklü olduğunu kontrol edin")