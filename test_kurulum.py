#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 OpenCV Kurulum Test Scripti
=============================

Bu script, OpenCV ve gerekli kütüphanelerin doğru kurulup kurulmadığını test eder.

Kullanım:
    python test_kurulum.py

Yazan: Eren Terzi
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, display_name=None):
    """Modül import testini yapar"""
    if display_name is None:
        display_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Bilinmiyor')
        print(f"✅ {display_name}: {version}")
        return True, module
    except ImportError as e:
        print(f"❌ {display_name}: Kurulu değil ({e})")
        return False, None

def test_opencv_functionality():
    """OpenCV temel fonksiyonalitesini test eder"""
    print("\n🔧 OpenCV Fonksiyonalite Testleri")
    print("-" * 35)
    
    try:
        import cv2
        import numpy as np
        
        # Test resmi oluştur
        test_resim = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Geometrik şekil çiz
        cv2.rectangle(test_resim, (50, 50), (150, 150), (0, 255, 0), -1)
        cv2.circle(test_resim, (225, 100), 50, (255, 0, 0), -1)
        
        # Metin ekle
        cv2.putText(test_resim, 'OpenCV Test', (75, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        print("✅ Resim oluşturma: Başarılı")
        
        # Renk uzayı dönüşümü
        gri_resim = cv2.cvtColor(test_resim, cv2.COLOR_BGR2GRAY)
        print("✅ Renk uzayı dönüşümü: Başarılı")
        
        # Filtreleme
        bulanik = cv2.GaussianBlur(test_resim, (15, 15), 0)
        print("✅ Gaussian blur: Başarılı")
        
        # Kenar algılama
        kenarlar = cv2.Canny(gri_resim, 50, 150)
        print("✅ Canny kenar algılama: Başarılı")
        
        # Resmi kaydet
        output_path = Path("test_output.jpg")
        cv2.imwrite(str(output_path), test_resim)
        print("✅ Resim kaydetme: Başarılı")
        
        # Resmi göstermeyi dene (GUI varsa)
        try:
            cv2.imshow('OpenCV Kurulum Testi', test_resim)
            print("✅ Resim gösterme: Başarılı")
            print("   💡 Test resmi görüntüleniyor... 3 saniye bekleyin veya bir tuşa basın.")
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        except cv2.error:
            print("⚠️ Resim gösterme: GUI desteksiz ortam (normal)")
        
        # Test dosyasını temizle
        if output_path.exists():
            output_path.unlink()
            
        return True
        
    except Exception as e:
        print(f"❌ OpenCV testi başarısız: {e}")
        return False

def test_jupyter():
    """Jupyter kurulumunu test eder"""
    print("\n📓 Jupyter Test")
    print("-" * 20)
    
    success, jupyter = test_import('jupyter')
    if success:
        try:
            # Jupyter notebook komutunu test et
            import subprocess
            result = subprocess.run(['jupyter', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Jupyter komut satırı: Çalışıyor")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"   {line}")
            else:
                print("⚠️ Jupyter komut satırı: Sorun var")
        except Exception as e:
            print(f"⚠️ Jupyter komut testi: {e}")

def test_matplotlib():
    """Matplotlib görselleştirme testini yapar"""
    print("\n📊 Matplotlib Test")
    print("-" * 25)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUI olmayan backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Basit grafik oluştur
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, 'b-', label='sin(x)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('OpenCV Kurulum Test Grafiği')
        plt.legend()
        plt.grid(True)
        
        # Grafiği kaydet
        plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Matplotlib grafik oluşturma: Başarılı")
        
        # Test dosyasını temizle
        Path('test_plot.png').unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Matplotlib testi başarısız: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🧪 OpenCV Dokümantasyon Projesi - Kurulum Testi")
    print("=" * 55)
    print(f"Python sürümü: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Temel modül testleri
    print("📦 Temel Paket Testleri")
    print("-" * 25)
    
    critical_modules = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    optional_modules = [
        ('jupyter', 'Jupyter'),
        ('sklearn', 'Scikit-learn'),
        ('PIL', 'Pillow'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'Tqdm'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
    ]
    
    # Kritik modüller
    critical_success = 0
    for module, name in critical_modules:
        success, _ = test_import(module, name)
        if success:
            critical_success += 1
    
    # Opsiyonel modüller
    optional_success = 0
    print(f"\n📚 Opsiyonel Paket Testleri")
    print("-" * 30)
    for module, name in optional_modules:
        success, _ = test_import(module, name)
        if success:
            optional_success += 1
    
    # Fonksiyonalite testleri
    opencv_test = test_opencv_functionality()
    matplotlib_test = test_matplotlib()
    test_jupyter()
    
    # Sonuç özeti
    print("\n" + "=" * 55)
    print("📋 TEST SONUÇLARI ÖZETİ")
    print("=" * 55)
    
    print(f"✅ Kritik paketler: {critical_success}/{len(critical_modules)}")
    print(f"📚 Opsiyonel paketler: {optional_success}/{len(optional_modules)}")
    print(f"🔧 OpenCV fonksiyonları: {'✅ Başarılı' if opencv_test else '❌ Başarısız'}")
    print(f"📊 Matplotlib: {'✅ Başarılı' if matplotlib_test else '❌ Başarısız'}")
    
    # Genel değerlendirme
    if critical_success == len(critical_modules) and opencv_test:
        print("\n🎉 TEBRİKLER! Kurulum başarıyla tamamlandı!")
        print("🚀 Artık OpenCV dokümantasyonunu kullanmaya başlayabilirsiniz!")
        print("\n📚 Sonraki adımlar:")
        print("   1. cd 01-Temeller")
        print("   2. python 02-ilk-program.py")
        print("   3. Alıştırmaları sırayla yapın")
        
        if optional_success < len(optional_modules) // 2:
            print("\n💡 İpucu: Daha iyi deneyim için ek paketleri kurun:")
            print("   pip install -r requirements.txt")
            
    elif critical_success >= len(critical_modules) - 1:
        print("\n⚠️ Kurulum kısmen başarılı!")
        print("Temel özellikler çalışıyor ama bazı sorunlar var.")
        print("Dokümantasyonun çoğunu kullanabilirsiniz.")
        
    else:
        print("\n❌ Kurulumda ciddi sorunlar var!")
        print("Lütfen INSTALL.md dosyasındaki talimatları takip edin.")
        print("Veya GitHub'da issue açın: https://github.com/erent8/opencv-doc/issues")
    
    print("\n" + "=" * 55)
    print("Test tamamlandı!")

if __name__ == "__main__":
    main()