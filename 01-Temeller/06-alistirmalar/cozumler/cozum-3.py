#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ Çözüm 3: Mini Proje - Akıllı Fotoğraf Düzenleyici
=================================================

Bu dosya, Alıştırma 3'ün örnek çözümlerini içerir.
Gerçek bir proje geliştirme deneyimi sunar.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
import json
from typing import List, Dict, Tuple, Optional

class FotografDuzenleyici:
    """✅ Akıllı Fotoğraf Düzenleyici Sınıfı - ÇÖZÜM"""
    
    def __init__(self):
        self.orijinal_resim = None
        self.guncel_resim = None
        self.gecmis = []  # Geri alma için işlem geçmişi
        self.max_gecmis = 10  # Maximum undo steps
        self.output_dir = Path("ciktiler")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🎨 Fotoğraf Düzenleyici başlatıldı!")
    
    def resim_yukle(self, dosya_yolu: str) -> bool:
        """✅ Güvenli resim yükleme fonksiyonu"""
        try:
            if isinstance(dosya_yolu, str):
                dosya_yolu = Path(dosya_yolu)
            
            if not dosya_yolu.exists():
                print(f"❌ Dosya bulunamadı: {dosya_yolu}")
                return False
            
            resim = cv2.imread(str(dosya_yolu))
            if resim is None:
                print(f"❌ Resim okunamadı: {dosya_yolu}")
                return False
            
            self.orijinal_resim = resim.copy()
            self.guncel_resim = resim.copy()
            self.gecmis = []  # Geçmişi temizle
            
            print(f"✅ Resim yüklendi: {dosya_yolu}")
            print(f"   Boyut: {resim.shape}")
            print(f"   Veri tipi: {resim.dtype}")
            return True
            
        except Exception as e:
            print(f"❌ Resim yükleme hatası: {e}")
            return False
    
    def _gecmise_kaydet(self):
        """İşlem geçmişine kaydet"""
        if self.guncel_resim is not None:
            self.gecmis.append(self.guncel_resim.copy())
            # Maksimum geçmiş sınırı
            if len(self.gecmis) > self.max_gecmis:
                self.gecmis.pop(0)
    
    def geri_al(self) -> bool:
        """✅ Son işlemi geri al"""
        if len(self.gecmis) == 0:
            print("❌ Geri alınacak işlem yok!")
            return False
        
        self.guncel_resim = self.gecmis.pop()
        print("↶ Son işlem geri alındı")
        return True
    
    def otomatik_duzelt(self) -> bool:
        """✅ Otomatik düzeltme fonksiyonu"""
        if self.guncel_resim is None:
            print("❌ Önce resim yükleyin!")
            return False
        
        self._gecmise_kaydet()
        
        # Histogram eşitleme ile kontrast düzeltme
        yuv = cv2.cvtColor(self.guncel_resim, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        duzeltilmis = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # Hafif keskinleştirme
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        duzeltilmis = cv2.filter2D(duzeltilmis, -1, kernel)
        
        self.guncel_resim = duzeltilmis
        print("✨ Otomatik düzeltme uygulandı")
        return True
    
    def parlaklik_ayarla(self, deger: int) -> bool:
        """✅ Parlaklık ayarlama (-100, +100)"""
        if self.guncel_resim is None:
            return False
        
        self._gecmise_kaydet()
        
        # Değeri sınırla
        deger = max(-100, min(100, deger))
        
        # Parlaklık uygula
        self.guncel_resim = cv2.convertScaleAbs(self.guncel_resim, alpha=1, beta=deger)
        print(f"💡 Parlaklık ayarlandı: {deger:+d}")
        return True
    
    def kontrast_ayarla(self, deger: float) -> bool:
        """✅ Kontrast ayarlama (0.5, 3.0)"""
        if self.guncel_resim is None:
            return False
        
        self._gecmise_kaydet()
        
        # Değeri sınırla
        deger = max(0.5, min(3.0, deger))
        
        # Kontrast uygula
        self.guncel_resim = cv2.convertScaleAbs(self.guncel_resim, alpha=deger, beta=0)
        print(f"🔆 Kontrast ayarlandı: {deger:.1f}")
        return True
    
    def bulaniklik_ekle(self, kuvvet: int) -> bool:
        """✅ Gaussian blur uygulama"""
        if self.guncel_resim is None:
            return False
        
        self._gecmise_kaydet()
        
        if kuvvet <= 0:
            return True
        
        # Kernel boyutu (tek sayı olmalı)
        ksize = kuvvet * 2 + 1
        self.guncel_resim = cv2.GaussianBlur(self.guncel_resim, (ksize, ksize), 0)
        print(f"🌫️ Bulanıklaştırma uygulandı: {kuvvet}")
        return True
    
    def keskinlestir(self) -> bool:
        """✅ Unsharp masking ile keskinleştirme"""
        if self.guncel_resim is None:
            return False
        
        self._gecmise_kaydet()
        
        # Gaussian blur uygula
        bulanik = cv2.GaussianBlur(self.guncel_resim, (0, 0), 2.0)
        
        # Unsharp mask
        self.guncel_resim = cv2.addWeighted(self.guncel_resim, 1.5, bulanik, -0.5, 0)
        print("✨ Keskinleştirme uygulandı")
        return True
    
    def kaydet(self, dosya_adi: str) -> bool:
        """Resmi kaydet"""
        if self.guncel_resim is None:
            return False
        
        kayit_yolu = self.output_dir / dosya_adi
        success = cv2.imwrite(str(kayit_yolu), self.guncel_resim)
        
        if success:
            print(f"💾 Resim kaydedildi: {kayit_yolu}")
        else:
            print(f"❌ Kaydetme hatası: {kayit_yolu}")
        
        return success

def gorev_1_sinif_tasarimi():
    """
    ✅ ÇÖZÜM 1: Fotoğraf Düzenleyici Sınıfını Test Etme
    """
    print("🎯 GÖREV 1: Sınıf Tasarımı")
    print("-" * 30)
    
    # Düzenleyici oluştur
    editor = FotografDuzenleyici()
    
    # Test resmi oluştur
    test_dir = Path("test-resimleri")
    test_dir.mkdir(exist_ok=True)
    
    test_resim = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
    cv2.rectangle(test_resim, (100, 100), (300, 200), (100, 150, 200), -1)
    test_yolu = test_dir / "test-resim.jpg"
    cv2.imwrite(str(test_yolu), test_resim)
    
    # Sınıf fonksiyonlarını test et
    print("\\n🧪 Sınıf Test Sonuçları:")
    print(f"   Resim yükleme: {'✅' if editor.resim_yukle(test_yolu) else '❌'}")
    print(f"   Parlaklık ayarlama: {'✅' if editor.parlaklik_ayarla(30) else '❌'}")
    print(f"   Kontrast ayarlama: {'✅' if editor.kontrast_ayarla(1.2) else '❌'}")
    print(f"   Keskinleştirme: {'✅' if editor.keskinlestir() else '❌'}")
    print(f"   Geri alma: {'✅' if editor.geri_al() else '❌'}")
    print(f"   Otomatik düzeltme: {'✅' if editor.otomatik_duzelt() else '❌'}")
    print(f"   Kaydetme: {'✅' if editor.kaydet('test-cikti.jpg') else '❌'}")
    
    # Sonucu göster
    if editor.guncel_resim is not None:
        cv2.imshow('Düzenlenmiş Resim', editor.guncel_resim)
        print("\\nDüzenlenmiş resim gösteriliyor... Herhangi bir tuşa basın.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("✅ Sınıf tasarımı test edildi!")

def gorev_2_batch_islem():
    """
    ✅ ÇÖZÜM 2: Toplu Resim İşleme (Batch Processing)
    """
    print("\\n🎯 GÖREV 2: Batch İşlem")
    print("-" * 25)
    
    # Test resimleri oluştur
    test_dir = Path("test-resimleri")
    test_dir.mkdir(exist_ok=True)
    
    # Farklı kalitede test resimleri oluştur
    test_resimleri = [
        ("dusuk-kontrast.jpg", np.random.randint(80, 120, (200, 300, 3), dtype=np.uint8)),
        ("asiri-parlak.jpg", np.ones((250, 350, 3), dtype=np.uint8) * 200),
        ("karanlik.jpg", np.ones((180, 250, 3), dtype=np.uint8) * 40),
        ("normal.jpg", np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    ]
    
    for isim, resim in test_resimleri:
        cv2.imwrite(str(test_dir / isim), resim)
    
    # Batch işlem
    islenmis_dir = Path("islenmis")
    islenmis_dir.mkdir(exist_ok=True)
    
    editor = FotografDuzenleyici()
    
    baslangic_zamani = time.time()
    islenen_sayi = 0
    toplam_boyut_oncesi = 0
    toplam_boyut_sonrasi = 0
    
    # Tüm jpg dosyalarını işle
    for resim_yolu in test_dir.glob("*.jpg"):
        print(f"\\n📸 İşleniyor: {resim_yolu.name}")
        
        # Orijinal boyut
        onceki_boyut = resim_yolu.stat().st_size
        toplam_boyut_oncesi += onceki_boyut
        
        # Resmi yükle ve işle
        if editor.resim_yukle(resim_yolu):
            # Otomatik düzeltmeler
            editor.otomatik_duzelt()
            
            # Boyut standardizasyonu (800x600)
            if editor.guncel_resim is not None:
                editor.guncel_resim = cv2.resize(editor.guncel_resim, (800, 600))
            
            # JPEG formatında kaydet
            cikti_yolu = islenmis_dir / f"islenmis_{resim_yolu.name}"
            if editor.kaydet(cikti_yolu.name.replace(cikti_yolu.name, str(cikti_yolu))):
                islenen_sayi += 1
                
                # İşlenmiş boyut
                if cikti_yolu.exists():
                    sonraki_boyut = cikti_yolu.stat().st_size
                    toplam_boyut_sonrasi += sonraki_boyut
    
    bitis_zamani = time.time()
    
    # İstatistikleri göster
    print("\\n📊 Batch İşlem İstatistikleri:")
    print(f"   İşlenen dosya sayısı: {islenen_sayi}")
    print(f"   Toplam işlem süresi: {bitis_zamani - baslangic_zamani:.2f} saniye")
    print(f"   Ortalama süre/dosya: {(bitis_zamani - baslangic_zamani) / max(1, islenen_sayi):.2f} saniye")
    
    if toplam_boyut_oncesi > 0:
        ortalama_oncesi = toplam_boyut_oncesi / islenen_sayi / 1024
        ortalama_sonrasi = toplam_boyut_sonrasi / islenen_sayi / 1024
        print(f"   Ortalama dosya boyutu (öncesi): {ortalama_oncesi:.1f} KB")
        print(f"   Ortalama dosya boyutu (sonrası): {ortalama_sonrasi:.1f} KB")
        print(f"   Boyut değişimi: {((ortalama_sonrasi - ortalama_oncesi) / ortalama_oncesi * 100):+.1f}%")
    
    print("✅ Batch işlem tamamlandı!")

def gorev_3_kalite_analizi():
    """
    ✅ ÇÖZÜM 3: Resim Kalite Analizi
    """
    print("\\n🎯 GÖREV 3: Kalite Analizi")
    print("-" * 30)
    
    def kalite_analiz_et(resim: np.ndarray) -> Dict:
        """Resim kalitesini analiz eder"""
        analiz = {}
        
        # Parlaklık analizi
        gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        ortalama_parlaklik = np.mean(gri)
        analiz['parlaklik'] = {
            'deger': ortalama_parlaklik,
            'durum': 'çok karanlık' if ortalama_parlaklik < 80 else 
                    'çok aydınlık' if ortalama_parlaklik > 180 else 'normal'
        }
        
        # Kontrast analizi
        std_sapma = np.std(gri)
        analiz['kontrast'] = {
            'deger': std_sapma,
            'durum': 'düşük' if std_sapma < 30 else 
                    'yüksek' if std_sapma > 80 else 'normal'
        }
        
        # Bulanıklık tespiti (Laplacian variance)
        laplacian_var = cv2.Laplacian(gri, cv2.CV_64F).var()
        analiz['keskinlik'] = {
            'deger': laplacian_var,
            'durum': 'bulanık' if laplacian_var < 100 else 'keskin'
        }
        
        # Renk dağılımı
        b, g, r = cv2.split(resim)
        renk_ortalama = [np.mean(b), np.mean(g), np.mean(r)]
        renk_std = [np.std(b), np.std(g), np.std(r)]
        
        analiz['renk_dagilimi'] = {
            'ortalama': renk_ortalama,
            'standart_sapma': renk_std,
            'dengeli': abs(max(renk_ortalama) - min(renk_ortalama)) < 30
        }
        
        return analiz
    
    def oneri_ver(analiz_sonucu: Dict) -> List[str]:
        """Analiz sonucuna göre öneri verir"""
        oneriler = []
        
        # Parlaklık önerileri
        if analiz_sonucu['parlaklik']['durum'] == 'çok karanlık':
            oneriler.append("• Parlaklığı artırın (+30 - +50)")
        elif analiz_sonucu['parlaklik']['durum'] == 'çok aydınlık':
            oneriler.append("• Parlaklığı azaltın (-30 - -50)")
        
        # Kontrast önerileri
        if analiz_sonucu['kontrast']['durum'] == 'düşük':
            oneriler.append("• Kontrast artırın (1.2 - 1.5)")
        elif analiz_sonucu['kontrast']['durum'] == 'yüksek':
            oneriler.append("• Kontrast azaltın (0.7 - 0.9)")
        
        # Keskinlik önerileri
        if analiz_sonucu['keskinlik']['durum'] == 'bulanık':
            oneriler.append("• Keskinleştirme filtresi uygulayın")
        
        # Renk dengesizliği
        if not analiz_sonucu['renk_dagilimi']['dengeli']:
            oneriler.append("• Renk dengesi düzeltmesi yapın")
        
        if not oneriler:
            oneriler.append("• Resim kalitesi iyi durumda!")
        
        return oneriler
    
    # Test resimlerini analiz et
    test_dir = Path("test-resimleri")
    
    for resim_yolu in test_dir.glob("*.jpg"):
        resim = cv2.imread(str(resim_yolu))
        if resim is None:
            continue
        
        print(f"\\n📋 Analiz: {resim_yolu.name}")
        print("-" * 40)
        
        analiz = kalite_analiz_et(resim)
        
        # Sonuçları yazdır
        print(f"Parlaklık: {analiz['parlaklik']['deger']:.1f} ({analiz['parlaklik']['durum']})")
        print(f"Kontrast: {analiz['kontrast']['deger']:.1f} ({analiz['kontrast']['durum']})")
        print(f"Keskinlik: {analiz['keskinlik']['deger']:.1f} ({analiz['keskinlik']['durum']})")
        print(f"Renk dengesi: {'✅' if analiz['renk_dagilimi']['dengeli'] else '❌'}")
        
        # Önerileri göster
        oneriler = oneri_ver(analiz)
        print("\\n💡 Öneriler:")
        for oneri in oneriler:
            print(f"   {oneri}")
        
        # Kalite raporunu dosyaya kaydet
        rapor_yolu = Path("ciktiler") / f"kalite_raporu_{resim_yolu.stem}.json"
        with open(rapor_yolu, 'w', encoding='utf-8') as f:
            json.dump(analiz, f, indent=2, ensure_ascii=False)
    
    print("\\n✅ Kalite analizi tamamlandı!")

def gorev_4_filtreleme_sistemi():
    """
    ✅ ÇÖZÜM 4: Gelişmiş Filtreleme Sistemi
    """
    print("\\n🎯 GÖREV 4: Filtreleme Sistemi")
    print("-" * 35)
    
    def vintage_filtre(resim: np.ndarray) -> np.ndarray:
        """Vintage fotoğraf efekti"""
        # Sepia efekti için renk matrisi
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        vintage = cv2.transform(resim, kernel)
        
        # Vintage için biraz bulanıklaştırma
        vintage = cv2.GaussianBlur(vintage, (3, 3), 0)
        
        # Vignette efekti (kenarları karartma)
        h, w = vintage.shape[:2]
        kernel_vignette = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - h/2)**2 + (j - w/2)**2)
                kernel_vignette[i, j] = 1 - min(distance / (min(h, w) * 0.6), 0.4)
        
        for channel in range(3):
            vintage[:, :, channel] = vintage[:, :, channel] * kernel_vignette
        
        return np.clip(vintage, 0, 255).astype(np.uint8)
    
    def soguk_ton_filtre(resim: np.ndarray) -> np.ndarray:
        """Soğuk ton filtresi (mavi vurgu)"""
        soguk = resim.copy().astype(np.float32)
        
        # Mavi kanalı artır, kırmızı azalt
        soguk[:, :, 0] = np.clip(soguk[:, :, 0] * 1.3, 0, 255)  # Mavi
        soguk[:, :, 1] = np.clip(soguk[:, :, 1] * 1.1, 0, 255)  # Yeşil
        soguk[:, :, 2] = np.clip(soguk[:, :, 2] * 0.8, 0, 255)  # Kırmızı
        
        return soguk.astype(np.uint8)
    
    def sicak_ton_filtre(resim: np.ndarray) -> np.ndarray:
        """Sıcak ton filtresi (sarı/turuncu vurgu)"""
        sicak = resim.copy().astype(np.float32)
        
        # Kırmızı ve yeşil kanalları artır, mavi azalt
        sicak[:, :, 0] = np.clip(sicak[:, :, 0] * 0.7, 0, 255)  # Mavi
        sicak[:, :, 1] = np.clip(sicak[:, :, 1] * 1.2, 0, 255)  # Yeşil
        sicak[:, :, 2] = np.clip(sicak[:, :, 2] * 1.3, 0, 255)  # Kırmızı
        
        return sicak.astype(np.uint8)
    
    def yuksek_kontrast_filtre(resim: np.ndarray) -> np.ndarray:
        """Yüksek kontrast (dramatik)"""
        return cv2.convertScaleAbs(resim, alpha=1.5, beta=-20)
    
    def yumusak_filtre(resim: np.ndarray) -> np.ndarray:
        """Yumuşak (soft focus)"""
        # Hafif bulanıklaştırma
        yumusak = cv2.GaussianBlur(resim, (15, 15), 0)
        # Orijinal ile harmanlama
        return cv2.addWeighted(resim, 0.6, yumusak, 0.4, 0)
    
    # Test resmi yükle
    test_dir = Path("test-resimleri")
    test_resim_yolu = test_dir / "normal.jpg"
    
    if not test_resim_yolu.exists():
        # Test resmi oluştur
        test_resim = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.rectangle(test_resim, (100, 100), (300, 200), (150, 200, 100), -1)
        cv2.imwrite(str(test_resim_yolu), test_resim)
    
    orijinal = cv2.imread(str(test_resim_yolu))
    
    # Filtreleri uygula
    filtreler = [
        ("Orijinal", orijinal),
        ("Vintage", vintage_filtre(orijinal)),
        ("Soğuk Ton", soguk_ton_filtre(orijinal)),
        ("Sıcak Ton", sicak_ton_filtre(orijinal)),
        ("Yüksek Kontrast", yuksek_kontrast_filtre(orijinal)),
        ("Yumuşak", yumusak_filtre(orijinal))
    ]
    
    # Karşılaştırmalı gösterim
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Filtreleme Sistemi Karşılaştırması', fontsize=16)
    
    for i, (filtre_adi, filtrelenmis) in enumerate(filtreler):
        row = i // 3
        col = i % 3
        
        axes[row, col].imshow(cv2.cvtColor(filtrelenmis, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(filtre_adi)
        axes[row, col].axis('off')
        
        # Filtrelenmiş resmi kaydet
        cikti_yolu = Path("ciktiler") / f"filtre_{filtre_adi.lower().replace(' ', '_')}.jpg"
        cv2.imwrite(str(cikti_yolu), filtrelenmis)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Filtreleme sistemi tamamlandı!")
    print("💾 Tüm filtreler 'ciktiler/' klasörüne kaydedildi")

def gorev_5_interaktif_arayuz():
    """
    ✅ ÇÖZÜM 5: İnteraktif Kullanıcı Arayüzü
    """
    print("\\n🎯 GÖREV 5: İnteraktif Arayüz")
    print("-" * 35)
    
    # Test resmi yükle
    test_dir = Path("test-resimleri")
    test_resim_yolu = test_dir / "normal.jpg"
    
    if not test_resim_yolu.exists():
        print("❌ Test resmi bulunamadı!")
        return
    
    orijinal_resim = cv2.imread(str(test_resim_yolu))
    if orijinal_resim is None:
        print("❌ Resim okunamadı!")
        return
    
    guncel_resim = orijinal_resim.copy()
    gecmis_resimler = [orijinal_resim.copy()]
    
    # Ana pencere
    pencere_adi = 'Interaktif Fotograf Duzenleyici'
    cv2.namedWindow(pencere_adi, cv2.WINDOW_NORMAL)
    
    # Trackbar'ları oluştur
    cv2.createTrackbar('Parlaklik', pencere_adi, 100, 200, lambda x: None)  # 0-200 (100=normal)
    cv2.createTrackbar('Kontrast', pencere_adi, 100, 200, lambda x: None)   # 0-200 (100=normal)
    cv2.createTrackbar('Doygunluk', pencere_adi, 100, 200, lambda x: None)  # 0-200 (100=normal)
    cv2.createTrackbar('Sicaklik', pencere_adi, 100, 200, lambda x: None)   # 0-200 (100=normal)
    
    # Mouse callback için global değişkenler
    mouse_pos = None
    mouse_dragging = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos, mouse_dragging, guncel_resim, gecmis_resimler
        
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos = (x, y)
            mouse_dragging = True
            gecmis_resimler.append(guncel_resim.copy())
            
        elif event == cv2.EVENT_MOUSEMOVE and mouse_dragging:
            # Bölgesel parlaklık artırma (basit örnek)
            cv2.circle(guncel_resim, (x, y), 20, (255, 255, 255), -1)
            
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_dragging = False
    
    cv2.setMouseCallback(pencere_adi, mouse_callback)
    
    print("🎛️ İnteraktif Kontroller:")
    print("   Trackbar'lar: Real-time ayar")
    print("   Mouse: Bölgesel düzeltme")
    print("   'r': Reset (sıfırla)")
    print("   's': Save (kaydet)")
    print("   'u': Undo (geri al)")  
    print("   'f': Filter menüsü")
    print("   ESC: Çıkış")
    
    while True:
        # Trackbar değerlerini al
        parlaklik = cv2.getTrackbarPos('Parlaklik', pencere_adi) - 100  # -100 to +100
        kontrast = cv2.getTrackbarPos('Kontrast', pencere_adi) / 100.0  # 0 to 2.0
        doygunluk = cv2.getTrackbarPos('Doygunluk', pencere_adi) / 100.0  # 0 to 2.0
        sicaklik = cv2.getTrackbarPos('Sicaklik', pencere_adi) - 100  # -100 to +100
        
        # İşlemleri uygula
        islenmis = orijinal_resim.copy().astype(np.float32)
        
        # Parlaklık ve kontrast
        islenmis = islenmis * kontrast + parlaklik
        
        # Sıcaklık ayarlama (basit yaklaşım)
        if sicaklik > 0:  # Sıcak ton
            islenmis[:, :, 2] = np.clip(islenmis[:, :, 2] + sicaklik * 0.3, 0, 255)  # Kırmızı
            islenmis[:, :, 1] = np.clip(islenmis[:, :, 1] + sicaklik * 0.1, 0, 255)  # Yeşil
        else:  # Soğuk ton
            islenmis[:, :, 0] = np.clip(islenmis[:, :, 0] - sicaklik * 0.3, 0, 255)  # Mavi
        
        # Doygunluk ayarlama (HSV üzerinden)
        if doygunluk != 1.0:
            hsv = cv2.cvtColor(np.clip(islenmis, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * doygunluk, 0, 255)
            islenmis = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        guncel_resim = np.clip(islenmis, 0, 255).astype(np.uint8)
        
        # Göster
        cv2.imshow(pencere_adi, guncel_resim)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Reset
            cv2.setTrackbarPos('Parlaklik', pencere_adi, 100)
            cv2.setTrackbarPos('Kontrast', pencere_adi, 100)
            cv2.setTrackbarPos('Doygunluk', pencere_adi, 100)
            cv2.setTrackbarPos('Sicaklik', pencere_adi, 100)
            guncel_resim = orijinal_resim.copy()
            gecmis_resimler = [orijinal_resim.copy()]
            print("↺ Tüm ayarlar sıfırlandı")
        elif key == ord('s'):  # Save
            kayit_yolu = Path("ciktiler") / "interaktif_duzenlenmis.jpg"
            cv2.imwrite(str(kayit_yolu), guncel_resim)
            print(f"💾 Resim kaydedildi: {kayit_yolu}")
        elif key == ord('u'):  # Undo
            if len(gecmis_resimler) > 1:
                gecmis_resimler.pop()
                guncel_resim = gecmis_resimler[-1].copy()
                print("↶ Son işlem geri alındı")
        elif key == ord('f'):  # Filter menu
            print("🎨 Filtre menüsü: 1=Vintage, 2=Soğuk, 3=Sıcak")
    
    cv2.destroyAllWindows()
    print("✅ İnteraktif arayüz kapatıldı!")

def gorev_6_performans_optimizasyonu():
    """
    ✅ ÇÖZÜM 6: Performans Optimizasyonu ve Benchmarking
    """
    print("\\n🎯 GÖREV 6: Performans Optimizasyonu")
    print("-" * 40)
    
    def benchmark_resize(resim: np.ndarray, boyutlar: List[Tuple[int, int]]) -> Dict:
        """Resize yöntemlerini karşılaştır"""
        yontemler = [
            ('INTER_LINEAR', cv2.INTER_LINEAR),
            ('INTER_CUBIC', cv2.INTER_CUBIC),
            ('INTER_AREA', cv2.INTER_AREA),
            ('INTER_LANCZOS4', cv2.INTER_LANCZOS4)
        ]
        
        sonuclar = {}
        
        for yontem_adi, yontem in yontemler:
            sureler = []
            
            for boyut in boyutlar:
                baslangic = time.perf_counter()
                cv2.resize(resim, boyut, interpolation=yontem)
                bitis = time.perf_counter()
                sureler.append(bitis - baslangic)
            
            sonuclar[yontem_adi] = {
                'ortalama_sure': np.mean(sureler),
                'min_sure': np.min(sureler),
                'max_sure': np.max(sureler)
            }
        
        return sonuclar
    
    def benchmark_blur(resim: np.ndarray, kernel_boyutlari: List[int]) -> Dict:
        """Blur yöntemlerini karşılaştır"""
        yontemler = [
            ('Gaussian', lambda img, k: cv2.GaussianBlur(img, (k, k), 0)),
            ('Box', lambda img, k: cv2.boxFilter(img, -1, (k, k))),
            ('Bilateral', lambda img, k: cv2.bilateralFilter(img, k, 75, 75))
        ]
        
        sonuclar = {}
        
        for yontem_adi, yontem_func in yontemler:
            sureler = []
            
            for kernel_boyutu in kernel_boyutlari:
                if kernel_boyutu % 2 == 0:  # Tek sayı olmalı
                    kernel_boyutu += 1
                
                baslangic = time.perf_counter()
                try:
                    yontem_func(resim, kernel_boyutu)
                    bitis = time.perf_counter()
                    sureler.append(bitis - baslangic)
                except:
                    sureler.append(float('inf'))  # Hata durumunda
            
            if sureler:
                sonuclar[yontem_adi] = {
                    'ortalama_sure': np.mean([s for s in sureler if s != float('inf')]),
                    'min_sure': np.min([s for s in sureler if s != float('inf')] or [0]),
                    'max_sure': np.max([s for s in sureler if s != float('inf')] or [0])
                }
        
        return sonuclar
    
    def kalite_metrikler(orijinal: np.ndarray, islenmis: np.ndarray) -> Dict:
        """Kalite metriklerini hesapla"""
        # PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((orijinal - islenmis) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # SSIM yaklaşımı (basitleştirilmiş)
        mu1 = np.mean(orijinal)
        mu2 = np.mean(islenmis)
        sigma1 = np.var(orijinal)
        sigma2 = np.var(islenmis)
        sigma12 = np.mean((orijinal - mu1) * (islenmis - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        return {'psnr': psnr, 'ssim': ssim}
    
    # Test resmi oluştur
    test_resim = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    
    print("🧪 Resize Yöntemleri Benchmark:")
    print("-" * 30)
    
    resize_boyutlari = [(200, 200), (400, 400), (800, 800)]
    resize_sonuclari = benchmark_resize(test_resim, resize_boyutlari)
    
    for yontem, sonuc in resize_sonuclari.items():
        print(f"{yontem:15}: {sonuc['ortalama_sure']*1000:.2f}ms (ort.)")
    
    print("\\n🌫️ Blur Yöntemleri Benchmark:")
    print("-" * 30)
    
    blur_kerneller = [5, 15, 25]
    blur_sonuclari = benchmark_blur(test_resim, blur_kerneller)
    
    for yontem, sonuc in blur_sonuclari.items():
        print(f"{yontem:15}: {sonuc['ortalama_sure']*1000:.2f}ms (ort.)")
    
    # Kalite testi
    print("\\n📊 Kalite Metrikleri Testi:")
    print("-" * 30)
    
    # Farklı işlemlerle kalite karşılaştırması
    islemler = [
        ("Orijinal", test_resim),
        ("JPEG Sıkıştırma", cv2.resize(cv2.resize(test_resim, (250, 250)), (500, 500))),
        ("Gaussian Blur", cv2.GaussianBlur(test_resim, (5, 5), 0))
    ]
    
    for islem_adi, islenmis_resim in islemler[1:]:  # İlki orijinal
        metrikler = kalite_metrikler(test_resim, islenmis_resim)
        print(f"{islem_adi:15}: PSNR={metrikler['psnr']:.2f}dB, SSIM={metrikler['ssim']:.3f}")
    
    # Performans grafiği
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Resize performansı
    yontemler = list(resize_sonuclari.keys())
    sureler = [resize_sonuclari[y]['ortalama_sure']*1000 for y in yontemler]
    
    ax1.bar(yontemler, sureler)
    ax1.set_title('Resize Yöntemleri Performansı')
    ax1.set_ylabel('Süre (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Blur performansı
    blur_yontemler = list(blur_sonuclari.keys())
    blur_sureler = [blur_sonuclari[y]['ortalama_sure']*1000 for y in blur_yontemler]
    
    ax2.bar(blur_yontemler, blur_sureler)
    ax2.set_title('Blur Yöntemleri Performansı')
    ax2.set_ylabel('Süre (ms)')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Performans analizi tamamlandı!")

def main():
    """Ana çözüm programı - Mini Proje Yöneticisi"""
    print("✅ OpenCV Alıştırma 3 - ÇÖZÜMLER")
    print("=" * 50)
    print("Mini Proje: Akıllı Fotoğraf Düzenleyici\\n")
    
    # Proje menüsü
    while True:
        print("\\n" + "="*50)
        print("📸 Akıllı Fotoğraf Düzenleyici - Çözüm Menüsü")
        print("="*50)
        print("1. Sınıf Tasarımı Test")
        print("2. Batch İşlem Sistemi")
        print("3. Kalite Analizi Modülü")
        print("4. Filtreleme Sistemi")
        print("5. İnteraktif Arayüz")
        print("6. Performans Optimizasyonu")
        print("7. Tüm Modülleri Gözden Geçir")
        print("0. Çıkış")
        
        secim = input("\\nBir çözüm modülü seçin (0-7): ").strip()
        
        if secim == "0":
            print("🎉 Tüm çözümler incelendi! Harika öğrenme!")
            break
        elif secim == "1":
            gorev_1_sinif_tasarimi()
        elif secim == "2":
            gorev_2_batch_islem()
        elif secim == "3":
            gorev_3_kalite_analizi()
        elif secim == "4":
            gorev_4_filtreleme_sistemi()
        elif secim == "5":
            gorev_5_interaktif_arayuz()
        elif secim == "6":
            gorev_6_performans_optimizasyonu()
        elif secim == "7":
            print("\\n🔍 Tüm Modüller Özeti:")
            print("   ✅ Sınıf tasarımı - OOP prensipleri")
            print("   ✅ Batch işlem - Otomatizasyon")
            print("   ✅ Kalite analizi - Görüntü metrikleri")
            print("   ✅ Filtreleme - Yaratıcı efektler")
            print("   ✅ İnteraktif arayüz - Kullanıcı deneyimi")
            print("   ✅ Performans - Optimizasyon teknikleri")
            print("\\n💡 Bu proje gerçek dünya uygulamalarının temelidir!")
        else:
            print("❌ Geçersiz seçim! Lütfen 0-7 arasında bir sayı girin.")

if __name__ == "__main__":
    main()

# 📝 PROJE ÇÖZÜM NOTLARI:
# 1. Bu çözümler profesyonel kod standartlarında yazılmıştır
# 2. Her modül bağımsız test edilebilir
# 3. Hata kontrolü ve exception handling eklenmiştir
# 4. Type hints kullanılarak kod okunabilirliği artırılmıştır
# 5. Performans metrikleri ve kalite ölçümleri gerçek dünya standartlarındadır
# 6. Modüler yapı sayesinde genişletilebilir
# 7. Bu proje bir portfolio projesi olarak kullanılabilir