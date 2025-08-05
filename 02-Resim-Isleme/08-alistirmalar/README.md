# 🎯 02-Resim-İşleme Alıştırmaları

Bu klasör, resim işleme konularında pratik yapmak için alıştırmalar içerir.

## 📚 Alıştırmaların Kapsamı

Bu alıştırmalar şu konuları kapsar:
- **Geometrik Transformasyonlar** (döndürme, ölçekleme, perspektif)
- **Filtreleme** (blur, median, bilateral)
- **Morfolojik İşlemler** (erozyon, dilatasyon, opening, closing)
- **Histogram İşlemleri** (eşitleme, CLAHE)
- **Kontrast ve Parlaklık** (gamma, otomatik ayarlama)
- **Gürültü Azaltma** (Gaussian, salt&pepper temizleme)
- **Kenar Algılama** (Canny, Sobel, Laplacian)

## 🎪 Alıştırma Yapısı

### 📖 Alıştırma 1: Temel Resim İşleme
**Zorluk:** ⭐⭐ (Orta)  
**Süre:** 45-60 dakika  
**Konular:** Geometrik transformasyon, filtreleme, histogram

**Görevler:**
1. Resim döndürme ve ölçekleme
2. Farklı blur türleri uygulama
3. Histogram eşitleme ve analiz
4. Sonuçları karşılaştırma

### 📖 Alıştırma 2: İleri Resim İyileştirme
**Zorluk:** ⭐⭐⭐ (İleri)  
**Süre:** 60-90 dakika  
**Konular:** Gürültü azaltma, kontrast düzeltme, morfoloji

**Görevler:**
1. Multi-tip gürültü temizleme
2. Otomatik kontrast düzeltme
3. Morfolojik işlemlerle şekil analizi
4. Kombine filtreleme teknikleri

### 📖 Alıştırma 3: Kenar Algılama ve Analiz
**Zorluk:** ⭐⭐⭐⭐ (Uzman)  
**Süre:** 90-120 dakika  
**Konular:** Kenar algılama, şekil tanıma, performans analizi

**Görevler:**
1. Multi-method kenar algılama
2. Kenar kalitesi analizi
3. Şekil sayma ve sınıflandırma
4. Performans benchmarking

## 📁 Dosya Yapısı

```
08-alistirmalar/
├── README.md                    # Bu dosya
├── alistirma-1.md              # Temel resim işleme (Markdown)
├── alistirma-1.py              # Temel resim işleme (Python)
├── alistirma-2.md              # İleri resim iyileştirme (Markdown)
├── alistirma-2.py              # İleri resim iyileştirme (Python)
├── alistirma-3.md              # Kenar algılama ve analiz (Markdown)
├── alistirma-3.py              # Kenar algılama ve analiz (Python)
├── cozumler/                   # Çözüm dosyaları
│   ├── cozum-1.py              # Alıştırma 1 çözümü
│   ├── cozum-2.py              # Alıştırma 2 çözümü
│   └── cozum-3.py              # Alıştırma 3 çözümü
└── test-resimleri/             # Test için örnek resimler
    ├── resim_olusturucu.py     # Test resimlerini oluşturur
    ├── normal.jpg              # Normal resim
    ├── dusuk_kontrast.jpg      # Düşük kontrastlı
    ├── gurultulu.jpg           # Gürültülü resim
    ├── perspektif.jpg          # Perspektif bozulmuş
    └── kenar_test.jpg          # Kenar algılama testi
```

## 🚀 Nasıl Çalışılır?

### 1. Hazırlık
```bash
# Test resimlerini oluştur
cd 02-Resim-Isleme/08-alistirmalar/test-resimleri
python resim_olusturucu.py
```

### 2. Alıştırmaları Çöz

**📖 Okumak için (Önerilen):**
- [alistirma-1.md](alistirma-1.md) - Temel resim işleme
- [alistirma-2.md](alistirma-2.md) - İleri resim iyileştirme  
- [alistirma-3.md](alistirma-3.md) - Kenar algılama ve analiz

**🐍 Çalıştırmak için:**
```bash
# Alıştırma 1'i çalıştır
python alistirma-1.py

# Alıştırma 2'yi çalıştır  
python alistirma-2.py

# Alıştırma 3'ü çalıştır
python alistirma-3.py
```

### 3. Çözümlerle Karşılaştır
```bash
# Çözümleri incele
python cozumler/cozum-1.py
python cozumler/cozum-2.py
python cozumler/cozum-3.py
```

## 📋 Kontrol Listesi

### ✅ Alıştırma 1: Temel Resim İşleme
- [ ] Resmi 45° döndürme
- [ ] %75 oranında küçültme
- [ ] Gaussian blur (3 farklı boyut)
- [ ] Median filter (salt&pepper gürültü)
- [ ] Histogram eşitleme
- [ ] PSNR hesaplama ve karşılaştırma

### ✅ Alıştırma 2: İleri Resim İyileştirme
- [ ] Karma gürültü (Gaussian + salt&pepper) temizleme
- [ ] Otomatik kontrast ayarlama
- [ ] CLAHE uygulaması
- [ ] Gamma düzeltme
- [ ] Morfolojik gürültü temizleme
- [ ] Filtreleme pipeline oluşturma

### ✅ Alıştırma 3: Kenar Algılama ve Analiz
- [ ] 4 farklı kenar algılama yöntemi
- [ ] Otomatik threshold hesaplama
- [ ] Kenar kalitesi metrikleri
- [ ] Geometrik şekil sayma
- [ ] Performans analizi (süre, bellek)
- [ ] En iyi yöntem seçimi

## 💡 İpuçları

### 🎯 Genel İpuçları
- Her adımda sonuçları görselleştirin
- Parametreleri systematik olarak test edin
- Orijinal resimle sürekli karşılaştırın
- Çözümünüzü çalıştırmadan önce test edin

### 🔧 Teknik İpuçları
- `cv2.imshow()` ile ara sonuçları kontrol edin
- `matplotlib` ile çoklu görselleştirme yapın
- `numpy` fonksiyonlarını optimize kullanın
- Büyük resimlerle test ederken boyut küçültün

### ⚠️ Dikkat Edilecekler
- Veri tiplerini kontrol edin (uint8, float32)
- Pixel değerlerinin 0-255 aralığında olduğundan emin olun
- Kernel boyutlarının tek sayı olması gerektiğini unutmayın
- Memory overflow'a karşı dikkatli olun

## 🎖️ Zorluk Seviyeleri

- **⭐ Başlangıç:** Temel OpenCV fonksiyonları
- **⭐⭐ Orta:** Parametre optimizasyonu, kombinasyonlar
- **⭐⭐⭐ İleri:** Manuel implementasyonlar, algoritma karşılaştırma
- **⭐⭐⭐⭐ Uzman:** Performance optimization, custom algorithms

## 🏆 Değerlendirme Kriterleri

### Kod Kalitesi (25%)
- Okunabilirlik ve organizasyon
- Yorum satırları ve dokümantasyon
- Error handling

### Teknik Uygulama (40%)
- Algoritmaların doğru kullanımı
- Parametre seçimi ve optimizasyon
- Sonuçların tutarlılığı

### Analiz ve Yorumlama (25%)
- Sonuçların görselleştirilmesi
- Performans karşılaştırmaları
- İstatistiksel analiz

### Yaratıcılık (10%)
- Alternatif yaklaşımlar
- İyileştirme önerileri
- Bonus özellikler

## 🚀 Bonus Görevler

Her alıştırma için ekstra puanlar:

### Alıştırma 1 Bonus
- Interaktif parametre ayarlama
- Batch processing
- Farklı dosya formatları desteği

### Alıştırma 2 Bonus  
- Adaptive filtering
- Multi-scale processing
- Custom gürültü modelleri

### Alıştırma 3 Bonus
- Real-time processing
- Machine learning entegrasyonu
- 3D visualization

## 📚 Faydalı Kaynaklar

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Reference](https://numpy.org/doc/)
- [Matplotlib Tutorials](https://matplotlib.org/tutorials/)
- [Digital Image Processing by Gonzalez & Woods](https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003225)

## 🤝 Yardım ve Destek

Takıldığınız noktalarda:
1. Önce çözüm dosyalarına bakın
2. Ana dokümantasyondaki örnekleri inceleyin
3. OpenCV dokümantasyonunu okuyun
4. Stack Overflow'da benzer problemleri arayın

**Mutlu kodlamalar!** 🎉

---

*Bu alıştırmalar Eren Terzi tarafından hazırlanmıştır.*