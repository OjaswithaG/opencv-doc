# 📁 Examples - Örnek Resimler

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
