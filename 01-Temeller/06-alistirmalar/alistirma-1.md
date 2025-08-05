# 🎯 Alıştırma 1: Temel Kurulum ve Kontrol

**Zorluk Seviyesi:** 🔰 Başlangıç  
**Tahmini Süre:** 15-20 dakika  

Bu alıştırma OpenCV'nin temel kurulum ve kontrol işlemlerini içerir. Her görevi tamamladıktan sonra sonucu kontrol edin.

## 📚 Gerekli Kütüphaneler

```python
import cv2
import numpy as np
import sys  # Python sürümü için
```

## 🎯 Görevler

Bu alıştırmada OpenCV'nin temel kullanımını öğreneceksiniz.

---

## 🎯 GÖREV 1: OpenCV Kurulum Kontrolü

OpenCV'nin doğru şekilde kurulduğunu kontrol edin.

### Yapılacaklar:
1. OpenCV sürümünü yazdırın
2. NumPy sürümünü yazdırın  
3. Python sürümünü yazdırın
4. Kurulumun başarılı olduğunu belirten bir mesaj yazdırın

### Beklenen Çıktı:
```
OpenCV Sürümü: 4.x.x
NumPy Sürümü: 1.x.x
Python Sürümü: 3.x.x
✅ Kurulum başarılı!
```

### Kodunuzu Yazın:
```python
def gorev_1_kurulum_kontrolu():
    print("🎯 GÖREV 1: Kurulum Kontrolü")
    print("-" * 30)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.__version__, np.__version__, sys.version kullanın
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- `cv2.__version__` OpenCV sürümünü verir
- `np.__version__` NumPy sürümünü verir  
- `sys.version` Python sürümünü verir

---

## 🎯 GÖREV 2: Basit Resim Oluşturma

NumPy ile basit bir resim oluşturun.

### Yapılacaklar:
1. 200x300 piksel boyutunda siyah bir resim oluşturun
2. Resmin şekil bilgisini yazdırın  
3. Resmin veri tipini yazdırın
4. Resmi 'siyah_resim' ismiyle gösterin

### Beklenen Çıktı:
```
Resim şekli: (200, 300, 3)
Veri tipi: uint8
```

### Kodunuzu Yazın:
```python
def gorev_2_basit_resim_olusturma():
    print("\n🎯 GÖREV 2: Basit Resim Oluşturma")
    print("-" * 35)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: np.zeros() kullanın, dtype=np.uint8 belirtin
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- `np.zeros((yükseklik, genişlik, 3), dtype=np.uint8)` ile siyah resim
- `resim.shape` ile boyut bilgisi
- `resim.dtype` ile veri tipi
- `cv2.imshow('isim', resim)` ile gösterme

---

## 🎯 GÖREV 3: Renkli Resim Oluşturma

İki yarısı farklı renkte olan bir resim oluşturun.

### Yapılacaklar:
1. 300x400 piksel boyutunda beyaz bir resim oluşturun
2. Sol yarısını kırmızı (BGR: 0,0,255) yapın
3. Sağ yarısını mavi (BGR: 255,0,0) yapın
4. Resmi 'renkli_resim' ismiyle gösterin

### Beklenen Sonuç: 
Sol yarısı kırmızı, sağ yarısı mavi resim

### Kodunuzu Yazın:
```python
def gorev_3_renkli_resim_olusturma():
    print("\n🎯 GÖREV 3: Renkli Resim Oluşturma")
    print("-" * 35)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: np.ones() * 255 ile beyaz resim, indeksleme ile renk atama
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- `np.ones((yükseklik, genişlik, 3), dtype=np.uint8) * 255` ile beyaz resim
- OpenCV BGR format kullanır (Blue, Green, Red)
- Sol yarı: `resim[:, :genişlik//2] = [0, 0, 255]`
- Sağ yarı: `resim[:, genişlik//2:] = [255, 0, 0]`

---

## 🎯 GÖREV 4: Geometrik Şekiller Çizme

OpenCV çizim fonksiyonlarını kullanarak şekiller çizin.

### Yapılacaklar:
1. 400x400 piksel boyutunda siyah bir resim oluşturun
2. Merkezde yeşil dolu bir daire çizin (yarıçap: 50)
3. Sol üst köşede kırmızı dolu bir dikdörtgen çizin (50x50)
4. Sağ alt köşede beyaz bir çizgi çizin (köşegen)
5. Resmi 'geometrik_sekiller' ismiyle gösterin

### Kullanılacak Fonksiyonlar:
- `cv2.circle()`
- `cv2.rectangle()`
- `cv2.line()`

### Kodunuzu Yazın:
```python
def gorev_4_geometrik_sekiller():
    print("\n🎯 GÖREV 4: Geometrik Şekiller")
    print("-" * 30)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.circle(resim, merkez, yarıçap, renk, kalınlık)
    #        kalınlık=-1 dolu şekil çizer
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- Daire: `cv2.circle(resim, (merkez_x, merkez_y), yarıçap, renk, -1)`
- Dikdörtgen: `cv2.rectangle(resim, (x1,y1), (x2,y2), renk, -1)`
- Çizgi: `cv2.line(resim, (x1,y1), (x2,y2), renk, kalınlık)`
- Renkler BGR formatında: (B, G, R)
- Yeşil: (0, 255, 0), Kırmızı: (0, 0, 255), Beyaz: (255, 255, 255)

---

## 🎯 GÖREV 5: Metin Ekleme

Resme farklı konumlarda ve renklerde metinler ekleyin.

### Yapılacaklar:
1. 300x500 piksel boyutunda beyaz bir resim oluşturun
2. Üst kısma "OpenCV" yazsın (kırmızı renk)
3. Orta kısma "Alıştırma 1" yazsın (mavi renk)  
4. Alt kısma "Tamamlandı!" yazsın (yeşil renk)
5. Resmi 'metin_resim' ismiyle gösterin

### Kullanılacak Fonksiyon:
- `cv2.putText()`

### Kodunuzu Yazın:
```python
def gorev_5_metin_ekleme():
    print("\n🎯 GÖREV 5: Metin Ekleme")
    print("-" * 25)
    
    # TODO: Buraya kodunuzu yazın
    # İpucu: cv2.putText(resim, metin, konum, font, ölçek, renk, kalınlık)
    #        font = cv2.FONT_HERSHEY_SIMPLEX
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**İpuçları:**
- `cv2.putText(resim, "metin", (x, y), font, scale, renk, thickness)`
- Font: `cv2.FONT_HERSHEY_SIMPLEX`
- Konum (x, y): metnin sol alt köşesi
- Scale: 1.0 normal boyut
- Kalınlık: 2 veya 3 uygun

---

## 🎯 GÖREV 6: İnteraktif Pencere

Klavye girişi ile resmin rengini değiştiren interaktif bir pencere oluşturun.

### Yapılacaklar:
1. 250x400 piksel boyutunda gri bir resim oluşturun (değer: 128)
2. Resmi gösterin
3. Herhangi bir tuşa basılmasını bekleyin
4. 'r' tuşuna basılırsa resmi kırmızı yapın
5. 'g' tuşuna basılırsa resmi yeşil yapın
6. 'b' tuşuna basılırsa resmi mavi yapın
7. ESC tuşuna basılırsa çıkış yapın

### Kontroller:
- **r**: Kırmızı
- **g**: Yeşil
- **b**: Mavi
- **ESC**: Çıkış

### Kodunuzu Yazın:
```python
def gorev_6_interaktif_pencere():
    print("\n🎯 GÖREV 6: İnteraktif Pencere")
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
```

**İpuçları:**
- `while True:` döngüsü kullanın
- `key = cv2.waitKey(0) & 0xFF` ile tuş oku
- `key == ord('r')` ile 'r' tuşu kontrolü
- `key == 27` ESC tuşu (ASCII kodu)
- Her renk değişikliğinde `cv2.imshow()` ile yeniden göster

---

## 🎨 BONUS GÖREV: Sanat Eseri Oluşturma

Yaratıcılığınızı konuşturun! OpenCV fonksiyonlarını kullanarak özgün bir sanat eseri oluşturun.

### Kullanabileceğiniz Fonksiyonlar:
- `cv2.circle()`, `cv2.rectangle()`, `cv2.line()`
- `cv2.ellipse()`, `cv2.polylines()`
- `cv2.putText()`
- Farklı renkler ve şekiller

### Kısıtlama:
En az 5 farklı geometrik şekil kullanın!

### Kodunuzu Yazın:
```python
def bonus_gorev_sanat_eseri():
    print("\n🎨 BONUS GÖREV: Sanat Eseri")
    print("-" * 30)
    print("Yaratıcılığınızı konuşturun!")
    
    # TODO: Buraya yaratıcı kodunuzu yazın
    # Kendi sanat eserinizi oluşturun!
    
    pass  # Bu satırı silin ve kodunuzu yazın
```

**Öneriler:**
- Mandala desenleri
- Geometrik sanat
- Abstract kompozisyonlar
- Logo tasarımı
- Renk geçişleri

---

## 🖥️ Ana Program

Tüm görevleri çalıştıran ana fonksiyon:

```python
def main():
    """Ana program - tüm görevleri çalıştırır"""
    print("🎯 OpenCV Alıştırma 1: Temel Kurulum ve Kontrol")
    print("=" * 55)
    print("Bu alıştırmada OpenCV'nin temel kullanımını öğreneceksiniz.\n")
    
    try:
        # Görevleri sırayla çalıştır
        gorev_1_kurulum_kontrolu()
        gorev_2_basit_resim_olusturma()
        gorev_3_renkli_resim_olusturma()
        gorev_4_geometrik_sekiller()
        gorev_5_metin_ekleme()
        gorev_6_interaktif_pencere()
        
        # Bonus görev (opsiyonel)
        bonus_cevap = input("\nBonus görevi yapmak ister misiniz? (e/h): ")
        if bonus_cevap.lower() == 'e':
            bonus_gorev_sanat_eseri()
        
        print("\n🎉 Tebrikler! Alıştırma 1'i tamamladınız!")
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
```

---

## ✅ Kontrol Listesi

Alıştırmayı tamamladıktan sonra kontrol edin:

- [ ] OpenCV düzgün kuruldu ve çalışıyor
- [ ] Basit resim oluşturabiliyorum
- [ ] Geometrik şekiller çizebiliyorum
- [ ] Metin ekleyebiliyorum
- [ ] İnteraktif pencere kontrolü yapabiliyorum
- [ ] (Bonus) Yaratıcı sanat eseri oluşturdum

## 💡 İpuçları

### Genel İpuçları
- Her görevde `pass` satırını silip kendi kodunuzu yazın
- Hata alırsanız hata mesajını dikkatlice okuyun
- İpuçlarını takip edin
- Kendi yaratıcı fikirlerinizi de ekleyebilirsiniz

### Teknik İpuçları
- OpenCV BGR format kullanır (Blue, Green, Red)
- Resim boyutları (yükseklik, genişlik, kanal) formatındadır
- `cv2.waitKey(0)` bir tuşa basılmasını bekler
- `cv2.destroyAllWindows()` tüm pencereleri kapatır

### Hata Durumları
- Import hataları: kütüphanelerin kurulu olduğundan emin olun
- Boyut hataları: indeksleme işlemlerinde sınırları kontrol edin
- Veri tipi hataları: `dtype=np.uint8` kullanmayı unutmayın

## 🎯 Çözüm

Takıldığınızda çözüm dosyasına bakabilirsiniz:
```bash
python cozumler/cozum-1.py
```

## 🚀 Sonraki Adım

Bu alıştırmayı tamamladıktan sonra **Alıştırma 2: Resim İşlemleri ve Renk Uzayları** adımına geçebilirsiniz.

---

*Bu alıştırma Eren Terzi tarafından hazırlanmıştır.*