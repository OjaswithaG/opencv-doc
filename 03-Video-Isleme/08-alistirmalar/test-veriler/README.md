# 🎬 Test Verileri

Bu klasör video işleme alıştırmaları için test dosyalarını ve yardımcı araçları içerir.

## 📁 İçerik

### `video_olusturucu.py`
Test videoları oluşturmak için kullanılan araç. Webcam yoksa veya kontrollü test senaryoları için kullanışlıdır.

**Özellikler:**
- Hareket simülasyonu
- Farklı hızlarda objeler
- Sahne değişiklikleri
- Gürültü ekleme

**Kullanım:**
```bash
python video_olusturucu.py
```

## 🎯 Test Senaryoları

### Alıştırma 1 İçin (Güvenlik Sistemi)
- **Temel Hareket**: Objeler ekranda hareket eder
- **Gürültü Testi**: Küçük değişiklikler (ignore edilmeli)  
- **Çoklu Olay**: Ardışık hareket olayları
- **Uzun Kayıt**: Sürekli hareket (performans testi)

### Alıştırma 2 İçin (Nesne Takibi)
- **Tek Nesne**: Basit tracking testi
- **Çoklu Nesne**: Multiple object tracking
- **Crossing**: Nesnelerin çizgiyi geçmesi
- **Occlusion**: Nesnelerin birbirini örtmesi

### Alıştırma 3 İçin (Kalite Analizi)
- **Kalite Değişimi**: Frame kalitesinde değişiklikler
- **Scene Change**: Ani sahne değişiklikleri
- **Motion Analysis**: Farklı hareket türleri
- **Performance**: Yoğun işlem gerektiren senaryolar

## 🛠️ Kendi Test Verilerinizi Oluşturma

1. `video_olusturucu.py` dosyasını çalıştırın
2. İstediğiniz parametreleri ayarlayın
3. Test videosunu generate edin
4. Alıştırmalarınızda kullanın

## 💡 İpuçları

- Webcam yerine test videoları kullanmak daha kontrollü test imkanı sağlar
- Farklı senaryoları test etmek algoritmanızı geliştirir
- Performance testleri için uzun videolar kullanın
- Edge case'ler için extreme senaryolar oluşturun