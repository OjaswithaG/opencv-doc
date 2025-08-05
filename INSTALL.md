# 🚀 OpenCV Dokümantasyon Projesi - Kurulum Rehberi

Bu rehber, OpenCV dokümantasyon projesini bilgisayarınıza kurmanız için adım adım talimatlar içerir.

## 📋 Sistem gereksinimleri

### Minimum Gereksinimler
- **Python**: 3.7 veya üzeri
- **İşletim Sistemi**: Windows 10, macOS 10.14, Ubuntu 18.04 veya üzeri
- **RAM**: En az 4GB (8GB önerilir)
- **Disk Alanı**: En az 2GB boş alan

### Önerilen Gereksinimler
- **Python**: 3.8 veya üzeri
- **RAM**: 8GB veya üzeri
- **GPU**: NVIDIA CUDA destekli (derin öğrenme için)

## 🛠️ Kurulum Seçenekleri

### Seçenek 1: Hızlı Kurulum (Önerilen)

```bash
# Projeyi klonlayın
git clone https://github.com/erent8/opencv-doc.git
cd opencv-dokumantasyon

# Minimal paketleri kurun
pip install -r requirements-minimal.txt
```

### Seçenek 2: Tam Kurulum

```bash
# Tüm paketleri kurun (daha uzun sürer)
pip install -r requirements.txt
```

### Seçenek 3: Sanal Ortam ile Kurulum (En Güvenli)

```bash
# Sanal ortam oluşturun
python -m venv opencv_env

# Sanal ortamı aktifleştirin
# Windows:
opencv_env\Scripts\activate
# macOS/Linux:
source opencv_env/bin/activate

# Paketleri kurun
pip install -r requirements.txt
```

### Seçenek 4: Geliştirici Kurulumu

```bash
# Geliştirme modunda kurulum
pip install -e .

# Veya tüm ekstralar ile
pip install -e ".[full,dev,ml]"
```

## 🔧 İşletim Sistemi Özel Talimatları

### Windows 10/11

```powershell
# Python'un kurulu olduğunu kontrol edin
python --version

# pip'i güncelleyin
python -m pip install --upgrade pip

# Visual C++ Build Tools gerekebilir
# Microsoft Visual C++ Build Tools'u indirin ve kurun

# Projeyi kurun
git clone https://github.com/erent8/opencv-doc.git
cd opencv-doc
pip install -r requirements.txt
```

### macOS

```bash
# Homebrew ile Python kurulumu (eğer yoksa)
brew install python

# Xcode command line tools (gerekebilir)
xcode-select --install

# Projeyi kurun
git clone https://github.com/erent8/opencv-doc.git
cd opencv-dokumantasyon
pip3 install -r requirements.txt
```

### Ubuntu/Debian Linux

```bash
# Sistem paketlerini güncelleyin
sudo apt update

# Python ve pip kurulumu
sudo apt install python3 python3-pip python3-venv

# Sistem bağımlılıkları
sudo apt install python3-opencv libopencv-dev

# Projeyi kurun
git clone https://github.com/erent8/opencv-doc.git
cd opencv-dokumantasyon
pip3 install -r requirements.txt
```

## ✅ Kurulum Doğrulaması

Kurulumun başarılı olduğunu doğrulamak için:

```python
# test_kurulum.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("🧪 OpenCV Kurulum Testi")
print("=" * 30)

# OpenCV sürümü
print(f"✅ OpenCV sürümü: {cv2.__version__}")

# NumPy sürümü
print(f"✅ NumPy sürümü: {np.__version__}")

# Matplotlib sürümü
print(f"✅ Matplotlib sürümü: {plt.matplotlib.__version__}")

# Test resmi oluştur
test_resim = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(test_resim, (25, 25), (75, 75), (0, 255, 0), -1)

# Resmi göster
cv2.imshow('Kurulum Testi', test_resim)
print("✅ Test resmi gösteriliyor...")
cv2.waitKey(2000)  # 2 saniye bekle
cv2.destroyAllWindows()

print("🎉 Kurulum başarılı!")
```

Bu test scriptini çalıştırın:

```bash
python test_kurulum.py
```

## 📚 İlk Adımlar

Kurulum tamamlandıktan sonra:

1. **Temel örnekleri çalıştırın:**
   ```bash
   cd 01-Temeller
   python 02-ilk-program.py
   ```

2. **Jupyter Notebook'ları başlatın:**
   ```bash
   jupyter notebook
   ```

3. **Test resimlerini oluşturun:**
   ```bash
   cd 01-Temeller/examples
   python ornek_resim_olusturucu.py
   ```

## 🐛 Sık Karşılaşılan Sorunlar

### Sorun 1: `ImportError: No module named cv2`

**Çözüm:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

### Sorun 2: `Microsoft Visual C++ 14.0 is required` (Windows)

**Çözüm:**
- Microsoft Visual C++ Build Tools'u indirin ve kurun
- Veya Visual Studio Community'yi kurun

### Sorun 3: `Permission denied` (Linux/macOS)

**Çözüm:**
```bash
# Kullanıcı dizinine kurulum
pip install --user -r requirements.txt

# Veya sudo ile
sudo pip install -r requirements.txt
```

### Sorun 4: Jupyter Notebook açılmıyor

**Çözüm:**
```bash
# Jupyter'ı tekrar kurun
pip uninstall jupyter
pip install jupyter

# Port değiştirerek deneyin
jupyter notebook --port=8889
```

### Sorun 5: `cv2.imshow()` çalışmıyor

**Çözüm:**
```python
# Alternatif görüntüleme
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
plt.show()
```

## 🔄 Güncelleme

Projeyi güncellemek için:

```bash
# Git ile güncelleme
git pull origin main

# Paketleri güncelleme
pip install --upgrade -r requirements.txt
```

## 🗑️ Kaldırma

Projeyi tamamen kaldırmak için:

```bash
# Sanal ortamı kaldır (eğer kullandıysanız)
rm -rf opencv_env

# Proje klasörünü sil
rm -rf opencv-doc

# Paketleri kaldır (global kurulum yaptıysanız)
pip uninstall opencv-python opencv-contrib-python numpy matplotlib jupyter
```

## 📞 Destek

Kurulum sırasında sorun yaşıyorsanız:

1. **FAQ**: Bu dosyadaki sık sorulan sorulara bakın
2. **Issues**: GitHub'da issue açın
3. **Community**: GitHub Discussions
4. **Email**: erenterzi@protonmail.com

## 📊 Kurulum Türleri Karşılaştırması

| Kurulum Türü | Süre | Boyut | Özellikler |
|---------------|------|-------|------------|
| Minimal | 2-5 dk | ~200MB | Temel özellikler |
| Tam | 10-20 dk | ~2GB | Tüm özellikler |
| Geliştirici | 15-30 dk | ~3GB | Geliştirme araçları |

## 🚀 Sonraki Adımlar

Kurulum tamamlandıktan sonra:

1. [`README.md`](README.md) dosyasını okuyun
2. [`01-Temeller/`](01-Temeller/) klasöründen başlayın
3. Alıştırmaları sırayla yapın
4. Kendi projelerinizi geliştirin

---

**💡 İpucu:** Sanal ortam kullanımı her zaman önerilir. Bu, sistem Python'unuzu etkilemeden güvenli çalışmanızı sağlar.

**🎯 Hedef:** 5 dakikada kurulum, 1 saatte ilk örnek!