# 🚀 OpenCV'ye Giriş ve Kurulum

## 🤔 OpenCV Nedir?

**OpenCV** (Open Source Computer Vision Library), görüntü işleme ve bilgisayarlı görü uygulamaları için geliştirilmiş açık kaynaklı bir kütüphanedir.

### ✨ Temel Özellikler

- 📸 **Görüntü İşleme**: Filtreleme, dönüştürme, iyileştirme
- 🎥 **Video İşleme**: Video analizi, hareket algılama
- 🤖 **Makine Öğrenmesi**: Sınıflandırma, tespit algoritmaları
- 👁️ **Bilgisayarlı Görü**: Nesne tanıma, yüz algılama
- 📊 **Geometri**: 3D rekonstrüksiyon, kamera kalibrasyonu

### 🏭 Kullanım Alanları

#### 🚗 **Otomotiv Sektörü**
- Otonom sürüş sistemleri
- Trafik işareti tanıma
- Şerit takip sistemleri

#### 🏭 **Endüstri 4.0**
- Kalite kontrol sistemleri
- Robot görüş sistemleri
- Üretim hattı izleme

#### 🏥 **Tıp ve Sağlık**
- Medikal görüntü analizi
- Röntgen, MR analizi
- Mikroskopi görüntü işleme

#### 📱 **Mobil ve Web**
- Fotoğraf editörü uygulamaları
- QR kod okuyucular
- Artırılmış gerçeklik (AR)

#### 🔒 **Güvenlik**
- Yüz tanıma sistemleri  
- Hareket algılama
- Plaka tanıma sistemleri

## 💻 Kurulum

### 🐍 Python için Kurulum

#### 1. **Temel OpenCV Kurulumu**
```bash
pip install opencv-python
```

#### 2. **Ekstra Modüller ile Kurulum (Önerilen)**
```bash
pip install opencv-contrib-python
```

#### 3. **Gerekli Ek Kütüphaneler**
```bash
pip install numpy matplotlib
```

#### 4. **Jupyter Notebook (Opsiyonel)**
```bash
pip install jupyter
```

### 🔧 Kurulum Kontrolü

Kurulumun başarılı olup olmadığını kontrol etmek için:

```python
import cv2
import numpy as np

print("OpenCV Sürümü:", cv2.__version__)
print("NumPy Sürümü:", np.__version__)

# Başarılı kurulum test resmi
test_image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imshow('Test', test_image)
cv2.waitKey(1000)  # 1 saniye bekle
cv2.destroyAllWindows()
print("✅ Kurulum başarılı!")
```

### 🖥️ Farklı İşletim Sistemleri

#### Windows 🪟
```bash
# Anaconda ile
conda install -c conda-forge opencv

# pip ile
pip install opencv-python
```

#### macOS 🍎
```bash
# Homebrew ile
brew install opencv

# pip ile
pip3 install opencv-python
```

#### Linux 🐧
```bash
# Ubuntu/Debian
sudo apt-get install python3-opencv

# pip ile
pip3 install opencv-python
```

## 🏗️ Geliştirme Ortamı Kurulumu

### 1. **IDE Önerileri**

#### 🐍 **Python için:**
- **PyCharm** - Güçlü Python IDE
- **Visual Studio Code** - Hafif ve genişletilebilir
- **Jupyter Notebook** - İnteraktif geliştirme
- **Spyder** - Bilimsel hesaplama odaklı

#### ⚙️ **Ayarlar:**
```python
# VS Code settings.json için öneriler
{
    "python.defaultInterpreterPath": "python",
    "python.linting.enabled": true,
    "python.formatting.provider": "black"
}
```

### 2. **Sanal Ortam (Virtual Environment)**

Projenizi izole etmek için sanal ortam oluşturun:

```bash
# Sanal ortam oluşturma
python -m venv opencv_env

# Aktifleştirme (Windows)
opencv_env\Scripts\activate

# Aktifleştirme (macOS/Linux)
source opencv_env/bin/activate

# Kurulum
pip install opencv-contrib-python numpy matplotlib
```

## 🧪 İlk Test

Kurulumunuzu test etmek için bu basit kodu çalıştırın:

```python
import cv2
import numpy as np

# Renkli bir resim oluştur
height, width = 300, 400
image = np.zeros((height, width, 3), dtype=np.uint8)

# Renkleri tanımla
blue = (255, 0, 0)    # BGR formatında mavi
green = (0, 255, 0)   # BGR formatında yeşil  
red = (0, 0, 255)     # BGR formatında kırmızı

# Renk çizgileri ekle
image[:100, :] = blue    # Üst kısım mavi
image[100:200, :] = green # Orta kısım yeşil
image[200:, :] = red     # Alt kısım kırmızı

# Metin ekle
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'OpenCV Kurulumu Basarili!', 
           (50, 150), font, 0.7, (255, 255, 255), 2)

# Resmi göster
cv2.imshow('OpenCV Test', image)
cv2.waitKey(0)  # Bir tuşa basılmasını bekle
cv2.destroyAllWindows()

print("🎉 Tebrikler! OpenCV başarıyla kuruldu.")
```

## 🔍 Sık Karşılaşılan Sorunlar

### ❌ **Sorun 1**: `ImportError: No module named cv2`
**Çözüm**: OpenCV kurulu değil
```bash
pip install opencv-python
```

### ❌ **Sorun 2**: `error: Microsoft Visual C++ 14.0 is required`
**Çözüm**: Visual C++ Build Tools kurulumu gerekli
- Microsoft Visual C++ Build Tools indirin ve kurun

### ❌ **Sorun 3**: `cv2.imshow` çalışmıyor
**Çözüm**: GUI backend problemi
```python
# Alternatif görüntüleme yöntemi
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

### ❌ **Sorun 4**: `ImportError: No module named numpy`
**Çözüm**: NumPy kurulumu
```bash
pip install numpy
```

## 📚 Faydalı Kaynaklar

### 🌐 **Resmi Dokümantasyon**
- [OpenCV Resmi Sitesi](https://opencv.org/)
- [OpenCV Python Dokümanları](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### 📖 **Öğrenme Kaynakları**
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [GitHub OpenCV](https://github.com/opencv/opencv)

### 🎥 **Video Eğitimler**
- YouTube OpenCV playlists
- Coursera Computer Vision kursları

## ✅ Kontrol Listesi

Devam etmeden önce aşağıdakileri kontrol edin:

- [ ] OpenCV başarıyla kuruldu (`cv2.__version__` çalışıyor)
- [ ] NumPy kurulu (`import numpy` çalışıyor)
- [ ] Test kodu başarıyla çalıştı
- [ ] IDE/editör hazır
- [ ] Örnek resimler indirilebilir durumda

## 🚀 Sonraki Adım

Kurulum tamamlandı! Şimdi [İlk OpenCV Programınızı](02-ilk-program.py) yazma zamanı!

---

**💡 İpucu:** Kurulum sırasında sorun yaşarsanız, önce pip ve Python sürümünüzü güncelleyin:
```bash
pip install --upgrade pip
python --version
```