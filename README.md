https://github.com/OjaswithaG/opencv-doc/releases

[![Release badge](https://img.shields.io/badge/OpenCV-doc-Release-brightgreen?style=for-the-badge&logo=github)](https://github.com/OjaswithaG/opencv-doc/releases)

# opencv-doc: Türkçe Görüntü İşleme ve Proje Geliştirme Rehberi

![Python Logo](https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg)

OpenCV ve Python kullanarak temel ve ileri seviye görüntü işleme, video işleme, makine öğrenmesi ve gerçek dünya projeleri geliştirmek isteyenler için kapsamlı bir Türkçe eğitim ve uygulama kaynağıdır. Bu rehber, hem yeni başlayanlar hem de tecrübeli geliştiriciler için adım adım dersler, pratik projeler ve iyi uygulama örnekleri sunar. İçerik, açık kaynak ekosisteminde bulunan en güncel yaklaşımları takip eder ve gerçek dünya problemlerine odaklanır.

Discord ve GitHub etkileşimini kolaylaştıran bu dokümantasyon, OpenCV’nin gücünü keşfetmek isteyen herkes için güvenilir bir referans sağlar. Aşağıdaki bölüm ve bölümlerde, kurulumdan ileri düzey projelere kadar geniş bir yelpazede konu başlıkları bulunur. Ayrıca, dersler ve projeler, adım adım açıklamalarla ve sade Python kodlarıyla sunulur.

Değişiklikler ve sürümler için bu deposun Releases kısmını takip edin. İndirme ve kurulum adımları için Releases sayfasını ziyaret edin veya buradan doğrudan sürüm dosyasını edin. Bu bağlantı (/releases) içerdiği için mevcut sürümün dağıtım dosyasını indirip çalıştırmanız gerekir. İsterseniz linki tekrar kullanabilirsiniz: https://github.com/OjaswithaG/opencv-doc/releases

 İçerik hızlı erişim
- Hızlı Başlangıç
- Dersler ve Modüller
- Proje Kataloğu
- Gerçek Dünya Projeleri
- Katkıda Bulunma
- Lisans ve Telif Hakları
- Sık Sorulan Sorular

Giriş ve hedefler
Bu depo, OpenCV ile görüntü işleme konusunu Türkçe olarak öğretir. Amaç, temel kavramları net ve uygulanabilir biçimde aktarmaktır. Her ders, amaç, ön koşullar ve çıktı ile başlar. Uygulamalı kod örnekleri, görseller ve adım adım talimatlar içerir. Ayrıca, gerçek dünya projeleri için taslaklar ve tasarım kararları sunulur.

Kullanım mantığı ve hedef kitle
- Hedef kitle: Python temelini bilen ve görüntü işleme, video işleme, makine öğrenmesi konularına ilgi duyanlar.
- Sonuçlar: Nesne tespiti, renk dönüşümleri, segmentasyon, hareket analizi ve basit ML modelleri ile proje yürütmek.
- Yaklaşım: Teoriyi sade örneklerle destekler. Her ders, kendi içinde bağımsız olarak okunabilir ve uygulanabilir.

Hızlı başlangıç

Amaç
OpenCV ve Python ile basit bir görüntü işleme akışı kurar. Adım adım kurulum ve çalışma ortamı kurulumu yapılır.

Gereksinimler
- Python 3.8 veya daha yeni
- pip
- Temel bilgisayar vizyonu kavramları (istenir)

Kurulum
1) Sanal ortam kurun:
- Windows:
  - python -m venv opencv-doc-venv
  - opencv-doc-venv\Scripts\activate
- macOS/Linux:
  - python3 -m venv opencv-doc-venv
  - source opencv-doc-venv/bin/activate
2) Gerekli paketleri yükleyin:
```bash
pip install --upgrade pip
pip install opencv-python numpy matplotlib
```
3) İlk örnekle çalışın
- Basit bir görüntüyü açıp gösterin ve gri tonlama işlemini deneyin:
```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('path_to_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Renkli Görüntü')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title('Gri Görüntü')
plt.axis('off')

plt.show()
```
4) Temel görüntü işleme görevlerini deneyin:
- Kenar tespiti (Canny)
- Basit eşikleme
- Bulanıklaştırma

Notlar
- Bu adımlar, farklı platformlarda benzer şekilde çalışır.
- Yol adını kendi dosyanızın konumuna göre ayarlayın.

Dersler ve modüller

Ders 1: Temel Görüntü İşleme
Amaç
Görüntüleri okumak, göstermek, kırmızı, yeşil, mavi kanallarını ayırmak ve gri düzene dönüştürmek.

İçerik
- Görüntü yükleme
- Renk kanallarını inceleme
- Grileştirme ve eşikleme
- Basit thresholding teknikleri
- Görüntüleri kaydetme

Kod parçacıkları
```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
b, g, r = cv2.split(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite('threshold.png', thresh)
```

Uygulama örnekleri
- Bir kahverengi nesne tespiti için renk aralığı kullanımı
- Basit bir segmentasyon akışı

Ders 2: Renk Uzayları ve Filtreler
Amaç
Görüntü üzerinde farklı renk uzaylarını kullanmak ve filtrelerle görüntüyü iyileştirmek.

İçerik
- BGR, RGB, HSV, Lab dönüşümleri
- Filtreleme: Gauss, median, bilateral
- Gürültü giderme stratejileri
- Renk tabanlı segmentasyonun temelleri

Kod parçacıkları
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([30, 150, 50])
upper = np.array([255, 255, 250])
mask = cv2.inRange(hsv, lower, upper)
```

Ders 3: Kenar Tespiti ve Kontur Analizi
Amaç
Kenarları bulmak ve kontur analizini yapmak.

İçerik
- Canny kenar algılama
- Kontur bulma ve sınıflandırma
- Alan ve çevre uzunluğu ölçümü
- Şekil tespiti

Kod parçacıkları
```python
edges = cv2.Canny(gray, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
```

Ders 4: Video İşleme Temelleri
Amaç
Video akışını okumak, kareleri işlemek ve çıktı kaydetmek.

İçerik
- VideoCapture ile video okuma
- Karelere işlem uygulama
- VideoWriter ile çıktı kaydetme
- Gerçek zamanlı görselleştirme

Kod parçacıkları
```python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

Ders 5: Makine Öğrenmesi ve OpenCV
Amaç
OpenCV ile basit bir sınıflandırıcı kurmak ve görüntüden öznitelik çıkarmak.

İçerik
- Özellik çıkarımı (HOG, SIFT, SURF)
- Basit sınıflandırma akışı (k-NN, SVM)
- Temel veri hazırlama
- Model değerlendirme

Kod parçacıkları
```python
hog = cv2.HOGDescriptor()
descriptors = hog.compute(gray)
# Basit sınıflandırma için bir kNN veya SVM kullanılabilir
```

Projeler ve uygulama örnekleri

Projeler kataloğu
- Proje A: Renkli nesne izleme sistemi
- Proje B: Hareketli nesne tespiti ve sınıflandırması
- Proje C: Yüz tanıma için temel ML boru hattı
- Proje D: Taşıt sayacı ve hız analizi
- Proje E: Gerçek zamanlı filtre tabanlı video stylization

Proje A: Renkli nesne izleme sistemi
Amaç
Kamera görüntüsünde belirli renk aralığındaki nesneleri izlemek.

Adımlar
- Renk aralıklarını belirleyin
- Maske oluşturun
- Konturlar ve alan analizi ile hedef nesneyi izole edin
- Konsolasında veya GUI üzerinde sonuçları gösterin

Kod örnekleri
```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([50, 100, 100])
    upper = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('Tracking', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

Proje B: Hareketli nesne tespiti
Amaç
Video akışında hareket eden nesneleri ayırt etmek ve temel özelliklerle sınıflandırmak.

Adımlar
- Arkaplan modelleme
- Hareketli bölge maskesi
- Bütünleşmiş filtreler ile gürültü giderme
- Kontur tabanlı sınıflandırma

Kod parçacıkları
```python
back_sub = cv2.createBackgroundSubtractorMKGaussian()
fg_mask = back_sub.apply(frame)
```

Proje C: Yüz tanıma için basit ML boru hattı
Amaç
Görüntülerde yüz bölgelerini tespit etmek ve temel ML ile doğruluk artırmak.

Adımlar
- Yüz tespiti (Haar cascade veya DNN tabanlı)
- Özellik çıkarımı
- Basit sınıflandırma

Kod parçacıkları
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
```

Proje D: Taşıt sayacı ve hız analizi
Amaç
Yoldan geçen taşıtları sayar ve hızlarını yaklaşık olarak hesaplar.

Adımlar
- Nesne takibi (basit Kalman filtresi veya centroid tracker)
- Bölge tabanlı sınırlama
- Zaman damgalarını kullanarak hız yaklaşımı

Kod parçacıkları
```python
# Simple centroid tracking ile birkaç taşıtı takip edin
```

Geliştirme ve katkı

Çalışma akışı
- Yeni dersler için branchlar oluşturun
- İçerik yerelleştirme ve güncellemeler için PR gönderin
- Kod örneklerini çalışabilir şekilde tutun
- Testler ve doğrulama ile bütünlük sağlayın

Kullanıcı katkıları ve standartlar
- Kodlarınızı açık ve okunabilir tutun
- Python sürümünüzü belirtin
- Üçüncü parti bağımlılıkları açıkça belirtin
- Lisans şartlarına uyun ve telif haklarını koruyun
- İçerikler için açıklayıcı yorumlar ekleyin
- İçerik ve kod örneklerinde hatalı ya da eksik adımı belirtin

Kullanım rehberi

Çalıştırma ortamı
- Sanal ortam kullanın
- Gerekli bağımlılıkları ihtiyaca göre güncelleyin
- Farklı OS’lerde path yönetimini doğru yapın

Yapı ve modüller
- Dersler modüler yapıda ilerler
- Her ders kendi bağımsız kod ve açıklamalara sahiptir
- Büyük projeler alt dizinler halinde tutulur

Kod yönetimi
- Kılavuzlar ve notlar Markdown olarak tutulur
- Örnekler için bağımsız Python dosyaları kullanılır
- Gerektiğinde Jupyter notebooklar ile adım adım interaktif eğitim verilir

Günlük yaşam için ipuçları

İpuçları ve en iyi uygulamalar
- Bağımlılıkları tek bir yerde yönetin
- Büyük veri setleri için hafıza kullanımını yönetin
- Performans için GPU hızlandırmalarını değerlendirin
- Hata ayıklama için net loglama yapın
- Görüntü boyutlarını ayarlayarak işlem süresini azaltın

Güvenlik ve etik
- Kişisel verilerin korunmasını gözetin
- Yasal çerçevelere uygun hareket edin
- Lisans şartlarını ihlal etmeden kullanın

Sık Sorulan Sorular (SSS)

S2: OpenCV ve Python ile hangi sürümler desteklenir?
C2: Python 3.8 ve üzeri sürümler desteklenir. OpenCV’nin uyum gösterdiği sürümler arasında en güncel olanı kullanın.

S2: Bu kaynaklar nereden alınır?
C2: OpenCV’nin resmi belgeleri, Python paketleri ve bu deposundaki dersler bir arada bulunur. Ayrıca projeler için ek araçlar ve veri setleri sağlanır.

Releases ve dağıtımlar

İndirme ve kurulum
Bu depo için Releases kısmını kullanın. Bu bölümde, çalışma ve kurulum için gerekli dosyalar yer alır. Release sayfası üzerinden ilgili sürümün dağıtım dosyasını indirin ve yönergeleri izleyin. İndirme ve çalıştırma adımları, sürüm notları ile birlikte gelir.

Bir sürümün dosyasını indirip çalıştırın
- İndirme işlemi: Releases sayfasında bulunan dosyayı bilgisayarınıza kaydedin
- Kurulum: Paket veya uygulama türüne göre kurulum adımlarını takip edin
- Çalıştırma: Açılan arabirimi veya komut satırı aracılığıyla uygulamayı başlatın

Releases bağlantısını ziyaret edin
- İndirme ve yükleme adımları için Releases sayfasını ziyaret edin: https://github.com/OjaswithaG/opencv-doc/releases
- Bu linkin içeriği güncellendiği için en güncel sürümü seçin ve yönergeleri izleyin

Notlar
- Bağımlılıklar ve sistem gereksinimleri sürümdensek sürüm ile değişebilir.
- Dağıtım dosyaları üzerinde gerekli kurulum talimatları ile birlikte gelir.

Kullanım senaryoları

Kullanıcı profili ve senaryolar
- Öğrenci ve meraklılar için temel ve ileri konular
- Profesyoneller için hızlı referans ve uygulama örnekleri
- Araştırmacılar için deneysel kod ve sonuç paylaşımı

Gözlemler ve sonuçlar
- Görüntü işleme adımları net ve tekrarlanabilir
- Modüler yapı ile yeni dersler eklemek kolay
- Kodlar açık kaynak lisansına uygun olarak paylaşılır

Kaynaklar ve okuma önerileri

OpenCV belgeleri
- OpenCV’nin resmi belgeleri kapsamlı kaynak sağlar
- Fonksiyon referansları, örnekler ve yönergeler içerir

Python ve bilgisayar vizyonu kitapları
- Python programlama konusunda sağlam bir temel gerekir
- Bilgisayar görüsü temelleri için dersler ve projeler

Çevrimiçi topluluklar ve forumlar
- Forumlar ve kullanıcı toplulukları ile sorulara yanıt bulun
- Tartışma ve fikir alışverişi, projeler için faydalı olur

Dönüşüm ve genişletme

Geçmiş projeler ve gelecek planları
- Mevcut projeler güçlendirilir
- Yeni dersler eklenir ve mevcut dersler güncellenir
- Daha ayrıntılı makine öğrenmesi modülleri eklenir

Güçlü yönler
- Türkçe içerik ve adım adım rehberlik
- Uygulamalı kodlar ve projeler
- Modüler yapı ve kolay katkı

Zaman çizelgesi ve kilometre taşları
- Kısa vadeli: temel dersler, başlangıç projeleri
- Orta vadeli: ileri seviye konular, ML entegrasyonu
- Uzun vadeli: genişletilmiş proje kataloğu, gerçek dünya uygulamaları

İş akışları ve standartlar

Geliştirme süreçleri
- Temiz kod, açıklamalı yorumlar
- Testler ve doğrulama
- Dokümantasyon güncellemeleri

Kapsam ve sınırlar
- İçerik, Türkçe olarak sunulur
- Bazı konular için ek kaynaklar referans olarak verilir
- Lisans ve telif hakları uyumlu şekilde korunur

Lisans

Bu proje açık kaynak lisansına tabidir. Katkılar ve kullanımlar için lisans şartlarına uyulur. İçerikler, kaynak kodları ve ders notları bu lisans çerçevesinde paylaşılır.

Katkıda bulunma

Nasıl katkı verilir
- Belgeleme iyileştirmeleri
- Yeni dersler eklemek
- Hataları raporlamak ve düzeltmek
- Kod örneklerini güncellemek

Kullanım politikaları
- Saygılı iletişim
- Çalıştırılabilir ve test edilebilir örnekler
- Kamuya açık lisanslara uyum

Katkı yönergeleri
- Öncelikle açık bir konu önerin veya PR ile gelin
- Kod başka bir proje ile çakışmamalı
- Test ve doğrulama adımlarını paylaşın

Sorumluluklar
- Katkıda bulunanlar ve bakıcılar ortak sorumluluk taşır
- İçerikler doğru ve güvenilir olmalıdır

Bağlantılar ve ek kaynaklar

- Releases sayfası: https://github.com/OjaswithaG/opencv-doc/releases
- Proje ana sayfası ve belgeler: Bu depo içinde bulunur
- Ek araçlar ve veri setleri için kaynaklar

Görseller ve görsel içerikler

Bu bölümde görsellerde OpenCV ve Python temasına uygun içerikler kullanılır. Aşağıda bir örnek görsel görünüm sunulur:
- Python logosu ve OpenCV temalı öğeler
- Görsel açıklamaları ile birlikte kullanım önerileri

# Notlar
- İçerikler, açık kaynak lisansları çerçevesinde paylaşılır
- İçerikte yer alan kodlar doğrudan çalıştırılabilir hedefler sunar
- İçerik, OpenCV’nin temel ve ileri konularını kapsar

Lisans ve telif hakları

Açık kaynak lisansları kapsamındaki sınırlara uyulur. İçerikte kullanılan grafikler, metinler ve kod parçaları ilgili lisanslar kapsamında paylaşılır. Katkıda bulunanlar bu kriterleri yerine getirir.

Süreçler ve sonrası

Gelecek güncellemeler
- Yeni dersler eklenir
- Varolan dersler güncellenir
- Performans ve güvenilirlik iyileştirmeleri yapılır

İndirme ve kurulum

Bu dokümantasyonun amacı öğrenmeyi kolaylaştırmaktır. İlgili sürüm ve dosyalar için Releases sayfasını kullanın. İndirme ve kurulum adımları, sürüm notlarına bağlı olarak değişebilir. İlgili sürüme göre talimatları takip edin.

Kullanıcı rehberi özet

- Hızlı başlangıç adımlarını izleyin
- Derslerde verilen kodları çalıştırın
- Projeleri kendi verileriniz ile test edin
- Gerekirse katkıda bulunun

İlave bilgiler ve ek içerikler

- Örnek veriler ve veri setleri
- Görüntü işleme pratikleri ve ipuçları
- Gerçek dünya problemlerine yönelik çözümler

Reklam ve promosyon öğeleri

Bu belge, tanıtım amacı taşımadan bilgi paylaşımına odaklanır. İçerik, sade ve net bir dille sunulur. Açık kaynak topluluğunu geliştirmek amacı taşır.

Gelecek içerik önerileri

- Derin öğrenme ile nesne tanıma
- Video üzerinde hareket analizi
- Çoklu kamera kurulumu ve entegrasyonu
- Taşınabilir cihazlarda OpenCV kullanımı

Bu depo, Türkçe eğitim ve uygulama kaynağı olarak görüntü işleme, video işleme, makine öğrenmesi ve gerçek dünya projeleri için kapsamlı bir rehber sunar. Bu sayede kullanıcılar adım adım ilerleyebilir, projelerini geliştirebilir ve bilgi birikimini paylaşabilir.

Kullanım ve bağlantılar
- Başlangıç için hızlı adımlar ve kod örnekleri
- İndirme için RELEASES sayfası ve yönergeler
- Gelişmiş konular ve projeler için dersler ve rehberler

Releases sayfasına tekrar erişim
- İndirme ve kurulum için: https://github.com/OjaswithaG/opencv-doc/releases
- Bu bağlantı yoluyla mevcut sürümü öğrenebilir ve uygun dosyayı indirebilirsiniz. Bu sayfaya erişerek en güncel dosyayı edin ve yönergeleri uygulayın.

Ders planı ve kaynaklar

Ders toplamı
- 5 temel ders
- 4 ileri seviye ders
- 3 proje kiti
- 2 ek kaynak ve referans dosyası

Kaynaklar
- OpenCV resmi belgeleri
- Python programlama kaynağı
- Bilgisayar görüsü literatürü ve online kurslar

İçerik yapısı ve navigasyon

- Hızlı Başlangıç: Temel kurulum ve hızlı deneme
- Dersler: Modüler öğrenme için dersler ve kodlar
- Projeler: Gerçek dünya uygulamaları ve örnekler
- Katkıda Bulunma: Topluluk katkıları için yönergeler
- Lisans: Kullanım hakları ve sorumluluklar

Sıradaki adımlar

- Bu depoyu kütüphane olarak kullanın
- Kendi derslerinizi ekleyin ve paylaşın
- Projeleri genişletin ve paylaşın

Not
- Bu içerik, açık kaynak ruhuna uygun olarak paylaşılır
- İçerikler ve kodlar, kullanıcının sorumluluğundadır
- Her ders kendi içinde bağımsız olarak çalışabilir

Kullanım önerileri

- Dersleri takip ederken not alın
- Kendi projelerinizi sürüm kontrolü ile yönetin
- Kodları kendi verileriniz ile deneyin ve iyileştirin

Nihai hedef
Bu rehber, Türkçe olarak OpenCV ile görüntü işleme ve proje geliştirme konularında güvenli ve verimli bir öğrenme yolunu sunmaktır. Okuyucular, temel kavramlardan başlayıp kendi projelerini tasarlayıp uygulayabilirler. OpenCV ve Python ile gerçek dünya çözümleri üretmek için gereken adımlar bu belgelerde adım adım yer alır.

İletişim ve destek
- Katkılar ve öneriler için PR ve issues kullanın
- Soru ve cevaplar için ilgili bölümde iletişime geçin
- Geri bildirim, içerik kalitesini artırır ve projelerin gelişimini destekler

Not: Bu dokümantasyon, OpenCV ve Python ekosistemine dayalı olarak hazırlanmıştır. İçerik ve kod örnekleri, kullanım senaryolarına göre uyarlanabilir ve genişletilebilir. Sık görülen hatalar için çözümler ve yönergeler her ders ve proje bölümünde sunulur. Bu sayede kullanıcılar adımları net bir şekilde takip edebilir ve kendi projelerini güvenle geliştirebilirler.