#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ OpenCV Yardımcı Fonksiyonları
===============================

Bu modül, OpenCV ile çalışırken sık kullanılan temel işlemler için
güvenli ve kolay kullanımlı wrapper fonksiyonları sağlar.

Yazan: Eren Terzi
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Union

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_image(image: np.ndarray) -> bool:
    """
    Resmin geçerli olup olmadığını kontrol eder
    
    Args:
        image: Kontrol edilecek resim
        
    Returns:
        bool: Resim geçerliyse True, değilse False
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False
    
    return image.size > 0

def get_image_info(image: np.ndarray) -> dict:
    """
    Resim hakkında detaylı bilgi döndürür
    
    Args:
        image: Analiz edilecek resim
        
    Returns:
        dict: Resim bilgileri
        
    Raises:
        ValueError: Geçersiz resim için
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'size': image.size,
        'ndim': image.ndim
    }
    
    if len(image.shape) == 3:
        info['height'], info['width'], info['channels'] = image.shape
    else:
        info['height'], info['width'] = image.shape
        info['channels'] = 1
    
    info['min_value'] = np.min(image)
    info['max_value'] = np.max(image)
    info['mean_value'] = np.mean(image)
    
    return info

def safe_resize(image: np.ndarray, 
                width: Optional[int] = None, 
                height: Optional[int] = None,
                scale: Optional[float] = None,
                inter: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Güvenli resim boyutlandırma
    
    Args:
        image: Boyutlandırılacak resim
        width: Hedef genişlik
        height: Hedef yükseklik  
        scale: Ölçek faktörü (width/height yerine)
        inter: İnterpolasyon yöntemi
        
    Returns:
        np.ndarray: Boyutlandırılmış resim
        
    Raises:
        ValueError: Geçersiz parametreler için
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    h, w = image.shape[:2]
    
    # Ölçek faktörü kullanılıyorsa
    if scale is not None:
        width = int(w * scale)
        height = int(h * scale)
    
    # Sadece genişlik verilmişse, en-boy oranını koru
    elif width is not None and height is None:
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    # Sadece yükseklik verilmişse, en-boy oranını koru
    elif height is not None and width is None:
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    
    # Her ikisi de verilmemişse hata
    elif width is None and height is None:
        raise ValueError("En az bir boyut parametresi gerekli")
    
    # Minimum boyut kontrolü
    width = max(1, width)
    height = max(1, height)
    
    return cv2.resize(image, (width, height), interpolation=inter)

def safe_rotate(image: np.ndarray, 
                angle: float,
                center: Optional[Tuple[int, int]] = None,
                scale: float = 1.0,
                border_value: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Güvenli resim döndürme
    
    Args:
        image: Döndürülecek resim
        angle: Döndürme açısı (derece)
        center: Döndürme merkezi (None ise resim merkezi)
        scale: Ölçek faktörü
        border_value: Sınır dolgu rengi
        
    Returns:
        np.ndarray: Döndürülmüş resim
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    h, w = image.shape[:2]
    
    # Merkez belirleme
    if center is None:
        center = (w // 2, h // 2)
    
    # Döndürme matrisi
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Döndürme işlemi
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                            borderValue=border_value)
    
    return rotated

def safe_crop(image: np.ndarray, 
              x: int, y: int, 
              width: int, height: int) -> np.ndarray:
    """
    Güvenli resim kırpma
    
    Args:
        image: Kırpılacak resim
        x: Başlangıç x koordinatı
        y: Başlangıç y koordinatı
        width: Kırpma genişliği
        height: Kırpma yüksekliği
        
    Returns:
        np.ndarray: Kırpılmış resim
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    h, w = image.shape[:2]
    
    # Koordinat sınırlarını kontrol et
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    
    # Boyut sınırlarını kontrol et
    width = min(width, w - x)
    height = min(height, h - y)
    
    return image[y:y+height, x:x+width]

def auto_contrast(image: np.ndarray, 
                  clip_percent: float = 1.0) -> np.ndarray:
    """
    Otomatik kontrast ayarlama
    
    Args:
        image: İşlenecek resim
        clip_percent: Kırpma yüzdesi (0-50 arası)
        
    Returns:
        np.ndarray: Kontrast ayarlanmış resim
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    # Gri tonlama dönüşümü
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Histogram hesaplama
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Kırpma değerleri
    total_pixels = gray.shape[0] * gray.shape[1]
    clip_pixels = total_pixels * clip_percent / 100
    
    # Alt ve üst sınırları bul
    cumsum = np.cumsum(hist)
    
    low_val = 0
    high_val = 255
    
    for i in range(256):
        if cumsum[i] >= clip_pixels:
            low_val = i
            break
    
    for i in range(255, -1, -1):
        if cumsum[i] <= total_pixels - clip_pixels:
            high_val = i
            break
    
    # Kontrast uygula
    if len(image.shape) == 3:
        result = image.copy()
        for channel in range(3):
            result[:, :, channel] = np.clip(
                (image[:, :, channel] - low_val) * 255 / (high_val - low_val),
                0, 255
            ).astype(np.uint8)
    else:
        result = np.clip(
            (image - low_val) * 255 / (high_val - low_val),
            0, 255
        ).astype(np.uint8)
    
    return result

def blend_images(img1: np.ndarray, 
                 img2: np.ndarray, 
                 alpha: float = 0.5) -> np.ndarray:
    """
    İki resmi harmanlama
    
    Args:
        img1: İlk resim
        img2: İkinci resim
        alpha: Harmanlanma oranı (0.0-1.0)
        
    Returns:
        np.ndarray: Harmanlanmış resim
    """
    if not is_valid_image(img1) or not is_valid_image(img2):
        raise ValueError("Geçersiz resim formatı")
    
    # Boyutları eşitle
    if img1.shape != img2.shape:
        h1, w1 = img1.shape[:2]
        img2 = cv2.resize(img2, (w1, h1))
    
    # Harmanlama
    return cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)

def create_border(image: np.ndarray,
                  top: int, bottom: int, left: int, right: int,
                  border_type: int = cv2.BORDER_CONSTANT,
                  value: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Resme sınır ekler
    
    Args:
        image: İşlenecek resim
        top, bottom, left, right: Sınır kalınlıkları
        border_type: Sınır tipi
        value: Sınır rengi
        
    Returns:
        np.ndarray: Sınırlı resim
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    return cv2.copyMakeBorder(image, top, bottom, left, right,
                             border_type, value=value)

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Histogram eşitleme
    
    Args:
        image: İşlenecek resim
        
    Returns:
        np.ndarray: Histogram eşitlenmiş resim
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    if len(image.shape) == 3:
        # Renkli resim için YUV dönüşümü
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # Gri tonlama resim
        return cv2.equalizeHist(image)

def calculate_brightness(image: np.ndarray) -> float:
    """
    Resmin parlaklığını hesaplar
    
    Args:
        image: Analiz edilecek resim
        
    Returns:
        float: Ortalama parlaklık (0-255 arası)
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return np.mean(gray)

def adjust_brightness(image: np.ndarray, 
                     brightness: int = 0) -> np.ndarray:
    """
    Parlaklık ayarlama
    
    Args:
        image: İşlenecek resim
        brightness: Parlaklık değişimi (-100 ile 100 arası)
        
    Returns:
        np.ndarray: Parlaklık ayarlanmış resim
    """
    if not is_valid_image(image):
        raise ValueError("Geçersiz resim formatı")
    
    # Parlaklık değerini sınırla
    brightness = max(-100, min(100, brightness))
    
    # Parlaklık uygula
    if brightness >= 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness
    
    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow
    
    return cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

# Test fonksiyonu
def test_helpers():
    """Yardımcı fonksiyonları test eder"""
    print("🧪 OpenCV Helpers Test Başladı...")
    
    # Test resmi oluştur
    test_image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    
    # Fonksiyonları test et
    print("✅ is_valid_image:", is_valid_image(test_image))
    print("✅ get_image_info:", get_image_info(test_image)['shape'])
    
    resized = safe_resize(test_image, width=150)
    print("✅ safe_resize:", resized.shape)
    
    rotated = safe_rotate(test_image, 45)
    print("✅ safe_rotate:", rotated.shape)
    
    cropped = safe_crop(test_image, 50, 50, 100, 100)
    print("✅ safe_crop:", cropped.shape)
    
    brightness = calculate_brightness(test_image)
    print(f"✅ calculate_brightness: {brightness:.2f}")
    
    print("🎉 Tüm testler başarılı!")

if __name__ == "__main__":
    test_helpers()