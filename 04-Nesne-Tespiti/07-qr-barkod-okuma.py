#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📱 QR ve Barkod Okuma - OpenCV QR Code & Barcode Detection
========================================================

Bu modül QR code ve barkod okuma yöntemlerini kapsar:
- QR Code Detection & Decoding
- Barcode Reading (multiple formats)
- Real-time scanning
- Batch processing
- Code generation (bonus)

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import time
from collections import deque
import json
import os

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False
    print("⚠️ qrcode kütüphanesi bulunamadı. QR kod oluşturma deaktif.")
    print("   Kurulum: pip install qrcode[pil]")

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("⚠️ pyzbar kütüphanesi bulunamadı. Gelişmiş barkod okuma deaktif.")
    print("   Kurulum: pip install pyzbar")

class QRBarcodeDetector:
    """QR kod ve barkod tespit sınıfı"""
    
    def __init__(self):
        # OpenCV QR detector
        self.qr_detector = cv2.QRCodeDetector()
        
        # Detection history
        self.detection_history = deque(maxlen=100)
        self.scan_results = []
        
        # Statistics
        self.stats = {
            'total_scans': 0,
            'successful_scans': 0,
            'qr_codes': 0,
            'barcodes': 0,
            'unique_codes': set()
        }
    
    def detect_qr_opencv(self, frame):
        """OpenCV ile QR kod tespiti"""
        try:
            # QR detection
            data, bbox, _ = self.qr_detector.detectAndDecode(frame)
            
            if data and bbox is not None:
                # Convert bbox to integer
                bbox = bbox.astype(int)
                
                return [{
                    'data': data,
                    'type': 'QR_CODE',
                    'bbox': bbox,
                    'confidence': 1.0,
                    'method': 'opencv'
                }]
            
        except Exception as e:
            print(f"OpenCV QR detection error: {e}")
        
        return []
    
    def detect_codes_pyzbar(self, frame):
        """Pyzbar ile QR ve barkod tespiti"""
        if not PYZBAR_AVAILABLE:
            return []
        
        try:
            # Convert to grayscale for better detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect barcodes and QR codes
            detected_codes = pyzbar.decode(gray)
            
            results = []
            for code in detected_codes:
                # Extract data
                data = code.data.decode('utf-8')
                code_type = code.type
                
                # Create bounding box
                x, y, w, h = code.rect
                bbox = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                
                results.append({
                    'data': data,
                    'type': code_type,
                    'bbox': bbox,
                    'confidence': 1.0,
                    'method': 'pyzbar'
                })
            
            return results
            
        except Exception as e:
            print(f"Pyzbar detection error: {e}")
            return []
    
    def detect_all_codes(self, frame):
        """Tüm methodları kullanarak kod tespiti"""
        all_results = []
        
        # OpenCV QR detection
        opencv_results = self.detect_qr_opencv(frame)
        all_results.extend(opencv_results)
        
        # Pyzbar detection
        pyzbar_results = self.detect_codes_pyzbar(frame)
        all_results.extend(pyzbar_results)
        
        # Remove duplicates (same data)
        unique_results = []
        seen_data = set()
        
        for result in all_results:
            if result['data'] not in seen_data:
                unique_results.append(result)
                seen_data.add(result['data'])
        
        return unique_results
    
    def update_statistics(self, results):
        """İstatistikleri güncelle"""
        self.stats['total_scans'] += 1
        
        if results:
            self.stats['successful_scans'] += 1
            
            for result in results:
                code_type = result['type']
                data = result['data']
                
                if code_type == 'QR_CODE' or code_type == 'QRCODE':
                    self.stats['qr_codes'] += 1
                else:
                    self.stats['barcodes'] += 1
                
                self.stats['unique_codes'].add(data)
    
    def draw_detections(self, frame, results):
        """Tespit edilen kodları çiz"""
        for result in results:
            bbox = result['bbox']
            data = result['data']
            code_type = result['type']
            method = result['method']
            
            # Color based on type
            if code_type == 'QR_CODE' or code_type == 'QRCODE':
                color = (0, 255, 0)  # Green for QR
            else:
                color = (255, 0, 0)  # Blue for barcode
            
            # Draw bounding box
            if len(bbox) == 4:  # Rectangle format
                cv2.polylines(frame, [bbox], True, color, 3)
                
                # Center point
                center = np.mean(bbox, axis=0).astype(int)
                cv2.circle(frame, tuple(center), 5, color, -1)
                
                # Label
                label = f"{code_type} ({method})"
                cv2.putText(frame, label, (bbox[0][0], bbox[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Data (truncated if too long)
                data_text = data[:30] + "..." if len(data) > 30 else data
                cv2.putText(frame, data_text, (bbox[0][0], bbox[2][1] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def save_scan_results(self, filename="scan_results.json"):
        """Tarama sonuçlarını kaydet"""
        results_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': {
                'total_scans': self.stats['total_scans'],
                'successful_scans': self.stats['successful_scans'],
                'qr_codes': self.stats['qr_codes'],
                'barcodes': self.stats['barcodes'],
                'unique_codes': len(self.stats['unique_codes'])
            },
            'scan_results': self.scan_results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Sonuçlar kaydedildi: {filename}")
        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")

class QRGenerator:
    """QR kod oluşturucu"""
    
    @staticmethod
    def generate_qr_code(data, filename=None, size=10, border=4):
        """QR kod oluştur"""
        if not QRCODE_AVAILABLE:
            print("❌ qrcode kütüphanesi gerekli!")
            return None
        
        try:
            # QR code oluştur
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=size,
                border=border,
            )
            
            qr.add_data(data)
            qr.make(fit=True)
            
            # PIL image oluştur (RGB modunda)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # PIL image'ı RGB moduna çevir (boolean'dan kaçınmak için)
            if qr_img.mode != 'RGB':
                qr_img = qr_img.convert('RGB')
            
            # NumPy array'e çevir
            qr_array = np.array(qr_img)
            
            # Veri tipini kontrol et ve uint8'e çevir
            if qr_array.dtype == bool:
                qr_array = qr_array.astype(np.uint8) * 255
            elif qr_array.dtype != np.uint8:
                qr_array = qr_array.astype(np.uint8)
            
            # BGR formatına çevir
            if len(qr_array.shape) == 2:  # Grayscale
                qr_bgr = cv2.cvtColor(qr_array, cv2.COLOR_GRAY2BGR)
            else:
                qr_bgr = cv2.cvtColor(qr_array, cv2.COLOR_RGB2BGR)
            
            # Kaydet
            if filename:
                cv2.imwrite(filename, qr_bgr)
                print(f"✅ QR kod kaydedildi: {filename}")
            
            return qr_bgr
            
        except Exception as e:
            print(f"❌ QR kod oluşturma hatası: {e}")
            return None

def ornek_1_realtime_qr_scanner():
    """
    Örnek 1: Real-time QR ve barkod tarayıcı
    """
    print("\n🎯 Örnek 1: Real-time QR & Barcode Scanner")
    print("=" * 45)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = QRBarcodeDetector()
    
    # Scanning settings
    auto_save = True
    scan_delay = 1.0  # seconds between scans of same code
    last_scan_time = {}
    
    print("📱 Real-time QR & Barcode Scanner")
    print("Kontroller:")
    print("  s: Sonuçları kaydet")
    print("  c: Geçmiş temizle")
    print("  a: Auto-save toggle")
    print("  ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Detect codes
        results = detector.detect_all_codes(frame)
        
        detection_time = (time.time() - start_time) * 1000
        
        # Process results
        current_time = time.time()
        new_detections = []
        
        for result in results:
            data = result['data']
            
            # Check if we scanned this recently
            if (data not in last_scan_time or 
                current_time - last_scan_time[data] > scan_delay):
                
                new_detections.append(result)
                last_scan_time[data] = current_time
                
                # Add to scan results
                detector.scan_results.append({
                    'timestamp': time.strftime('%H:%M:%S'),
                    'data': data,
                    'type': result['type'],
                    'method': result['method']
                })
        
        # Update statistics
        detector.update_statistics(results)
        
        # Draw detections
        display_frame = detector.draw_detections(frame.copy(), results)
        
        # Info panel
        cv2.putText(display_frame, f"Detection Time: {detection_time:.1f}ms", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Codes Found: {len(results)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Total Scans: {detector.stats['total_scans']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Successful: {detector.stats['successful_scans']}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Unique Codes: {len(detector.stats['unique_codes'])}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Auto-save indicator
        save_text = "Auto-save: ON" if auto_save else "Auto-save: OFF"
        cv2.putText(display_frame, save_text, (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Recent scans
        if detector.scan_results:
            cv2.putText(display_frame, "RECENT SCANS:", (10, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            recent_scans = detector.scan_results[-5:]  # Last 5 scans
            for i, scan in enumerate(recent_scans):
                scan_text = f"{scan['timestamp']}: {scan['data'][:20]}..."
                cv2.putText(display_frame, scan_text, (10, 245 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Method availability
        methods_x = display_frame.shape[1] - 150
        cv2.putText(display_frame, "METHODS:", (methods_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        opencv_status = "✅ OpenCV" 
        pyzbar_status = "✅ Pyzbar" if PYZBAR_AVAILABLE else "❌ Pyzbar"
        
        cv2.putText(display_frame, opencv_status, (methods_x, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(display_frame, pyzbar_status, (methods_x, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (0, 255, 0) if PYZBAR_AVAILABLE else (0, 0, 255), 1)
        
        # New detections notification
        if new_detections:
            cv2.putText(display_frame, f"NEW: {len(new_detections)} codes!", 
                       (display_frame.shape[1]//2 - 80, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow('Real-time QR & Barcode Scanner', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            detector.save_scan_results()
        elif key == ord('c'):
            detector.scan_results.clear()
            detector.stats = {
                'total_scans': 0,
                'successful_scans': 0,
                'qr_codes': 0,
                'barcodes': 0,
                'unique_codes': set()
            }
            print("🧹 Geçmiş temizlendi")
        elif key == ord('a'):
            auto_save = not auto_save
            status = "ON" if auto_save else "OFF"
            print(f"🔄 Auto-save: {status}")
    
    # Final save
    if auto_save and detector.scan_results:
        detector.save_scan_results()
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_qr_generator():
    """
    Örnek 2: QR kod oluşturucu
    """
    print("\n🎯 Örnek 2: QR Code Generator")
    print("=" * 30)
    
    if not QRCODE_AVAILABLE:
        print("❌ qrcode kütüphanesi gerekli!")
        print("   Kurulum: pip install qrcode[pil]")
        return
    
    generator = QRGenerator()
    
    print("🏗️ QR Code Generator")
    
    while True:
        print("\n" + "="*40)
        print("QR Kod oluşturmak istediğiniz metni girin:")
        print("(Boş bırakırsanız örnekler gösterilir)")
        
        user_input = input("Metin: ").strip()
        
        if not user_input:
            # Show examples
            examples = [
                "Merhaba Dünya!",
                "https://www.opencv.org",
                "Tel: +90 555 123 4567",
                "OpenCV QR Test",
                "📱 QR Code Demo"
            ]
            
            print("\n📋 Örnek QR kodlar oluşturuluyor...")
            
            for i, text in enumerate(examples):
                filename = f"qr_example_{i+1}.png"
                qr_img = generator.generate_qr_code(text, filename)
                
                if qr_img is not None:
                    # Show preview
                    resized = cv2.resize(qr_img, (300, 300), interpolation=cv2.INTER_NEAREST)
                    
                    # Add text label
                    label_img = np.ones((50, 300, 3), dtype=np.uint8) * 255
                    cv2.putText(label_img, text[:40], (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Combine
                    combined = np.vstack([label_img, resized])
                    
                    cv2.imshow(f'QR Code: {text[:20]}...', combined)
                    
                    key = cv2.waitKey(2000) & 0xFF
                    if key == 27:  # ESC
                        cv2.destroyAllWindows()
                        return
                    
                    cv2.destroyAllWindows()
            
            break
        
        else:
            # Generate custom QR code
            filename = f"custom_qr_{int(time.time())}.png"
            qr_img = generator.generate_qr_code(user_input, filename)
            
            if qr_img is not None:
                # Show preview
                resized = cv2.resize(qr_img, (400, 400), interpolation=cv2.INTER_NEAREST)
                
                # Add text info
                info_img = np.ones((100, 400, 3), dtype=np.uint8) * 255
                cv2.putText(info_img, "Generated QR Code:", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(info_img, user_input[:50], (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(info_img, f"File: {filename}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
                
                # Combine
                combined = np.vstack([info_img, resized])
                
                cv2.imshow('Generated QR Code', combined)
                
                print(f"✅ QR kod oluşturuldu ve kaydedildi: {filename}")
                print("📱 Test etmek için telefon kameranızla tarayın!")
                print("ESC: Çıkış, herhangi bir tuş: Devam")
                
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                
                if key == 27:  # ESC
                    break
    
    print("👋 QR Generator kapatılıyor...")

def ornek_3_batch_processing():
    """
    Örnek 3: Toplu işleme (resim dosyalarından kod okuma)
    """
    print("\n🎯 Örnek 3: Batch Processing")
    print("=" * 30)
    
    detector = QRBarcodeDetector()
    
    # Test images directory
    test_dir = "test_codes"
    os.makedirs(test_dir, exist_ok=True)
    
    # Check for test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(test_dir, file))
    
    if not image_files:
        print(f"❌ {test_dir} klasöründe resim dosyası bulunamadı!")
        print("Test için QR/barkod içeren resimler ekleyin:")
        print("  - .jpg, .jpeg, .png, .bmp formatları desteklenir")
        return
    
    print(f"📁 {len(image_files)} resim dosyası bulundu")
    print("📊 Toplu işleme başlıyor...")
    
    results_summary = []
    
    for i, image_path in enumerate(image_files):
        print(f"\n🔍 İşleniyor: {os.path.basename(image_path)} ({i+1}/{len(image_files)})")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Resim yüklenemedi: {image_path}")
                continue
            
            # Detect codes
            start_time = time.time()
            results = detector.detect_all_codes(image)
            processing_time = (time.time() - start_time) * 1000
            
            # Process results
            file_result = {
                'filename': os.path.basename(image_path),
                'codes_found': len(results),
                'processing_time': processing_time,
                'codes': []
            }
            
            if results:
                print(f"✅ {len(results)} kod bulundu")
                
                # Display image with detections
                display_image = detector.draw_detections(image.copy(), results)
                
                # Add filename to image
                cv2.putText(display_image, os.path.basename(image_path), (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Resize for display
                height, width = display_image.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_image = cv2.resize(display_image, (new_width, new_height))
                
                cv2.imshow('Batch Processing', display_image)
                
                # Print results
                for j, result in enumerate(results):
                    print(f"   {j+1}. {result['type']}: {result['data']}")
                    
                    file_result['codes'].append({
                        'type': result['type'],
                        'data': result['data'],
                        'method': result['method']
                    })
                
                # Wait for key press
                key = cv2.waitKey(2000) & 0xFF
                if key == 27:  # ESC
                    break
                
            else:
                print("❌ Kod bulunamadı")
            
            results_summary.append(file_result)
            
        except Exception as e:
            print(f"❌ Hata: {e}")
    
    cv2.destroyAllWindows()
    
    # Summary report
    print("\n" + "="*50)
    print("📊 TOPLU İŞLEME RAPORU")
    print("="*50)
    
    total_files = len(results_summary)
    successful_files = sum(1 for r in results_summary if r['codes_found'] > 0)
    total_codes = sum(r['codes_found'] for r in results_summary)
    avg_time = sum(r['processing_time'] for r in results_summary) / total_files if total_files > 0 else 0
    
    print(f"📁 Toplam dosya: {total_files}")
    print(f"✅ Başarılı dosya: {successful_files} ({successful_files/total_files*100:.1f}%)")
    print(f"🔍 Toplam kod: {total_codes}")
    print(f"⏱️ Ortalama süre: {avg_time:.1f}ms")
    
    # Save detailed results
    batch_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_files': total_files,
            'successful_files': successful_files,
            'total_codes': total_codes,
            'average_processing_time': avg_time
        },
        'file_results': results_summary
    }
    
    results_file = "batch_results.json"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Detaylı sonuçlar kaydedildi: {results_file}")
    except Exception as e:
        print(f"❌ Kaydetme hatası: {e}")

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("📱 OpenCV QR ve Barkod Okuma Demo")
        print("="*50)
        print("1. 📱 Real-time QR & Barcode Scanner")
        print("2. 🏗️ QR Code Generator")
        print("3. 📊 Batch Processing (Resim dosyalarından)")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-3): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_realtime_qr_scanner()
            elif secim == "2":
                ornek_2_qr_generator()
            elif secim == "3":
                ornek_3_batch_processing()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-3 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("📱 OpenCV QR ve Barkod Okuma")
    print("Bu modül QR kod ve barkod okuma tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (real-time scanning için)")
    print("\n📦 İsteğe bağlı (gelişmiş özellikler için):")
    print("   - qrcode[pil] (pip install qrcode[pil]) - QR kod oluşturma")
    print("   - pyzbar (pip install pyzbar) - Gelişmiş barkod okuma")
    print("\n📝 Notlar:")
    print("   - OpenCV built-in QR detector kullanılır")
    print("   - İyi aydınlatma ve net görüntü önemli")
    print("   - Farklı açılardan deneyebilirsiniz")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. OpenCV QR detector varsayılan olarak mevcut
# 2. Pyzbar multiple barcode formatlarını destekler
# 3. QR kod oluşturma için qrcode kütüphanesi gerekli
# 4. JSON format scan results için ideal
# 5. Batch processing büyük veri setleri için kullanışlı