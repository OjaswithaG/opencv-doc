#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 DNN Nesne Tespiti - OpenCV Deep Neural Network Object Detection
===============================================================

Bu modul modern derin ogrenme tabanli nesne tespit yontemlerini kapsar:
- YOLO (You Only Look Once) v3, v4, v5
- SSD (Single Shot MultiBox Detector)
- MobileNet-SSD (Mobile optimized)
- Pre-trained COCO models
- Real-time inference

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
import urllib.request
from pathlib import Path

class DNNObjectDetector:
    """DNN tabanlı nesne tespit sınıfı"""
    
    def __init__(self):
        self.models = {}
        self.class_names = {}
        self.colors = {}
        self.model_configs = {
            'yolov3': {
                'config': 'models/yolov3.cfg',
                'weights': 'models/yolov3.weights',
                'classes': 'models/coco.names',
                'input_size': (416, 416),
                'scale': 1/255.0,
                'mean': [0, 0, 0],
                'swap_rb': True
            },
            'yolov4': {
                'config': 'models/yolov4.cfg',
                'weights': 'models/yolov4.weights',
                'classes': 'models/coco.names',
                'input_size': (608, 608),
                'scale': 1/255.0,
                'mean': [0, 0, 0],
                'swap_rb': True
            },
            'ssd_mobilenet': {
                'config': 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt',
                'weights': 'models/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb',
                'classes': 'models/coco.names',
                'input_size': (320, 320),
                'scale': 1.0,
                'mean': [127.5, 127.5, 127.5],
                'swap_rb': True
            }
        }
        
        # Backup: Haar Cascade (no download needed)
        self.cascade_classifiers = {
            'face': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'body': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml'),
            'eye': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        }
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        self.load_coco_classes()
        
    def load_coco_classes(self):
        """COCO class isimlerini yükle"""
        coco_classes = [
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
            'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        self.class_names['coco'] = coco_classes
        
        # Generate colors for each class
        np.random.seed(42)
        self.colors['coco'] = np.random.randint(0, 255, size=(len(coco_classes), 3), dtype="uint8")
    
    def download_model_if_needed(self, model_name):
        """Model dosyalarini indir (eger mevcut degilse)"""
        print(f"📥 {model_name} model dosyalari kontrol ediliyor...")
        
        urls = {
            'yolov3': {
                'weights': 'https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights',
                'config': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
                'fallback_weights': [
                    'https://drive.google.com/uc?export=download&id=1Lf6biCF6hGEOtx0dC_GzL6w5AyGgU6xW',
                    'https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt'
                ]
            },
            'yolov4': {
                'weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights',
                'config': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg'
            }
        }
        
        if model_name in urls:
            config_path = self.model_configs[model_name]['config']
            weights_path = self.model_configs[model_name]['weights']
            
            # Download config
            if not os.path.exists(config_path):
                print(f"⬇️ Indiriliyor: {config_path}")
                try:
                    urllib.request.urlretrieve(urls[model_name]['config'], config_path)
                    print(f"✅ Indirildi: {config_path}")
                except Exception as e:
                    print(f"❌ Indirme hatasi: {e}")
                    return False
            
            # Download weights (large file)
            if not os.path.exists(weights_path):
                print(f"⬇️ Indiriliyor (buyuk dosya): {weights_path}")
                print("⚠️ Bu islem zaman alabilir...")
                
                # Ana URL'yi dene
                success = False
                try:
                    urllib.request.urlretrieve(urls[model_name]['weights'], weights_path)
                    print(f"✅ Indirildi: {weights_path}")
                    success = True
                except Exception as e:
                    print(f"❌ Indirme hatasi (ana URL): {e}")
                    
                    # Fallback URL'leri dene
                    if 'fallback_weights' in urls[model_name]:
                        for i, fallback_url in enumerate(urls[model_name]['fallback_weights']):
                            print(f"🔄 Alternatif URL deneniyor ({i+1})...")
                            try:
                                urllib.request.urlretrieve(fallback_url, weights_path)
                                print(f"✅ Indirildi (alternatif URL): {weights_path}")
                                success = True
                                break
                            except Exception as e2:
                                print(f"❌ Alternatif URL hatasi: {e2}")
                
                if not success:
                    print("❌ Tum indirme denemeler basarisiz!")
                    print("💡 Cozum onerileri:")
                    print("   1. Internet baglantinizi kontrol edin")
                    print("   2. Model dosyalarini manuel indirin:")
                    print(f"      - {urls[model_name]['weights']}")
                    print(f"      - Dosyayi {weights_path} konumuna kaydedin")
                    print("   3. Baska bir model secin (yolov4 deneyin)")
                    print("   4. Basit nesne tespiti icin Haar Cascade kullanin")
                    print("      (04-Nesne-Tespiti/01-klasik-nesne-tespiti.py)")
                    return False
        
        return True
    
    def load_model(self, model_name):
        """Model yukle"""
        if model_name not in self.model_configs:
            print(f"❌ Desteklenmeyen model: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        
        # Check if files exist, download if needed
        if not self.download_model_if_needed(model_name):
            return False
        
        try:
            if model_name.startswith('yolo'):
                # YOLO models (Darknet)
                net = cv2.dnn.readNetFromDarknet(config['config'], config['weights'])
            elif 'ssd' in model_name or 'mobilenet' in model_name:
                # TensorFlow models
                net = cv2.dnn.readNetFromTensorflow(config['weights'], config['config'])
            else:
                print(f"❌ Bilinmeyen model turu: {model_name}")
                return False
            
            # Set backend and target
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.models[model_name] = net
            print(f"✅ Model yuklendi: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Model yukleme hatasi: {e}")
            return False
    
    def detect_objects(self, frame, model_name, confidence_threshold=0.5, nms_threshold=0.4):
        """Nesne tespiti yap"""
        if model_name not in self.models:
            return []
        
        net = self.models[model_name]
        config = self.model_configs[model_name]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=config['scale'],
            size=config['input_size'],
            mean=config['mean'],
            swapRB=config['swap_rb']
        )
        
        # Set input
        net.setInput(blob)
        
        # Forward pass
        layer_names = net.getLayerNames()
        
        if model_name.startswith('yolo'):
            # YOLO output layers
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(output_layers)
            return self._process_yolo_outputs(outputs, frame, confidence_threshold, nms_threshold)
        
        elif 'ssd' in model_name or 'mobilenet' in model_name:
            # SSD output
            outputs = net.forward()
            return self._process_ssd_outputs(outputs, frame, confidence_threshold)
        
        return []
    
    def _process_yolo_outputs(self, outputs, frame, confidence_threshold, nms_threshold):
        """YOLO ciktilarini isle"""
        height, width = frame.shape[:2]
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # YOLO format: center_x, center_y, width, height
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    box_width = int(detection[2] * width)
                    box_height = int(detection[3] * height)
                    
                    # Convert to corner format
                    x = int(center_x - box_width / 2)
                    y = int(center_y - box_height / 2)
                    
                    boxes.append([x, y, box_width, box_height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names['coco'][class_id] if class_id < len(self.class_names['coco']) else 'unknown'
                })
        
        return detections
    
    def _process_ssd_outputs(self, outputs, frame, confidence_threshold):
        """SSD ciktilarini isle"""
        height, width = frame.shape[:2]
        detections = []
        
        for i in range(outputs.shape[2]):
            confidence = outputs[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                class_id = int(outputs[0, 0, i, 1])
                
                # SSD format: normalized coordinates
                x1 = int(outputs[0, 0, i, 3] * width)
                y1 = int(outputs[0, 0, i, 4] * height)
                x2 = int(outputs[0, 0, i, 5] * width)
                y2 = int(outputs[0, 0, i, 6] * height)
                
                detections.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names['coco'][class_id] if class_id < len(self.class_names['coco']) else 'unknown'
                })
        
        return detections
    
    def detect_with_cascade(self, frame, cascade_type='face'):
        """Haar Cascade ile basit tespit (yedek yontem)"""
        if cascade_type not in self.cascade_classifiers:
            print(f"❌ Desteklenmeyen cascade tipi: {cascade_type}")
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = self.cascade_classifiers[cascade_type]
        
        # Detect objects
        objects = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in objects:
            detections.append({
                'bbox': (x, y, w, h),
                'confidence': 0.8,  # Haar Cascade doesn't provide confidence
                'class_id': 0,
                'class_name': cascade_type
            })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Tespitleri ciz"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Color
            if class_id < len(self.colors['coco']):
                color = tuple(map(int, self.colors['coco'][class_id]))
            else:
                color = (255, 255, 255)
            
            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

def ornek_1_yolo_detection():
    """
    Örnek 1: YOLO ile nesne tespiti
    """
    print("\n🎯 Örnek 1: YOLO Object Detection")
    print("=" * 35)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = DNNObjectDetector()
    
    # Model selection
    available_models = ['yolov3', 'yolov4']
    current_model = 'yolov3'
    
    # Load initial model
    print(f"🔄 {current_model} yükleniyor...")
    if not detector.load_model(current_model):
        print("❌ Model yüklenemedi!")
        return
    
    # Parameters
    confidence_threshold = 0.5
    nms_threshold = 0.4
    
    print("🧠 YOLO Object Detection")
    print("Kontroller:")
    print("  1-2: Model değiştir (YOLOv3, YOLOv4)")
    print("  +/-: Confidence threshold")
    print("  n/b: NMS threshold")
    print("  ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Object detection
        detections = detector.detect_objects(frame, current_model, confidence_threshold, nms_threshold)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Draw detections
        display_frame = detector.draw_detections(frame.copy(), detections)
        
        # Info panel
        cv2.putText(display_frame, f"Model: {current_model.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Objects: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Inference: {inference_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Confidence: {confidence_threshold:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"NMS: {nms_threshold:.2f}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS calculation
        fps = 1000 / inference_time if inference_time > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Class statistics
        class_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Show top classes
        y_offset = 220
        cv2.putText(display_frame, "DETECTED CLASSES:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        for i, (class_name, count) in enumerate(list(class_counts.items())[:5]):
            y_offset += 25
            cv2.putText(display_frame, f"{class_name}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('YOLO Object Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('1'):
            new_model = 'yolov3'
            if new_model != current_model:
                print(f"🔄 Loading {new_model}...")
                if detector.load_model(new_model):
                    current_model = new_model
        elif key == ord('2'):
            new_model = 'yolov4'
            if new_model != current_model:
                print(f"🔄 Loading {new_model}...")
                if detector.load_model(new_model):
                    current_model = new_model
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
        elif key == ord('-'):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
        elif key == ord('n'):
            nms_threshold = min(0.9, nms_threshold + 0.05)
        elif key == ord('b'):
            nms_threshold = max(0.1, nms_threshold - 0.05)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_multi_model_comparison():
    """
    Örnek 2: Çoklu model karşılaştırması
    """
    print("\n🎯 Örnek 2: Multi-Model Comparison")
    print("=" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = DNNObjectDetector()
    
    # Load multiple models
    models_to_load = ['yolov3']  # Start with one model
    loaded_models = []
    
    for model_name in models_to_load:
        print(f"🔄 Loading {model_name}...")
        if detector.load_model(model_name):
            loaded_models.append(model_name)
    
    if not loaded_models:
        print("❌ Hiç model yüklenemedi!")
        return
    
    print("⚡ Multi-model comparison")
    print("Farklı modellerin performansını karşılaştırın")
    
    # Comparison data
    model_stats = {model: {'total_time': 0, 'frame_count': 0, 'avg_detections': 0} 
                   for model in loaded_models}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Split screen for multiple models
        if len(loaded_models) == 1:
            # Single model - full screen
            model = loaded_models[0]
            
            start_time = time.time()
            detections = detector.detect_objects(frame, model, 0.5, 0.4)
            inference_time = (time.time() - start_time) * 1000
            
            display_frame = detector.draw_detections(frame.copy(), detections)
            
            # Update stats
            model_stats[model]['total_time'] += inference_time
            model_stats[model]['frame_count'] += 1
            model_stats[model]['avg_detections'] = ((model_stats[model]['avg_detections'] * 
                                                    (model_stats[model]['frame_count'] - 1) + 
                                                    len(detections)) / model_stats[model]['frame_count'])
            
            # Info
            cv2.putText(display_frame, f"Model: {model.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Time: {inference_time:.1f}ms", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Objects: {len(detections)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Average stats
            if model_stats[model]['frame_count'] > 0:
                avg_time = model_stats[model]['total_time'] / model_stats[model]['frame_count']
                cv2.putText(display_frame, f"Avg Time: {avg_time:.1f}ms", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Avg Objects: {model_stats[model]['avg_detections']:.1f}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Frames: {model_stats[model]['frame_count']}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        else:
            # Multiple models - split screen (future enhancement)
            display_frame = frame.copy()
            cv2.putText(display_frame, "Multi-model comparison coming soon!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Multi-Model Comparison', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n📊 Model Performance Comparison:")
    print("-" * 40)
    for model, stats in model_stats.items():
        if stats['frame_count'] > 0:
            avg_time = stats['total_time'] / stats['frame_count']
            avg_fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"{model.upper():10}: {avg_time:6.1f}ms avg, {avg_fps:5.1f} FPS, {stats['avg_detections']:.1f} objects avg")

def ornek_3_object_filtering():
    """
    Örnek 3: Nesne filtreleme ve sınıflandırma
    """
    print("\n🎯 Örnek 3: Object Filtering")
    print("=" * 30)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = DNNObjectDetector()
    
    # Load model
    model_name = 'yolov3'
    if not detector.load_model(model_name):
        print("❌ Model yüklenemedi!")
        return
    
    # Filter settings
    target_classes = {'person', 'car', 'bicycle', 'motorbike', 'cat', 'dog'}
    confidence_threshold = 0.5
    show_all = False
    
    print("🔍 Object filtering demo")
    print("a: Show all/filtered toggle")
    print("ESC: Çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Detect all objects
        all_detections = detector.detect_objects(frame, model_name, confidence_threshold, 0.4)
        
        # Filter detections
        if show_all:
            filtered_detections = all_detections
        else:
            filtered_detections = [d for d in all_detections if d['class_name'] in target_classes]
        
        inference_time = (time.time() - start_time) * 1000
        
        # Draw detections
        display_frame = detector.draw_detections(frame.copy(), filtered_detections)
        
        # Info panel
        mode_text = "ALL OBJECTS" if show_all else "FILTERED"
        cv2.putText(display_frame, f"Mode: {mode_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Total Objects: {len(all_detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Shown Objects: {len(filtered_detections)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Time: {inference_time:.1f}ms", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Target classes info
        if not show_all:
            cv2.putText(display_frame, "TARGET CLASSES:", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            y_offset = 185
            for class_name in list(target_classes)[:6]:
                cv2.putText(display_frame, f"• {class_name}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
        
        # Class distribution
        class_counts = {}
        for detection in filtered_detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            stats_x = display_frame.shape[1] - 200
            cv2.putText(display_frame, "DETECTED:", (stats_x, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            y_offset = 55
            for class_name, count in class_counts.items():
                cv2.putText(display_frame, f"{class_name}: {count}", (stats_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
        
        cv2.imshow('Object Filtering', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('a'):
            show_all = not show_all
            mode = "ALL" if show_all else "FILTERED"
            print(f"🔄 Mode: {mode}")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_4_haar_cascade_fallback():
    """
    Ornek 4: Haar Cascade ile Basit Nesne Tespiti (Yedek Yontem)
    YOLO indirme sorunlari yasandiginda alternatif
    """
    print("\n🔄 Haar Cascade Nesne Tespiti (Yedek Yontem)")
    print("=" * 45)
    print("Bu yontem YOLO indirilemediginde kullanilir")
    print("Sadece yuz, vucut ve goz tespiti yapar")
    print("ESC: Cikis, F: Yuz, B: Vucut, E: Goz")
    
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("❌ Webcam acilamadi!")
        return
    
    detector = DNNObjectDetector() 
    current_type = 'face'
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        # Haar Cascade ile tespit
        detections = detector.detect_with_cascade(frame, current_type)
        
        # Tespitleri ciz
        detector.draw_detections(frame, detections)
        
        # UI
        cv2.putText(frame, f"Tespit Tipi: {current_type.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Tespit Sayisi: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "F:Yuz B:Vucut E:Goz ESC:Cikis", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Haar Cascade Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('f') or key == ord('F'):
            current_type = 'face'
            print("🔄 Yuz tespiti aktif")
        elif key == ord('b') or key == ord('B'):
            current_type = 'body'
            print("🔄 Vucut tespiti aktif")
        elif key == ord('e') or key == ord('E'):
            current_type = 'eye'
            print("🔄 Goz tespiti aktif")
    
    cap.release()
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menusu"""
    while True:
        print("\n" + "="*50)
        print("🧠 OpenCV DNN Nesne Tespiti Demo")
        print("="*50)
        print("1. 🎯 YOLO Object Detection")
        print("2. ⚡ Multi-Model Comparison")  
        print("3. 🔍 Object Filtering & Classification")
        print("4. 🔄 Haar Cascade (Yedek Yontem)")
        print("0. ❌ Cikis")
        
        try:
            secim = input("\nSeciminizi yapin (0-4): ").strip()
            
            if secim == "0":
                print("👋 Gorusmek uzere!")
                break
            elif secim == "1":
                ornek_1_yolo_detection()
            elif secim == "2":
                ornek_2_multi_model_comparison()
            elif secim == "3":
                ornek_3_object_filtering()
            elif secim == "4":
                ornek_4_haar_cascade_fallback()
            else:
                print("❌ Gecersiz secim! Lutfen 0-4 arasinda bir sayi girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandirildi.")
            break
        except Exception as e:
            print(f"❌ Hata olustu: {e}")

def main():
    """Ana fonksiyon"""
    print("🧠 OpenCV DNN Nesne Tespiti")
    print("Bu modul modern deep learning nesne tespit tekniklerini ogretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Internet baglantisi (model indirme icin)")
    print("   - Webcam (onerilen)")
    print("\n⚠️ Notlar:")
    print("   - Ilk calistirmada model dosyalari indirilir (buyuk dosyalar)")
    print("   - GPU destegi icin opencv-python-gpu gerekebilir")
    print("   - YOLO modelleri CPU'da yavas calisabilir")
    print("   - Model indirme basarisiz olursa Haar Cascade alternatifi mevcut")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. YOLO modelleri darknet formatında
# 2. SSD modelleri TensorFlow formatında  
# 3. Model indirme internet gerektirir
# 4. GPU backend cok daha hizli
# 5. NMS threshold overlapping detection'lari filtreler