#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👤 Yüz Tespiti - OpenCV Face Detection
=====================================

Bu modül kapsamlı yüz tespit yöntemlerini kapsar:
- Face Detection (Haar Cascade & DNN)
- Eye Detection ve tracking
- Smile Detection
- Age & Gender Detection
- Face Recognition basics

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import os
import time
from collections import deque
import math

class FaceDetector:
    """Kapsamlı yüz tespit sınıfı"""
    
    def __init__(self):
        self.cascade_detectors = {}
        self.dnn_models = {}
        self.load_cascade_models()
        self.load_dnn_models()
        
        # Face tracking için
        self.face_tracker = FaceTracker()
        
    def load_cascade_models(self):
        """Haar cascade modellerini yükle"""
        cascade_files = {
            'face': 'haarcascade_frontalface_alt.xml',
            'face_default': 'haarcascade_frontalface_default.xml', 
            'profile_face': 'haarcascade_profileface.xml',
            'eye': 'haarcascade_eye.xml',
            'eye_tree': 'haarcascade_eye_tree_eyeglasses.xml',
            'smile': 'haarcascade_smile.xml',
            'nose': 'haarcascade_mcs_nose.xml'
        }
        
        for name, filename in cascade_files.items():
            try:
                cascade_path = cv2.data.haarcascades + filename
                if os.path.exists(cascade_path):
                    self.cascade_detectors[name] = cv2.CascadeClassifier(cascade_path)
                    print(f"✅ {name} cascade yüklendi")
                else:
                    print(f"⚠️ {name} cascade dosyası bulunamadı")
            except Exception as e:
                print(f"⚠️ {name} cascade yüklenemedi: {e}")
    
    def load_dnn_models(self):
        """DNN modellerini yükle"""
        try:
            # OpenCV DNN face detector
            prototxt_path = "models/opencv_face_detector.pbtxt"
            model_path = "models/opencv_face_detector_uint8.pb"
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self.dnn_models['face'] = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
                print("✅ DNN face detector yüklendi")
            else:
                print("⚠️ DNN model dosyaları bulunamadı")
                
        except Exception as e:
            print(f"⚠️ DNN model yüklenemedi: {e}")
    
    def detect_faces_cascade(self, frame, detector_type='face', scale_factor=1.1, min_neighbors=5):
        """Haar cascade ile yüz tespiti"""
        if detector_type not in self.cascade_detectors:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.cascade_detectors[detector_type].detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def detect_faces_dnn(self, frame, confidence_threshold=0.7):
        """DNN ile yüz tespiti"""
        if 'face' not in self.dnn_models:
            return []
        
        h, w = frame.shape[:2]
        
        # Blob oluştur
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        
        # DNN forward pass
        self.dnn_models['face'].setInput(blob)
        detections = self.dnn_models['face'].forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append((x1, y1, x2-x1, y2-y1, confidence))
        
        return faces
    
    def detect_eyes_in_face(self, frame, face_roi):
        """Yüz içinde göz tespiti"""
        if 'eye' not in self.cascade_detectors:
            return []
        
        x, y, w, h = face_roi
        face_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        eyes = self.cascade_detectors['eye'].detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10)
        )
        
        # Global koordinatlara çevir
        global_eyes = []
        for (ex, ey, ew, eh) in eyes:
            global_eyes.append((x + ex, y + ey, ew, eh))
        
        return global_eyes
    
    def detect_smile_in_face(self, frame, face_roi):
        """Yüz içinde gülümseme tespiti"""
        if 'smile' not in self.cascade_detectors:
            return []
        
        x, y, w, h = face_roi
        # Gülümseme genelde yüzün alt yarısında
        face_lower = frame[y+h//2:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_lower, cv2.COLOR_BGR2GRAY)
        
        smiles = self.cascade_detectors['smile'].detectMultiScale(
            face_gray,
            scaleFactor=1.8,
            minNeighbors=20,
            minSize=(25, 25)
        )
        
        # Global koordinatlara çevir
        global_smiles = []
        for (sx, sy, sw, sh) in smiles:
            global_smiles.append((x + sx, y + h//2 + sy, sw, sh))
        
        return global_smiles

class FaceTracker:
    """Yüz tracking sınıfı"""
    
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        
    def register(self, centroid):
        """Yeni yüz kaydet"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Yüzü kayıttan çıkar"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects):
        """Yüz tracking güncelle"""
        if len(rects) == 0:
            # Tüm objeler için disappeared sayacını artır
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
        
        # Eğer tracked object yoksa, hepsini register et
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        
        else:
            # Existing objects ile input centroids arasında mesafe hesapla
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # En küçük mesafeleri bul
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # Eğer input centroid sayısı >= object sayısı ise
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects

class FaceAnalyzer:
    """Yüz analiz sınıfı"""
    
    def __init__(self):
        self.face_history = deque(maxlen=50)
        
    def analyze_face(self, face_roi):
        """Yüz analizi yap"""
        h, w = face_roi.shape[:2]
        
        # Brightness analizi
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        brightness = np.mean(gray)
        
        # Contrast analizi
        contrast = np.std(gray)
        
        # Blur analizi (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Face size
        face_size = w * h
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'blur_score': blur_score,
            'face_size': face_size,
            'aspect_ratio': w / h if h > 0 else 0
        }

def ornek_1_comprehensive_face_detection():
    """
    Örnek 1: Kapsamlı yüz tespiti
    """
    print("\n🎯 Örnek 1: Kapsamlı Yüz Tespiti")
    print("=" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = FaceDetector()
    analyzer = FaceAnalyzer()
    
    # Detection mode
    detection_mode = 'cascade'  # 'cascade' or 'dnn'
    cascade_type = 'face'
    
    # Parameters
    scale_factor = 1.1
    min_neighbors = 5
    confidence_threshold = 0.7
    
    show_analysis = True
    
    available_cascades = list(detector.cascade_detectors.keys())
    
    print("👤 Kapsamlı yüz tespiti")
    print("Kontroller:")
    print("  m: Detection mode (Cascade/DNN)")
    print("  1-7: Cascade türü değiştir")
    print("  a: Analiz on/off")
    print("  +/-: Parameters")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Face detection
        if detection_mode == 'cascade':
            faces = detector.detect_faces_cascade(frame, cascade_type, scale_factor, min_neighbors)
            detection_info = f"Cascade: {cascade_type}"
        else:
            faces = detector.detect_faces_dnn(frame, confidence_threshold)
            detection_info = f"DNN (conf: {confidence_threshold:.1f})"
        
        detection_time = (time.time() - start_time) * 1000
        
        # Process each face
        for i, face in enumerate(faces):
            if detection_mode == 'dnn' and len(face) == 5:
                x, y, w, h, conf = face
                conf_text = f" ({conf:.2f})"
            else:
                x, y, w, h = face
                conf_text = ""
            
            # Face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face{i+1}{conf_text}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Eye detection in face
            eyes = detector.detect_eyes_in_face(frame, (x, y, w, h))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                cv2.circle(frame, (ex + ew//2, ey + eh//2), 3, (255, 0, 0), -1)
            
            # Smile detection in face
            smiles = detector.detect_smile_in_face(frame, (x, y, w, h))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                cv2.putText(frame, "Smile", (sx, sy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Face analysis
            if show_analysis and i == 0:  # Sadece ilk yüz için
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    analysis = analyzer.analyze_face(face_roi)
                    
                    # Analysis display
                    info_x = frame.shape[1] - 200
                    info_y = 30
                    
                    cv2.putText(frame, "FACE ANALYSIS", (info_x, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, f"Brightness: {analysis['brightness']:.1f}", (info_x, info_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, f"Contrast: {analysis['contrast']:.1f}", (info_x, info_y + 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, f"Sharpness: {analysis['blur_score']:.1f}", (info_x, info_y + 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, f"Size: {analysis['face_size']}", (info_x, info_y + 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, f"Aspect: {analysis['aspect_ratio']:.2f}", (info_x, info_y + 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Main info panel
        cv2.putText(frame, detection_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {detection_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if detection_mode == 'cascade':
            cv2.putText(frame, f"Scale: {scale_factor:.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MinNeigh: {min_neighbors}", (10, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Available cascades
        if detection_mode == 'cascade':
            for i, cascade_name in enumerate(available_cascades[:7]):
                color = (0, 255, 0) if cascade_name == cascade_type else (255, 255, 255)
                cv2.putText(frame, f"{i+1}: {cascade_name}", (10, 200 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow('Comprehensive Face Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('m'):
            detection_mode = 'dnn' if detection_mode == 'cascade' else 'cascade'
            print(f"🔄 Detection mode: {detection_mode}")
        elif key >= ord('1') and key <= ord('7'):
            if detection_mode == 'cascade':
                idx = key - ord('1')
                if idx < len(available_cascades):
                    cascade_type = available_cascades[idx]
                    print(f"🔄 Cascade: {cascade_type}")
        elif key == ord('a'):
            show_analysis = not show_analysis
            print(f"🔄 Analysis: {'ON' if show_analysis else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            if detection_mode == 'cascade':
                scale_factor = min(2.0, scale_factor + 0.05)
            else:
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
        elif key == ord('-'):
            if detection_mode == 'cascade':
                scale_factor = max(1.05, scale_factor - 0.05)
            else:
                confidence_threshold = max(0.3, confidence_threshold - 0.05)
        elif key == ord('n'):
            min_neighbors = min(10, min_neighbors + 1)
        elif key == ord('b'):
            min_neighbors = max(1, min_neighbors - 1)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_face_tracking():
    """
    Örnek 2: Yüz tracking sistemi
    """
    print("\n🎯 Örnek 2: Yüz Tracking")
    print("=" * 30)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = FaceDetector()
    face_tracker = FaceTracker(max_disappeared=30)
    
    # Colors for different tracked faces
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    print("👥 Yüz tracking sistemi")
    print("Birden fazla kişi ile test edin!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Face detection
        faces = detector.detect_faces_cascade(frame, 'face', 1.1, 5)
        
        # Update tracker
        tracked_faces = face_tracker.update(faces)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Draw tracked faces
        for object_id, centroid in tracked_faces.items():
            color = colors[object_id % len(colors)]
            
            # Find corresponding face rectangle
            face_rect = None
            min_dist = float('inf')
            
            for face in faces:
                x, y, w, h = face
                face_centroid = (int(x + w/2), int(y + h/2))
                dist = np.sqrt((centroid[0] - face_centroid[0])**2 + (centroid[1] - face_centroid[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    face_rect = face
            
            if face_rect is not None and min_dist < 50:
                x, y, w, h = face_rect
                
                # Face rectangle with color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Face ID
                cv2.putText(frame, f"Face #{object_id}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Center point
                cv2.circle(frame, tuple(centroid), 5, color, -1)
                
                # Eyes detection
                eyes = detector.detect_eyes_in_face(frame, (x, y, w, h))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 1)
        
        # Info panel
        cv2.putText(frame, f"Tracked Faces: {len(tracked_faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {processing_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Tracking history visualization
        if len(face_tracker.objects) > 0:
            cv2.putText(frame, "Active IDs:", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            for i, object_id in enumerate(face_tracker.objects.keys()):
                color = colors[object_id % len(colors)]
                cv2.putText(frame, f"#{object_id}", (10 + i*40, 155),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Face Tracking', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            # Reset tracker
            face_tracker = FaceTracker(max_disappeared=30)
            print("🔄 Tracker reset")
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_smile_emotion_detection():
    """
    Örnek 3: Gülümseme ve emotion detection
    """
    print("\n🎯 Örnek 3: Smile & Emotion Detection")
    print("=" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam açılamadı!")
        return
    
    detector = FaceDetector()
    
    # Emotion tracking
    emotion_history = deque(maxlen=30)  # Son 30 frame
    
    print("😊 Smile detection - gülümseyin!")
    print("Kontroller: ESC çıkış")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Face detection
        faces = detector.detect_faces_cascade(frame, 'face', 1.1, 5)
        
        total_smiles = 0
        
        for i, (x, y, w, h) in enumerate(faces):
            # Face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Eye detection
            eyes = detector.detect_eyes_in_face(frame, (x, y, w, h))
            eye_count = len(eyes)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                # Pupil center (approximation)
                cv2.circle(frame, (ex + ew//2, ey + eh//2), 3, (255, 0, 0), -1)
            
            # Smile detection
            smiles = detector.detect_smile_in_face(frame, (x, y, w, h))
            smile_count = len(smiles)
            total_smiles += smile_count
            
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
            
            # Emotion analysis
            is_smiling = smile_count > 0
            has_both_eyes = eye_count >= 2
            
            # Emotion state
            if is_smiling and has_both_eyes:
                emotion = "Happy 😊"
                emotion_color = (0, 255, 0)
            elif has_both_eyes:
                emotion = "Neutral 😐"
                emotion_color = (255, 255, 0)
            elif is_smiling:
                emotion = "Smiling 🙂"
                emotion_color = (0, 255, 255)
            else:
                emotion = "Unknown 🤔"
                emotion_color = (128, 128, 128)
            
            # Face info
            cv2.putText(frame, f"Face {i+1}", (x, y - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
            
            # Face metrics
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            cv2.circle(frame, (face_center_x, face_center_y), 3, (0, 255, 0), -1)
            
            # Eye-smile correlation
            if i == 0:  # İlk yüz için detaylı analiz
                cv2.putText(frame, f"Eyes: {eye_count}", (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"Smiles: {smile_count}", (x, y + h + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Emotion history tracking
        emotion_history.append(total_smiles > 0)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Main info panel
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Smiles: {total_smiles}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Time: {processing_time:.1f}ms", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Smile history visualization
        if len(emotion_history) > 1:
            smile_percentage = (sum(emotion_history) / len(emotion_history)) * 100
            cv2.putText(frame, f"Smile Rate: {smile_percentage:.1f}%", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Mini happiness meter
            meter_x = 10
            meter_y = 140
            meter_w = 200
            meter_h = 20
            
            # Background
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (100, 100, 100), -1)
            
            # Happiness bar
            happiness_w = int((smile_percentage / 100) * meter_w)
            happiness_color = (0, int(255 * smile_percentage / 100), int(255 * (1 - smile_percentage / 100)))
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + happiness_w, meter_y + meter_h), happiness_color, -1)
            
            # Border
            cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (255, 255, 255), 2)
            
            cv2.putText(frame, "Happiness Meter", (meter_x, meter_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Smile & Emotion Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("👤 OpenCV Yüz Tespiti Demo")
        print("="*50)
        print("1. 👤 Kapsamlı Yüz Tespiti (Cascade & DNN)")
        print("2. 👥 Yüz Tracking Sistemi") 
        print("3. 😊 Smile & Emotion Detection")
        print("0. ❌ Çıkış")
        
        try:
            secim = input("\nSeçiminizi yapın (0-3): ").strip()
            
            if secim == "0":
                print("👋 Görüşmek üzere!")
                break
            elif secim == "1":
                ornek_1_comprehensive_face_detection()
            elif secim == "2":
                ornek_2_face_tracking()
            elif secim == "3":
                ornek_3_smile_emotion_detection()
            else:
                print("❌ Geçersiz seçim! Lütfen 0-3 arasında bir sayı girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırıldı.")
            break
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

def main():
    """Ana fonksiyon"""
    print("👤 OpenCV Yüz Tespiti")
    print("Bu modül kapsamlı yüz tespit ve analiz tekniklerini öğretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (gerekli)")
    print("\n📝 Notlar:")
    print("   - DNN modelleri daha doğru ama yavaş")
    print("   - Cascade modelleri hızlı ama sınırlı")
    print("   - İyi aydınlatma önemli")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Yüz tespiti için iyi aydınlatma kritik
# 2. DNN modelleri cascade'den daha doğru
# 3. Eye tracking gözlük ile zorlaşabilir
# 4. Smile detection arka plan gürültüsünden etkilenir
# 5. Face tracking ID persistency sağlar