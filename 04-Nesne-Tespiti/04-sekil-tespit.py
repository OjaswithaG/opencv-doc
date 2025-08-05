#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📐 Şekil Tespiti - OpenCV Shape Detection
========================================

Bu modül geometrik şekil tespit yöntemlerini kapsar:
- Circle Detection (Hough Transform)
- Rectangle & Square Detection
- Triangle Detection
- Polygon Detection ve Classification
- Contour-based Shape Analysis

Yazan: Eren Terzi
Tarih: 2024
"""

import cv2
import numpy as np
import math
import time
from collections import defaultdict

class ShapeDetector:
    """Kapsamlı şekil tespit sınıfı"""
    
    def __init__(self):
        self.shape_stats = defaultdict(int)
        self.min_area = 500
        self.approximation_epsilon = 0.02
        
    def detect_circles_hough(self, frame, param1=50, param2=30, min_radius=10, max_radius=100):
        """Hough Transform ile daire tespiti"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gürültü azaltma
        blurred = cv2.medianBlur(gray, 5)
        
        # HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        return []
    
    def detect_shapes_contour(self, frame):
        """Contour tabanlı şekil tespiti"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold (adaptive)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Contour detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            # Contour approximation
            epsilon = self.approximation_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            
            # Shape classification
            shape_info = self.classify_shape(approx, contour)
            shape_info.update({
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w//2, y + h//2)
            })
            
            shapes.append(shape_info)
        
        return shapes, thresh
    
    def classify_shape(self, approx, contour):
        """Şekil sınıflandırması"""
        vertices = len(approx)
        
        # Temel geometrik özellikler
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h if h > 0 else 0
        
        # Extent (contour area / bounding rect area)
        extent = area / (w * h) if (w * h) > 0 else 0
        
        # Solidity (contour area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Circularity
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Shape classification
        shape_name = "unknown"
        confidence = 0.0
        
        if vertices == 3:
            shape_name = "triangle"
            confidence = 0.9
            
            # Triangle type analysis
            triangle_type = self.analyze_triangle(approx)
            shape_name = f"{triangle_type}_triangle"
            
        elif vertices == 4:
            # Rectangle/Square analysis
            if 0.85 <= aspect_ratio <= 1.15:
                shape_name = "square"
                confidence = 0.95
            else:
                shape_name = "rectangle"
                confidence = 0.9
                
        elif vertices == 5:
            shape_name = "pentagon"
            confidence = 0.8
            
        elif vertices == 6:
            shape_name = "hexagon"
            confidence = 0.8
            
        elif vertices > 6:
            # Circle/Ellipse detection
            if circularity > 0.7:
                if 0.9 <= aspect_ratio <= 1.1:
                    shape_name = "circle"
                    confidence = 0.85
                else:
                    shape_name = "ellipse"
                    confidence = 0.8
            else:
                shape_name = f"polygon_{vertices}"
                confidence = 0.6
        
        # Low vertex count but high circularity -> probably circle
        elif vertices <= 6 and circularity > 0.8:
            shape_name = "circle"
            confidence = circularity
        
        return {
            'shape': shape_name,
            'vertices': vertices,
            'confidence': confidence,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity,
            'circularity': circularity,
            'area': area,
            'perimeter': perimeter
        }
    
    def analyze_triangle(self, triangle_points):
        """Üçgen analizi"""
        if len(triangle_points) != 3:
            return "unknown"
        
        # Kenar uzunlukları
        points = triangle_points.reshape(3, 2)
        sides = []
        
        for i in range(3):
            p1 = points[i]
            p2 = points[(i+1) % 3]
            side_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(side_length)
        
        sides.sort()
        a, b, c = sides
        
        # Triangle type classification
        if abs(a - b) < 5 and abs(b - c) < 5:
            return "equilateral"  # Eşkenar
        elif abs(a - b) < 5 or abs(b - c) < 5 or abs(a - c) < 5:
            return "isosceles"    # İkizkenar
        else:
            # Right triangle check (Pisagor)
            if abs(a*a + b*b - c*c) < 10:
                return "right"    # Dik
            else:
                return "scalene"  # Çeşitkenar
    
    def detect_specific_shapes(self, frame, target_shape="all"):
        """Belirli şekil türlerini tespit et"""
        shapes, thresh = self.detect_shapes_contour(frame)
        
        if target_shape == "all":
            return shapes, thresh
        
        # Filter by target shape
        filtered_shapes = []
        for shape in shapes:
            if target_shape in shape['shape'].lower():
                filtered_shapes.append(shape)
        
        return filtered_shapes, thresh

class CircleDetector:
    """Özelleşmiş daire tespit sınıfı"""
    
    def __init__(self):
        self.circles_history = []
        
    def detect_circles_advanced(self, frame):
        """Gelişmiş daire tespiti"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple methods for circle detection
        results = []
        
        # Method 1: HoughCircles
        circles_hough = self.hough_circles(gray)
        for circle in circles_hough:
            circle['method'] = 'hough'
            results.append(circle)
        
        # Method 2: Contour-based
        circles_contour = self.contour_circles(gray)
        for circle in circles_contour:
            circle['method'] = 'contour'
            results.append(circle)
        
        # Merge similar circles
        merged_circles = self.merge_similar_circles(results)
        
        return merged_circles
    
    def hough_circles(self, gray):
        """Hough transform ile daire tespiti"""
        blurred = cv2.medianBlur(gray, 5)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=200
        )
        
        result = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                result.append({
                    'center': (x, y),
                    'radius': r,
                    'confidence': 0.8,
                    'area': math.pi * r * r
                })
        
        return result
    
    def contour_circles(self, gray):
        """Contour tabanlı daire tespiti"""
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 500:
                continue
            
            # Circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity > 0.7:  # Circular threshold
                # Minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                circles.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'confidence': circularity,
                    'area': area
                })
        
        return circles
    
    def merge_similar_circles(self, circles, distance_threshold=20):
        """Benzer daireleri birleştir"""
        if len(circles) <= 1:
            return circles
        
        merged = []
        used = set()
        
        for i, circle1 in enumerate(circles):
            if i in used:
                continue
                
            similar_circles = [circle1]
            
            for j, circle2 in enumerate(circles[i+1:], i+1):
                if j in used:
                    continue
                
                # Distance between centers
                dx = circle1['center'][0] - circle2['center'][0]
                dy = circle1['center'][1] - circle2['center'][1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < distance_threshold:
                    similar_circles.append(circle2)
                    used.add(j)
            
            # Merge similar circles (average)
            if len(similar_circles) == 1:
                merged.append(similar_circles[0])
            else:
                # Calculate average
                avg_x = sum(c['center'][0] for c in similar_circles) / len(similar_circles)
                avg_y = sum(c['center'][1] for c in similar_circles) / len(similar_circles)
                avg_r = sum(c['radius'] for c in similar_circles) / len(similar_circles)
                avg_conf = sum(c['confidence'] for c in similar_circles) / len(similar_circles)
                
                merged.append({
                    'center': (int(avg_x), int(avg_y)),
                    'radius': int(avg_r),
                    'confidence': avg_conf,
                    'area': math.pi * avg_r * avg_r,
                    'method': 'merged'
                })
            
            used.add(i)
        
        return merged

def ornek_1_comprehensive_shape_detection():
    """
    Örnek 1: Kapsamlı şekil tespiti
    """
    print("\n🎯 Örnek 1: Kapsamlı Şekil Tespiti")
    print("=" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam acilamadi!")
        return
    
    detector = ShapeDetector()
    
    # Parameters
    target_shape = "all"
    show_details = True
    
    shape_types = ["all", "circle", "triangle", "rectangle", "square", "polygon"]
    current_shape_idx = 0
    
    print("📐 Şekil tespiti - farklı geometrik şekiller gösterin!")
    print("Kontroller:")
    print("  1-6: Şekil filtresi")
    print("  d: Detay gösterimi on/off")
    print("  +/-: Min area")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Shape detection
        shapes, thresh = detector.detect_specific_shapes(frame, target_shape)
        
        detection_time = (time.time() - start_time) * 1000
        
        # Draw shapes
        shape_counts = defaultdict(int)
        
        for i, shape in enumerate(shapes):
            contour = shape['contour']
            bbox = shape['bbox']
            shape_name = shape['shape']
            confidence = shape['confidence']
            center = shape['center']
            
            shape_counts[shape_name] += 1
            
            # Shape color based on type
            if 'circle' in shape_name:
                color = (255, 0, 0)  # Red
            elif 'triangle' in shape_name:
                color = (0, 255, 0)  # Green
            elif 'square' in shape_name or 'rectangle' in shape_name:
                color = (0, 0, 255)  # Blue
            elif 'polygon' in shape_name:
                color = (255, 255, 0)  # Yellow
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw contour
            cv2.drawContours(frame, [contour], -1, color, 2)
            
            # Bounding box
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            
            # Center point
            cv2.circle(frame, center, 5, color, -1)
            
            # Label
            label = f"{shape_name}"
            if show_details:
                label += f" ({confidence:.2f})"
            
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Shape ID
            cv2.putText(frame, f"#{i+1}", (center[0]-10, center[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Detailed info for first few shapes
            if show_details and i < 3:
                info_x = 10
                info_y = 150 + i * 100
                
                cv2.putText(frame, f"Shape {i+1}:", (info_x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Type: {shape_name}", (info_x, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Vertices: {shape['vertices']}", (info_x, info_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Area: {int(shape['area'])}", (info_x, info_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (info_x, info_y + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Circ: {shape['circularity']:.2f}", (info_x, info_y + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Main info panel
        cv2.putText(frame, f"Filter: {target_shape}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Shapes: {len(shapes)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Min Area: {detector.min_area}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {detection_time:.1f}ms", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Shape statistics
        stats_x = frame.shape[1] - 200
        stats_y = 30
        cv2.putText(frame, "SHAPE STATS:", (stats_x, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        y_offset = stats_y + 25
        for shape_type, count in shape_counts.items():
            cv2.putText(frame, f"{shape_type}: {count}", (stats_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        # Threshold preview (küçük pencere)
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh_small = cv2.resize(thresh_colored, (160, 120))
        
        frame[frame.shape[0]-130:frame.shape[0]-10, frame.shape[1]-170:frame.shape[1]-10] = thresh_small
        cv2.rectangle(frame, (frame.shape[1]-170, frame.shape[0]-130), 
                     (frame.shape[1]-10, frame.shape[0]-10), (255, 255, 255), 2)
        cv2.putText(frame, "Threshold", (frame.shape[1]-165, frame.shape[0]-135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow('Comprehensive Shape Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key >= ord('1') and key <= ord('6'):
            idx = key - ord('1')
            if idx < len(shape_types):
                target_shape = shape_types[idx]
                current_shape_idx = idx
                print(f"🔄 Shape filter: {target_shape}")
        elif key == ord('d'):
            show_details = not show_details
            print(f"🔄 Details: {'ON' if show_details else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            detector.min_area = min(2000, detector.min_area + 100)
        elif key == ord('-'):
            detector.min_area = max(100, detector.min_area - 100)
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_2_advanced_circle_detection():
    """
    Örnek 2: Gelişmiş daire tespiti
    """
    print("\n🎯 Örnek 2: Gelişmiş Daire Tespiti")
    print("=" * 40)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam acilamadi!")
        return
    
    circle_detector = CircleDetector()
    
    print("⭕ Gelismis daire tespiti - yuvarlak objeler gosterin!")
    print("ESC: Cikis")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Advanced circle detection
        circles = circle_detector.detect_circles_advanced(frame)
        
        detection_time = (time.time() - start_time) * 1000
        
        # Draw circles
        for i, circle in enumerate(circles):
            center = circle['center']
            radius = circle['radius']
            confidence = circle['confidence']
            method = circle['method']
            
            # Color based on method
            if method == 'hough':
                color = (255, 0, 0)  # Red
            elif method == 'contour':
                color = (0, 255, 0)  # Green
            else:  # merged
                color = (0, 255, 255)  # Yellow
            
            # Circle
            cv2.circle(frame, center, radius, color, 2)
            cv2.circle(frame, center, 2, color, -1)
            
            # Label
            label = f"C{i+1} ({method})"
            cv2.putText(frame, label, (center[0] - 30, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Confidence
            cv2.putText(frame, f"{confidence:.2f}", (center[0] - 15, center[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Circle info
            if i < 3:  # İlk 3 daire için detay
                info_x = 10
                info_y = 150 + i * 80
                
                cv2.putText(frame, f"Circle {i+1}:", (info_x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Method: {method}", (info_x, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Radius: {radius}", (info_x, info_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Area: {int(circle['area'])}", (info_x, info_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (info_x, info_y + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Main info
        cv2.putText(frame, f"Circles: {len(circles)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {detection_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Method legend
        legend_x = frame.shape[1] - 150
        cv2.putText(frame, "METHODS:", (legend_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, "Hough", (legend_x, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(frame, "Contour", (legend_x, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "Merged", (legend_x, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.imshow('Advanced Circle Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

def ornek_3_shape_matching_game():
    """
    Örnek 3: Şekil eşleştirme oyunu
    """
    print("\n🎯 Örnek 3: Şekil Eşleştirme Oyunu")
    print("=" * 35)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam acilamadi!")
        return
    
    detector = ShapeDetector()
    
    # Game state
    target_shapes = ["circle", "triangle", "square", "rectangle"]
    current_target = 0
    score = 0
    target_found = False
    target_timer = 0
    game_timer = time.time()
    
    print("🎮 Sekil Eslestirme Oyunu!")
    print("Hedef sekli gostererek puan kazanin!")
    print("ESC: Cikis, R: Reset")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # Shape detection
        shapes, _ = detector.detect_shapes_contour(frame)
        
        # Game logic
        target_shape = target_shapes[current_target]
        
        # Check for target shape
        found_target = False
        target_shape_info = None
        
        for shape in shapes:
            if target_shape in shape['shape'].lower():
                found_target = True
                target_shape_info = shape
                break
        
        # Game state update
        if found_target and not target_found:
            target_found = True
            target_timer = current_time
            score += 10
            print(f"✅ {target_shape.title()} found! Score: {score}")
        
        elif target_found and (current_time - target_timer > 2.0):
            # Move to next target
            current_target = (current_target + 1) % len(target_shapes)
            target_found = False
            print(f"🎯 Next target: {target_shapes[current_target].title()}")
        
        # Draw all shapes
        for shape in shapes:
            contour = shape['contour']
            center = shape['center']
            shape_name = shape['shape']
            
            # Color: green if target, blue otherwise
            if target_shape in shape_name.lower():
                color = (0, 255, 0)  # Green - target found
                thickness = 3
            else:
                color = (255, 0, 0)  # Blue - other shapes
                thickness = 2
            
            cv2.drawContours(frame, [contour], -1, color, thickness)
            cv2.circle(frame, center, 5, color, -1)
            
            # Shape label
            cv2.putText(frame, shape_name, (center[0] - 30, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Game UI
        game_duration = current_time - game_timer
        
        # Target display
        cv2.putText(frame, f"TARGET: {target_shape.upper()}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        # Score and time
        cv2.putText(frame, f"Score: {score}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {game_duration:.1f}s", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Target status
        if target_found:
            status_text = "TARGET FOUND! ✅"
            status_color = (0, 255, 0)
            
            # Countdown
            countdown = 2.0 - (current_time - target_timer)
            cv2.putText(frame, f"Next in: {countdown:.1f}s", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            status_text = "SEARCHING... 🔍"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Target shape visualization (corner)
        target_viz_x = frame.shape[1] - 120
        target_viz_y = 50
        
        cv2.putText(frame, "TARGET:", (target_viz_x, target_viz_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw target shape representation
        if target_shape == "circle":
            cv2.circle(frame, (target_viz_x + 30, target_viz_y + 20), 20, (0, 255, 255), 2)
        elif target_shape == "triangle":
            pts = np.array([[target_viz_x + 30, target_viz_y], 
                           [target_viz_x + 10, target_viz_y + 40],
                           [target_viz_x + 50, target_viz_y + 40]], np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        elif target_shape == "square":
            cv2.rectangle(frame, (target_viz_x + 10, target_viz_y + 5), 
                         (target_viz_x + 50, target_viz_y + 45), (0, 255, 255), 2)
        elif target_shape == "rectangle":
            cv2.rectangle(frame, (target_viz_x + 5, target_viz_y + 15), 
                         (target_viz_x + 55, target_viz_y + 35), (0, 255, 255), 2)
        
        cv2.imshow('Shape Matching Game', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            # Reset game
            score = 0
            current_target = 0
            target_found = False
            game_timer = time.time()
            print("🔄 Game reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final score
    print(f"\n🏆 Final Score: {score}")
    print(f"⏱️ Total Time: {game_duration:.1f} seconds")
    if score > 0:
        print(f"📊 Points per second: {score/game_duration:.2f}")

def demo_menu():
    """Demo menüsü"""
    while True:
        print("\n" + "="*50)
        print("📐 OpenCV Sekil Tespiti Demo")
        print("="*50)
        print("1. 📐 Kapsamli Sekil Tespiti")
        print("2. ⭕ Gelismis Daire Tespiti")
        print("3. 🎮 Sekil Eslestirme Oyunu")
        print("0. ❌ Cikis")
        
        try:
            secim = input("\nSeciminizi yapin (0-3): ").strip()
            
            if secim == "0":
                print("👋 Gorusmek uzere!")
                break
            elif secim == "1":
                ornek_1_comprehensive_shape_detection()
            elif secim == "2":
                ornek_2_advanced_circle_detection()
            elif secim == "3":
                ornek_3_shape_matching_game()
            else:
                print("❌ Gecersiz secim! Lutfen 0-3 arasinda bir sayi girin.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandirildi.")
            break
        except Exception as e:
            print(f"❌ Hata olustu: {e}")

def main():
    """Ana fonksiyon"""
    print("📐 OpenCV Sekil Tespiti")
    print("Bu modul geometrik sekil tespit tekniklerini ogretir.")
    print("\n💡 Gereksinimler:")
    print("   - OpenCV (pip install opencv-python)")
    print("   - NumPy (pip install numpy)")
    print("   - Webcam (onerilen)")
    print("\n📝 Notlar:")
    print("   - Iyi kontrast onemli")
    print("   - Duz arka plan daha iyi sonuc verir")
    print("   - Sekillerin net gorunmesi gerekli")
    
    demo_menu()

if __name__ == "__main__":
    main()

# 📝 NOTLAR:
# 1. Contour approximation epsilon değeri hassaslığı belirler
# 2. Hough Transform daireler için çok etkili
# 3. Adaptive threshold değişken aydınlatmada iyi
# 4. Circularity metric daire tespiti için kritik
# 5. Shape classification confidence skorları önemli