# 📦 Alıştırma 3: Envanter Yönetim Sistemi

**Seviye:** ⭐⭐⭐⭐⭐ (Uzman)  
**Süre:** 4-5 saat  
**Hedef:** QR kod tabanlı kapsamlı envanter takip sistemi geliştirme

## 🎯 Alıştırma Hedefi

Bu alıştırmada **endüstriyel seviyede bir envanter yönetim sistemi** oluşturacaksınız. Sistem QR kod/barkod okuma, veritabanı entegrasyonu, stok takibi ve raporlama özellikleri içerecek.

## 🛠️ Teknik Gereksinimler

### Kullanılacak Teknolojiler
- **QR/Barcode Detection** (OpenCV + Pyzbar)
- **Database Management** (SQLite/PostgreSQL)
- **Inventory Management** (Stock tracking, transactions)
- **Report Generation** (PDF/Excel export)
- **GUI Development** (Tkinter/PyQt for management interface)

### Gerekli Kütüphaneler
```python
import cv2
import numpy as np
import sqlite3
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import qrcode
from reportlab.pdfgen import canvas
import pandas as pd
```

## 📋 Sistem Özellikleri

### 🔧 Temel Özellikler (Zorunlu)

#### 1️⃣ QR/Barcode Management
- [ ] **Code scanning** - QR kod ve barkod okuma
- [ ] **Code generation** - ürünler için QR kod oluşturma
- [ ] **Batch scanning** - toplu kod okuma
- [ ] **Code validation** - kod formatı doğrulama

#### 2️⃣ Product Management
- [ ] **Product registration** - yeni ürün ekleme
- [ ] **Product information** - detaylı ürün bilgileri
- [ ] **Category management** - ürün kategorileri
- [ ] **Supplier tracking** - tedarikçi bilgileri

#### 3️⃣ Inventory Operations
- [ ] **Stock in/out** - giriş/çıkış işlemleri
- [ ] **Stock level tracking** - stok seviyesi takibi
- [ ] **Low stock alerts** - düşük stok uyarıları
- [ ] **Inventory audit** - stok sayım işlemleri

#### 4️⃣ Database Integration
- [ ] **SQLite database** - local veritabanı
- [ ] **Transaction logging** - işlem kayıtları
- [ ] **Data backup/restore** - yedekleme sistemi
- [ ] **Data validation** - veri doğrulama

### 🌟 Gelişmiş Özellikler (Bonus)

#### 5️⃣ Advanced Analytics
- [ ] **Inventory reports** - detaylı raporlar
- [ ] **Trend analysis** - stok trend analizi
- [ ] **ABC analysis** - ürün önem sınıflandırması
- [ ] **Turnover analysis** - devir hızı analizi

#### 6️⃣ System Integration
- [ ] **Excel import/export** - Excel entegrasyonu
- [ ] **PDF report generation** - PDF rapor oluşturma
- [ ] **Email notifications** - otomatik bildirimler
- [ ] **Multi-user support** - çoklu kullanıcı desteği

#### 7️⃣ Mobile Integration
- [ ] **Mobile scanning app** - mobil tarama uygulaması
- [ ] **Offline mode** - internet bağlantısı olmadan çalışma
- [ ] **Cloud synchronization** - bulut senkronizasyonu
- [ ] **REST API** - harici sistem entegrasyonu

## 🏗️ Sistem Mimarisi

### Veritabanı Şeması
```sql
-- Products table
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    barcode TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    category_id INTEGER,
    supplier_id INTEGER,
    unit_price REAL,
    min_stock_level INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories (id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers (id)
);

-- Inventory table
CREATE TABLE inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 0,
    location TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products (id)
);

-- Transactions table
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    transaction_type TEXT NOT NULL, -- 'IN', 'OUT', 'AUDIT'
    quantity INTEGER NOT NULL,
    unit_price REAL,
    total_amount REAL,
    reference_number TEXT,
    notes TEXT,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products (id)
);

-- Categories table
CREATE TABLE categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Suppliers table
CREATE TABLE suppliers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    contact_person TEXT,
    email TEXT,
    phone TEXT,
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Sınıf Yapısı
```python
class InventoryManagementSystem:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.qr_scanner = QRBarcodeScanner()
        self.report_generator = ReportGenerator()
        self.gui = InventoryGUI(self)

class DatabaseManager:
    def __init__(self, db_path="inventory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        # Create tables if not exist
        pass
    
    def add_product(self, product_data):
        # Add new product to database
        pass
    
    def update_inventory(self, product_id, quantity, transaction_type):
        # Update inventory levels
        pass

class Product:
    def __init__(self, barcode, name, description, category_id, supplier_id, unit_price):
        self.barcode = barcode
        self.name = name
        self.description = description
        self.category_id = category_id
        self.supplier_id = supplier_id
        self.unit_price = unit_price

class InventoryTransaction:
    def __init__(self, product_id, transaction_type, quantity, unit_price, reference_number, notes):
        self.product_id = product_id
        self.transaction_type = transaction_type  # 'IN', 'OUT', 'AUDIT'
        self.quantity = quantity
        self.unit_price = unit_price
        self.reference_number = reference_number
        self.notes = notes
        self.timestamp = datetime.datetime.now()
```

## 📊 Sistem Modülleri

### 🔍 1. Scanning Module
```python
class QRBarcodeScanner:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()
        self.scan_history = []
    
    def scan_single_code(self, frame):
        # Single code scanning
        pass
    
    def scan_batch_codes(self, video_source):
        # Batch scanning from video
        pass
    
    def validate_code(self, code_data):
        # Code format validation
        pass
```

### 📦 2. Product Management Module
```python
class ProductManager:
    def __init__(self, db_manager):
        self.db = db_manager
    
    def add_product(self, product_data):
        # Add new product with auto-generated QR code
        pass
    
    def update_product(self, product_id, updates):
        # Update product information
        pass
    
    def search_products(self, query):
        # Search products by name, barcode, category
        pass
    
    def generate_product_qr(self, product_id):
        # Generate QR code for product
        pass
```

### 📈 3. Inventory Management Module
```python
class InventoryManager:
    def __init__(self, db_manager):
        self.db = db_manager
        self.alert_threshold = 0.1  # 10% of max stock
    
    def stock_in(self, product_id, quantity, unit_price, reference):
        # Process incoming stock
        pass
    
    def stock_out(self, product_id, quantity, reference):
        # Process outgoing stock
        pass
    
    def check_low_stock(self):
        # Check for low stock items
        pass
    
    def generate_stock_report(self):
        # Generate current stock report
        pass
```

### 📊 4. Report Generation Module
```python
class ReportGenerator:
    def __init__(self, db_manager):
        self.db = db_manager
    
    def generate_inventory_report(self, format='pdf'):
        # Generate inventory status report
        pass
    
    def generate_transaction_report(self, start_date, end_date):
        # Generate transaction history report
        pass
    
    def generate_low_stock_report(self):
        # Generate low stock alert report
        pass
    
    def export_to_excel(self, data, filename):
        # Export data to Excel format
        pass
```

## 🖥️ Kullanıcı Arayüzü

### Ana Menü
```
┌─────────────────────────────────────────┐
│           ENVANTER YÖNETİMİ             │
├─────────────────────────────────────────┤
│  📱 QR/Barkod Tarama                   │
│  📦 Ürün Yönetimi                      │
│  📈 Stok İşlemleri                     │
│  📊 Raporlar                           │
│  ⚙️  Ayarlar                           │
│  🔄 Yedekleme/Geri Yükleme            │
└─────────────────────────────────────────┘
```

### QR Tarama Ekranı
- **Live scanning** - canlı kamera görüntüsü
- **Scan results** - taranan kodların listesi
- **Product preview** - ürün bilgileri önizlemesi
- **Quick actions** - hızlı stok giriş/çıkış butonları

### Ürün Yönetimi Ekranı
- **Product list** - ürün listesi (filtrelenebilir)
- **Add/Edit forms** - ürün ekleme/düzenleme formları
- **Category management** - kategori yönetimi
- **Supplier management** - tedarikçi yönetimi

### Stok İşlemleri Ekranı
- **Stock in/out forms** - giriş/çıkış formları
- **Current stock levels** - mevcut stok seviyeleri
- **Low stock alerts** - düşük stok uyarıları
- **Transaction history** - işlem geçmişi

## 🧪 Test Senaryoları

### 📝 Test Case 1: Product Registration
1. **20 farklı ürün tanımla** (farklı kategorilerde)
2. **Her ürün için QR kod oluştur**
3. **Ürün bilgilerinin doğru kaydedildiğini kontrol et**
4. **QR kodları tarayarak ürünleri doğrula**

### 📝 Test Case 2: Stock Operations
1. **10 ürün için stok girişi yap** (farklı miktarlarda)
2. **5 ürün için stok çıkışı yap**
3. **Stok seviyelerinin doğru güncellendiğini kontrol et**
4. **Transaction log'un doğru tutulduğunu doğrula**

### 📝 Test Case 3: Low Stock Management
1. **Ürünler için minimum stok seviyesi belirle**
2. **Stok seviyesini minimumun altına düşür**
3. **Low stock alert'lerinin çalıştığını kontrol et**
4. **Alert system'inin responsive olduğunu doğrula**

### 📝 Test Case 4: Batch Scanning
1. **10 farklı QR kod hazırla** (printed)
2. **Batch scanning modunu kullanarak hepsini tara**
3. **Scanning accuracy'yi ölç** (recognition rate)
4. **Performance'ı değerlendir** (scan speed)

### 📝 Test Case 5: Report Generation
1. **1 haftalık işlem geçmişi oluştur** (100+ transaction)
2. **Inventory report oluştur** (PDF ve Excel formatında)
3. **Transaction report oluştur** (date range filtered)
4. **Report accuracy'yi manuel olarak doğrula**

### 📝 Test Case 6: Database Operations
1. **1000 ürün ve 5000 transaction ekle** (stress test)
2. **Database backup al**
3. **Database'i sıfırla ve backup'tan geri yükle**
4. **Data integrity'yi kontrol et**

## 💻 İmplementasyon Adımları

### 🏃‍♂️ Temel Sistem (2 saat)
1. **Database schema oluştur**
2. **Temel CRUD operations implement et**
3. **QR scanning functionality**
4. **Basit console interface**

### 🚀 Orta Seviye (3-4 saat)
1. **GUI interface oluştur** (Tkinter)
2. **Product management system**
3. **Inventory operations** (stock in/out)
4. **Basic reporting** (console output)

### 🔥 İleri Seviye (5 saat)
1. **Advanced GUI features**
2. **PDF/Excel report generation**
3. **Low stock alert system**
4. **Batch operations**
5. **Data validation ve error handling**

### 🚀 Uzman Seviye (Bonus)
1. **Multi-user support**
2. **REST API development**
3. **Mobile app integration**
4. **Cloud synchronization**

## 🎯 Değerlendirme Kriterleri

### ✅ Temel Kriterler (60 puan)
- [ ] **Database operations working** (15p)
- [ ] **QR scanning functional** (15p)
- [ ] **Product management** (10p)
- [ ] **Basic inventory operations** (10p)
- [ ] **GUI interface** (10p)

### 🌟 Gelişmiş Kriterler (30 puan)
- [ ] **Report generation** (10p)
- [ ] **Low stock alerts** (5p)
- [ ] **Batch operations** (5p)
- [ ] **Data validation** (5p)
- [ ] **Error handling** (5p)

### 🏆 Bonus Kriterler (35 puan ek)
- [ ] **Advanced analytics** (10p)
- [ ] **Excel import/export** (8p)
- [ ] **Multi-user support** (7p)
- [ ] **API development** (10p)

## 💡 İpuçları ve Trickler

### 🎯 Database Design
```python
# Tip: Connection pooling kullanın
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row  # Dict-like access
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# Usage
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
```

### ⚡ QR Code Generation
```python
# Tip: Consistent QR code generation
def generate_product_qr(product_id, product_name):
    qr_data = {
        'type': 'PRODUCT',
        'id': product_id,
        'name': product_name,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(json.dumps(qr_data))
    qr.make(fit=True)
    
    return qr.make_image(fill_color="black", back_color="white")
```

### 🎨 GUI Development
```python
# Tip: Reusable form components
class FormField:
    def __init__(self, parent, label, field_type='text', required=False):
        self.frame = ttk.Frame(parent)
        self.label = ttk.Label(self.frame, text=label)
        self.required = required
        
        if field_type == 'text':
            self.widget = ttk.Entry(self.frame)
        elif field_type == 'number':
            self.widget = ttk.Spinbox(self.frame, from_=0, to=999999)
        elif field_type == 'dropdown':
            self.widget = ttk.Combobox(self.frame)
        
        self.label.pack(side='left', padx=(0, 10))
        self.widget.pack(side='right', fill='x', expand=True)
        
    def get_value(self):
        return self.widget.get()
    
    def set_value(self, value):
        self.widget.delete(0, 'end')
        self.widget.insert(0, str(value))
```

### 📊 Report Generation
```python
# Tip: Template-based PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

def generate_inventory_pdf(inventory_data, filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    
    # Table data
    table_data = [['Product', 'Barcode', 'Quantity', 'Unit Price', 'Total Value']]
    for item in inventory_data:
        table_data.append([
            item['name'], 
            item['barcode'], 
            str(item['quantity']),
            f"${item['unit_price']:.2f}",
            f"${item['total_value']:.2f}"
        ])
    
    # Create table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#40466e'),
        ('TEXTCOLOR', (0, 0), (-1, 0), '#1a1a1a'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), '#f7f7f7'),
        ('GRID', (0, 0), (-1, -1), 1, '#CCCCCC')
    ]))
    
    elements.append(table)
    doc.build(elements)
```

## 🚨 Yaygın Hatalar ve Çözümleri

### ❌ Problem: Database Locking
**Çözüm**: Connection pooling ve proper transaction management
```python
# SQLite timeout ayarla
conn = sqlite3.connect('inventory.db', timeout=30.0)

# Transaction'ları minimize et
with get_db_connection() as conn:
    cursor = conn.cursor()
    # Multiple operations in single transaction
    cursor.execute("INSERT INTO ...")
    cursor.execute("UPDATE ...")
    # Auto-commit on exit
```

### ❌ Problem: QR Code Reading Failures
**Çözüm**: Multiple detection methods, image preprocessing
```python
def robust_qr_scan(frame):
    results = []
    
    # Method 1: Direct detection
    results.extend(cv2_qr_detect(frame))
    
    # Method 2: Preprocessed image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    results.extend(cv2_qr_detect(enhanced))
    
    # Method 3: Pyzbar (if available)
    if PYZBAR_AVAILABLE:
        results.extend(pyzbar_detect(frame))
    
    return remove_duplicates(results)
```

### ❌ Problem: Large Dataset Performance
**Çözüm**: Database indexing, pagination, caching
```sql
-- Database indexes
CREATE INDEX idx_products_barcode ON products(barcode);
CREATE INDEX idx_inventory_product_id ON inventory(product_id);
CREATE INDEX idx_transactions_product_id ON transactions(product_id);
CREATE INDEX idx_transactions_created_at ON transactions(created_at);
```

## 📚 Referans Kodlar

### Database Manager Template
```python
class DatabaseManager:
    def __init__(self, db_path="inventory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    barcode TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    category_id INTEGER,
                    supplier_id INTEGER,
                    unit_price REAL,
                    min_stock_level INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add more tables...
    
    def add_product(self, product_data):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO products (barcode, name, description, category_id, supplier_id, unit_price, min_stock_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                product_data['barcode'],
                product_data['name'],
                product_data['description'],
                product_data['category_id'],
                product_data['supplier_id'],
                product_data['unit_price'],
                product_data['min_stock_level']
            ))
            
            product_id = cursor.lastrowid
            
            # Initialize inventory record
            cursor.execute('''
                INSERT INTO inventory (product_id, quantity)
                VALUES (?, 0)
            ''', (product_id,))
            
            return product_id
```

### GUI Application Template
```python
class InventoryGUI:
    def __init__(self, inventory_system):
        self.system = inventory_system
        self.root = tk.Tk()
        self.root.title("Envanter Yönetim Sistemi")
        self.root.geometry("1200x800")
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main menu frame
        menu_frame = ttk.Frame(self.root)
        menu_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        # Menu buttons
        ttk.Button(menu_frame, text="QR/Barkod Tarama", 
                  command=self.open_scanning_window).pack(fill='x', pady=5)
        ttk.Button(menu_frame, text="Ürün Yönetimi",
                  command=self.open_product_management).pack(fill='x', pady=5)
        ttk.Button(menu_frame, text="Stok İşlemleri",
                  command=self.open_inventory_operations).pack(fill='x', pady=5)
        ttk.Button(menu_frame, text="Raporlar",
                  command=self.open_reports).pack(fill='x', pady=5)
        
        # Main content frame
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
    
    def open_scanning_window(self):
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Create scanning interface
        scanning_frame = ttk.LabelFrame(self.content_frame, text="QR/Barkod Tarama")
        scanning_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add scanning controls and camera view
        # Implementation here...
    
    def run(self):
        self.root.mainloop()
```

## 🎊 Tebrikler!

Bu alıştırmayı tamamladığınızda:
- ✅ **Enterprise-level system** geliştirme becerisi
- ✅ **Database design** ve **optimization** deneyimi
- ✅ **GUI application development** becerisi
- ✅ **System integration** ve **API development** deneyimi
- ✅ **Professional software architecture** tasarım becerisi

**🏆 Bu son alıştırmayla tüm Nesne Tespiti bölümünü tamamladınız!**

---
**⏰ Tahmini Süre**: 4-5 saat  
**🎯 Zorluk**: Uzman seviye  
**🏆 Kazanım**: Envanter yönetim sistemi geliştirme becerisi

## 🌟 Final Challenge

Tüm 3 alıştırmayı tamamladıysanız, **bonus final challenge**:

**🚀 Mega System Integration**: 3 sistemi birleştirerek **"Akıllı Tesis Yönetim Sistemi"** oluşturun:
- Hibrit tespit sistemi → **Obje tanıma**
- Güvenlik sistemi → **Personel takibi** 
- Envanter sistemi → **Ekipman yönetimi**

Bu mega projeyi tamamlarsanız **💎 Diamond Level** sertifikayı hak edeceksiniz!