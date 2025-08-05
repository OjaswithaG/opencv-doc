#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV Dokümantasyon Projesi - Setup Script
===========================================

Bu script, projeyi Python paketi olarak kurmanızı sağlar.

Kullanım:
    pip install -e .        # Development modunda kurulum
    pip install .           # Normal kurulum
    python setup.py install # Klasik kurulum

Yazan: Eren Terzi
"""

from setuptools import setup, find_packages
from pathlib import Path

# README dosyasını oku
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Requirements dosyasını oku
def read_requirements(filename):
    """Requirements dosyasından paket listesi okutur"""
    requirements = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Yorumları ve boş satırları atla
                if line and not line.startswith('#'):
                    # Platform-specific requirements'ları atla (basitlik için)
                    if ';' not in line:
                        requirements.append(line)
    except FileNotFoundError:
        print(f"Warning: {filename} dosyası bulunamadı!")
    
    return requirements

setup(
    # Temel bilgiler
    name="opencv-dokumantasyon",
    version="1.0.0",
    author="Eren Terzi",
    author_email="erenterzi@protonmail.com",
    description="OpenCV öğrenmek için kapsamlı Türkçe dokümantasyon ve örnekler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erent8/opencv-doc",
    
    # Paket yapılandırması
    packages=find_packages(),
    include_package_data=True,
    
    # Python sürüm gereksinimleri
    python_requires=">=3.7",
    
    # Bağımlılıklar
    install_requires=read_requirements("requirements-minimal.txt"),
    
    # Ek bağımlılık grupları
    extras_require={
        "full": read_requirements("requirements.txt"),
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.812",
        ],
        "ml": [
            "scikit-learn>=0.24.0",
            "tensorflow>=2.6.0",
            "torch>=1.9.0",
        ],
        "web": [
            "flask>=2.0.0",
            "streamlit>=0.80.0",
        ]
    },
    
    # Sınıflandırma
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: Turkish",
    ],
    
    # Anahtar kelimeler
    keywords="opencv, computer-vision, image-processing, tutorial, documentation, turkce, turkish",
    
    # Entry points (komut satırı araçları)
    entry_points={
        "console_scripts": [
            "opencv-test=utils.opencv_helpers:test_helpers",
        ],
    },
    
    # Paket verileri
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
        "assets": ["*.jpg", "*.png", "*.mp4"],
        "examples": ["*.jpg", "*.png"],
        "test-resimleri": ["*.jpg", "*.png"],
    },
    
    # Proje URL'leri
    project_urls={
        "Bug Reports": "https://github.com/erent8/opencv-doc/issues",
        "Source": "https://github.com/erent8/opencv-doc",
        "Documentation": "https://github.com/erent8/opencv-doc",
        "Tutorial": "https://github.com/erent8/opencv-doc",
    },
)