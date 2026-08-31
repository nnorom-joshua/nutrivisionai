"""
setup.py — NutriVision AI package installer
"""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="nutrivision-ai",
    version="1.0.0",
    author="Wonder Bassey Paul",
    author_email="wonder@schooldevtechnologies.com",
    description="Real-Time AI-Powered Automated Nutritional Analysis System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheTechApostle/nutrivision-ai",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "streamlit>=1.28.0",
        "ultralytics>=8.0.0",
        "optuna>=3.3.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pillow>=10.0.0",
        "plotly>=5.15.0",
        "albumentations>=1.3.0",
        "timm>=0.9.0",
        "tqdm>=4.65.0",
        "opencv-python-headless>=4.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "nutrivision-train=train:main",
            "nutrivision-app=streamlit_app.app:main",
        ]
    },
    include_package_data=True,
    package_data={
        "": ["*.toml", "*.json", "*.csv"],
    },
)
