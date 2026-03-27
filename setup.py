#!/usr/bin/env python3
"""
Setup script for the EchoPilot agent project.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="echopilot",
    version="1.0.0",
    author="EchoPilot Team",
    author_email="team@echopilot.ai",
    description="Minimal echocardiography ReAct agent built on EchoPilot tools",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/echopilot/echopilot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchvision[cuda]>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "echopilot=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "echopilot": ["configs/*.json", "assets/*"],
    },
)
