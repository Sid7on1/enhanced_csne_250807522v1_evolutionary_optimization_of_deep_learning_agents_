import setuptools
from setuptools import find_packages
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enhanced_cs_ne_2508_agent",
    version="1.0.0",
    author="XR Eye Tracking Team",
    author_email="dev@example.com",
    description="Evolutionary Optimization of Deep Learning Agents for XR Eye Tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/enhanced_cs_ne_2508_agent",  # Replace with your GitHub page
    project_urls={
        "Bug Reports": "https://github.com/example/enhanced_cs_ne_2508_agent/issues",
        "Funding": "https://donate.example.com",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch==1.12.0",
        "numpy==1.23.5",
        "pandas==1.5.2",
        "gym==0.21.0",  # Environment handling
        "matplotlib==3.6.0",  # For visualization
        "seaborn==0.11.2",  # Visualization
        "scipy==1.9.3",  # Scientific computing
        "tqdm==4.65.0",  # Progress bars
    ],
    entry_points={
        "console_scripts": [
            "enhanced_agent = enhanced_cs_ne_2508_agent.cli:main",
        ]
    },
)

print("Setup file created successfully. You can now build and install the package using 'pip install .' or 'python setup.py install'.")