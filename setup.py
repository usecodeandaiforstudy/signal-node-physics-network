from setuptools import setup, find_packages

setup(
    name="signal-node-physics-network",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9",
        "numpy>=1.20",
        "matplotlib>=3.4",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "pandas>=1.3",
    ],
    python_requires=">=3.8",
)
