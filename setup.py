"""Waveguide photonic design library – package configuration."""

from setuptools import setup, find_packages

setup(
    name="waveguide",
    version="0.1.0",
    description=(
        "Silicon photonics design library: SM-to-MM adiabatic tapers and "
        "low-loss asymmetric MZI on the ANT NanoSOI platform."
    ),
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
