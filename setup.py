from setuptools import setup, find_packages

setup(
    name="transformer-arch-demo",
    version="1.0.0",
    description="A comprehensive demonstration of transformer architectures and their use cases",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0",
        "jupyter>=1.0.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "tokenizers>=0.13.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
