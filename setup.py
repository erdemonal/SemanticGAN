from setuptools import setup, find_packages

setup(
    name="semantic-gan",
    version="0.1.1",
    description="A library for Knowledge Graph Completion using Wasserstein GANs",
    author="Erdem Önal",
    url="https://github.com/erdemonal/SemanticGAN",
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas",
        "tqdm",
        "scikit-learn",
        "plotly",
        "matplotlib>=3.7.0",
    ],
    python_requires=">=3.7",
)
