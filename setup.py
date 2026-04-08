from setuptools import setup, find_packages

setup(
    name="md-MulT-EHR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "torch-geometric",
        "pyyaml",
        "pandas>=1.3.2",
        "scikit-learn>=0.24.2",
        "tqdm",
        "numpy",
        "matplotlib",
        "networkx>=2.6.3",
        "wandb",
        "plotly",
        "scipy",
        "pandarallel>=1.5.3",
        "mne>=1.0.3",
        "rdkit>=2022.03.4",
    ],
)
