from setuptools import setup, find_packages

setup(
    name="md-MulT-EHR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "pyyaml",
        "pandas",
        "scikit-learn",
        "tqdm",
        "numpy",
        "matplotlib",
        "networkx",
        "wandb",
        "plotly",
        "scipy",
        "pandarallel",
    ],
)
