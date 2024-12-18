from setuptools import setup, find_packages

setup(
    name="videoautoencoder",
    version="0.1.0",
    description="A video autoencoder library for video compression and reconstruction on low memory graphics cards",
    author="Rivera.ai/Fredy Rivera",
    author_email="riveraaai200678@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "av",
        "tqdm",
        "torchmetrics",
    ],
    python_requires=">=3.8",
)