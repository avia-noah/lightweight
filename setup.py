from setuptools import setup, find_packages

setup(
    name="lightweight",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "tqdm",
        "numpy",
        "pillow",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
)
