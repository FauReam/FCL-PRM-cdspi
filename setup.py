"""Setup script for FCL-PRM package."""

from setuptools import find_packages, setup

setup(
    name="fclprm",
    version="0.1.0",
    description="Federated Continual Process Reward Model",
    author="",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "scikit-learn>=1.5.0",
        "scipy>=1.13.0",
    ],
    extras_require={
        "privacy": ["opacus>=1.5.0"],
        "dev": ["pytest>=8.2.0", "black>=24.4.0", "isort>=5.13.0", "mypy>=1.10.0"],
        "track": ["wandb>=0.17.0"],
    },
)
