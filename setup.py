# setup.py
from setuptools import setup, find_packages

setup(
    name="project-aayu",  # Use lowercase and hyphens for PyPI best practice
    version="0.1.0",
    description="A modular ML framework for health risk prediction",
    author="Ruchira Lakshan",
    author_email="ruchiralakshanm@gmail.com",
    url="https://github.com/ruchiralak/health_predictor_framework",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
        "pyyaml",
        "streamlit"
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
          'project_aayu=app:main',     # Optional GUI entry via command
        ],
    },
)
