from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="data-science-prefect",
    version="1.0.0",
    author="Student",
    author_email="student@example.com",
    description="Cloud-based Data Science/ML Application with Prefect Orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/data-science-prefect",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "azure-storage-blob>=12.17.0",
            "google-cloud-storage>=2.10.0",
        ],
        "advanced-ml": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ds-pipeline=workflows.training_flow:main",
            "ds-api=api.api_server:main",
            "ds-deploy=scripts.deploy:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
)