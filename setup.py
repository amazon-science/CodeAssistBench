"""
CodeAssistBench packaging setup.

For more details on how to operate this file, check
https://w.amazon.com/index.php/Python/Brazil
"""

import os

from setuptools import setup

# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append(
        (os.path.relpath(root, "configuration"), [os.path.join(root, f) for f in files])
    )

setup(
    name="codeassistbench",
    version="1.0.0",
    author="Amazon Science Team",
    description="A comprehensive Python package for CodeAssistBench (CAB)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=["cab_evaluation", "cab_evaluation.core", "cab_evaluation.agents", 
              "cab_evaluation.workflows", "cab_evaluation.utils", "cab_evaluation.prompts"],
    package_dir={"": "src"},
    install_requires=[
        "boto3>=1.34.0",
        "botocore>=1.34.0", 
        "openai>=1.3.0",
        "docker>=6.1.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "rank-bm25>=0.2.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0", 
            "mypy>=1.5.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0"
        ]
    },
    python_requires=">=3.8",
    data_files=data_files,
    include_package_data=True,
    zip_safe=False
)
