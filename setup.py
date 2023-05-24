import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.mf file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

setup(
    name="acamda", # Replace with your own username
    version="0.0.1",
    description="The acamda is a tool for robust and efficient data augmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires = [
        "pytorch-lightning==1.2.7",
        "torch==1.8.1",
        "gym==1.6.0",
        "tqdm==4.60.0",
        "keras==2.3.1",
        "pandas==1.1.5",
        "tensorflow==2.1.0",
        "matplotlib==3.4.1",
        "torchvision",
        "ipdb"
    ],
    tests_require=[
        "pytest"
    ],
)