from setuptools import find_packages, setup
from setuptools.command.install import install
import sys
import os

# circleci.py version
VERSION = "0.1.11"

def readme():
    """print long description"""
    with open('README.rst') as f:
        return f.read()

setup(
    name='PyIng',
    packages=find_packages(include=['PyIng']),
    version=VERSION,
    description='Parses ingredient names into Name, Unit and Quantity',
    author='Will White',
    license='MIT',
    install_requires=["tflite-runtime>=2.5", "numpy"],
    url='https://github.com/whitew1994WW/PyIng',
    test_suite='tests',
    python_requires='>=3.7',
    long_description=readme()
)
