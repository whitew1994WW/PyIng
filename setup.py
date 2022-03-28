from setuptools import find_packages, setup
from setuptools.command.install import install
import sys
import os

# circleci.py version
VERSION = "0.1.7"

def readme():
    """print long description"""
    with open('README.rst') as f:
        return f.read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setup(
    name='PyIng',
    packages=find_packages(include=['PyIng']),
    version='0.1.0',
    description='Parses ingredient names into Name, Unit and Quantity',
    author='Will White',
    license='MIT',
    install_requires=["tflite-runtime>=2.5", "numpy"],
    url='https://github.com/whitew1994WW/PyIng',
    test_suite='tests',
    python_requires='>=3.7',
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
