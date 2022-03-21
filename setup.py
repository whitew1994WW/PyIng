from setuptools import find_packages, setup

setup(
    name='PyIng',
    packages=find_packages(include=['PyIng']),
    version='0.1.0',
    description='Parses ingredient names into Name, Unit and Quantity',
    author='Will White',
    license='MIT',
    install_requires=[],
    setup_requires=[],
    tests_require=[],
    test_suite='tests',
)