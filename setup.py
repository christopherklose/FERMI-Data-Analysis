from setuptools import setup, find_packages

setup(
    name='process_FERMI',
    version='2023.10-1',
    packages=find_packages(include=['process_FERMI', 'process_FERMI.*'])
)
