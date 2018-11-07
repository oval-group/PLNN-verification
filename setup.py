import sys
import glob

from os import path
from setuptools import setup, find_packages
from setuptools.extension import Extension

if sys.version_info < (3,6):
    sys.exit("Sorry, only Python >= 3.6 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='PLNN-verification',
    version='0.0.2',
    description='Verification of Piecewise Linear Neural Networks',
    author='Rudy Bunel',
    author_email='rudy@robots.ox.ac.uk',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=['sh', 'numpy', 'torch==0.4.0', 'scipy'],
    extras_require={
        'tests': ['mypy', 'flake8'],
        'dev': ['ipython', 'ipdb']
    },
    scripts=glob.glob('tools/*.py'),
)
