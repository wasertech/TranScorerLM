#!/usr/bin/env python

import os
from setuptools import setup, find_packages
import scorer

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    long_description = f.read()


setup(
    name='transcorer',
    author='Danny Waser',
    version=scorer.__version__,
    license='LICENSE',
    url='https://github.com/wasertech/TranScorerLM',
    description='Transformer as Scorer (Language Model) for STT accoustic models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages('.'),
    python_requires='>=3.8,<3.11',
    install_requires = [
        'transformers>=4.29.0',
        #'torch~=1.13.1',
        'datasets>=1.18.0',
    ],
    entry_points={
        'console_scripts': [
            'trainscorer = scorer.train:train',
            'transcorer = scorer.main:main',
        ]
    },
)
