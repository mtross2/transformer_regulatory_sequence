#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:35:58 2024

@author: mtross
"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name='gene_expression_prediction',
    version='1.0',
    packages=find_packages(),
    author='Michael Tross',
    author_email='mikeytross16@gmail.com',
    description='A package for gene expression prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mtross2/transformer_regulatory_sequence',
    install_requires=requirements,
)
