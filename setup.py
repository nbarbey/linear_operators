#!/usr/bin/env python
#from setuptools import Extension, setup
from distutils.core import setup
setup(name='linear_operators',
      version='0.2.0',
      description='LinearOperators and Iterative algorithms',
      author='Nicolas Barbey',
      author_email='nicolas.barbey@cea.fr',
      requires = ['numpy', 'scipy', ],
      packages=['linear_operators', 'linear_operators.wrappers', 'linear_operators.iterative'],
      )
