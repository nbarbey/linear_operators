#!/usr/bin/env python
#from setuptools import Extension, setup
from distutils.core import setup
setup(name='lo',
      version='0.2.0',
      description='LinearOperators and Iterative algorithms',
      author='Nicolas Barbey',
      author_email='nicolas.barbey@cea.fr',
      requires = ['numpy', 'scipy', ],
      packages=['lo', 'lo.wrappers', 'lo.iterative'],
      )
