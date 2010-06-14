#!/usr/bin/env python
from setuptools import Extension, setup
setup(name='lo',
      version='0.1',
      description='LinearOperators and Iterative algorithms',
      author='Nicolas Barbey',
      author_email='nicolas.barbey@cea.fr',
      install_requires = ['numpy>=1.3.0', 'scipy', ],
      packages=['lo'],
      )
