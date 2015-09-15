try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from Cython.Distutils import build_ext

import numpy

ext = Extension("utils",
                sources = ['utils.pyx',
                           'discrete_gamma.c',
                           'lnl_calc.c'],
                include_dirs = [numpy.get_include()],
               )

setup(cmdclass={'build_ext':build_ext},
      name="utilities",
      author='Kevin Gori',
      author_email='kgori@ebi.ac.uk',
      description='Wrapper of utility c functions',
      url='',
      version="0.0.1",
      ext_modules = [ext],
      install_requires = ['cython'],
     )
