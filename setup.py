try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, Extension

from Cython.Distutils import build_ext

import numpy

ext = Extension("phylo_utils.likcalc",
                sources = ['extensions/likcalc.pyx',
                           'extensions/discrete_gamma.c'],
                include_dirs = [numpy.get_include()],
               )

setup(cmdclass={'build_ext':build_ext},
      name="phylo_utils",
      author='Kevin Gori',
      author_email='kgori@ebi.ac.uk',
      description='Phylogenetics calculations in python',
      url='',
      version="0.0.3",
      ext_modules = [ext],
      install_requires = ['cython', 'numpy', 'dendropy'],
      packages=find_packages(),
     )
