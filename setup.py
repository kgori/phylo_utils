try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, Extension

from Cython.Distutils import build_ext

import numpy
import platform, re, subprocess

def is_clang(bin):
    proc = subprocess.Popen([bin, '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    output = str(b'\n'.join([stdout, stderr]).decode('ascii', 'ignore'))
    print(output)
    return not re.search(r'clang', output) is None

class my_build_ext(build_ext):
    def build_extensions(self):
        binary = self.compiler.compiler[0]
        if is_clang(binary):
            for e in self.extensions:
                if platform.system() == 'Darwin':
                    e.extra_compile_args.append('-mmacosx-version-min=10.7')
                    e.extra_link_args.append('-mmacosx-version-min=10.7')
                # for list_ in (e.extra_compile_args, e.extra_link_args):
                #     list_.remove('-fopenmp')
                #     list_.append('-openmp')
        build_ext.build_extensions(self)

# extra_compile_args=['-fopenmp'],
# extra_link_args=['-fopenmp'])
ext_modules = [
    Extension("phylo_utils.discrete_gamma",
              sources = ['src/c_discrete_gamma.c',
                         'src/discrete_gamma.pyx'],
              define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
              include_dirs = [numpy.get_include()]),
    Extension("phylo_utils.optimisation",
              sources = ['src/optimisation.pyx'],
              define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
              include_dirs = [numpy.get_include()]),
    Extension("phylo_utils.simulation",
              sources = ['src/simulation.pyx'],
              define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
              include_dirs = [numpy.get_include()])]


setup(cmdclass={'build_ext': my_build_ext},
      name="phylo_utils",
      author='Kevin Gori',
      author_email='kcg25@cam.ac.uk',
      description='Phylogenetics calculations in python',
      url='',
      version="0.9.9000",
      ext_modules = ext_modules,
      install_requires = ['cython', 'numpy', 'scipy', 'dendropy', 'numba'],
      packages=find_packages())
