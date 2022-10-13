try:
    from setuptools import setup
    from setuptools import Extension
    from Cython.Build import cythonize
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("_occupancy_fingerprinter",["_occupancy_fingerprinter.pyx"])]

setup(
    name= '_occupancy_fingerprinter',
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = ext_modules,
    )
