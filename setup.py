from setuptools import setup

from setuptools import setup, Extension

module = Extension ('PyRAD_SRAD', sources=['PyRAD_SRAD.pyx'])

setup(
    name='cythonTest',
    version='1.0',
    author='jetbrains',
    ext_modules=[module]
)
