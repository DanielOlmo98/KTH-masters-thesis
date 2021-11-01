from setuptools import setup, Extension

module1 = Extension('HMF', sources=['noise_filtering\HMF.pyx'])
module2 = Extension('PyRAD_SRAD', sources=['noise_filtering\SRAD\PyRAD_SRAD.pyx'])

setup(
    name='cythonTest',
    version='1.0',
    author='Daniel Olmo',
    ext_modules=[module1, module2]
)
