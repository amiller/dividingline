from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("dividingline.dividingline_cy",
                       ["dividingline/dividingline_cy.pyx"])]

setup(name='DividingLine',
      version='0.1',
      author='Andrew Miller',
      email='amiller@cs.ucf.edu',
      packages=['dividingline'],
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
      install_requires=['distribute', 'cython', 'numpy', 'scipy'])
