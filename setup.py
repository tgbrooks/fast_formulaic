import numpy
import setuptools
from Cython.Build import cythonize

csc = setuptools.Extension(
    name="csc",
    sources=["src/fast_formulaic/csc.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"],
)

setuptools.setup(
    modules=["src/fast_formulaic"],
    ext_modules=cythonize([csc]),
)
