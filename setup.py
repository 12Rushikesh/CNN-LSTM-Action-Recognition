from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

ext_modules = [
    Extension(
        name="kalmar_fsm_cython",
        sources=["cython.pyx"],
        language="c++",
    )
]

setup(
    name="kalmar_fsm_cython",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"}
    ),
)
