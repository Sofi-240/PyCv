from setuptools import setup, Extension
import numpy

setup(
    name="PyCv",
    version="0.2",
    packages=['pycv', 'pycv._lib', 'pycv._lib.core'],
    ext_modules=[
        Extension(
            "pycv._lib.core.ops",
            [
                "pycv/_lib/core/ops.c",
                "pycv/_lib/core/ops_support.c",
                "pycv/_lib/core/filters.c",
                "pycv/_lib/core/morphology.c"
            ],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        )
    ]
)